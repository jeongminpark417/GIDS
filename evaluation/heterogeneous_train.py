import sklearn.metrics

import dgl
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np
from models import *
from mlperf_model import RGNN
from dataloader import IGB260MDGLDataset, OGBDGLDataset
from dataloader import IGBHeteroDGLDataset, IGBHeteroDGLDatasetMassive, OGBHeteroDGLDatasetMassive

import csv 
import argparse, datetime
import warnings

import torch.cuda.nvtx as t_nvtx
import nvtx
import threading
import gc

import GIDS
from GIDS import GIDS_DGLDataLoader

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

torch.manual_seed(0)
dgl.seed(0)
warnings.filterwarnings("ignore")


@nvtx.annotate("fetch_data_chunk()", color="blue")
def fetch_data_chunk(test, out_t, page_size, stream_id):
    test.fetch_from_backing_memory_chunk(out_t.data_ptr(), page_size, stream_id)


def print_times(transfer_time, train_time, e2e_time):
    print("transfer time: ", transfer_time)
    print("train time: ", train_time)
    print("e2e time: ", e2e_time)


def track_acc_GIDS(g, category, args, device, dim , label_array=None, key_offset=None):

    GIDS_Loader = None
    GIDS_Loader = GIDS.GIDS(
        page_size = args.page_size,
        off = args.offset,
        num_ele = args.num_ele,
        num_ssd = args.num_ssd,
        cache_size = args.cache_size,
        window_buffer = args.window_buffer,
        wb_size = args.wb_size,
        accumulator_flag = args.accumulator,
        cache_dim = args.cache_dim,
        #ssd_list=[5],
        heterograph = True,
        heterograph_map = key_offset
    )
    
    dim = args.emb_size

    if(args.accumulator):
        GIDS_Loader.set_required_storage_access(args.bw, args.l_ssd, args.l_system, args.num_ssd, args.peak_percent)


    if(args.cpu_buffer):
        num_nodes = g.number_of_nodes()
        num_pinned_nodes = int(num_nodes * args.cpu_buffer_percent)
        GIDS_Loader.cpu_backing_buffer(dim, num_pinned_nodes)
        pr_ten = torch.load(args.pin_file)
        GIDS_Loader.set_cpu_buffer(pr_ten, num_pinned_nodes)


    sampler = dgl.dataloading.NeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])


    # g.ndata['features'] = g.ndata['feat']
    # g.ndata['labels'] = g.ndata['label']

    train_nid = torch.nonzero(g.nodes[category].data['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.nodes[category].data['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.nodes[category].data['test_mask'], as_tuple=True)[0]
    
    in_feats = dim

    if args.model_type == 'rgcn':
        model = RGCN(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rsage':
        model = RSAGE(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rgat':
        # model = RGAT(g.etypes, in_feats, args.hidden_channels,
        #     args.num_classes, args.num_layers, 0.2, args.num_heads).to(device)

         model = RGNN(g.etypes,
               in_feats,
               args.hidden_channels,
               args.num_classes,
               num_layers=args.num_layers,
               dropout=0.2,
               model='rgat',
               heads=args.num_heads,
               node_type='paper').to(device)


    #train_dataloader = dgl.dataloading.DataLoader(
    train_dataloader =  GIDS_DGLDataLoader(
        g,
        {category: train_nid},
        sampler,
        args.batch_size,
        dim,
        GIDS_Loader,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_alternate_streams=False
    )

    val_dataloader = dgl.dataloading.DataLoader(
        g, {category: val_nid}, sampler,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=args.num_workers)


    test_dataloader =  GIDS_DGLDataLoader(
        g,
        {category: test_nid},
        sampler,
        args.batch_size,
        dim,
        GIDS_Loader,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_alternate_streams=False
    )


    # test_dataloader = dgl.dataloading.DataLoader(
    #     g, {category: test_nid}, sampler,
    #     batch_size=args.batch_size,
    #     shuffle=True, drop_last=False,
    #     num_workers=args.num_workers)


    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate
        #, weight_decay=args.decay
        )

    warm_up_iter = 35000
    # Setup is Done
    for epoch in tqdm.tqdm(range(args.epochs)):
        epoch_start = time.time()
        epoch_loss = 0
        train_acc = 0
        model.train()

        batch_input_time = 0
        train_time = 0
        transfer_time = 0
        e2e_time = 0
        e2e_time_start = time.time()

        for step, (input_nodes, seeds, blocks, ret) in enumerate(train_dataloader):

    
            # p = blocks[0].srcdata[dgl.NID]['paper'].cpu()
            # a = blocks[0].srcdata[dgl.NID]['author'].cpu()
            # print(f"paper node: {p} author node:{a} emb: {ret}")
            # p_orig_feat = g.ndata['feat']['paper'][p]
            # print(f"paper feat: {p_orig_feat}")
           
            if(step == warm_up_iter):
                print("warp up done")
                train_dataloader.print_stats()
                train_dataloader.print_timer()
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0
                e2e_time_start = time.time()
        
            
            # Features are fetched by the baseline GIDS dataloader in ret

            batch_inputs = ret
            transfer_start = time.time() 

            batch_labels = (blocks[-1].dstdata['label']['paper']).to(device) 
           
            blocks = [block.int().to(device) for block in blocks]
            transfer_time = transfer_time +  time.time()  - transfer_start
 
            # Model Training Stage
            train_start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()                  
            train_time = train_time + time.time() - train_start
          
            if(step == warm_up_iter + 100):
                print("Performance for 100 iteration after 1000 iteration")
                e2e_time += time.time() - e2e_time_start 
                train_dataloader.print_stats()
                train_dataloader.print_timer()
                print_times(transfer_time, train_time, e2e_time)
             
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0
                
                #Just testing 100 iterations remove the next line if you do not want to halt
                #break
         

            if (step % 1000 == 0):
                train_acc =sklearn.metrics.accuracy_score(batch_labels.cpu().numpy(),
                  batch_pred.argmax(1).detach().cpu().numpy())*100
                print(f"Step {step}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.2f}%")

            # if(step == 5000):
            #     break




       
    model.eval()
    predictions = []
    labels = []
    counter = 0
    eval_acc = 0
    with torch.no_grad():
        #for _, _, blocks in test_dataloader:
        for step, (input_nodes, seeds, blocks, ret) in enumerate(test_dataloader):

            
            # blocks = [block.to(device) for block in blocks]
            # inputs = blocks[0].srcdata['feat']
     
            batch_labels = (blocks[-1].dstdata['label']['paper']).to(device) 
            batch_inputs = ret

            blocks = [block.int().to(device) for block in blocks]
 

            if(args.data == 'IGB'):
                labels.append(blocks[-1].dstdata['label']['paper'].cpu().numpy())
            elif(args.data == 'OGB'):
                out_label = torch.index_select(label_array, 0, b[1]).flatten()
                labels.append(out_label.numpy())
            #predict = model(blocks, inputs).argmax(1).cpu().numpy()
            predict = model(blocks, batch_inputs)
            #predictions.append(predict)

            train_acc = sklearn.metrics.accuracy_score(batch_labels.cpu().numpy(),
                  predict.argmax(1).detach().cpu().numpy())*100
            eval_acc += train_acc
            if(counter % 1000 == 0):
                print(f"Step {step}, Eval Acc: {train_acc:.2f}%")

            # if(counter == 5000):
            #     break

            counter += 1
    

        # predictions = np.concatenate(predictions)
        # labels = np.concatenate(labels)
        # test_acc = sklearn.metrics.accuracy_score(labels, predictions)*100

        test_acc = eval_acc/counter
    print("Test Acc {:.2f}%".format(test_acc))




    
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['tiny', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983, 172, 348,349, 350, 153, 152], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--data', type=str, default='IGB')
    parser.add_argument('--emb_size', type=int, default=1024)
    
    # Model
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['rgat', 'rsage', 'rgcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Model parameters 
    parser.add_argument('--fan_out', type=str, default='10,15')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=2)

    #GIDS parameter
    parser.add_argument('--GIDS', action='store_true', help='Enable GIDS Dataloader')
    parser.add_argument('--num_ssd', type=int, default=1)
    parser.add_argument('--cache_size', type=int, default=8)
    parser.add_argument('--uva', type=int, default=0)
    parser.add_argument('--uva_graph', type=int, default=0)
    parser.add_argument('--wb_size', type=int, default=6)

    parser.add_argument('--device', type=int, default=0)

    #GIDS Optimization
    parser.add_argument('--accumulator', action='store_true', help='Enable Storage Access Accmulator')
    parser.add_argument('--bw', type=float, default=5.8, help='SSD peak bandwidth in GB/s')
    parser.add_argument('--l_ssd', type=float, default=11.0, help='SSD latency in microseconds')
    parser.add_argument('--l_system', type=float, default=20.0, help='System latency in microseconds')
    parser.add_argument('--peak_percent', type=float, default=0.95)

    parser.add_argument('--num_iter', type=int, default=1)

    parser.add_argument('--cpu_buffer', action='store_true', help='Enable CPU Feature Buffer')
    parser.add_argument('--cpu_buffer_percent', type=float, default=0.2, help='CPU feature buffer size (0.1 for 10%)')
    parser.add_argument('--pin_file', type=str, default="/mnt/nvme16/pr_full.pt", 
        help='Pytorch Tensor File for the list of nodes that will be pinned in the CPU feature buffer')

    parser.add_argument('--window_buffer', action='store_true', help='Enable Window Buffering')



    #GPU Software Cache Parameters
    parser.add_argument('--page_size', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0, help='Offset for the feature data stored in the SSD') 
    parser.add_argument('--num_ele', type=int, default=100, help='Number of elements in the dataset (Total Size / sizeof(Type)') 
    parser.add_argument('--cache_dim', type=int, default=1024) #CHECK


    args = parser.parse_args()
    print("GIDS DataLoader Setting")
    print("GIDS: ", args.GIDS)
    print("CPU Feature Buffer: ", args.cpu_buffer)
    print("Window Buffering: ", args.window_buffer)
    print("Storage Access Accumulator: ", args.accumulator)

    labels = None
    key_offset = None
    dim = 1024
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    if(args.data == 'IGB'):
        dim = 1024
        print("Dataset: IGB")
        if(args.dataset_size == 'full' or args.dataset_size == 'large'):
            dataset = IGBHeteroDGLDatasetMassive(args)
            # User need to fill this out for their dataset based how it is stored in SSD
            if(args.dataset_size == 'full'):
                key_offset = {
                    'paper' : 0,
                    'author' : 269346174,
                    'fos' : 546567057,
                    'institute' : 547280017,
                    'journal' : 546593975,
                    'conference' : 546643027
                }
            else:
                key_offset = {
                    'paper' : 0,
                    'author' : 100000000,
                    'fos' : 100000000 + 116959896,
                    'institute' : 100000000 + 116959896 + 649707,
                    'journal' : 100000000 + 116959896 + 649707 + 26524,
                    'conference' : 100000000 + 116959896 + 649707 + 26524 + 48820
                }

        else:
            dataset = IGBHeteroDGLDataset(args)
            if(args.dataset_size == 'small'):
                key_offset = {
                    'paper' : 0,
                    'author' : 1000000,
                    'fos' : 1000000 + 192606,
                    'institute' : 1000000 + 192606 + 190449,
                    # 'journal' : 1000000 + 192606 + 190449 + 14751,
                    # 'conference' : 1000000 + 192606 + 190449 + 14751 + 15277
                }
            elif(args.dataset_size == 'medium'):
                key_offset = {
                    'paper' : 0,
                    'author' : 10000000,
                    'fos' : 10000000 + 15544654,
                    'institute' : 10000000 + 15544654 + 415054,
                    # 'journal' : 10000000 + 15544654 + 415054 + 23256,
                    # 'conference' : 10000000 + 15544654 + 415054 + 23256 + 37565
                }
            elif(args.dataset_size == 'tiny'):
                key_offset = {
                    'paper' : 0,
                    'author' : 100000,
                    'fos' : 100000 + 357041,
                    'institute' : 100000 + 357041 + 84220
                }
            else:
                key_offset = None
                print("key_offset is not set")
                exit()

        g = dataset[0]
        g = g.formats('csc')
    elif(args.data == "OGB"):
        print("Dataset: OGB")
        dataset = OGBHeteroDGLDatasetMassive(args)
        g = dataset[0]
        g = g.formats('csc')
    else:
        g=None
        dataset=None
    
    # nt = g.ntypes

    # for t in nt:
    #     num_t = g.num_nodes(t)
    #     print("type: ", t, " num: ", num_t)


    category = g.predict
    print(f"GIDS trainign start key pffset: {key_offset}")
    track_acc_GIDS(g, category, args, device, dim, labels, key_offset)
    #track_acc(g, args, device, labels)




