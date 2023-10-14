import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np
from models import *
from dataloader import IGB260MDGLDataset, OGBDGLDataset
import csv 
import warnings

import torch.cuda.nvtx as t_nvtx
import nvtx
import threading
import gc
import GIDS
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

torch.manual_seed(0)
dgl.seed(0)
warnings.filterwarnings("ignore")


@nvtx.annotate("fetch_data_chunk()", color="blue")
def fetch_data_chunk(test, out_t, page_size, stream_id):
    test.fetch_from_backing_memory_chunk(out_t.data_ptr(), page_size, stream_id)

def track_acc(g, args, device, label_array=None):

    bam_flag = False
    pin_flag = False
    cpu_cache_flag = False
    uva_flag = False
    wb_flag = False
    uva_graph = False
    if(args.bam == 1):
        bam_flag = True
    if(args.pin == 1):
        pin_flag = True
    if(args.cpu_cache == 1):
        cpu_cache_flag = True
    if(args.uva == 1):
        uva_flag = True
    if(args.window == 1):
        wb_flag = True
    if(args.uva_graph == 1):
        uva_graph = True

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [int(fanout) for fanout in args.fan_out.split(',')])
    dim = 1024
    if(args.data == 'IGB'):
        g.ndata['features'] = g.ndata['feat']
        g.ndata['labels'] = g.ndata['label']

        train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
        val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
        test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
        in_feats = g.ndata['features'].shape[1]
        dim = 1024

    elif(args.data == 'OGB'):
        g.ndata['features'] = g.ndata['feat']
        g.ndata['labels'] = g.ndata['label']

        train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
        val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
        test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
        in_feats = g.ndata['features'].shape[1]
        dim = 128

    else:
        train_nid = None
        val_nid = None
        test_nid = None
        in_feats = None
    train_dataloader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_uva=uva_flag,
        use_uva_graph=uva_graph,
        bam=bam_flag,
        feature_dim=dim,
        use_prefetch_thread=False,
        pin_prefetcher=False,
        window_buffer=wb_flag,
        window_buffer_size=args.wb_size,
        use_alternate_streams=False
        )
    if(bam_flag):
        if(args.data == "IGB"):
            train_dataloader.bam_init(0,4096,1024, 270*1000*1000*1024, args.cache_size, args.num_ssd, args.wb_queue_size)
        elif(args.data == "OGB"):
            train_dataloader.bam_init(0,128*4,128, 111059956 * 128 * 2,  args.cache_size, args.num_ssd)

    if(pin_flag):
        with open('/mnt/nvme14/IGB260M_bam/full_node_ranking.npy', 'rb') as f:
            pin_idx_f = np.load(f, allow_pickle=True)
            pin_size = int(args.pin_size * 1024 * 1024 / 4)
            pin_idx = pin_idx_f[0:pin_size]
            pin_idx = torch.tensor(pin_idx, dtype=torch.long)
            pin_idx = pin_idx.to(device)
            train_dataloader.pin_pages(pin_idx, 1024)
            train_dataloader.print_stats()


    val_dataloader = dgl.dataloading.DataLoader(
        g, val_nid, sampler,
        #device=device,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = dgl.dataloading.DataLoader(
        g, test_nid, sampler,
        #device=device,
        batch_size=args.batch_size,
        shuffle=True, drop_last=False,
        num_workers=args.num_workers)
    

    if args.model_type == 'gcn':
        model = GCN(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'sage':
        model = SAGE(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'gat':
        model = GAT(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers, args.num_heads).to(device)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    print("Model: ", model)
    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), 
        lr=args.learning_rate, weight_decay=args.decay)

    # Training loop
    best_accuracy = 0
    print("training start")
    print(torch.cuda.memory_allocated() / (1000*1000*1000))
    print(torch.cuda.memory_reserved() /  (1000*1000*1000))


    if(bam_flag and  wb_flag):
        print("GIDS evict tensor create")
        GIDS_loader = train_dataloader.get_GIDS()
        GIDS_loader.create_evict_tensors(args.batch_size * 10 * 5 * 5, dim)
        print("GIDS evict tensor create done")
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())

    training_start = time.time()
    for epoch in tqdm.tqdm(range(args.epochs)):
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        
        epoch_loss = 0
        gpu_mem_alloc = 0
        train_acc = 0
        epoch_start = time.time()
        idx = 0
        model.train()

        batch_input_time = 0
        train_time = 0
        transfer_time = 0
        e2e_time = 0
        e2e_time_start = time.time()
        for step, (input_nodes, seeds, blocks, ret) in enumerate(train_dataloader):
#            print("idx: ", idx)
#            print("idx: ", idx)
#            print(torch.cuda.memory_allocated() / (1000*1000*1000))
#            print(torch.cuda.memory_reserved() /  (1000*1000*1000))


            
            if(idx == 200):
                print("warp up done")
                e2e_time += time.time() - e2e_time_start
                if(bam_flag):
                    train_dataloader.print_stats()
                train_dataloader.print_timer()
                if(bam_flag == False):
                    print("feature aggreagation time: ", batch_input_time)
                print("transfer time: ", transfer_time)
                print("train time: ", train_time)
                print("e2e time: ", e2e_time)
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0
                e2e_time_start = time.time()
            idx += 1
            
            batch_input_start = time.time()
            if(bam_flag): 
                batch_inputs = ret
            else:
                batch_inputs = blocks[0].srcdata['feat']
           
            if(args.data == 'IGB'):
                batch_labels = blocks[-1].dstdata['labels']
            
            else:
                batch_labels = blocks[-1].dstdata['labels']
         

            agg_time = (time.time() - batch_input_start)
            batch_input_time += agg_time
            #print("batch input: ", batch_inputs) 

            transfer_start = time.time() 
            blocks = [block.int().to(device) for block in blocks]
            #if(bam_flag == False):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            transfer_time = transfer_time +  time.time()  -transfer_start
             
            
            if(idx == 1):
                print("batch len: ", len(batch_inputs))

            #move it to next tensor
            #fetch_data_chunk(GIDS_loader,prefetch_tensor,prefetch_tensor_size_bytes,0)
            
            if(bam_flag and wb_flag):
            #    GIDS_loader.prefetch_from_victim_queue(0)
                train_dataloader.set_wb_counter()
 
            train_start = time.time()

            with nvtx.annotate("train", color="yellow"):
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach()
            torch.cuda.synchronize()
            train_time = train_time + time.time() - train_start
            #train_acc += (sklearn.metrics.accuracy_score(batch_labels.cpu().numpy(), 
           #     batch_pred.argmax(1).detach().cpu().numpy())*100)

           
            del ret
            if(idx == 300):
                print("Performance for 100 iteration after 1000 iteration")
                e2e_time += time.time() - e2e_time_start 
                if(bam_flag):
                    train_dataloader.print_stats()
                train_dataloader.print_timer()
                if(bam_flag == False):
                    print("feature aggregation time: ", batch_input_time)
                print("transfer time: ", transfer_time)
                print("train time: ", train_time)
                print("e2e time: ", e2e_time)
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0
                return None
            #print("idx: ",idx)
            #if(bam_flag):
            #    train_dataloader.print_stats()
            #train_dataloader.print_timer()

        train_acc /= idx
        gpu_mem_alloc /= idx
        if(bam_flag):
            train_dataloader.print_stats()

        train_dataloader.print_timer()
        tqdm.tqdm.write(
                "Epoch {:03d} | Aggregation Time {:.4f} | Transfer Time {:.4f} | Train Time {:.4f} | GPU {:.1f} MB".format(
                    epoch,
                    batch_input_time,
                    transfer_time,
                    train_time,
                    gpu_mem_alloc
                )
        )

    
    
def test_PVP(args):
    test = GIDS.GIDS(4096,0,1024, 300*1000*1000*1024, cache_size=args.cache_size, wb_size=args.wb_size, wb_queue_size=args.wb_queue_size,num_ssd=args.num_ssd)
    test.init_cpu_meta()

    index_arr = []
    with open('/mnt/nvme14/IGB260M_bam/large_index.npy', 'rb') as f:
        index_arr.append( np.load(f,allow_pickle=True))

    c = 0
    for index_batch in index_arr:
        print("epoc: ", c)
        c= c+1
        feature_time =0
        index_num=0
        wb = []
        #c_index = torch.tensor(index, dtype=torch.long)
        print("index batch: ", len(index_batch))
        for i in range(args.wb_size + 101):

            index = index_batch[i]
            temp =[]
            gpu_index = index.to('cuda:0')
            
            if (i < 100):
                test.fetch_feature(gpu_index,1024)
                continue
            temp.append(gpu_index)
            wb.append(temp)

        for i in range(10):
            feature_start = time.time()
            test.set_wb_counter_list(wb)
            feature_time =  time.time() - feature_start

#            cpu_agg_start = time.time()
 #           test.cpu_aggregate()
 #           agg_time = time.time() - cpu_agg_start
  #          print("agg time: ", agg_time)
            
            #feature_start = time.time()
            #test_gpu = test.fetch_feature(gpu_index,dim)
            #feature_time = feature_time + time.time() - feature_start
            #index_num = index_num + len(index)
            print("feature time: ", feature_time)
        #test.print_stats()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['experimental', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983, 171,172, 173], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')

    # Model
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gat', 'sage', 'gcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Model parameters 
    parser.add_argument('--fan_out', type=str, default='10,15')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    

    parser.add_argument('--bam', type=int, default=0)
    parser.add_argument('--num_ssd', type=int, default=1)

    parser.add_argument('--cache_size', type=int, default=8)
    parser.add_argument('--pin', type=int, default=0)
    parser.add_argument('--pin_size', type=int, default=0)

    parser.add_argument('--cpu_cache', type=int, default=0)
    parser.add_argument('--uva', type=int, default=0)
    parser.add_argument('--uva_graph', type=int, default=0)
    parser.add_argument('--window', type=int, default=0)
    parser.add_argument('--wb_size', type=int, default=6)
    

    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--log_every', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--data', type=str, default='IGB')
    # Experiment (to be deleted)
    parser.add_argument('--emb_size', type=int, default=1024)
    parser.add_argument('--wb_queue_size', type=int, default=131072)    

    args = parser.parse_args()
    labels = None
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    

    
    test_PVP(args)



