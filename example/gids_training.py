import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np
from models import *
from dataloader import IGB260MDGLDataset, OGBDGLDataset
import csv 
import warnings

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

torch.manual_seed(0)
dgl.seed(0)
warnings.filterwarnings("ignore")

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
        window_buffer_size=args.wb_size
        )
    if(bam_flag):
        if(args.data == "IGB"):
            train_dataloader.bam_init(0,4096,1024, 300*1000*1000*1024, args.cache_size, args.num_ssd)
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

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), 
        lr=args.learning_rate, weight_decay=args.decay)

    # Training loop
    best_accuracy = 0
    print("training start")
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
        for step, (input_nodes, seeds, blocks, ret) in enumerate(train_dataloader):
            print("idx: ", idx)
            if(idx == 1000):
                print("warp up done")
                if(bam_flag):
                    train_dataloader.print_stats()
                train_dataloader.print_timer()
                print("agg time: ", batch_input_time)
                print("transfer time: ", transfer_time)
                print("train time: ", train_time)
                batch_input_time = 0
                transfer_time = 0
                train_time = 0

            idx += 1
            transfer_start = time.time() 
            blocks = [block.int().to(device) for block in blocks]
            transfer_time = transfer_time +  time.time()  -transfer_start
            
            batch_input_start = time.time()
            if(bam_flag): 
                batch_inputs = ret
            else:
                batch_inputs = blocks[0].srcdata['feat']
            agg_time = (time.time() - batch_input_start)
            batch_input_time += agg_time
            batch_input_start = time.time()
            agg_time = (time.time() - batch_input_start)
           
            if(args.data == 'IGB'):
                batch_labels = blocks[-1].dstdata['labels']
                
            else:
                batch_labels = blocks[-1].dstdata['labels']
            
            if(idx == 1):
                print("batch len: ", len(batch_inputs))
            train_start = time.time()

            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()
            train_time = train_time + time.time() - train_start
            train_acc += (sklearn.metrics.accuracy_score(batch_labels.cpu().numpy(), 
                batch_pred.argmax(1).detach().cpu().numpy())*100)

            gpu_mem_alloc += (
                torch.cuda.max_memory_allocated() / 1000000
                if torch.cuda.is_available()
                else 0
            )
            if(idx == 1100):
                print("1dx 1100")
                if(bam_flag):
                    train_dataloader.print_stats()
                train_dataloader.print_timer()
                print("agg time: ", batch_input_time)
                print("transfer time: ", transfer_time)
                print("train time: ", train_time)
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
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

        if epoch%args.log_every == 0:
            model.eval()
            predictions = []
            labels = []
            with torch.no_grad():
                for _, _, blocks,_ in val_dataloader:
                #for b,_ in val_dataloader:
                   # blocks = b[2]
                    blocks = [block.to(device) for block in blocks]
                    inputs = blocks[0].srcdata['feat']
                   
                    if(args.data == 'IGB'):
                        labels.append(blocks[-1].dstdata['label'].cpu().numpy())
                    elif(args.data == 'OGB'):
                        out_label = torch.index_select(label_array, 0, b[1]).flatten()
                        labels.append(out_label.numpy())

                    predictions.append(model(blocks, inputs).argmax(1).cpu().numpy())
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                val_acc = sklearn.metrics.accuracy_score(labels, predictions)*100
                if best_accuracy < val_acc:
                    best_accuracy = val_acc
                    if args.model_save:
                        torch.save(model.state_dict(), args.modelpath)

            tqdm.tqdm.write(
                "Epoch {:03d} | Loss {:.4f} | Train Acc {:.2f} | Val Acc {:.2f} | Time {} | GPU {:.1f} MB".format(
                    epoch,
                    epoch_loss,
                    train_acc,
                    val_acc,
                    str(datetime.timedelta(seconds = int(time.time() - epoch_start))),
                    gpu_mem_alloc
                )
            )
       
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for _, _, blocks,_ in test_dataloader:
        #for b,_ in test_dataloader:
        #    blocks = b[2]
            blocks = [block.to(device) for block in blocks]
            inputs = blocks[0].srcdata['feat']
     
            if(args.data == 'IGB'):
                labels.append(blocks[-1].dstdata['label'].cpu().numpy())
            elif(args.data == 'OGB'):
                out_label = torch.index_select(label_array, 0, b[1]).flatten()
                labels.append(out_label.numpy())


            #labels.append(blocks[-1].dstdata['label'].cpu().numpy())
            predict = model(blocks, inputs).argmax(1).cpu().numpy()
            predictions.append(predict)
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        test_acc = sklearn.metrics.accuracy_score(labels, predictions)*100
    print("Test Acc {:.2f}%".format(test_acc))
    print("Total time taken " + str(datetime.timedelta(seconds = int(time.time() - training_start))))


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

    parser.add_argument('--cache_size', type=int, default=10)
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
    

    args = parser.parse_args()
    labels = None
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)
        g = dataset[0]
        g  = g.formats('csc')
    elif(args.data == "OGB"):
        print("Dataset: OGB")
        dataset = OGBDGLDataset(args)
        g = dataset[0]
    else:
        g=None
        dataset=None
    
    print(g)
    track_acc(g, args, device, labels)




