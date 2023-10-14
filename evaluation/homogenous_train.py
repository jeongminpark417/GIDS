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
from ladies_sampler import LadiesSampler, normalized_edata

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

def track_acc_BaM(g, args, device, label_array=None):
    GIDS_Loader = None
    GIDS_Loader = GIDS.GIDS(
        page_size = args.page_size,
        off = args.offset,
        num_ele = args.num_ele,
        num_ssd = args.num_ssd,
        cache_size = args.cache_size,
        wb_size = args.wb_size, #rename
        
        cache_dim = args.cache_dim #should be removed
    )

    # Using BaM for the Graph Sampling Stage
    GIDS_Loader.init_graph_GIDS(512, 0, 512, 9 * 1000 * 1000 * 1000, 1)
    adj_mat =  g.adj_sparse('csc')
    GIDS_Loader.graph_GIDS.set_offsets(0,len(adj_mat[0]),len(adj_mat[0])+len(adj_mat[1]))

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
               [int(fanout) for fanout in args.fan_out.split(',')],
                GIDS_flag=True, 
                GIDS=GIDS_Loader)

    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
    in_feats = g.ndata['features'].shape[1]
    dim = args.dim


    train_dataloader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_uva=False,
        use_uva_graph=False,    # Need to rename
        bam=True,
        feature_dim=dim,
        use_prefetch_thread=False,
        pin_prefetcher=False,
        window_buffer=False,
        use_alternate_streams=False,
        GIDS=GIDS_Loader
    )

    val_dataloader = dgl.dataloading.DataLoader(
        g, val_nid, sampler,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = dgl.dataloading.DataLoader(
        g, test_nid, sampler,
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

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, weight_decay=args.decay
        )

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
            
            if(step == 1000):
                print("warp up done")
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0
                e2e_time_start = time.time()
        
            
            # Features are fetched by the baseline GIDS dataloader in ret
            batch_inputs = ret
            batch_labels = blocks[-1].dstdata['labels']
            

            transfer_start = time.time() 
            blocks = [block.int().to(device) for block in blocks]
            batch_labels = batch_labels.to(device)
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
          
            if(step == 1100):
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
                return None

  
    # Evaluation

    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for _, _, blocks,_ in test_dataloader:
            blocks = [block.to(device) for block in blocks]
            inputs = blocks[0].srcdata['feat']
     
            if(args.data == 'IGB'):
                labels.append(blocks[-1].dstdata['label'].cpu().numpy())
            elif(args.data == 'OGB'):
                out_label = torch.index_select(label_array, 0, b[1]).flatten()
                labels.append(out_label.numpy())
            predict = model(blocks, inputs).argmax(1).cpu().numpy()
            predictions.append(predict)

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        test_acc = sklearn.metrics.accuracy_score(labels, predictions)*100
    print("Test Acc {:.2f}%".format(test_acc))



def track_acc_GIDS(g, args, device, label_array=None):

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
       # ssd_list = [5],
        cache_dim = args.cache_dim
    
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




    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
    in_feats = g.ndata['features'].shape[1]

    sampler = None
    if (args.sample_type == 'NHS'):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [int(fanout) for fanout in args.fan_out.split(',')]
            )

    elif(args.sample_type == "ClusterGCN"):
        sampler = dgl.dataloading.ClusterGCNSampler(
            g,
            args.num_partitions
        )
        train_nid = torch.arange(args.num_partitions)
    #LADIES
    else:
        g.edata["w"] = normalized_edata(g)
        l_out = [int(_) for _ in args.ladouts.split(",")]
        sampler = LadiesSampler(l_out)


    train_dataloader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_uva=False,
        feature_dim=dim,
        use_prefetch_thread=False,
        pin_prefetcher=False,
        use_alternate_streams=False,

        use_uva_graph=args.uva_graph, 
        bam=True,
        #window_buffer=args.window_buffer,
        window_buffer=False,
        window_buffer_size=args.wb_size,
        GIDS=GIDS_Loader
    )

    val_dataloader = dgl.dataloading.DataLoader(
        g, val_nid, sampler,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = dgl.dataloading.DataLoader(
        g, test_nid, sampler,
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

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, weight_decay=args.decay
        )

    warm_up_iter = 10
    eval_iter = 10
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
            #if(step % 10 == 0):
            print("step: ", step)
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

            batch_labels = blocks[-1].dstdata['labels']
            

            blocks = [block.int().to(device) for block in blocks]
            batch_labels = batch_labels.to(device)
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
          
            if(step == warm_up_iter + eval_iter):
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
                return None


       
  
    # Evaluation

    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for _, _, blocks,_ in test_dataloader:
            blocks = [block.to(device) for block in blocks]
            inputs = blocks[0].srcdata['feat']
     
            if(args.data == 'IGB'):
                labels.append(blocks[-1].dstdata['label'].cpu().numpy())
            elif(args.data == 'OGB'):
                out_label = torch.index_select(label_array, 0, b[1]).flatten()
                labels.append(out_label.numpy())
            predict = model(blocks, inputs).argmax(1).cpu().numpy()
            predictions.append(predict)

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        test_acc = sklearn.metrics.accuracy_score(labels, predictions)*100
    print("Test Acc {:.2f}%".format(test_acc))



def track_acc(g, args, device, label_array=None):

    GIDS_Loader = None
    if(bam_flag):
        if(args.data == "IGB"):
            GIDS_Loader = GIDS.GIDS(4096, 0, 1024, 270*1000*1000*1024, args.num_ssd,  args.cache_size, args.wb_size, args.wb_queue_size, False, 0,  args.cpu_agg, args.cpu_agg_queue_size, merge_flag=False)
        elif(args.data == "OGB"):
            GIDS_Loader = GIDS.GIDS(128*4, 0 ,128, 111059956 * 128 * 2,  args.num_ssd, args.cache_size, args.wb_size, args.wb_queue_size, False, 0,  args.cpu_agg, args.cpu_agg_queue_size)


    print("SETTING OFFSETS")
#    GIDS_Loader.init_graph_GIDS(512, 0, 512, 9 * 1000 * 1000 * 1000, 1)
#    adj_mat =  g.adj_sparse('csc')
    
#    GIDS_Loader.graph_GIDS.set_offsets(0,len(adj_mat[0]),len(adj_mat[0])+len(adj_mat[1]))

#    sampler = dgl.dataloading.MultiLayerNeighborSampler(
#                [int(fanout) for fanout in args.fan_out.split(',')], GIDS_flag=bam_flag, GIDS=GIDS_Loader)

    num_nodes = g.number_of_nodes()
    num_pinned_nodes = int(num_nodes * 0.1)
    GIDS_Loader.cpu_backing_buffer(1024, num_pinned_nodes)
    pr_path = "/mnt/nvme16/pr_full.pt"
    pr_ten = torch.load(pr_path)
    #GIDS_Loader.set_cpu_buffer(pr_ten, num_pinned_nodes)

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
        use_alternate_streams=False,
        GIDS=GIDS_Loader
        )

    
    #if(bam_flag):
    #    if(args.data == "IGB"):
    #        train_dataloader.bam_init(0,4096,1024, 270*1000*1000*1024, args.cache_size, args.num_ssd, args.wb_queue_size, args.cpu_agg, args.cpu_agg_queue_size)
    #    elif(args.data == "OGB"):
    #        train_dataloader.bam_init(0,128*4,128, 111059956 * 128 * 2,  args.cache_size, args.num_ssd)

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
#                GIDS_loader.prefetch_from_victim_queue(0)
                train_dataloader.set_wb_counter(args.cpu_agg)
 
            train_start = time.time()

            with nvtx.annotate("train", color="yellow"):
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach()
            
            #torch.cuda.synchronize()
             
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

    
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['experimental', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983, 172], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--data', type=str, default='IGB')
    parser.add_argument('--emb_size', type=int, default=1024)
    
    # Model
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gat', 'sage', 'gcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Sampling
    parser.add_argument('--sample_type', type=str, default='NHS',
                        choices=['NHS', 'ClusterGCN', 'LADIES'])
    parser.add_argument('--num_partitions', type=int, default=1000)



    # Model parameters 
    parser.add_argument('--fan_out', type=str, default='10,5,5')
    parser.add_argument('--ladouts', type=str, default='64,64,64')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
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
    print("Sample Type: ", args.sample_type)


    labels = None
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)
        g = dataset[0]
        if(args.sample_type == "LADIES"):
            print("LADIES")
            g = g.formats('coo')
        else:
            g  = g.formats('csc')
    elif(args.data == "OGB"):
        print("Dataset: OGB")
        dataset = OGBDGLDataset(args)
        g = dataset[0]
        if(args.sample_type == "LADIES"):
            print("LADIES")
            g = g.formats('coo')
            g = g.long()
        else:
            g  = g.formats('csc')
    else:
        g=None
        dataset=None

    print("Fortmats", g.formats())
    track_acc_GIDS(g, args, device, labels)
    #track_acc(g, args, device, labels)




