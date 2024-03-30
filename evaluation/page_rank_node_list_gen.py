import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np
from models import *
from dataloader import IGB260MDGLDataset, OGBDGLDataset, IGBHeteroDGLDataset, IGBHeteroDGLDatasetMassive, OGBHeteroDGLDatasetMassive
import csv 
import warnings

import torch.cuda.nvtx as t_nvtx
import nvtx
import threading
import gc

import GIDS

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

import networkx as nx
import dgl.function as fn

torch.manual_seed(0)
dgl.seed(0)
warnings.filterwarnings("ignore")

def compute_pagerank_hetero(g, DAMP, K, key_offset):
    # Assuming N is the total number of nodes across all types
    N = sum([g.number_of_nodes(ntype) for ntype in g.ntypes])
    pv_init_value = torch.ones(N) / N
    degrees = torch.zeros(N, dtype=torch.float32)

    # Initialize PageRank values for each node type
    for ntype in g.ntypes:
        print("node type: ", ntype)
        print("offset : ",key_offset[ntype]) 
        n = g.number_of_nodes(ntype)
        offset = key_offset[ntype]
        g.nodes[ntype].data['pv'] = pv_init_value[offset:offset+n]



    func_dict = {}
    for etype in g.etypes:
        func_dict[etype] = (fn.copy_u('pv','m'), fn.sum('m', 'pv'))
    

    for k in range(K):
        for etype in g.etypes:
            #print("etype: ", etype)
        
            cur_degrees = g.in_degrees(v='__ALL__', etype=etype).type(torch.float32)
            if(etype == 'affiliated_to'):
                g.nodes['institute'].data['pv'] /=  cur_degrees
            elif(etype == 'cites'):
                g.nodes['paper'].data['pv'] /= cur_degrees
            elif(etype == 'topic'):
                g.nodes['fos'].data['pv'] /= cur_degrees
            elif(etype == 'written_by'):
                g.nodes['author'].data['pv'] /= cur_degrees
                



        g.multi_update_all(func_dict, cross_reducer="sum")

        for ntype in g.ntypes:
            g.nodes[ntype].data['pv'] = (1 - DAMP) / N + DAMP * g.nodes[ntype].data['pv']
    return pv_init_value


#    

def compute_pagerank(g, DAMP, K, N):
    g.ndata['pv'] = torch.ones(N) / N
    degrees = g.in_degrees(g.nodes()).type(torch.float32)
    for k in range(K):
        g.ndata['pv'] = g.ndata['pv'] / degrees
        g.update_all(message_func=fn.copy_u('pv', 'm'),
                     reduce_func=fn.sum('m', 'pv'))
        g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['pv']
    return g.ndata['pv']

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
    parser.add_argument('--out_path', type=str, default='./pr_full.pt', 
        help='output path for the node list')
    parser.add_argument('--damp', type=float, default=0.85, 
        help='Damp value for the page rank algorithm')
    parser.add_argument('--K', type=int, default=20, 
        help='K value for the page rank algorithm')
    parser.add_argument('--device', type=int, default=0, 
        help='cuda device number')
    parser.add_argument('--data', type=str, default='IGB', 
        choices=['IGB', 'OGB'], help='Dataset type')
    parser.add_argument('--uva_graph', type=int, default=0,help='0:non-uva, 1:uva')
    parser.add_argument('--emb_size', type=int, default=1024)
    
    parser.add_argument('--hetero', action='store_true', help='Heterogenous Graph')
 

    args = parser.parse_args()
    
    labels = None
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    

    if(args.hetero):
        if(args.data == 'IGB'):
            print("Dataset: IGBH")
            if(args.dataset_size == 'full' or args.dataset_size == 'large'):
                dataset = IGBHeteroDGLDatasetMassive(args)
            else:
                dataset = IGBHeteroDGLDataset(args)

            g = dataset[0]
            #g  = g.formats('csc')
        elif(args.data == "OGB"):
            print("Dataset: MAG")
            dataset = OGBHeteroDGLDatasetMassive(args)
            g = dataset[0]
            #g  = g.formats('csc')
        else:
            g=None
            dataset=None

    
    else:
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

    
    pr_val = None
    N = 0
    if(args.hetero):
        print("heterogeneous graph")

        key_offset = {
                    'paper' : 0,
                    'author' : 1000000,
                    'fos' : 1000000 + 192606,
                    'institute' : 1000000 + 192606 + 190449
                }

        pv = compute_pagerank_hetero(g, args.damp, args.K, key_offset) 
        N = len(pv)
    else:
        N = g.number_of_nodes()
        pv = compute_pagerank(g, args.damp, args.K, N)
    
    print("N: ", N)
    topk = int(N * 0.6)
    _, indices = torch.topk(pv, k=topk, largest=True)

    torch.save(indices, args.out_path)

