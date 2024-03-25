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

import networkx as nx
import dgl.function as fn

torch.manual_seed(0)
dgl.seed(0)
warnings.filterwarnings("ignore")

    
def compute_pagerank(g, DAMP, K, N):
    g.ndata['pv'] = torch.ones(N) / N
    degrees = g.in_degrees(g.nodes()).type(torch.float32)
    for k in range(K):
        g.ndata['pv'] = g.ndata['pv'] / degrees
        g.update_all(message_func=fn.copy_src(src='pv', out='m'),
                     reduce_func=fn.sum(msg='m', out='pv'))
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

    N = g.number_of_nodes()

    pv = compute_pagerank(g, args.damp, args.K, N)
    topk = int(N * 0.6)
    _, indices = torch.topk(pv, k=topk, largest=True)

    torch.save(indices, args.out_path)

