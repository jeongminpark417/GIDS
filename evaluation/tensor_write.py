import math
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
import threading


import GIDS
from GIDS import GIDS_DGLDataLoader

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M', 
        help='path containing the datasets')
    
    #GPU Software Cache Parameters
    parser.add_argument('--page_size', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0, help='Offset for the feature data stored in the SSD in byte(offset should be in page size granularity)') 
    parser.add_argument('--num_ele', type=int, default=100, help='Number of elements in the dataset (Total Size / sizeof(Type)') 
    parser.add_argument('--cache_dim', type=int, default=1024)
    parser.add_argument('--num_ssd', type=int, default=1)
    parser.add_argument('--cache_size', type=int, default=8)

    parser.add_argument('--ssd_list', type=str, default=None)


    parser.add_argument('--mmap', type=int, default=0) 
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'

    gids_ssd_list = None
    if (args.ssd_list != None):
        gids_ssd_list =  [int(ssd_list) for ssd_list in args.ssd_list.split(',')]

    print("GIDS SSD List: ", gids_ssd_list)

    GIDS_Loader = GIDS.GIDS(
        page_size = args.page_size,
        off = args.offset,
        num_ele = args.num_ele,
        num_ssd = args.num_ssd,
        cache_size = args.cache_size,
        cache_dim = args.cache_dim,
        ssd_list = gids_ssd_list
    )

    emb = np.load(args.path)
    emb_tensor = torch.tensor(emb).to(device)
    GIDS_Loader.store_tensor(emb_tensor, 0)






