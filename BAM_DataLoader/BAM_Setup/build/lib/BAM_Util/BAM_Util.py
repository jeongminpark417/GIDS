import torch
import numpy as np
import ctypes

import BAM_Feature_Store

class BAM_Util():
    def __init__(self):
        self.BAM_FS = BAM_Feature_Store.BAM_Feature_Store()
        self.BAM_FS.init_controllers()

    def fetch_feature(self, index, dim):
        index_ptr = index.data_ptr()
        index_size = len(index)
        return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device="cuda:0")
        print("python dim: ", dim , " index size ", index_size)
        self.BAM_FS.read_feature(return_torch.data_ptr(), index_ptr, index_size, dim)
        return return_torch




