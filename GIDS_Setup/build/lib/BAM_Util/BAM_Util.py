import torch
import numpy as np
import ctypes

import BAM_Feature_Store

class BAM_Util():
    def __init__(self, page_size=4096, off=0, cache_dim = 1024, num_ele = 300*1000*1000*1024, num_ssd = 1,  cache_size = 10, cpu_cache=False, cpu_cache_ptr=0):
        self.BAM_FS = BAM_Feature_Store.BAM_Feature_Store()
        self.cache_dim = cache_dim
        print("page size: ", page_size, " off: ", off)
        print("num sdd: ", num_ssd)
        self.BAM_FS.init_controllers(page_size, off, cache_size,num_ele, num_ssd, cpu_cache, cpu_cache_ptr)

    def fetch_feature(self, index, dim):
        index_ptr = index.data_ptr()
        index_size = len(index)
        return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device="cuda:0")
        self.BAM_FS.read_feature(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim)
        return return_torch


    def pin_pages(self, index, dim):
        num_index = len(index)
        index_ptr = index.data_ptr()
        self.BAM_FS.pin_pages(index_ptr,num_index,dim)

    def print_stats(self):
        self.BAM_FS.print_stats()
        return

    def set_cpu_pages(self, index):
        num_index = len(index)
        index_ptr = index.data_ptr()
        self.BAM_FS.set_cpu_pages(index_ptr, num_index)

    def hint_cache(self, input_list, output_list):
        input_set = set(input_list)
        output_set = set(output_list)
        pref_id = input_set.intersection(output_set)
        print("pref id: ", len(pref_id))
        pref_id = list(pref_id)
        num_pages = len(pref_id)
      #  print("input set: ", input_set)
       # print("output set: ", output_set)
        print("input size: ", len(input_list))
        print("output size: ", len(output_list))
        print("num pages: ", num_pages)
        id_tensor = torch.tensor(pref_id)
        val_tensor = torch.ones([num_pages,1], dtype=torch.uint8)
        id_tensor_ptr = id_tensor.to('cuda:0')
        val_tensor_ptr = val_tensor.to('cuda:0')
        self.BAM_FS.set_prefetching(id_tensor_ptr.data_ptr(), val_tensor_ptr.data_ptr(), num_pages)


   

    def window_buffer(self, batch, window):
        input_tensor = batch[0].to("cpu")
        input_set = set(input_tensor.tolist())
        wb = []
        for w_batch in window:
            wb.extend(w_batch[0].to("cpu").tolist())

        output_set = set(wb)
        pref_id = input_set.intersection(output_set)
        pref_id = list(pref_id)  
        num_pages = len(pref_id)

        #print("input size: ", len(input_tensor))
        #print("output size: ", len(wb))
        #print("num pages: ", num_pages)
        
        id_tensor = torch.tensor(pref_id)
        val_tensor = torch.ones([num_pages,1], dtype=torch.uint8)
        id_tensor_ptr = id_tensor.to('cuda:0')
        val_tensor_ptr = val_tensor.to('cuda:0')
        self.BAM_FS.set_prefetching(id_tensor_ptr.data_ptr(), val_tensor_ptr.data_ptr(), num_pages)

   

    def window_buffer2(self, batch, window):
        input_tensor = batch[0].to("cpu")
        input_set = set(input_tensor.tolist())
        wb = []
        for w_batch in window:
            wb.extend(w_batch[0].to("cpu").tolist())

        output_set = set(wb)
        pref_id = input_set.intersection(output_set)
        pref_id = list(pref_id)  
        num_pages = len(pref_id)

        #print("input size: ", len(input_tensor))
        #print("output size: ", len(wb))
        #print("num pages: ", num_pages)
        
        id_tensor = torch.tensor(pref_id)
        val_tensor = torch.ones([num_pages,1], dtype=torch.uint8)
        id_tensor_ptr = id_tensor.to('cuda:0')
        val_tensor_ptr = val_tensor.to('cuda:0')
        self.BAM_FS.set_window_buffering(id_tensor_ptr.data_ptr(), val_tensor_ptr.data_ptr(), num_pages)


