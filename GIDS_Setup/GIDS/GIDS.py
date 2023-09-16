import time
import torch
import numpy as np
import ctypes
import nvtx 

import BAM_Feature_Store

class GIDS():
    def __init__(self, page_size=4096, off=0, cache_dim = 1024, num_ele = 300*1000*1000*1024, num_ssd = 1,  cache_size = 10,  wb_size = 8, wb_queue_size = 131072, ddp=False, ctrl_idx=0, no_init=False, cpu_agg=False, cpu_agg_queue_size=0):
        self.wb_queue_size = wb_queue_size
        self.BAM_FS = BAM_Feature_Store.BAM_Feature_Store()
        self.cache_dim = cache_dim
        self.gids_device="cuda:" + str(ctrl_idx)
        self.WB_time = 0.0

        self.evict_feat_ten = None
        self.evict_node_ten = None
        self.evict_flag_ten = None
        self.evict_node_size = 0
        self.evict_feat_size = 0

        self.first_fill = True

        self.num_cl = int(num_ele/cache_dim)
        print("GIDS num cl: ", self.num_cl )

        self.batch_ptr_ten = None
        self.batch_len_ten = None

        self.batch_ptr_list = []
        self.batch_len_list = []
       

        print("page size: ", page_size, " off: ", off)
        print("num sdd: ", num_ssd)

        if(no_init):
            print("skip initalization")
            return

        if(ddp):
            print("multi GPU setup")
            self.BAM_FS.mgc_init_controllers(page_size, off, cache_size,num_ele, ctrl_idx)
        else:
            self.BAM_FS.init_controllers(page_size, off, cache_size,num_ele, num_ssd, wb_size, wb_queue_size, cpu_agg, cpu_agg_queue_size)

    def create_evict_tensors(self,  num_nodes, dim):
        self.evict_feat_size = self.wb_queue_size * dim
        self.evict_feat_ten = torch.zeros((self.evict_feat_size ),device=self.gids_device, dtype=torch.float)

        self.evict_node_size = num_nodes
        self.evict_node_ten = torch.zeros((self.evict_node_size),device=self.gids_device, dtype=torch.int)
        self.evict_flag_ten = torch.zeros((self.evict_node_size),device=self.gids_device, dtype=torch.int)

    

    def fetch_feature(self, index, dim):
        index_ptr = index.data_ptr()
        index_size = len(index)
        return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device)
        self.BAM_FS.read_feature(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim)
        return return_torch

    @nvtx.annotate("fetch_feature_with_wb()", color="red")
    def fetch_feature_with_wb(self, return_torch, index, dim):

        index_ptr = index.data_ptr()
        index_size = len(index)


            #Need to  clear
        self.evict_flag_ten = torch.zeros((self.evict_node_size),device=self.gids_device, dtype=torch.int)
        self.BAM_FS.fill_batch(self.evict_feat_ten.data_ptr(), self.evict_node_ten.data_ptr(), return_torch.data_ptr(), index.data_ptr(), self.evict_flag_ten.data_ptr(),  index_size, dim, self.first_fill)
            
        self.BAM_FS.read_feature_with_wb(return_torch.data_ptr(), index_ptr, self.evict_flag_ten.data_ptr(), self.evict_feat_ten.data_ptr(),  index_size, dim, self.cache_dim)

        if(self.first_fill):
            self.first_fill = False
        
        return
    



    #return return_torch
    
    def cpu_aggregate(self):
        self.BAM_FS.cpu_aggregate(self.cache_dim)

    @nvtx.annotate("Set WB counter", color="blue")
    def set_wb_counter(self, batch_array_tensor):

        self.batch_ptr_list = []
        self.batch_len_list = []
        first = True
        for batch in batch_array_tensor:
            if(first):
                first = False
                continue
            batch_ptr = batch[0].data_ptr()
            batch_len = len(batch[0])
            self.batch_ptr_list.append(int(batch_ptr))
            self.batch_len_list.append(batch_len)

        self.batch_ptr_ten = torch.tensor(self.batch_ptr_list, device=self.gids_device)
        self.batch_len_ten = torch.tensor(self.batch_len_list, device=self.gids_device)
#        torch.cuda.synchronize() 
        self.BAM_FS.set_wb_counter(self.batch_ptr_ten.data_ptr(), self.batch_len_ten.data_ptr(), self.evict_node_size)
        
        #del batch_ptr_ten
        #del batch_len_ten

        return

    @nvtx.annotate("Set WB counter_list", color="blue")
    def set_wb_counter_list(self, batch_array_tensor):

        self.batch_ptr_list = []
        self.batch_len_list = []
        first = True
        for batch in batch_array_tensor:
            if(first):
                first = False
                continue
            batch_ptr = batch[0].data_ptr()
            batch_len = len(batch[0])
            self.batch_ptr_list.append(int(batch_ptr))
            self.batch_len_list.append(batch_len)

        self.batch_ptr_ten = torch.tensor(self.batch_ptr_list, device=self.gids_device)
        self.batch_len_ten = torch.tensor(self.batch_len_list, device=self.gids_device)
        self.BAM_FS.set_wb_counter_list(self.batch_ptr_ten.data_ptr(), self.batch_len_ten.data_ptr(), self.evict_node_size)


        return

    def set_wb_counter_with_CPU(self, batch_array_tensor):
        batch1 = batch_array_tensor[0]
        batch_ptr = batch1[0].data_ptr()
        self.BAM_FS.set_wb_counter_with_CPU(batch_ptr, 0, 0)

    def init_cpu_meta(self):
        self.BAM_FS.init_cpu_meta(self.num_cl)
        
    def pin_pages(self, index, dim):
        num_index = len(index)
        index_ptr = index.data_ptr()
        self.BAM_FS.pin_pages(index_ptr,num_index,dim)

    def print_stats(self):
        wbtime = self.WB_time 
        self.WB_time = 0.0
        print("WB time: ", wbtime)
        self.BAM_FS.print_stats()
        return


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
        if isinstance(batch[0], dict):
            for key, value in batch[0].items():
                if(len(value) == 0):
                    next
                else:
                    input_tensor = value.to("cuda:0")
                    num_pages = len(input_tensor)
                    val_tensor = torch.ones([num_pages,1], dtype=torch.uint8)
                    val_tensor_ptr = val_tensor.to('cuda:0')
                    s_time = time.time()
                    self.BAM_FS.set_window_buffering(input_tensor.data_ptr(), val_tensor_ptr.data_ptr(), num_pages)
                    e_time = time.time()
                    self.WB_time += e_time - s_time
        else:
            input_tensor = batch[0].to("cuda:0")
            num_pages = len(input_tensor)

            val_tensor = torch.ones([num_pages,1], dtype=torch.uint8)
            val_tensor_ptr = val_tensor.to('cuda:0')
            s_time = time.time()
            self.BAM_FS.set_window_buffering(input_tensor.data_ptr(), val_tensor_ptr.data_ptr(), num_pages)
            e_time = time.time()
            self.WB_time += e_time - s_time

    def init_backing_memory(self, mem_size):
        self.BAM_FS.init_backing_memory(mem_size)
         
    def create_streams(self, num_streams):
        self.BAM_FS.create_streams(num_streams)

    def fetch_from_backing_memory(self, dp, bp, bp2, bs, cl, ntc):
        self.BAM_FS.fetch_from_backing_memory(dp, bp, bp2, bs, cl, ntc)

    def fetch_from_backing_memory_chunk(self,dp, cl, stream_id):
        self.BAM_FS.fetch_from_backing_memory_chunk(dp, cl, stream_id)

    def sync_streams(self,sync_stream):
        self.BAM_FS.sync_streams(sync_stream)

    def compute_test(self, out_ten, n):
        self.BAM_FS.compute_test(out_ten.data_ptr(), n)


    def print_wb_queue(self):
        self.BAM_FS.print_wb_queue()

    def update_time(self):
        self.BAM_FS.update_time()

    @nvtx.annotate("Prefetch()", color="green")
    def prefetch_from_victim_queue(self, stream_id):
         self.BAM_FS.prefetch_from_victim_queue(self.evict_feat_ten.data_ptr(), self.evict_node_ten.data_ptr(), stream_id)


    def fill_batch(self, feat_ten, node_ten, batch_ten, batch_node_ten, node_flag_ten, batch_size, dim):
        self.BAM_FS.fill_batch(feat_ten.data_ptr(), node_ten.data_ptr(), batch_ten.data_ptr(), batch_node_ten.data_ptr(), node_flag_ten.data_ptr(),  batch_size, dim)

