import torch
import numpy as np
import ctypes

import example

class BAM_Tensor_Loader():
    def __init__(self):
        self.filename = "f";
        self.cpu_tensor = torch.zeros([2,4], dtype=torch.float)
        self.gpu_tensor = self.cpu_tensor.to('cuda:0')
    
    def fetch_data(self):
        cptr = self.gpu_tensor.data_ptr()
        cpu_ptr = self.cpu_tensor.data_ptr()
        print("cptr: ", cptr)
        print("cpu ptr: ",self.cpu_tensor.data_ptr())
        print("cpu tensor: ", self.cpu_tensor)
        example.set_val(cpu_ptr, 5, 8)
        print("cpu tensor: ", self.cpu_tensor)

    def malloc(self, byte_size):
        cpu_ptr = example.c_malloc(byte_size)
        self.c_ptr = cpu_ptr
        return cpu_ptr

    def read_ptr_val(self, ptr):
        return example.read_ptr_val(ptr)
    def set_ptr_val(self, ptr, val):
        example.set_ptr_val(ptr, val)

    
    def set_cuda_val(self, idx, val):
        gpu_ptr = self.gpu_tensor.data_ptr()
        example.set_cuda_val(gpu_ptr, idx, val)

    def print_gpu_tensor(self):
        print(self.gpu_tensor)

class BAM_Tensor(torch.Tensor):
  
        
      #  super().__init__(**kwargs) 
    
    #super().__init__(*args, **kwargs)


    
    def __getitem__(self, idx): 
        print("BAM getitem")

        return_val = super().__getitem__( idx)
        print("idx: ", idx)
        if(idx == 1):
            return torch.tensor([9,9])
        else:
            return super().__getitem__( idx)

    def __new_getitem__():
        print("BAM new getitem")
        super().__getitem__()



