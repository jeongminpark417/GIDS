import torch
import numpy as np
import BAM_Util

from BAM_Util import BAM_Tensor_Loader


BAM_L = BAM_Tensor_Loader(2,100)
BAM_L.print_gpuT();
BAM_L.fetch_feature()
BAM_L.print_gpuT();

#cpu_ptr = BAM_L.malloc(12)
#print(BAM_L.read_ptr_val(cpu_ptr))
#BAM_L.set_ptr_val(cpu_ptr, 5.0)
#print(BAM_L.read_ptr_val(cpu_ptr))

#BAM_L.set_cuda_val(0,5)
#BAM_L.print_gpu_tensor()
#print("FIRST\n\n\n\n")
#print("fist val:", x)
#print("SECOND\n\n\n\n")
#print("second val: ",y[[0]])
#print("THIRD\n\n\n\n")
#print("third val: ", x[1])

