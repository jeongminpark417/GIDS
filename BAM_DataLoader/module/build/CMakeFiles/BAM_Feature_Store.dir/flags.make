# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# compile CUDA with /usr/local/cuda/bin/nvcc
CUDA_DEFINES = -DBAM_Feature_Store_EXPORTS

CUDA_INCLUDES = -I/opt/conda/include/python3.8 -I/opt/conda/include -I/root/BAM_Tensor/BAM_DataLoader/module/./include -I/root/BAM_Tensor/BAM_DataLoader/module/../../bam/include -I/root/BAM_Tensor/BAM_DataLoader/module/../../bam/include/freestanding/include

CUDA_FLAGS = -Xcompiler=-fPIC -std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -std=c++11
