# GPU Initiated Direct Storage Accesses (GIDS) Dataloader

This repository contains the implementation of the GIDS Dataloader, an open-source implementation for accelerating large-scale Graph Neural Network (GNN) workloads using GPU-initiated direct storage accesses. Contributions to the codebase are most welcome.

## Installation Prerequisites
Before proceeding with the installation of GIDS, ensure that the following libraries/packages are installed on your system:
  - DGL Framework
  - BaM
  - pybind11
  - pytorch


## Get Started
To use the GIDS Dataloader, users need to set up the environment for the BaM system and create the shared library. Please follow the instructions from the BaM repository.

  

Once the BaM system is set up, users can create the Python module for the GIDS dataloader by running the following commands:

``` 
cd gids_module
mkdir build
cmake .. && make -j
cd BAM_Feature_Store
python setup install
```
If cmake cannot automatically find the path for the installed libraries, try
```
cmake .. \
-DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
-DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
```

Afterwards, users can set up the Python interface for the GIDS dataloader by running the following command from the root directory:

```
cd GIDS_Setup
pip install .
```

## Running the Example Benchmark
The example code for the IGB, OGB, and MAG datasets are located in the evaluation directory. 

`homogenous_train.py` GNN training example code with GIDS and BaM dataloaders for homogenous graphs.\
`homogenous_train_baseline.py` GNN training example code with the mmap baseline dataloader for homogenous graphs.\
`heterogeneous_train.py` GNN training example code with GIDS and BaM dataloaders for heterogeneous graphs.\
`heterogeneous_train_baseline.py` GNN training example code with  mmap baseline dataloader for heterogeneous_train graphs.

To run with the pre-defined configurations for GIDS, you can run the following scripts in the evaluation directory.\
`run_GIDS_IGBH.sh` Script to run GNN training with GIDS dataloader on IGBH graph. \
`run_BaM_IGBH.sh` Script to run GNN training with GIDS dataloader on IGBH graph. \
`run_base_IGBH.sh` Script to run GNN training with the baseline mmap dataloader on IGBH graph.

## Graph Strucute Data Type
DGL supports the Compressed sparse column (CSC) and Coordinate list (COO) file formats for graph structure data. However, when both formats are enabled for the dataloader, the dataloader automatically converts COO to CSR format during the graph sampling process. This conversion consumes more than 700GB of memory for terabyte-scale datasets such as the IGB dataset. To address this issue, we use the CSC format and fix the format for the graph structure data.
