# GPU Initiated Direct Storage Accesses (GIDS) Dataloader

This repository contains the implementation of the GIDS Dataloader, an open-source implementation for accelerating large-scale Graph Neural Network (GNN) workloads using GPU-initiated direct storage accesses. Contributions to the codebase are most welcome.

## Installation Prerequisites
Before proceeding with the installation of GIDS, ensure that the following libraries/packages are installed on your system:
  - DGL Framework
  - BaM
  - pybind11
  - pytorch


## Get Started

### Storing Data to SSDs
Since the GIDS Dataloader reads data directly from SSDs, users need to store feature data into the SSDs initially. If you have a binary file, you can utilize the `readwrite_stripe` benchmark application in the BaM directory (window buffer branch) to write the data. Alternatively, if you have a NumPy tensor file, you can use `tensor_write.py` to store the binary values of the feature data into the SSDs. However, note that `tensor_write.py` requires loading the entire tensor data into CPU memory first. Thus, we encourge you to write data with `readwrite_stripe` benchmark application.

When utilizing multiple SSDs for the dataset, GIDS assumes that the feature data is striped with page granularity. Users can configure the order of the SSDs with the ssd_list flag when initializing GIDS. 

For heterogenous graphs, there are different types of nodes. To enable GIDS Dataloader to fetch correct feature data, users need to pass the offset for each node type and pass the offset dictionary to GIDS with `heterograph_map` flag. Moreover, `heterograph` flag should set to be True for heterogenous graph. For instance, if users want to use /dev/libnvm0 and /dev/libnvm1 for heterogenous graph, they can configure GIDS as follows:

```
import GIDS

GIDS_Loader = GIDS.GIDS(
  .....
  other arguments
  .....
  ssd_list = [0, 1],
  heterograph = True,
  heterograph_map = {
              'paper' : 0,
              'author' : 269346174,
              'fos' : 546567057,
              'institute' : 547280017
  }
)
```



### Installing GIDS
To use the GIDS Dataloader, users need to set up the environment for the BaM system and create the shared library. Please follow the instructions from the BaM repository.


  

Once the BaM system is set up, users can create the Python module for the GIDS dataloader by running the following commands:

``` 
cd gids_module
mkdir build && cd build
cmake .. && make -j
cd BAM_Feature_Store
python setup.py install
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

### Generate Node List for the Constant CPU Buffer 
The GIDS Dataloader stores highly reusable node feature data into the Constant CPU Buffer. We provide an example code, `page_rank_node_list_gen.py`, to generate the node list based on the reversed page rank values. Users can also manually generate the node list based on other graph properties.

The path of the node_list file should be defined with the `pin_file` flag for the example applications.


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
