# GPU Initiated Direct Storage Accesses (GIDS) Dataloader

This repository contains the implementation of the GIDS Dataloader, an open-source implementation for accelerating large-scale Graph Neural Network (GNN) workloads using GPU-initiated direct storage accesses. Contributions to the codebase are most welcome.


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
Afterwards, users can set up the Python interface for the GIDS dataloader by running the following command from the root directory:

```
cd GIDS_Setup
pip install .
```
Finally, users need to update the DGL dataloader by reinstalling the DGL library from the forked DGL repository.

The scripts for the IGB, OGB, and MAG datasets are located in the example directory. 


## Graph Strucute Data Type
DGL supports the CSC and COO file formats for graph structure data. However, when both formats are enabled for the dataloader, the dataloader automatically converts COO to CSR format during the graph sampling process. This conversion consumes more than 700GB of memory for terabyte-scale datasets such as the IGB dataset. To address this issue, we use the CSC format and fix the format for the graph structure data.
