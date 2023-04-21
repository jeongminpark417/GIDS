# GPU Initiated Direct Storage Accesses

This is the opencourse implementation of GIDS Dataloader. Contributions to the codebase are most welcome.

## Get Started
Users need to first set up the environment for the BaM system and create the shared library. 
Please follow the instruction from https://github.com/ZaidQureshi/bam/tree/bam_pin

Once the BaM system is set up, users need to create the python module for the GIDS dataloader.

``` 
cd gids_module
mkdir build
cmake .. && make -j
cd BAM_Feature_Store
python setup install
```
Then, users need to set up the python interface for the GIDS dataloader. From the root directory,

```
cd GIDS_Setup
pip install .
```
Finally, users need to update the dgl dataloader by reinstalling the dgl library from the forked DGL repo.

The scripts for IGB, OGB, and MAG datasets are in the `example` directory
