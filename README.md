# hpc4ds-autoencoder
Autoencoder developed from scratch, benchmarked on a single node with no parallelism, with parallelism (OpenMP), multi-node (MPI) and "hybrid" (OpenMP + MPI) with a data parallel setup.


## About
We want this project to be self-contained, so far all external libraries are included as git submodules.

Cloning the project:
```bash
git clone --recurse-submodules <url>
```
or afer cloning:
```bash
git submodule --init --recursive
```
