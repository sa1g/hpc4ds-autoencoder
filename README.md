# hpc4ds-autoencoder
Autoencoder developed from scratch, benchmarked on a single node with no parallelism, with parallelism (OpenMP), multi-node (MPI) and "hybrid" (OpenMP + MPI) with a data parallel setup.


## About
run only my tests
```bash
ctest -L AutoencoderTests
```
<!-- TODO: or with prefix: `autoencoder_`:
```bash
ctest -R "^autoencoder_"
``` -->

Building
```bash
mkdir build
cd build
cmake ..
make -j<n_cores>
```

