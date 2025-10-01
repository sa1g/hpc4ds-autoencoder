# hpc4ds-autoencoder

Autoencoder developed from scratch, benchmarked on a single node with no parallelism, with parallelism (OpenMP), multi-node (MPI) and "hybrid" (OpenMP + MPI) with a data parallel setup.

## Requirements
- C++20
- cmake >= 4.0.0
- protobuf
- openmp
- mpi

## Environment
The environment is build with Apptainer/Singuarity.  

Building the environment will take a while as protobuf needs to be compiled from source:  
```bash
apptainer build cpp_env.sif cpp_env.conf
```

## Building
Most libraries are dynamically linked, so everything must be run inside of singularity.

Optional flag: `WITH_OPENMP`, either `ON` or not set, to enable threading with omp.

```bash
singularity exec cpp_env.sif cmake -B build -DCMAKE_BUILD_TYPE=Release
singularity exec cpp_env.sif cmake --build build/ -j$(nproc)
cd build
singularity exec cpp_env.sif ./autoencoder
```


// mpirun -np 32 singularity exec my_container.sif ./my_app

https://docs.sylabs.io/guides/3.3/user-guide/mpi.html

## About

run only my tests

```bash
ctest -L autoencoder_tests
```

<!-- TODO: or with prefix: `autoencoder_`:
```bash
ctest -R "^autoencoder_"
``` -->



## Testing
```bash
cd build

# Either
ctest -L "autoencoder_tests"
# or
ctest --output-on-failure
# or whatever ctest config you want
```



## Course slides
- lecture 13 for instructions on what to put in the report
