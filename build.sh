#!/bin/bash

#PBS -l select=1:ncpus=1:mem=16gb
#PBS -q short_cpuQ
#PBS -l walltime=00:30:00

#PBS -N build
#PBS -o build.out
#PBS -e build.err

module load singularity-3.4.0
module load openmpi-4.0.4

cd hpc4ds-autoencoder

singularity exec singularity.sif cmake -S . -B build -DWITH_MPI=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
singularity exec singularity.sif cmake --build build/ -j$(nproc)
