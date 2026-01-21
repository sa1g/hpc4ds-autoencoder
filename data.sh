#!/bin/bash

#PBS -l select=1:ncpus=16:mem=16gb
#PBS -l walltime=2:00:00

#PBS -q short_cpuQ

#PBS -N data
#PBS -o data.out
#PBS -e data.err

module load python-3.8.13
module load cmake-3.15.4

cd ~/hpc4ds-autoencoder/
make -j$(nproc)
