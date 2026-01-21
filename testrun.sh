#!/bin/bash

#PBS -l select=4:ncpus=4:mpiprocs=4:mem=16gb
#PBS -l place=scatter
#PBS -l walltime=02:00:00

#PBS -q short_cpuQ

#PBS -N testrun
#PBS -o testrun.out
#PBS -e testrun.err

module load singularity-3.4.0
module load openmpi-4.0.4

cd hpc4ds-autoencoder
mpirun -np 16 \
  --mca pml ob1 \
  --mca btl tcp,self \
  --mca btl_tcp_if_exclude lo,docker0 \
  --mca mtl ^psm,psm2 \
  --mca btl_vader_single_copy_mechanism none \
  singularity exec singularity.sif ./build/autoencoder