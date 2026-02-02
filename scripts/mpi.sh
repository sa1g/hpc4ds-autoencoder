#!/bin/bash
#PBS -q short_cpuQ
#PBS -l walltime=06:00:00
#PBS -l place=scatter
#PBS -j oe
#PBS -N mpi_runs
#PBS -J 0-3

# Submit with:
# qsub -l select=8:ncpus=1:mem=16gb -v DATASET_NAME=mnist mpi.sh

set -euo pipefail

module load singularity-3.4.0
module load openmpi-4.0.4

cd hpc4ds-autoencoder

DATASET_NAME=${DATASET_NAME:-mnist}
BUILD_DIR="build_mpi_${DATASET_NAME}"

# -------------------------
# MPI configurations
# -------------------------
NODES_LIST=(1 2 4 8)
NODES=${NODES_LIST[$PBS_ARRAY_INDEX]}

echo "MPI job ${PBS_JOBID}: ${NODES} node(s)"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Directory $BUILD_DIR does not exist. Please run build.sh first."
    exit 1
fi

# -------------------------
# Run
# -------------------------
mpirun -np ${NODES} \
  --mca pml ob1 \
  --mca btl tcp,self \
  --mca btl_tcp_if_exclude lo,docker0 \
  --mca mtl ^psm,psm2 \
  --mca btl_vader_single_copy_mechanism none \
  singularity exec singularity.sif \
  ./${BUILD_DIR}/autoencoder