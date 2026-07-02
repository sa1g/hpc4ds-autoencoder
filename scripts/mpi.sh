#!/bin/bash
#PBS -q shortCPUQ
#PBS -l walltime=06:00:00
#PBS -l place=scatter
#PBS -j oe
#PBS -N mpi_runs
#PBS -J 0-4

# Submit with:
# qsub -l select=16:ncpus=1:mem=16gb -v DATASET_NAME=mnist scripts/mpi.sh

set -euo pipefail

ml OpenMPI/4.1.6-GCC-13.2.0

cd hpc4ds-autoencoder

DATASET_NAME=${DATASET_NAME:-mnist}
BUILD_ROOT="./build"
BUILD_DIR="${BUILD_ROOT}/build_mpi_${DATASET_NAME}"

# -------------------------
# MPI configurations
# -------------------------
NODES_LIST=(1 2 4 8 16)
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
    --mca btl_vader_single_copy_mechanism none \
    --mca btl_tcp_if_include 192.168.0.0/16 \
    --mca oob_tcp_if_include 192.168.0.0/16 \
    --map-by ppr:1:node \
    --hostfile $PBS_NODEFILE \
    singularity exec singularity.sif ./${BUILD_DIR}/autoencoder

# mpirun -np ${NODES} \
#     --hostfile $PBS_NODEFILE \
#     --map-by ppr:1:node \
#     --mca btl_tcp_if_include 192.168.0.0/16 \
#     --mca oob_tcp_if_include 192.168.0.0/16 \
#     singularity exec singularity.sif \
#     ./${BUILD_DIR}/autoencoder