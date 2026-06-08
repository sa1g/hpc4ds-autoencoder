#!/bin/bash
#PBS -q shortCPUQ
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -N omp_runs
#PBS -J 0-4

# Submit with:
# qsub -l select=1:ncpus=16:mem=16gb -v DATASET_NAME=mnist omp.sh

set -euo pipefail



cd hpc4ds-autoencoder

DATASET_NAME=${DATASET_NAME:-mnist}
BUILD_ROOT="./build"
BUILD_DIR="${BUILD_ROOT}/build_omp_${DATASET_NAME}"

CORES_LIST=(1 2 4 8 16)
CORES=${CORES_LIST[$PBS_ARRAY_INDEX]}

echo "OpenMP job ${PBS_JOBID}: ${CORES} threads"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Directory $BUILD_DIR does not exist. Please run build.sh first."
    exit 1
fi

# -------------------------
# Run
# -------------------------
export OMP_NUM_THREADS=${CORES}

singularity exec singularity.sif \
  ./${BUILD_DIR}/autoencoder