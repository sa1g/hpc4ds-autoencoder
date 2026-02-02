#!/bin/bash
#PBS -q short_cpuQ
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -N omp_runs
#PBS -J 0-3

# Submit with:
# qsub -l select=1:ncpus=8:mem=16gb -v DATASET_NAME=mnist omp.sh

set -euo pipefail

module load singularity-3.4.0

cd hpc4ds-autoencoder

DATASET_NAME=${DATASET_NAME:-mnist}
BUILD_DIR="build_omp_${DATASET_NAME}"

CORES_LIST=(1 2 4 8)
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