#!/bin/bash
#PBS -q short_cpuQ
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=1:mem=64gb
#PBS -j oe
#PBS -N seq_run

# Submit with:
# qsub -v DATASET_NAME=mnist sequential.sh

set -euo pipefail

module load singularity-3.4.0

cd hpc4ds-autoencoder

DATASET_NAME=${DATASET_NAME:-mnist}
BUILD_DIR="build_seq_${DATASET_NAME}"

echo "Sequential run – dataset: ${DATASET_NAME}"

# Check if build exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Directory $BUILD_DIR does not exist. Please run build.sh first."
    exit 1
fi

# -------------------------
# Run
# -------------------------
singularity exec singularity.sif \
  ./${BUILD_DIR}/autoencoder