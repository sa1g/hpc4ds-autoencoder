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

# -------------------------
# Compile
# -------------------------
if [ ! -d "$BUILD_DIR" ]; then
  singularity exec singularity.sif \
    cmake -S . -B ${BUILD_DIR} \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DDATASET_NAME=${DATASET_NAME}

  singularity exec singularity.sif \
    cmake --build ${BUILD_DIR} -j1
fi

# -------------------------
# Run (single core)
# -------------------------
singularity exec singularity.sif \
  ./${BUILD_DIR}/autoencoder
