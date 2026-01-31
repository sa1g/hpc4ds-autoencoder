#!/bin/bash
#PBS -q short_cpuQ
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -N omp_runs

# Submit with:
# qsub -l select=1:ncpus=8:mem=16gb -v DATASET_NAME=mnist omp.sh

set -euo pipefail

module load singularity-3.4.0

cd hpc4ds-autoencoder

DATASET_NAME=${DATASET_NAME:-mnist}
BUILD_DIR="build_omp_${DATASET_NAME}"

echo "OpenMP – dataset: ${DATASET_NAME}"

# -------------------------
# Compile
# -------------------------
if [ ! -d "$BUILD_DIR" ]; then
  singularity exec singularity.sif \
    cmake -S . -B ${BUILD_DIR} \
      -DWITH_OPENMP=ON \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DDATASET_NAME=${DATASET_NAME}

  singularity exec singularity.sif \
    cmake --build ${BUILD_DIR} -j1
fi

# -------------------------
# Runs
# -------------------------
for CORES in 1 2 4 8; do
  echo "Running OpenMP with ${CORES} threads"
  export OMP_NUM_THREADS=${CORES}

  singularity exec singularity.sif \
    ./${BUILD_DIR}/autoencoder
done
