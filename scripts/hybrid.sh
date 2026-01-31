#!/bin/bash
#PBS -q short_cpuQ
#PBS -l walltime=06:00:00
#PBS -l place=scatter
#PBS -j oe
#PBS -N hybrid_runs

# Submit with:
# qsub -l select=8:ncpus=8:mem=16gb -v DATASET_NAME=mnist hybrid.sh

set -euo pipefail

module load singularity-3.4.0
module load openmpi-4.0.4

cd hpc4ds-autoencoder

DATASET_NAME=${DATASET_NAME:-mnist}
BUILD_DIR="build_hybrid_${DATASET_NAME}"

echo "Hybrid MPI+OpenMP – dataset: ${DATASET_NAME}"

# -------------------------
# Compile
# -------------------------
if [ ! -d "$BUILD_DIR" ]; then
  singularity exec singularity.sif \
    cmake -S . -B ${BUILD_DIR} \
      -DWITH_MPI=ON \
      -DWITH_OPENMP=ON \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DDATASET_NAME=${DATASET_NAME}

  singularity exec singularity.sif \
    cmake --build ${BUILD_DIR} -j1
fi

# -------------------------
# Hybrid runs
# -------------------------
MAX_NODES=$(wc -l < "$PBS_NODEFILE")

for CORES in 1 2 4 8; do
  export OMP_NUM_THREADS=${CORES}

  for NODES in 1 2 4 8; do
    if [ "$NODES" -le "$MAX_NODES" ]; then
      echo "Hybrid run: ${NODES} node(s), ${CORES} thread(s) per rank"

      mpirun -np ${NODES} \
        singularity exec singularity.sif \
        ./${BUILD_DIR}/autoencoder
    fi
  done
done
