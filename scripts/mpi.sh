#!/bin/bash
#PBS -q short_cpuQ
#PBS -l walltime=06:00:00
#PBS -l place=scatter
#PBS -j oe
#PBS -N mpi_runs

# Submit with:
# qsub -l select=8:ncpus=1:mem=16gb -v DATASET_NAME=mnist mpi.sh

set -euo pipefail

module load singularity-3.4.0
module load openmpi-4.0.4

cd hpc4ds-autoencoder

# -------------------------
# Dataset
# -------------------------
DATASET_NAME=${DATASET_NAME:-mnist}
echo "Dataset: ${DATASET_NAME}"

# -------------------------
# Build directory
# -------------------------
BUILD_DIR="build_mpi_${DATASET_NAME}"

# -------------------------
# Compile (only if needed)
# -------------------------
if [ ! -d "$BUILD_DIR" ]; then
  echo "Building MPI version for dataset ${DATASET_NAME}..."
  singularity exec singularity.sif \
    cmake -S . -B ${BUILD_DIR} \
      -DWITH_MPI=ON \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DDATASET_NAME=${DATASET_NAME}

  singularity exec singularity.sif \
    cmake --build ${BUILD_DIR} -j1
else
  echo "Using existing build: ${BUILD_DIR}"
fi

# -------------------------
# MPI runs
# -------------------------
MAX_NODES=$(wc -l < "$PBS_NODEFILE")

for NODES in 1 2 4 8; do
  if [ "$NODES" -le "$MAX_NODES" ]; then
    echo "Running MPI with ${NODES} node(s)"

    mpirun -np ${NODES} \
      --mca pml ob1 \
      --mca btl tcp,self \
      --mca btl_tcp_if_exclude lo,docker0 \
      --mca mtl ^psm,psm2 \
      --mca btl_vader_single_copy_mechanism none \
      singularity exec singularity.sif \
      ./${BUILD_DIR}/autoencoder
  else
    echo "Skipping ${NODES} nodes (not allocated)"
  fi
done
