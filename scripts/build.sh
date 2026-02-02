#!/bin/bash
#PBS -q short_cpuQ
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe
#PBS -N build_all

# Submit with:
# qsub -v DATASET_NAME=mnist build.sh

set -euo pipefail

module load singularity-3.4.0

cd hpc4ds-autoencoder

DATASET_NAME=${DATASET_NAME:-mnist}
echo "Building for dataset: ${DATASET_NAME}"

# Define build directories
BUILD_SEQ="build_seq_${DATASET_NAME}"
BUILD_MPI="build_mpi_${DATASET_NAME}"
BUILD_OMP="build_omp_${DATASET_NAME}"
BUILD_HYB="build_hybrid_${DATASET_NAME}"

# --- Function to build a variant ---
build_variant() {
    local dir=$1
    shift
    local cmake_args=("$@")

    echo "----------------------------------------"
    echo "Building in: $dir"
    # Fix: Use :- to handle empty array safely with set -u
    echo "Args: ${cmake_args[*]:-}"
    
    # Clean previous build if it exists
    rm -rf "$dir"
    
    # Configure
    # Fix: Use "${cmake_args[@]:-}" to safely expand empty arrays
    singularity exec singularity.sif \
      cmake -S . -B "$dir" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DDATASET_NAME="${DATASET_NAME}" \
        "${cmake_args[@]:-}"

    # Compile
    singularity exec singularity.sif \
      cmake --build "$dir" -j4
}

# 1. SEQUENTIAL (No extra args, this triggered the error before)
build_variant "$BUILD_SEQ"

# 2. MPI
build_variant "$BUILD_MPI" -DWITH_MPI=ON

# 3. OPENMP
build_variant "$BUILD_OMP" -DWITH_OPENMP=ON

# 4. HYBRID
build_variant "$BUILD_HYB" -DWITH_MPI=ON -DWITH_OPENMP=ON

echo "----------------------------------------"
echo "All builds completed."