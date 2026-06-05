#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

cd "$PROJECT_ROOT"

DATASET_NAME=${DATASET_NAME:-mnist}
echo "Building for dataset: ${DATASET_NAME}"

# Common build root
BUILD_ROOT="./build"
mkdir -p "$BUILD_ROOT"

# Define build directories
BUILD_SEQ="${BUILD_ROOT}/build_seq_${DATASET_NAME}"
BUILD_MPI="${BUILD_ROOT}/build_mpi_${DATASET_NAME}"
BUILD_OMP="${BUILD_ROOT}/build_omp_${DATASET_NAME}"
BUILD_HYB="${BUILD_ROOT}/build_hybrid_${DATASET_NAME}"

# --- Function to build a variant ---
build_variant() {
    local dir=$1
    shift
    local cmake_args=("$@")

    echo "----------------------------------------"
    echo "Building in: $dir"
    echo "Args: ${cmake_args[*]:-}"

    # Clean previous build if it exists
    rm -rf "$dir"

    # Configure
    singularity exec singularity.sif \
      cmake -S . -B "$dir" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DDATASET_NAME="${DATASET_NAME}" \
        "${cmake_args[@]:-}"

    # Compile
    singularity exec singularity.sif \
      cmake --build "$dir" -j16
}

# 1. SEQUENTIAL (No extra args, this triggered the error before)
build_variant "$BUILD_SEQ" -DCMAKE_BUILD_TYPE=Release

# 2. MPI
build_variant "$BUILD_MPI" -DWITH_MPI=ON -DCMAKE_BUILD_TYPE=Release

# 3. OPENMP
build_variant "$BUILD_OMP" -DWITH_OPENMP=ON -DCMAKE_BUILD_TYPE=Release

# 4. HYBRID
build_variant "$BUILD_HYB" -DWITH_MPI=ON -DWITH_OPENMP=ON -DCMAKE_BUILD_TYPE=Release

echo "----------------------------------------"
echo "All builds completed."