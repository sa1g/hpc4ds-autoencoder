#!/bin/bash
#PBS -q shortCPUQ
#PBS -l walltime=01:00:00
#PBS -l select=16:ncpus=1:mem=2gb
#PBS -j oe
#PBS -N comm_bench
#PBS -J 0-4

# Submit with:
# qsub comm_bench.sh

set -euo pipefail
ml OpenMPI/4.1.6-GCC-13.2.0

cd hpc4ds-autoencoder
DATASET_NAME=${DATASET_NAME:-mnist}
BUILD_ROOT="./build"
BUILD_DIR="${BUILD_ROOT}/build_omp_${DATASET_NAME}/benchmarks"
LOG_DIR="${BUILD_ROOT}/logs"

# -------------------------
# MPI Configuration
# -------------------------
NODES_LIST=(1 2 4 8 16)
NODES=${NODES_LIST[$PBS_ARRAY_INDEX]}

echo "MPI job ${PBS_JOBID}: ${NODES} node(s)"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Directory $BUILD_DIR does not exist. Please run build.sh first."
    exit 1
fi

# -------------------------
# Logging setup
# -------------------------
if [ ! -d "$BUILD_DIR" ]; then
		echo "Error: Directory $BUILD_DIR does not exist. Please run build.sh first."
		exit 1
fi

if [ ! -x "${BUILD_DIR}/b_communication_many_calls" ]; then
	echo "Error: Benchmark executable ${BUILD_DIR}/b_communication_many_calls does not exist."
	exit 1
fi

mkdir -p "${LOG_DIR}"
JSON_FILE="${LOG_DIR}/comm_bench_${NODES}_nodes.json"
echo "Logging benchmark results to: ${JSON_FILE}"


# -------------------------
# Run
# -------------------------
mpirun -np ${NODES} singularity exec singularity.sif \
    ./${BUILD_DIR}/b_communication_many_calls \
    --benchmark_out="${JSON_FILE}" \ 
    --benchmark_out_format=json

echo "Wrote benchmark JSON to ${JSON_FILE}"
