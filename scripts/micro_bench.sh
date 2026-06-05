#!/bin/bash
#PBS -q shortCPUQ
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=16:mem=16gb
#PBS -j oe
#PBS -N micro_bench_runs
#PBS -J 0-34

# Submit with:
# qsub -v DATASET_NAME=mnist micro_bench.sh

set -euo pipefail



BUILD_ROOT="./build"
DATASET_NAME=${DATASET_NAME:-mnist}
BUILD_DIR="${BUILD_ROOT}/build_omp_${DATASET_NAME}/benchmarks"
LOG_DIR="${BUILD_ROOT}/logs"

BENCHMARKS=(
	b_dataloader
	b_linear
	b_mse
	b_relu
	b_sgd
	b_sigmoid
	b_train
)

CORES_LIST=(1 2 4 8 16)

# Default to 0 when not running as a PBS array job
IDX=${PBS_ARRAY_INDEX:-0}
BENCHMARK_INDEX=$((IDX / ${#CORES_LIST[@]}))
THREAD_INDEX=$((IDX % ${#CORES_LIST[@]}))

BENCHMARK=${BENCHMARKS[$BENCHMARK_INDEX]}
CORES=${CORES_LIST[$THREAD_INDEX]}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME=${PBS_JOBNAME:-micro_bench_runs}
RUN_BASE="${JOB_NAME}_${TIMESTAMP}_${BENCHMARK}_nthreads_${CORES}"

# Ensure log dir exists before redirecting output
mkdir -p "${LOG_DIR}"

RUN_LOG_FILE="${LOG_DIR}/${RUN_BASE}.out"
JSON_FILE="${LOG_DIR}/${RUN_BASE}.json"

exec > >(tee -a "${RUN_LOG_FILE}") 2>&1

echo "Micro benchmark job ${PBS_JOBID}: ${BENCHMARK} with ${CORES} threads"
echo "Logging to: ${RUN_LOG_FILE}"
echo "Benchmark JSON: ${JSON_FILE}"

if [ ! -d "$BUILD_DIR" ]; then
		echo "Error: Directory $BUILD_DIR does not exist. Please run build.sh first."
		exit 1
fi

if [ ! -x "${BUILD_DIR}/${BENCHMARK}" ]; then
	echo "Error: Benchmark executable ${BUILD_DIR}/${BENCHMARK} does not exist."
	exit 1
fi

mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS=${CORES}

singularity exec singularity.sif \
    ${BUILD_DIR}/${BENCHMARK} \
	--benchmark_out="${JSON_FILE}" \
	--benchmark_out_format=json

echo "Wrote benchmark JSON to ${JSON_FILE}"