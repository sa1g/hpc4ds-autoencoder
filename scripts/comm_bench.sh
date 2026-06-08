#!/bin/bash
#PBS -q shortCPUQ
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -N comm_bench
#PBS -J 0-4

# qsub -l select=16:ncpus=1:mem=2gb -v DATASET_NAME=mnist comm_bench.sh

set -euo pipefail
ml OpenMPI/4.1.6-GCC-13.2.0

cd hpc4ds-autoencoder
DATASET_NAME=${DATASET_NAME:-mnist}
BUILD_ROOT="./build"
BUILD_DIR="${BUILD_ROOT}/build_hybrid_${DATASET_NAME}/benchmarks"
LOG_DIR="logs"

# MPI Configuration
NODES_LIST=(1 2 4 8 16)
NODES=${NODES_LIST[$PBS_ARRAY_INDEX]}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME=${PBS_JOBNAME:-comm_bench}
RUN_BASE="${JOB_NAME}_${TIMESTAMP}_nnodes_${NODES}"

echo "MPI job ${PBS_JOBID}: ${NODES} node(s)"
echo "Allocated nodes: $(cat $PBS_NODEFILE | sort -u | wc -l)"
echo "Total cores: $(cat $PBS_NODEFILE | wc -l)"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Directory $BUILD_DIR does not exist. Please run build.sh first."
    exit 1
fi

if [ ! -x "${BUILD_DIR}/b_communication_many_calls" ]; then
    echo "Error: Benchmark executable does not exist."
    exit 1
fi

mkdir -p "${LOG_DIR}"
RUN_LOG_FILE="${LOG_DIR}/${RUN_BASE}.out"
JSON_FILE="${LOG_DIR}/${RUN_BASE}.json"

exec > >(tee -a "${RUN_LOG_FILE}") 2>&1

echo "Logging benchmark output to: ${RUN_LOG_FILE}"
echo "Logging benchmark results to: ${JSON_FILE}"

# Run with mpiexec (PBS/Torque's launcher)
mpirun -np ${NODES} \
    --hostfile $PBS_NODEFILE \
    --map-by ppr:1:node \
    --mca pml ob1 \
    --mca btl tcp,self \
    --mca btl_tcp_if_exclude lo,docker0 \
    --mca mtl ^psm,psm2 \
    --mca btl_vader_single_copy_mechanism none \
    singularity exec singularity.sif \
    ./${BUILD_DIR}/b_communication_many_calls \
    --benchmark_out="${JSON_FILE}" \
    --benchmark_out_format=json

echo "Wrote benchmark JSON to ${JSON_FILE}"