#!/bin/bash
#PBS -q short_cpuQ
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -N hybrid_runs
#PBS -J 0-15

# Submit with:
# qsub -l select=8:ncpus=8:mem=16gb -v DATASET_NAME=mnist hybrid.sh

set -euo pipefail

module load singularity-3.4.0
module load openmpi-4.0.4

cd hpc4ds-autoencoder

DATASET_NAME=${DATASET_NAME:-mnist}
BUILD_DIR="build_hybrid_${DATASET_NAME}"

# -------------------------
# Parameter grid
# -------------------------
CORES_LIST=(1 2 4 8)
NODES_LIST=(1 2 4 8)

IDX=${PBS_ARRAY_INDEX}

CORES=${CORES_LIST[$((IDX % 4))]}
NODES=${NODES_LIST[$((IDX / 4))]}

echo "Hybrid job ${PBS_JOBID}: ${NODES} node(s), ${CORES} OMP threads"

# -------------------------
# Resource sanity check
# -------------------------
REQ_CPUS=$((NODES * CORES))
ALLOC_CPUS=$(wc -l < "$PBS_NODEFILE")

if [ "$REQ_CPUS" -gt "$ALLOC_CPUS" ]; then
  echo "Requested ${REQ_CPUS} CPUs, got ${ALLOC_CPUS}. Skipping."
  exit 0
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Directory $BUILD_DIR does not exist. Please run build.sh first."
    exit 1
fi

# -------------------------
# Run
# -------------------------
export OMP_NUM_THREADS=${CORES}

# Note: Added MCA parameters for stability, consistent with MPI script
mpirun -np ${NODES} \
  --mca pml ob1 \
  --mca btl tcp,self \
  --mca btl_tcp_if_exclude lo,docker0 \
  --mca mtl ^psm,psm2 \
  --mca btl_vader_single_copy_mechanism none \
  singularity exec singularity.sif \
  ./${BUILD_DIR}/autoencoder