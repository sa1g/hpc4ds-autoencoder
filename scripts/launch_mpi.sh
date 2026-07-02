#!/bin/bash
NODE_COUNTS=(1 2 4 8 16)

for NODES in "${NODE_COUNTS[@]}"; do
    echo "Submitting job for $NODES nodes..."

    qsub -l select=$NODES:ncpus=1:mpiprocs=1 -l place=scatter -N "mpi_run_${NODES}" -q shortCPUQ -v NODES=$NODES,DATASET_NAME=mnist scripts/run_mpi.pbs
done