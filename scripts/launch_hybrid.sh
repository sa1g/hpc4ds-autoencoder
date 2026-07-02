#!/bin/bash
NODE_COUNTS=(1 2 4 8 16)
THREAD_COUNTS=(1 2 4 8 16)

for NODES in "${NODE_COUNTS[@]}"; do
    for THREADS in "${THREAD_COUNTS[@]}"; do
        echo "Submitting hybrid job for $NODES nodes and $THREADS threads..."

        qsub \
        -l select=$NODES:ncpus=$THREADS:mpiprocs=1 \
        -l place=scatter \
        -N "hybrid_${NODES}n_${THREADS}t" \
        -q shortCPUQ \
        -v NODES=$NODES,THREADS=$THREADS,DATASET_NAME=mnist \
        scripts/run_hybrid.pbs
    done
done