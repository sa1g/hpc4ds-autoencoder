#!/bin/bash
NODE_COUNTS=(1 2 4 8 16)

for NODES in "${NODE_COUNTS[@]}"; do
    echo "Submitting job for $NODES nodes..."

    qsub -l select=$NODES:ncpus=1:mpiprocs=1 -l place=scatter -N "comm_bench_${NODES}" -q shortCPUQ -v NODES=$NODES scripts/run_comm_bench.pbs
done