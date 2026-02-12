#!/bin/bash

#SBATCH --job-name=bench_P32
#SBATCH -N 2                       # Request 2 Nodes (CRITICAL for P=32)
#SBATCH --ntasks-per-node=16       # 16 tasks per node * 2 nodes = 32 tasks
#SBATCH --output=results_P32_%j.out
#SBATCH --error=results_P32_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:10:00

# --- Configuration ---
P=32
D1=2
D2=4
T=10
SEED=1000
M1=262144
M2=1048576

echo "=========================================="
echo "Starting Experiment for P=$P"
echo "Node Allocation: $SLURM_JOB_NUM_NODES nodes"
echo "=========================================="

module purge
module load compiler/gcc/12.3
module load compiler/openmpi/4.1.5

# --- Execution Loop (Repeat 5 times) ---
for i in {1..5}
do
    echo "Iteration $i of 5..."

    # Configuration 5: M = 262,144
    echo "  [P=$P][M=$M1] Running..."
    # Note: With P=32, this will span across the 2 nodes requested
    mpirun -np $P ./src_exec $M1 $D1 $D2 $T $SEED

    # Configuration 6: M = 1,048,576
    echo "  [P=$P][M=$M2] Running..."
    mpirun -np $P ./src_exec $M2 $D1 $D2 $T $SEED
done

echo "Experiment P=$P Complete."