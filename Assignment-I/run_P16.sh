#!/bin/bash

#SBATCH --job-name=bench_P16
#SBATCH -N 1                       # Request 1 Node
#SBATCH --ntasks-per-node=16       # 16 tasks (P=16 fits on 1 node)
#SBATCH --output=results_P16_%j.out
#SBATCH --error=results_P16_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:10:00

# --- Configuration ---
P=16
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

    # Configuration 3: M = 262,144
    echo "  [P=$P][M=$M1] Running..."
    mpirun -np $P ./src_exec $M1 $D1 $D2 $T $SEED

    # Configuration 4: M = 1,048,576
    echo "  [P=$P][M=$M2] Running..."
    mpirun -np $P ./src_exec $M2 $D1 $D2 $T $SEED
done

echo "Experiment P=$P Complete."