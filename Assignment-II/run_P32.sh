#!/bin/bash

#SBATCH --job-name=bench_P32
#SBATCH -N 1                       # Request 1 Node (CRITICAL for P=32)
#SBATCH --ntasks-per-node=32       # 32 tasks per node * 1 node = 32 tasks
#SBATCH --output=results_P32_%j.out
#SBATCH --error=results_P32_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:10:00

# --- Configuration ---
P=32
PX=4
PY=4
PZ=2
PPN=32
D=7
SEED=1000
ISOVALUE=500
T=5
F=2

echo "=========================================="
echo "Starting Experiment for P=$P"
echo "Node Allocation: $SLURM_JOB_NUM_NODES nodes"
echo "=========================================="

module purge
module load compiler/gcc/12.3
module load compiler/openmpi/4.1.5

mpicc src.c -o src_exec -lm -O3

# --- Execution Loop (Repeat 5 times) ---
for i in {1..5}
do
    echo "Iteration $i of 5..."

    # Configuration nx=ny=nz=120
    echo "  [P=$P][N=120] Running..."
    # Note: With P=32, this will span across the 2 nodes requested
    mpirun -np $P ./src_exec $D $PX $PY $PZ 120 120 120 $T $SEED $F $ISOVALUE

    # Configuration nx=ny=nz=240
    echo "  [P=$P][N=240] Running..."
    mpirun -np $P ./src_exec $D $PX $PY $PZ 240 240 240 $T $SEED $F $ISOVALUE
done

echo "Experiment P=$P Complete."