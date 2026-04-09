#!/bin/bash

#SBATCH --job-name=bench_P48
#SBATCH -N 1                       # Request 1 Node (48/48 = 1)
#SBATCH --ntasks-per-node=48       # ppn = 48
#SBATCH --output=results_P48_%j.out
#SBATCH --error=results_P48_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:10:00

# --- Configuration ---
P=48
PPN=48
PX=6; PY=4; PZ=2
D=7; T=5; SEED=1000; F=2; ISO=500
N1=120; N2=240

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

    # Configuration 1: nx = ny = nz = 120
    echo "  [P=$P][Grid=$N1] Running..."
    mpirun -np $P ./src_exec $D $PPN $PX $PY $PZ $N1 $N1 $N1 $T $SEED $F $ISO

    # Configuration 2: nx = ny = nz = 240
    echo "  [P=$P][Grid=$N2] Running..."
    mpirun -np $P ./src_exec $D $PPN $PX $PY $PZ $N2 $N2 $N2 $T $SEED $F $ISO
done

echo "Experiment P=$P Complete."
