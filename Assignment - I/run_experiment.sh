#!/bin/bash

#SBATCH --job-name=run_experiment
#SBATCH -N 2
#SBATCH --ntasks-per-node=16
#SBATCH --output=run_experiment_%j.out
#SBATCH --error=run_experiment_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:05:00

# Configuration
P=8
D1=2
D2=4
T=10 
SEED=1000

# Clear previous results

echo "Starting Benchmark for P=$P..."

module load compiler/oneapi-2024/mpi

# Execution Loop (1 to 5)
for i in {1..5}
do
    echo "  Iteration $i of 5..."
    
    # Run for M = 262,144
    echo "    Running M=262144..."
    mpirun -np $P -f hostfile ./src.o 262144 $D1 $D2 $T $SEED 
    
    # Run for M = 1,048,576
    echo "    Running M=1048576..."
    mpirun -np $P -f hostfile ./src.o 1048576 $D1 $D2 $T $SEED 
done

echo "P=$P Benchmark Complete. Data saved."