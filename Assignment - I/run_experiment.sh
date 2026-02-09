#!/bin/bash

# Configuration
P=8
D1=2
D2=4
T=10
SEED=1000
OUTPUT_FILE="results_P$P.txt"

# Clear previous results
> $OUTPUT_FILE

echo "Starting Benchmark for P=$P..."

# Execution Loop (1 to 5)
for i in {1..5}
do
    echo "  Iteration $i of 5..."
    
    # Run for M = 262,144
    echo "    Running M=262144..."
    mpirun -np $P -f hostfile ./src 262144 $D1 $D2 $T $SEED >> $OUTPUT_FILE
    
    # Run for M = 1,048,576
    echo "    Running M=1048576..."
    mpirun -np $P -f hostfile ./src 1048576 $D1 $D2 $T $SEED >> $OUTPUT_FILE
done

echo "P=$P Benchmark Complete. Data saved to $OUTPUT_FILE"