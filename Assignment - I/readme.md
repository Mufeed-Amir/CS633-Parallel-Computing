# CS633 — Assignment I

## Overview
This repository contains the work for Assignment I of CS633. Follow the instructions below to build and run.

### Compile and run (MPI)

1. Compile:
     ```
     mpicc src.c -o src.o
     ```
2. Run with 4 MPI processes:
     ```
     mpirun -np 4 ./src.o 2 1 2 1 100
     mpirun -np 4 ./src.o 262144 2 4 10 1000
     ```
3. Check nodes
     ```
          sinfo -p standard -N -o "%N"
     ```