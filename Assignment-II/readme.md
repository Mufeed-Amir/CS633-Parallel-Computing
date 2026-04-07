# CS633 --- Assignment I

## Overview

This repository contains the implementation for **Assignment II** of CS633.\
The project is designed to be executed on an HPC cluster using **MPI (Message Passing Interface)** for parallel processing.

Workflow

-   Transfer files to the cluster
-   Configure the environment
-   Compile the application
-   Submit SLURM jobs
-   Validate outputs
-   Retrieve results

------------------------------------------------------------------------

## Prerequisites

Before proceeding, ensure:

-   You have SSH access to the cluster.
-   MPI-compatible modules are available.
-   A directory named `bekar_chole_wale` exists in your **login home directory**.

------------------------------------------------------------------------

## Project Structure

    bekar_chole_wale/
    │
    ├── src_main.c
    ├── run_P32.sh
    ├── run_P48.sh
    ├── run_P64.sh
    ├── run_P96.sh
    └── src_exec (generated after compilation)

------------------------------------------------------------------------

## 1. Transfer Files to the Cluster

Run from your local machine:

``` bash
scp -P 4422 src_main.c run_P32.sh run_P48.sh run_P64.sh run_P96.sh iitk_21@paramrudra.cdacdelhi.in:~/bekar_chole_wale/
```

Verify the transfer:

``` bash
ls -l ~/bekar_chole_wale
```

------------------------------------------------------------------------

## 2. Prepare the Environment

Always begin with a clean module environment to avoid dependency
conflicts.

``` bash
module purge
module load compiler/gcc/12.3
module load compiler/openmpi/4.1.5
```

------------------------------------------------------------------------

## 3. Compile the Application

``` bash
mpicc -O3 -ffast-math -march=native src_main.c -o src_exec -lm
mpicc src_main.c -o src_exec -lm -O3
```
> First one is recommended for the best output

### Recommended Checks

Confirm the executable exists:

``` bash
ls -lh src_exec
```

Optional validation:

``` bash
which mpicc
mpicc --version
```

------------------------------------------------------------------------

## 4. Submit SLURM Jobs

Submit each configuration separately:

``` bash
sbatch run_P32.sh
sbatch run_P48.sh
sbatch run_P64.sh
sbatch run_P96.sh
```

Check job status:

``` bash
squeue -u $USER
```

------------------------------------------------------------------------

## 5. Verify Execution Results

After completion, inspect:

### Standard Output

``` bash
cat results_P<process>_<jobID>.out
```

### Error Log

``` bash
cat results_P<process>_<jobID>.err
```

> Note: Warnings related to network transports (e.g., InfiniBand) are
> often non-fatal for single-node jobs.

------------------------------------------------------------------------

## 6. Retrieve Results to Local Machine

``` bash
scp -P 4422 iitk_21@paramrudra.cdacdelhi.in:"~/bekar_chole_wale/*.out" ./results/
```

------------------------------------------------------------------------