#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <float.h>

/* Helper: Allocates memory with error handling. */
static double *allocate_buffer(size_t size)
{
     double *ptr = (double *)malloc(size * sizeof(double));
     if (!ptr)
     {
          fprintf(stderr, "Memory allocation failed\n");
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
     }
     return ptr;
}

int main(int argc, char *argv[])
{
     MPI_Init(&argc, &argv);

     int rank, P;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &P);

     /* Argument Validation */
     if (argc != 6)
     {
          if (rank == 0)
               printf("Error: Expected 5 arguments.\n");
          MPI_Finalize();
          exit(1);
     }

     /* Parse Input Parameters */
     int M = atoi(argv[1]);    // Buffer size per transmission
     int D1 = atoi(argv[2]);   // Stride/Offset for Path 1
     int D2 = atoi(argv[3]);   // Stride/Offset for Path 2
     int T = atoi(argv[4]);    // Iteration count
     int seed = atoi(argv[5]); // PRNG Seed

     /* --- Topology Definition --- */

     /* Determine Downstream Neighbors (Targets) */
     int target_D1 = rank + D1;
     int target_D2 = rank + D2;

     /* Determine Upstream Neighbors (Sources) */
     int source_D1 = rank - D1;
     int source_D2 = rank - D2;

     /* Establish Roles: Determine if this rank acts as a sender or receiver */
     int i_am_sender_D1 = (target_D1 < P) ? 1 : 0;
     int i_am_sender_D2 = (target_D2 < P) ? 1 : 0;

     int i_am_receiver_D1 = (source_D1 >= 0) ? 1 : 0;
     int i_am_receiver_D2 = (source_D2 >= 0) ? 1 : 0;

     /* --- Resource Allocation: Communication Buffers --- */

     /* 1. Outgoing Buffers (Source Role)
      * distinct buffers required for D1 and D2 paths as data diverges after processing. */
     double *data_to_send_D1 = allocate_buffer(M);
     double *data_to_send_D2 = allocate_buffer(M);

     /* 2. Incoming Buffers (Sink Role)
      * Storage for raw data received from upstream neighbors. */
     double *recv_buf_D1 = allocate_buffer(M); // Input from (Rank - D1)
     double *recv_buf_D2 = allocate_buffer(M); // Input from (Rank - D2)

     /* 3. Return Buffers (Source Awaiting Reply)
      * Storage for processed results returned from downstream neighbors. */
     double *data_recvd_from_D1 = allocate_buffer(M);
     double *data_recvd_from_D2 = allocate_buffer(M);

     /* --- State Initialization --- */

     /* Initialize Pseudo-Random Number Generator deterministically */
     srand(seed);
     for (int i = 0; i < M; i++)
     {
          /* Generate base values based on rank weight */
          double val = (double)rand() * (rank + 1) / 10000.0;

          /* Initial Load: T=0 broadcasts raw values to both paths */
          data_to_send_D1[i] = val;
          data_to_send_D2[i] = val;

          /* Zero-initialize receive buffers to ensure clean state */
          data_recvd_from_D1[i] = 0.0;
          data_recvd_from_D2[i] = 0.0;
          recv_buf_D1[i] = 0.0;
          recv_buf_D2[i] = 0.0;
     }

     /* --- Main Execution Loop: Synchronous Pipeline --- */
     double start_time = MPI_Wtime();

     for (int t = 0; t < T; t++)
     {

          /* ============================================================
           * PHASE A: Forward Propagation (Data Distribution)
           * Protocol: Receive upstream requirements -> Send downstream tasks
           * ============================================================ */

          /* 1. Ingest Data from Upstream (Left Neighbors) */
          if (i_am_receiver_D1)
          {
               MPI_Recv(recv_buf_D1, M, MPI_DOUBLE, source_D1, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }

          if (i_am_receiver_D2)
          {
               MPI_Recv(recv_buf_D2, M, MPI_DOUBLE, source_D2, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }

          /* 2. Dispatch Data to Downstream (Right Neighbors) */
          if (i_am_sender_D1)
          {
               MPI_Send(data_to_send_D1, M, MPI_DOUBLE, target_D1, 100, MPI_COMM_WORLD);
          }

          if (i_am_sender_D2)
          {
               MPI_Send(data_to_send_D2, M, MPI_DOUBLE, target_D2, 101, MPI_COMM_WORLD);
          }

          /* ============================================================
           * PHASE B: Local Kernel Execution (Transformation)
           * ============================================================ */

          /* Kernel D1: Quadratic Transformation */
          if (i_am_receiver_D1)
          {
               for (int i = 0; i < M; i++)
               {
                    recv_buf_D1[i] = recv_buf_D1[i] * recv_buf_D1[i];
               }
          }

          /* Kernel D2: Logarithmic Transformation (with domain safety check) */
          if (i_am_receiver_D2)
          {
               for (int i = 0; i < M; i++)
               {
                    if (recv_buf_D2[i] > 0)
                         recv_buf_D2[i] = log(recv_buf_D2[i]);
                    else
                         recv_buf_D2[i] = 0;
               }
          }

          /* ============================================================
           * PHASE C: Backward Propagation (Result Aggregation)
           * Protocol: Return local results -> Collect downstream results
           * ============================================================ */

          /* 1. Return Processed Data to Upstream Sources */
          if (i_am_receiver_D1)
          {
               MPI_Send(recv_buf_D1, M, MPI_DOUBLE, source_D1, 200, MPI_COMM_WORLD);
          }

          if (i_am_receiver_D2)
          {
               MPI_Send(recv_buf_D2, M, MPI_DOUBLE, source_D2, 201, MPI_COMM_WORLD);
          }

          /* 2. Collect Results from Downstream Targets */
          if (i_am_sender_D1)
          {
               MPI_Recv(data_recvd_from_D1, M, MPI_DOUBLE, target_D1, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }

          if (i_am_sender_D2)
          {
               MPI_Recv(data_recvd_from_D2, M, MPI_DOUBLE, target_D2, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }

          /* ============================================================
           * PHASE D: State Transition (Next Iteration Prep)
           * ============================================================ */

          if (i_am_sender_D1)
          {
               for (int i = 0; i < M; i++)
               {
                    /* Aggregate results from active paths */
                    double combined = data_recvd_from_D1[i];
                    if (i_am_sender_D2)
                    {
                         combined += data_recvd_from_D2[i];
                    }

                    /* Generate inputs for next iteration using heuristic formula */
                    data_to_send_D1[i] = (double)((unsigned long long)combined % 100000);
                    data_to_send_D2[i] = combined * 100000.0;
               }
          }

     } /* End Iterative Loop */

     /* --- Global Statistics: Maximum Value Reduction --- */
     double local_max_D1 = -DBL_MAX;
     double local_max_D2 = -DBL_MAX;

     /* Scan local return buffers for maxima (Senders only) */
     if (i_am_sender_D1)
     {
          for (int i = 0; i < M; i++)
          {
               if (data_recvd_from_D1[i] > local_max_D1)
                    local_max_D1 = data_recvd_from_D1[i];
          }
     }

     if (i_am_sender_D2)
     {
          for (int i = 0; i < M; i++)
          {
               if (data_recvd_from_D2[i] > local_max_D2)
                    local_max_D2 = data_recvd_from_D2[i];
          }
     }

     /* Reduce local maxima to global maxima across all ranks */
     double global_max_D1, global_max_D2;
     MPI_Reduce(&local_max_D1, &global_max_D1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
     MPI_Reduce(&local_max_D2, &global_max_D2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

     double end_time = MPI_Wtime();

     /* Output Results (Rank 0 only) */
     if (rank == 0)
     {
          printf("%lf %lf %lf\n", global_max_D1, global_max_D2, end_time - start_time);
     }

     MPI_Finalize();

     return 0;
}