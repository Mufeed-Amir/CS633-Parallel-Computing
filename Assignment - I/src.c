/*
 * MPI Pipeline Simulation with Odd-Even Communication & Tree Reduction
 *
 * Description:
 * Simulates a dual-dependency pipeline (D1, D2) across P processors.
 * - Uses blocking MPI_Send/MPI_Recv with an Odd-Even (Red-Black) ordering
 * strategy to prevent deadlock and serialization.
 * - Implements a custom O(log P) bitwise tree reduction for global maximums.
 *
 * Usage:
 * mpirun -np <P> ./program <M> <D1> <D2> <T> <seed>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <float.h>

/* --- Constants for MPI Message Tags --- */
#define TAG_FWD_D1 100
#define TAG_FWD_D2 101
#define TAG_RET_D1 200
#define TAG_RET_D2 201
#define TAG_RED_D1 500
#define TAG_RED_D2 501

/* * Function: allocate_buffer
 * -------------------------
 * Allocates a double array of size 'size'.
 * Aborts the MPI environment if allocation fails.
 */
static double *allocate_buffer(size_t size)
{
     double *ptr = (double *)malloc(size * sizeof(double));
     if (!ptr)
     {
          fprintf(stderr, "[Error] Memory allocation failed for size %zu\n", size);
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
     }
     return ptr;
}

static inline int computeTarget(const int rank, const int d){
	return rank + d;
}

static inline int computeSource(const int rank, const int d){
	return rank - d;
}

static inline int iAmSender(const int rank, const int d, const int P){
	return computeTarget(rank, d) < P;
}

static inline int iAmReceiver(const int rank, const int d, const int P){
	return computeSource(rank, d) >= 0;
}

void doCommunication(const int rank, const int d, const int P, const int M, double* recv_buf, double* data_to_send, int tag, int is_forward){
     int i_am_receiver = iAmReceiver(rank, d, P);
     int i_am_sender = iAmSender(rank, d, P);
     int target = computeTarget(rank, d);
     int source = computeSource(rank, d);
     
     if (i_am_receiver && !i_am_sender)
     {
          // Tail
          if(is_forward){
               MPI_Recv(recv_buf, M, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
          else{
               MPI_Send(data_to_send, M, MPI_DOUBLE, source, tag, MPI_COMM_WORLD);
          }
     }
     else if (i_am_sender && !i_am_receiver)
     {
          // Head
          if(is_forward){
               MPI_Send(data_to_send, M, MPI_DOUBLE, target, tag, MPI_COMM_WORLD);
          }
          else{
               MPI_Recv(recv_buf, M, MPI_DOUBLE, target, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
     }
     else if (i_am_sender && i_am_receiver)
     {
          // Middle: Parity Check
          if ((rank / d) % 2 != 0)
          { // Odd parity block
               if(is_forward){
                    MPI_Recv(recv_buf, M, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(data_to_send, M, MPI_DOUBLE, target, tag, MPI_COMM_WORLD);
               }
               else{
                    MPI_Send(data_to_send, M, MPI_DOUBLE, source, tag, MPI_COMM_WORLD);
                    MPI_Recv(recv_buf, M, MPI_DOUBLE, target, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
               }
          }
          else
          { // Even parity block
               if(is_forward){
                    MPI_Send(data_to_send, M, MPI_DOUBLE, target, tag, MPI_COMM_WORLD);
                    MPI_Recv(recv_buf, M, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
               }
               else{
                    MPI_Recv(recv_buf, M, MPI_DOUBLE, target, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(data_to_send, M, MPI_DOUBLE, source, tag, MPI_COMM_WORLD);
               }
          }
     }
}

int main(int argc, char *argv[])
{
     /* --- MPI Initialization --- */
     MPI_Init(&argc, &argv);

     int rank, P;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &P);

     /* --- Input Validation --- */
     if (argc != 6)
     {
          if (rank == 0)
          {
               fprintf(stderr, "Usage: %s <M> <D1> <D2> <T> <seed>\n", argv[0]);
          }
          MPI_Finalize();
          exit(EXIT_FAILURE);
     }

     /* --- Argument Parsing --- */
     int M = atoi(argv[1]);    // Array size
     int D1 = atoi(argv[2]);   // Dependency offset 1
     int D2 = atoi(argv[3]);   // Dependency offset 2
     int T = atoi(argv[4]);    // Time steps
     int seed = atoi(argv[5]); // Random seed

     /* --- Memory Allocation --- */
     double *data_to_send_D1 = allocate_buffer(M);
     double *data_to_send_D2 = allocate_buffer(M);
     double *recv_buf_D1 = allocate_buffer(M);
     double *recv_buf_D2 = allocate_buffer(M);
     double *data_recvd_from_D1 = allocate_buffer(M);
     double *data_recvd_from_D2 = allocate_buffer(M);

     /* --- Data Initialization --- */
     srand(seed);
     for (int i = 0; i < M; i++)
     {
          double val = (double)rand() * (rank + 1) / 10000.0;
          data_to_send_D1[i] = val;
          data_to_send_D2[i] = val;
          data_recvd_from_D1[i] = 0.0;
          data_recvd_from_D2[i] = 0.0;
          recv_buf_D1[i] = 0.0;
          recv_buf_D2[i] = 0.0;
     }

     double start_time = MPI_Wtime();

     /* =========================================================================
      * Main Simulation Loop
      * ========================================================================= */
     for (int t = 0; t < T; t++)
     {
          /* * PHASE 1: Forward Communication (Tasks)
           * Strategy: Use blocking calls with Odd-Even ordering to prevent deadlock.
           * - Middle nodes (Sender & Receiver) alternate Send/Recv order based on parity.
           * - Edges (Only Sender or Only Receiver) execute directly.
           */

          /* --- D1 Chain --- */
          doCommunication(rank, D1, P, M, recv_buf_D1, data_to_send_D1, TAG_FWD_D1, 1);

          /* --- D2 Chain --- */
          doCommunication(rank, D2, P, M, recv_buf_D2, data_to_send_D2, TAG_FWD_D2, 1);

          /* * PHASE 2: Computation
           */
          if (iAmReceiver(rank, D1, P))
          {
               for (int i = 0; i < M; i++)
               {
                    recv_buf_D1[i] *= recv_buf_D1[i];
               }
          }

          if (iAmReceiver(rank, D2, P))
          {
               for (int i = 0; i < M; i++)
               {
                    recv_buf_D2[i] = (recv_buf_D2[i] > 0) ? log(recv_buf_D2[i]) : 0;
               }
          }

          /* * PHASE 3: Backward Communication (Results)
           * Note: Logic flips. The 'Receiver' of tasks becomes the 'Sender' of results.
           */

          /* --- D1 Chain Results --- */
          doCommunication(rank, D1, P, M, data_recvd_from_D1, recv_buf_D1, TAG_RET_D1, 0);

          /* --- D2 Chain Results --- */
          doCommunication(rank, D2, P, M, data_recvd_from_D2, recv_buf_D2, TAG_RET_D2, 0);

          /* * PHASE 4: Update for Next Iteration
           */
          if (iAmSender(rank, D1, P))
          {
               for (int i = 0; i < M; i++)
               {
                    data_to_send_D1[i] = (double)((unsigned long long)data_recvd_from_D1[i] % 100000);
               }
          }

          if (iAmSender(rank, D2, P))
          {
               for (int i = 0; i < M; i++)
               {
                    data_to_send_D2[i] = data_recvd_from_D2[i] * 100000.0;
               }
          }
     } /* End of Time Loop */

     /* =========================================================================
      * Analysis & Reduction
      * ========================================================================= */

     /* 1. Calculate Local Maxima */
     double local_max_D1 = -DBL_MAX;
     double local_max_D2 = -DBL_MAX;

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

     /* 2. Global Reduction (Manual Tree Implementation)
      * Performs a bitwise butterfly/tree reduction.
      * Complexity: O(log P)
      */
     double global_max_D1 = local_max_D1;
     double global_max_D2 = local_max_D2;

     int step = 1;
     while (step < P)
     {
          int partner = rank ^ step; // XOR finds the neighbor in the current tree level

          if (partner < P)
          {
               if (rank < partner)
               {
                    // Low rank receives and aggregates
                    double recv_val_D1, recv_val_D2;
                    MPI_Recv(&recv_val_D1, 1, MPI_DOUBLE, partner, TAG_RED_D1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&recv_val_D2, 1, MPI_DOUBLE, partner, TAG_RED_D2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    global_max_D1 = fmax(global_max_D1, recv_val_D1);
                    global_max_D2 = fmax(global_max_D2, recv_val_D2);
               }
               else
               {
                    // High rank sends and retires from the reduction
                    MPI_Send(&global_max_D1, 1, MPI_DOUBLE, partner, TAG_RED_D1, MPI_COMM_WORLD);
                    MPI_Send(&global_max_D2, 1, MPI_DOUBLE, partner, TAG_RED_D2, MPI_COMM_WORLD);
                    break;
               }
          }
          step <<= 1; // Move to next level of tree
     }

     /* 3. Time Measurement */
     double end_time = MPI_Wtime();
     double local_time = end_time - start_time;
     double max_time;

     MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

     /* --- Final Output --- */
     if (rank == 0)
     {
          printf("%lf %lf %lf\n", global_max_D1, global_max_D2, max_time);
     }

     /* --- Cleanup --- */
     free(data_to_send_D1);
     free(data_to_send_D2);
     free(recv_buf_D1);
     free(recv_buf_D2);
     free(data_recvd_from_D1);
     free(data_recvd_from_D2);

     MPI_Finalize();
     return 0;
}
