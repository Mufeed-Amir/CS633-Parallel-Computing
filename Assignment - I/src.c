/*
 * MPI Pipeline Simulation with Odd-Even Communication & Tree Reduction
 *
 * Description:
 * Simulates a dual-dependency pipeline (D1, D2) across P processors.
 * - Uses blocking MPI_Send/MPI_Recv with an Odd-Even (Red-Black) ordering
 * strategy to prevent deadlock and serialization.
 * - Implements a custom O(log P) bitwise tree reduction for global maximums.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <float.h>

/* --- MPI Message Tags (kept unique by phase and dependency stream) --- */
#define TAG_FWD_D1 100 /* Forward pass data movement for dependency D1 */
#define TAG_FWD_D2 101 /* Forward pass data movement for dependency D2 */
#define TAG_RET_D1 200 /* Return/backward pass data movement for dependency D1 */
#define TAG_RET_D2 201 /* Return/backward pass data movement for dependency D2 */
#define TAG_RED_D1 500 /* Tree-reduction messages (global max) for dependency D1 */
#define TAG_RED_D2 501 /* Tree-reduction messages (global max) for dependency D2 */

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

/*
These are a set of hempler functions to compute the source and target ranks for communication
based on the current rank, the distance (d), and the total number of processors (P).
They also determine whether the current rank is a sender or receiver in the communication step.
This abstraction helps to keep the main communication logic clean and focused on the
actual data movement rather than the details of rank calculations.
*/
static inline int computeTarget(const int rank, const int d) { return rank + d; }
static inline int computeSource(const int rank, const int d) { return rank - d; }

/*
These functions determine the role of the current rank in the communication step.
A rank is considered a sender if its computed target is within the valid range of ranks (less than P),
and it is considered a receiver if its computed source is within the valid range (greater than or equal to 0).
This logic helps to identify which ranks will be sending data and which will be receiving data during the communication phases.
*/
static inline int iAmSender(const int rank, const int d, const int P) { return computeTarget(rank, d) < P; }
static inline int iAmReceiver(const int rank, const int d, const int P) { return computeSource(rank, d) >= 0; }

/**
 * @brief Handles point-to-point MPI communication for a specific algorithmic step.
 *
 * This function orchestrates data movement between the current rank and its
 * computed source/target neighbors. It supports both forward and reverse
 * communication passes and includes logic to prevent deadlocks during
 * bidirectional exchange.
 *
 * @param rank          The rank of the calling MPI process.
 * @param d             The current stride or distance (often used in recursive doubling/halving).
 * @param P             Total number of processors.
 * @param M             Size of the message (number of doubles).
 * @param recv_buf      Buffer to store received data.
 * @param data_to_send  Buffer containing data to transmit.
 * @param tag           MPI tag to distinguish this specific communication context.
 * @param is_forward    Boolean flag:
 * 1 (True) for forward pass (Head -> Tail),
 * 0 (False) for reverse pass (Tail -> Head).
 */
void doCommunication(const int rank, const int d, const int P, const int M, double *recv_buf, double *data_to_send, int tag, int is_forward)
{
     /*Role Identification:
      * - i_am_receiver: True if this rank is a receiver in the current communication step.
      * - i_am_sender: True if this rank is a sender in the current communication step.
      */
     int i_am_receiver = iAmReceiver(rank, d, P);
     int i_am_sender = iAmSender(rank, d, P);

     /* Communication Partners:
      * - target: The rank to which this process will send data in Phase 1 and will receive data in Phase 2.
      * - source: The rank from which this process will receive data in Phase 1 and will send data in Phase 2.
      */
     int target = computeTarget(rank, d);
     int source = computeSource(rank, d);

     // This is the core communication logic that handles all cases (Sender, Receiver, Middle) with proper ordering to avoid deadlock.
     if (i_am_receiver && !i_am_sender)
     {
          // The last block of proceeses (Receivers only) - They only receive in the forward pass and send in the reverse pass.
          if (is_forward)
               MPI_Recv(recv_buf, M, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          else
               MPI_Send(data_to_send, M, MPI_DOUBLE, source, tag, MPI_COMM_WORLD);
     }
     else if (i_am_sender && !i_am_receiver)
     {
          // The first block of processes (Senders only) - They only send in the forward pass and receive in the reverse pass.
          if (is_forward)
               MPI_Send(data_to_send, M, MPI_DOUBLE, target, tag, MPI_COMM_WORLD);
          else
               MPI_Recv(recv_buf, M, MPI_DOUBLE, target, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
     }
     else if (i_am_sender && i_am_receiver)
     {
          // Middle block of processes (Both Sender and Receiver) - They must perform both send and receive operations.
          // The odd vs even parity of the block determines the order of send/recv to prevent deadlock and this happens in 2 rounds
          if ((rank / d) % 2 != 0)
          {
               // Odd parity block - They perform send/recv in one order during the forward pass and reverse it during the backward pass.
               if (is_forward)
               {
                    MPI_Recv(recv_buf, M, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(data_to_send, M, MPI_DOUBLE, target, tag, MPI_COMM_WORLD);
               }
               else
               {
                    MPI_Send(data_to_send, M, MPI_DOUBLE, source, tag, MPI_COMM_WORLD);
                    MPI_Recv(recv_buf, M, MPI_DOUBLE, target, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
               }
          }
          else
          {
               // Even parity block - They perform send/recv in the opposite order of the odd block to 
               // ensure that at least one side is always sending while the other is receiving, thus preventing deadlock.
               if (is_forward)
               {
                    MPI_Send(data_to_send, M, MPI_DOUBLE, target, tag, MPI_COMM_WORLD);
                    MPI_Recv(recv_buf, M, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
               }
               else
               {
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
               fprintf(stderr, "Usage: %s <M> <D1> <D2> <T> <seed>\n", argv[0]);
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
                    recv_buf_D1[i] *= recv_buf_D1[i];
          }

          if (iAmReceiver(rank, D2, P))
          {
               for (int i = 0; i < M; i++)
                    recv_buf_D2[i] = (recv_buf_D2[i] > 0) ? log(recv_buf_D2[i]) : 0;
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
                    data_to_send_D1[i] = (double)((unsigned long long)data_recvd_from_D1[i] % 100000);
          }

          if (iAmSender(rank, D2, P))
          {
               for (int i = 0; i < M; i++)
                    data_to_send_D2[i] = data_recvd_from_D2[i] * 100000.0;
          }
     } /* End of Time Loop */

     /* =========================================================================
      * Analysis & Reduction
      * ========================================================================= */

     /* 1. Calculate Local Maxima initialized to lowest possible double size */
     double local_max_D1 = -DBL_MAX;
     double local_max_D2 = -DBL_MAX;

     // Only the ranks that are senders in the last iteration will have valid data to compute local maxima from the received results.
     if (iAmSender(rank, D1, P))
     {
          for (int i = 0; i < M; i++)
          {
               if (data_recvd_from_D1[i] > local_max_D1)
                    local_max_D1 = data_recvd_from_D1[i];
          }
     }

     if (iAmSender(rank, D2, P))
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
     double global_max_D1 = local_max_D1, global_max_D2 = local_max_D2;

     /* * --- Global Maximum Reduction (Binary Tree Strategy) ---
      * This loop aggregates the local maximums from all P processes into rank 0.
      * It uses a butterfly-style tree where the number of active processes
      * halves at each iteration, resulting in log2(P) communication steps.
      */
     int step = 1;
     while (step < P)
     {
          /* * Use bitwise XOR to identify the communication partner at this level.
           * For example, in step 1: Rank 0 pairs with 1, 2 with 3, etc.
           * In step 2: Rank 0 pairs with 2, 4 with 6, etc.
           */
          int partner = rank ^ step;

          // Ensure the partner exists (relevant if P is not a power of 2)
          if (partner < P)
          {
               /* * Standard Tree Reduction Logic:
                * The lower rank in each pair acts as the 'accumulator' (Receiver).
                * The higher rank sends its data and then exits the loop (Retires).
                */
               if (rank < partner)
               {
                    // Current rank is the Receiver/Aggregator for this branch
                    double recv_val_D1, recv_val_D2;

                    // Blocking receive from the higher-ranked partner
                    MPI_Recv(&recv_val_D1, 1, MPI_DOUBLE, partner, TAG_RED_D1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&recv_val_D2, 1, MPI_DOUBLE, partner, TAG_RED_D2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // Update the running global maximums using the received values
                    global_max_D1 = fmax(global_max_D1, recv_val_D1);
                    global_max_D2 = fmax(global_max_D2, recv_val_D2);
               }
               else
               {
                    // Current rank is the Sender for this branch.
                    // Transfer local/accumulated state to the lower-ranked partner.
                    MPI_Send(&global_max_D1, 1, MPI_DOUBLE, partner, TAG_RED_D1, MPI_COMM_WORLD);
                    MPI_Send(&global_max_D2, 1, MPI_DOUBLE, partner, TAG_RED_D2, MPI_COMM_WORLD);

                    // Once data is sent, this rank is no longer needed in the reduction tree.
                    break;
               }
          }

          /* * Shift the step bit left (multiply by 2) to move to the next level of the tree.
           * This doubles the 'distance' between communication partners for the next round.
           */
          step <<= 1;
     }

     /* 3. Time Measurement & max_time*/
     double end_time = MPI_Wtime(), max_time;
     double local_time = end_time - start_time;
     MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

     /* --- Final Output --- */
     if (rank == 0)
          printf("%lf %lf %lf\n", global_max_D1, global_max_D2, max_time);

     /* --- Cleanup to free allocated memory --- */
     free(data_to_send_D1);
     free(data_to_send_D2);
     free(recv_buf_D1);
     free(recv_buf_D2);
     free(data_recvd_from_D1);
     free(data_recvd_from_D2);

     MPI_Finalize();
     return 0;
}
