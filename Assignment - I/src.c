#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <float.h>

/* Allocates memory safely */
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

     /* Validate args */
     if (argc != 6)
     {
          if (rank == 0)
          {
               printf("Error: Expected 5 arguments.\n");
          }
          MPI_Finalize();
          exit(1);
     }

     /* Read inputs */
     int M = atoi(argv[1]);
     int D1 = atoi(argv[2]);
     int D2 = atoi(argv[3]);
     int T = atoi(argv[4]);
     int seed = atoi(argv[5]);

     /* Neighbor ranks */
     int target_D1 = rank + D1;
     int target_D2 = rank + D2;
     int source_D1 = rank - D1;
     int source_D2 = rank - D2;

     /* Role flags */
     int i_am_sender_D1 = (target_D1 < P);
     int i_am_sender_D2 = (target_D2 < P);
     int i_am_receiver_D1 = (source_D1 >= 0);
     int i_am_receiver_D2 = (source_D2 >= 0);

     /* Buffers */
     double *data_to_send_D1 = allocate_buffer(M);
     double *data_to_send_D2 = allocate_buffer(M);
     double *recv_buf_D1 = allocate_buffer(M);
     double *recv_buf_D2 = allocate_buffer(M);
     double *data_recvd_from_D1 = allocate_buffer(M);
     double *data_recvd_from_D2 = allocate_buffer(M);

     /* Init data */
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

     for (int t = 0; t < T; t++)
     {
          /* Receive */
          if (i_am_receiver_D1)
          {
               MPI_Recv(recv_buf_D1, M, MPI_DOUBLE, source_D1, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }

          if (i_am_receiver_D2)
          {
               MPI_Recv(recv_buf_D2, M, MPI_DOUBLE, source_D2, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }

          /* Send */
          if (i_am_sender_D1)
          {
               MPI_Send(data_to_send_D1, M, MPI_DOUBLE, target_D1, 100, MPI_COMM_WORLD);
          }

          if (i_am_sender_D2)
          {
               MPI_Send(data_to_send_D2, M, MPI_DOUBLE, target_D2, 101, MPI_COMM_WORLD);
          }

          /* Compute */
          if (i_am_receiver_D1)
          {
               for (int i = 0; i < M; i++)
               {
                    recv_buf_D1[i] *= recv_buf_D1[i];
               }
          }

          if (i_am_receiver_D2)
          {
               for (int i = 0; i < M; i++)
               {
                    recv_buf_D2[i] = (recv_buf_D2[i] > 0) ? log(recv_buf_D2[i]) : 0;
               }
          }

          /* Return results */
          if (i_am_receiver_D1)
          {
               MPI_Send(recv_buf_D1, M, MPI_DOUBLE, source_D1, 200, MPI_COMM_WORLD);
          }

          if (i_am_receiver_D2)
          {
               MPI_Send(recv_buf_D2, M, MPI_DOUBLE, source_D2, 201, MPI_COMM_WORLD);
          }

          /* Collect results */
          if (i_am_sender_D1)
          {
               MPI_Recv(data_recvd_from_D1, M, MPI_DOUBLE, target_D1, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }

          if (i_am_sender_D2)
          {
               MPI_Recv(data_recvd_from_D2, M, MPI_DOUBLE, target_D2, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }

          /* Prepare next iteration */
          if (i_am_sender_D1)
          {
               for (int i = 0; i < M; i++)
               {
                    double combined = data_recvd_from_D1[i];
                    if (i_am_sender_D2)
                    {
                         combined += data_recvd_from_D2[i];
                    }

                    data_to_send_D1[i] = (double)((unsigned long long)combined % 100000);
                    data_to_send_D2[i] = combined * 100000.0;
               }
          }
     }

     /* Local maxima */
     double local_max_D1 = -DBL_MAX;
     double local_max_D2 = -DBL_MAX;

     if (i_am_sender_D1)
     {
          for (int i = 0; i < M; i++)
          {
               if (data_recvd_from_D1[i] > local_max_D1)
               {
                    local_max_D1 = data_recvd_from_D1[i];
               }
          }
     }

     if (i_am_sender_D2)
     {
          for (int i = 0; i < M; i++)
          {
               if (data_recvd_from_D2[i] > local_max_D2)
               {
                    local_max_D2 = data_recvd_from_D2[i];
               }
          }
     }

     /* Global reduction */
     double global_max_D1, global_max_D2;
     MPI_Reduce(&local_max_D1, &global_max_D1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
     MPI_Reduce(&local_max_D2, &global_max_D2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

     double end_time = MPI_Wtime();

     /* Print result */
     if (rank == 0)
     {
          printf("%lf %lf %lf\n", global_max_D1, global_max_D2, end_time - start_time);
     }

     free(data_to_send_D1);
     free(data_to_send_D2);
     free(recv_buf_D1);
     free(recv_buf_D2);
     free(data_recvd_from_D1);
     free(data_recvd_from_D2);
     
     MPI_Finalize();
     return 0;
}
