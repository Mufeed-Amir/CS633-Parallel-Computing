#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Macro mapping 3D coordinates (z,y,x) to a 1D flat array considering the halo (ghost) cell depth
#define IDX(z, y, x, dim_x, dim_y, halo) ((x) + (y) * ((dim_x) + 2 * (halo)) + (z) * ((dim_x) + 2 * (halo)) * ((dim_y) + 2 * (halo)))

// Optimal L1 Cache Block Size for double precision floating point operations
#define CACHE_BLOCK_SIZE 64

// 1. Manually resolve 3D Cartesian topology neighbors based on process ID
void find_process_neighbors(int process_id, int px, int py, int pz,
                              int *neighbor_left, int *neighbor_right, int *neighbor_bottom,
                              int *neighbor_top, int *neighbor_back, int *neighbor_front)
{
     int coord_x = process_id % px;
     int coord_y = (process_id / px) % py;
     int coord_z = process_id / (px * py);

     *neighbor_left = (coord_x > 0) ? process_id - 1 : MPI_PROC_NULL;
     *neighbor_right = (coord_x < px - 1) ? process_id + 1 : MPI_PROC_NULL;
     *neighbor_bottom = (coord_y > 0) ? process_id - px : MPI_PROC_NULL;
     *neighbor_top = (coord_y < py - 1) ? process_id + px : MPI_PROC_NULL;
     *neighbor_back = (coord_z > 0) ? process_id - (px * py) : MPI_PROC_NULL;
     *neighbor_front = (coord_z < pz - 1) ? process_id + (px * py) : MPI_PROC_NULL;
}

// 2. Stage outgoing boundary data into contiguous memory buffers
void stage_outgoing_halos(double *data_buffer, int nx, int ny, int nz, int halo_depth,
                          double *send_left, double *send_right, double *send_bottom,
                          double *send_top, double *send_back, double *send_front)
{
     int ptr_l = 0, ptr_r = 0, ptr_b = 0, ptr_t = 0, ptr_bk = 0, ptr_fr = 0;

     for (int z = halo_depth; z < nz + halo_depth; z++)
     {
          for (int y = halo_depth; y < ny + halo_depth; y++)
          {
               for (int x = halo_depth; x < 2 * halo_depth; x++)
                    send_left[ptr_l++] = data_buffer[IDX(z, y, x, nx, ny, halo_depth)];
               for (int x = nx; x < nx + halo_depth; x++)
                    send_right[ptr_r++] = data_buffer[IDX(z, y, x, nx, ny, halo_depth)];
          }
     }
     for (int z = halo_depth; z < nz + halo_depth; z++)
     {
          for (int y = halo_depth; y < 2 * halo_depth; y++)
          {
               for (int x = halo_depth; x < nx + halo_depth; x++)
                    send_bottom[ptr_b++] = data_buffer[IDX(z, y, x, nx, ny, halo_depth)];
          }
          for (int y = ny; y < ny + halo_depth; y++)
          {
               for (int x = halo_depth; x < nx + halo_depth; x++)
                    send_top[ptr_t++] = data_buffer[IDX(z, y, x, nx, ny, halo_depth)];
          }
     }
     for (int z = halo_depth; z < 2 * halo_depth; z++)
     {
          for (int y = halo_depth; y < ny + halo_depth; y++)
          {
               for (int x = halo_depth; x < nx + halo_depth; x++)
                    send_back[ptr_bk++] = data_buffer[IDX(z, y, x, nx, ny, halo_depth)];
          }
     }
     for (int z = nz; z < nz + halo_depth; z++)
     {
          for (int y = halo_depth; y < ny + halo_depth; y++)
          {
               for (int x = halo_depth; x < nx + halo_depth; x++)
                    send_front[ptr_fr++] = data_buffer[IDX(z, y, x, nx, ny, halo_depth)];
          }
     }
}

// 3. Integrate incoming contiguous network buffers into local ghost cells
void integrate_incoming_halos(double *data_buffer, int nx, int ny, int nz, int halo_depth,
                              double *recv_left, double *recv_right, double *recv_bottom,
                              double *recv_top, double *recv_back, double *recv_front,
                              int neighbor_left, int neighbor_right, int neighbor_bottom,
                              int neighbor_top, int neighbor_back, int neighbor_front)
{
     int ptr_l = 0, ptr_r = 0, ptr_b = 0, ptr_t = 0, ptr_bk = 0, ptr_fr = 0;

     for (int z = halo_depth; z < nz + halo_depth; z++)
     {
          for (int y = halo_depth; y < ny + halo_depth; y++)
          {
               if (neighbor_left != MPI_PROC_NULL)
               {
                    for (int x = 0; x < halo_depth; x++)
                         data_buffer[IDX(z, y, x, nx, ny, halo_depth)] = recv_left[ptr_l++];
               }
               if (neighbor_right != MPI_PROC_NULL)
               {
                    for (int x = nx + halo_depth; x < nx + 2 * halo_depth; x++)
                         data_buffer[IDX(z, y, x, nx, ny, halo_depth)] = recv_right[ptr_r++];
               }
          }
     }
     for (int z = halo_depth; z < nz + halo_depth; z++)
     {
          if (neighbor_bottom != MPI_PROC_NULL)
          {
               for (int y = 0; y < halo_depth; y++)
               {
                    for (int x = halo_depth; x < nx + halo_depth; x++)
                         data_buffer[IDX(z, y, x, nx, ny, halo_depth)] = recv_bottom[ptr_b++];
               }
          }
          if (neighbor_top != MPI_PROC_NULL)
          {
               for (int y = ny + halo_depth; y < ny + 2 * halo_depth; y++)
               {
                    for (int x = halo_depth; x < nx + halo_depth; x++)
                         data_buffer[IDX(z, y, x, nx, ny, halo_depth)] = recv_top[ptr_t++];
               }
          }
     }
     if (neighbor_back != MPI_PROC_NULL)
     {
          for (int z = 0; z < halo_depth; z++)
          {
               for (int y = halo_depth; y < ny + halo_depth; y++)
               {
                    for (int x = halo_depth; x < nx + halo_depth; x++)
                         data_buffer[IDX(z, y, x, nx, ny, halo_depth)] = recv_back[ptr_bk++];
               }
          }
     }
     if (neighbor_front != MPI_PROC_NULL)
     {
          for (int z = nz + halo_depth; z < nz + 2 * halo_depth; z++)
          {
               for (int y = halo_depth; y < ny + halo_depth; y++)
               {
                    for (int x = halo_depth; x < nx + halo_depth; x++)
                         data_buffer[IDX(z, y, x, nx, ny, halo_depth)] = recv_front[ptr_fr++];
               }
          }
     }
}

// 4. Compute Subdomain with CACHE BLOCKING (Tiled Loops) for maximum CPU efficiency
long long compute_subdomain_region(
    double *read_buffer, double *write_buffer,
    int z_start, int z_end, int y_start, int y_end, int x_start, int x_end,
    int nx, int ny, int nz,
    int halo_depth, int d, double isovalue)
{
     long long regional_iso_count = 0;

     // Cache Blocking (Loop Tiling) execution
     for (int block_z = z_start; block_z <= z_end; block_z += CACHE_BLOCK_SIZE)
     {
          int limit_z = (block_z + CACHE_BLOCK_SIZE - 1 < z_end) ? block_z + CACHE_BLOCK_SIZE - 1 : z_end;

          for (int block_y = y_start; block_y <= y_end; block_y += CACHE_BLOCK_SIZE)
          {
               int limit_y = (block_y + CACHE_BLOCK_SIZE - 1 < y_end) ? block_y + CACHE_BLOCK_SIZE - 1 : y_end;

               for (int block_x = x_start; block_x <= x_end; block_x += CACHE_BLOCK_SIZE)
               {
                    int limit_x = (block_x + CACHE_BLOCK_SIZE - 1 < x_end) ? block_x + CACHE_BLOCK_SIZE - 1 : x_end;

                    // Inner Math Computation per Cache Tile
                    for (int z = block_z; z <= limit_z; z++)
                    {
                         for (int y = block_y; y <= limit_y; y++)
                         {
                              for (int x = block_x; x <= limit_x; x++)
                              {
                                   int flat_idx = IDX(z, y, x, nx, ny, halo_depth);

                                   // --- A. Marching Cubes Surface Detection ---
                                   double vertex[8];
                                   vertex[0] = read_buffer[flat_idx];
                                   vertex[1] = read_buffer[IDX(z, y, x + 1, nx, ny, halo_depth)];
                                   vertex[2] = read_buffer[IDX(z, y + 1, x, nx, ny, halo_depth)];
                                   vertex[3] = read_buffer[IDX(z, y + 1, x + 1, nx, ny, halo_depth)];
                                   vertex[4] = read_buffer[IDX(z + 1, y, x, nx, ny, halo_depth)];
                                   vertex[5] = read_buffer[IDX(z + 1, y, x + 1, nx, ny, halo_depth)];
                                   vertex[6] = read_buffer[IDX(z + 1, y + 1, x, nx, ny, halo_depth)];
                                   vertex[7] = read_buffer[IDX(z + 1, y + 1, x + 1, nx, ny, halo_depth)];

                                   int vertices_below_threshold = 0;
                                   for (int k = 0; k < 8; k++)
                                   {
                                        if (vertex[k] < isovalue)
                                             vertices_below_threshold++;
                                   }

                                   if (vertices_below_threshold > 0 && vertices_below_threshold < 8)
                                   {
                                        regional_iso_count++;
                                   }

                                   // --- B. Variable D-Point Stencil ---
                                   double accumulation = read_buffer[flat_idx];

                                   for (int step = 1; step <= halo_depth; step++)
                                   {
                                        accumulation += read_buffer[IDX(z, y, x - step, nx, ny, halo_depth)];
                                        accumulation += read_buffer[IDX(z, y, x + step, nx, ny, halo_depth)];
                                        accumulation += read_buffer[IDX(z, y - step, x, nx, ny, halo_depth)];
                                        accumulation += read_buffer[IDX(z, y + step, x, nx, ny, halo_depth)];
                                        accumulation += read_buffer[IDX(z - step, y, x, nx, ny, halo_depth)];
                                        accumulation += read_buffer[IDX(z + step, y, x, nx, ny, halo_depth)];
                                   }

                                   write_buffer[flat_idx] = accumulation / (double)d;
                              }
                         }
                    }
               }
          }
     }
     return regional_iso_count;
}

// --- MAIN FUNCTION ---

int main(int argc, char *argv[])
{
     int process_id, num_processes;
     MPI_Init(&argc, &argv);
     MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
     MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

     if (argc != 13)
     {
          if (process_id == 0)
               printf("Usage: %s d ppn px py pz nx ny nz T seed F isovalue\n", argv[0]);
          MPI_Finalize();
          return 1;
     }

     // input parsing
     int d = atoi(argv[1]);
     int ppn = atoi(argv[2]);
     int px = atoi(argv[3]);
     int py = atoi(argv[4]);
     int pz = atoi(argv[5]);
     int nx = atoi(argv[6]);
     int ny = atoi(argv[7]);
     int nz = atoi(argv[8]);
     int T = atoi(argv[9]);
     int seed = atoi(argv[10]);
     int F = atoi(argv[11]);
     double isovalue = atof(argv[12]);

     if ((d - 1) % 6 != 0)
     {
          if (process_id == 0)
               printf("Error: Invalid stencil point configuration.\n");
          MPI_Finalize();
          return 1;
     }
     int halo_depth = (d - 1) / 6;

     if (num_processes != px * py * pz)
     {
          if (process_id == 0)
               printf("Error: Process topology mismatch.\n");
          MPI_Finalize();
          return 1;
     }

     int neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front;
     find_process_neighbors(process_id, px, py, pz, &neighbor_left, &neighbor_right,
                              &neighbor_bottom, &neighbor_top, &neighbor_back, &neighbor_front);

     // Memory Allocation
     size_t total_allocation_size = (nx + 2 * halo_depth) * (ny + 2 * halo_depth) * (nz + 2 * halo_depth);
     double **read_buffer = (double **)malloc(F * sizeof(double *));
     double **write_buffer = (double **)malloc(F * sizeof(double *));

     for (int i = 0; i < F; i++)
     {
          read_buffer[i] = (double *)calloc(total_allocation_size, sizeof(double));
          write_buffer[i] = (double *)calloc(total_allocation_size, sizeof(double));
     }

     // Mathematical Initialization
     srand(seed);
     int arrSize = nx * ny * nz;
     for (int i = 0; i < F; i++)
     {
          for (int j = 0; j < arrSize; j++)
          {
               int z_in = j / (nx * ny);
               int y_in = (j / nx) % ny;
               int x_in = j % nx;
               int flat_idx = IDX(z_in + halo_depth, y_in + halo_depth, x_in + halo_depth, nx, ny, halo_depth);
               // process_id represents myrank in the formula
               read_buffer[i][flat_idx] = (double)rand() * (process_id + 1) / (110426.0 + i + j);
          }
     }

     // Network Buffers
     int halo_vol_x = halo_depth * ny * nz;
     int halo_vol_y = nx * halo_depth * nz;
     int halo_vol_z = nx * ny * halo_depth;

     double *send_l = malloc(halo_vol_x * sizeof(double)), *recv_l = malloc(halo_vol_x * sizeof(double));
     double *send_r = malloc(halo_vol_x * sizeof(double)), *recv_r = malloc(halo_vol_x * sizeof(double));
     double *send_b = malloc(halo_vol_y * sizeof(double)), *recv_b = malloc(halo_vol_y * sizeof(double));
     double *send_t = malloc(halo_vol_y * sizeof(double)), *recv_t = malloc(halo_vol_y * sizeof(double));
     double *send_bk = malloc(halo_vol_z * sizeof(double)), *recv_bk = malloc(halo_vol_z * sizeof(double));
     double *send_fr = malloc(halo_vol_z * sizeof(double)), *recv_fr = malloc(halo_vol_z * sizeof(double));

     long long *local_results = (long long *)malloc(T * F * sizeof(long long));
     long long *global_results = (process_id == 0) ? (long long *)malloc(T * F * sizeof(long long)) : NULL;

     // 7-Region Decomposition Boundaries (Array Coordinates)
     int D_L = halo_depth;
     int D_Rx = halo_depth + nx - 1;
     int D_Ry = halo_depth + ny - 1;
     int D_Rz = halo_depth + nz - 1;

     int I_L = 2 * halo_depth;
     int I_Rx = nx - 1;
     int I_Ry = ny - 1;
     int I_Rz = nz - 1;

     MPI_Barrier(MPI_COMM_WORLD);
     double clock_start = MPI_Wtime();

     // Simulation Loop
     for (int t = 0; t < T; t++)
     {
          for (int i = 0; i < F; i++)
          {
               MPI_Request network_requests[12];
               int active_requests = 0;

               stage_outgoing_halos(read_buffer[i], nx, ny, nz, halo_depth, send_l, send_r, send_b, send_t, send_bk, send_fr);

               // 1. INITIATE NON-BLOCKING COMMUNICATION
               if (neighbor_left != MPI_PROC_NULL)
               {
                    MPI_Isend(send_l, halo_vol_x, MPI_DOUBLE, neighbor_left, 0, MPI_COMM_WORLD, &network_requests[active_requests++]);
                    MPI_Irecv(recv_l, halo_vol_x, MPI_DOUBLE, neighbor_left, 1, MPI_COMM_WORLD, &network_requests[active_requests++]);
               }
               if (neighbor_right != MPI_PROC_NULL)
               {
                    MPI_Isend(send_r, halo_vol_x, MPI_DOUBLE, neighbor_right, 1, MPI_COMM_WORLD, &network_requests[active_requests++]);
                    MPI_Irecv(recv_r, halo_vol_x, MPI_DOUBLE, neighbor_right, 0, MPI_COMM_WORLD, &network_requests[active_requests++]);
               }
               if (neighbor_bottom != MPI_PROC_NULL)
               {
                    MPI_Isend(send_b, halo_vol_y, MPI_DOUBLE, neighbor_bottom, 2, MPI_COMM_WORLD, &network_requests[active_requests++]);
                    MPI_Irecv(recv_b, halo_vol_y, MPI_DOUBLE, neighbor_bottom, 3, MPI_COMM_WORLD, &network_requests[active_requests++]);
               }
               if (neighbor_top != MPI_PROC_NULL)
               {
                    MPI_Isend(send_t, halo_vol_y, MPI_DOUBLE, neighbor_top, 3, MPI_COMM_WORLD, &network_requests[active_requests++]);
                    MPI_Irecv(recv_t, halo_vol_y, MPI_DOUBLE, neighbor_top, 2, MPI_COMM_WORLD, &network_requests[active_requests++]);
               }
               if (neighbor_back != MPI_PROC_NULL)
               {
                    MPI_Isend(send_bk, halo_vol_z, MPI_DOUBLE, neighbor_back, 4, MPI_COMM_WORLD, &network_requests[active_requests++]);
                    MPI_Irecv(recv_bk, halo_vol_z, MPI_DOUBLE, neighbor_back, 5, MPI_COMM_WORLD, &network_requests[active_requests++]);
               }
               if (neighbor_front != MPI_PROC_NULL)
               {
                    MPI_Isend(send_fr, halo_vol_z, MPI_DOUBLE, neighbor_front, 5, MPI_COMM_WORLD, &network_requests[active_requests++]);
                    MPI_Irecv(recv_fr, halo_vol_z, MPI_DOUBLE, neighbor_front, 4, MPI_COMM_WORLD, &network_requests[active_requests++]);
               }

               // 2. COMPUTE INNER CORE (Overlapping calculation while network transmits data)
               long long field_count = 0;
               field_count += compute_subdomain_region(read_buffer[i], write_buffer[i], I_L, I_Rz, I_L, I_Ry, I_L, I_Rx, nx, ny, nz, halo_depth, d, isovalue);

               // 3. WAIT FOR NETWORK COMPLETION
               MPI_Waitall(active_requests, network_requests, MPI_STATUSES_IGNORE);

               // 4. INTEGRATE RECEIVED HALOS
               integrate_incoming_halos(read_buffer[i], nx, ny, nz, halo_depth, recv_l, recv_r, recv_b, recv_t, recv_bk, recv_fr, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front);

               // 5. COMPUTE BOUNDARY SHELLS (Using exact non-overlapping geometrical regions)
               field_count += compute_subdomain_region(read_buffer[i], write_buffer[i], D_L, I_L - 1, D_L, D_Ry, D_L, D_Rx, nx, ny, nz, halo_depth, d, isovalue);   // Z-Bot
               field_count += compute_subdomain_region(read_buffer[i], write_buffer[i], I_Rz + 1, D_Rz, D_L, D_Ry, D_L, D_Rx, nx, ny, nz, halo_depth, d, isovalue); // Z-Top
               field_count += compute_subdomain_region(read_buffer[i], write_buffer[i], I_L, I_Rz, D_L, I_L - 1, D_L, D_Rx, nx, ny, nz, halo_depth, d, isovalue);   // Y-Bot
               field_count += compute_subdomain_region(read_buffer[i], write_buffer[i], I_L, I_Rz, I_Ry + 1, D_Ry, D_L, D_Rx, nx, ny, nz, halo_depth, d, isovalue); // Y-Top
               field_count += compute_subdomain_region(read_buffer[i], write_buffer[i], I_L, I_Rz, I_L, I_Ry, D_L, I_L - 1, nx, ny, nz, halo_depth, d, isovalue);   // X-Left
               field_count += compute_subdomain_region(read_buffer[i], write_buffer[i], I_L, I_Rz, I_L, I_Ry, I_Rx + 1, D_Rx, nx, ny, nz, halo_depth, d, isovalue); // X-Right

               local_results[t * F + i] = field_count;

               // Pointer Swap
               double *temp_ptr = read_buffer[i];
               read_buffer[i] = write_buffer[i];
               write_buffer[i] = temp_ptr;
          }
     }

     MPI_Reduce(local_results, global_results, T * F, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
     MPI_Barrier(MPI_COMM_WORLD);
     double clock_end = MPI_Wtime();

     if (process_id == 0)
     {
          for (int t = 0; t < T; t++)
          {
               for (int i = 0; i < F; i++)
               {
                    printf("%lld ", global_results[t * F + i]);
               }
               printf("\n");
          }
          printf("%f\n", clock_end - clock_start);
     }

     for (int i = 0; i < F; i++)
     {
          free(read_buffer[i]);
          free(write_buffer[i]);
     }
     free(read_buffer);
     free(write_buffer);
     free(send_l);
     free(recv_l);
     free(send_r);
     free(recv_r);
     free(send_b);
     free(recv_b);
     free(send_t);
     free(recv_t);
     free(send_bk);
     free(recv_bk);
     free(send_fr);
     free(recv_fr);
     free(local_results);

     if (process_id == 0)
     {
          free(global_results);
     }
          
     MPI_Finalize();
     return 0;
}
