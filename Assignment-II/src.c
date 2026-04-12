#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Macro mapping 3D coordinates (z,y,x) to a 1D flat array considering the halo cell depth */
#define IDX(z, y, x, dim_x, dim_y, halo) ((x) + (y) * ((dim_x) + 2 * (halo)) + (z) * ((dim_x) + 2 * (halo)) * ((dim_y) + 2 * (halo)))

/* Optimal L1 and L2 Cache Block Size for double precision floating point operations this is done to optimize time */
#define CACHE_BLOCK_SIZE 16

/* Manually resolve 3D Cartesian topology neighbors based on rank of the process 
 * Firstly we translate the rank to the (x, y, z) coordinates of the 3D domain.
 * Then depending on these coordinates we find out the cooresponding ranks of
 * the neighbours. These are passed using output pointers in the function definition
*/
void find_process_neighbors(int myrank, int px, int py, int pz,
                              int *neighbor_left, int *neighbor_right, int *neighbor_bottom,
                              int *neighbor_top, int *neighbor_back, int *neighbor_front)
{
     int coord_x = myrank % px;
     int coord_y = (myrank / px) % py;
     int coord_z = myrank / (px * py);

	 /* 
	  * If any process is at a domain boundary return MPI_PROC_NULL
	  * MPI_PROC_NULL means don't do any comm with the corresponding
	  * neighbour
	 */
     *neighbor_left = (coord_x > 0) ? myrank - 1 : MPI_PROC_NULL;
     *neighbor_right = (coord_x < px - 1) ? myrank + 1 : MPI_PROC_NULL;
     *neighbor_bottom = (coord_y > 0) ? myrank - px : MPI_PROC_NULL;
     *neighbor_top = (coord_y < py - 1) ? myrank + px : MPI_PROC_NULL;
     *neighbor_back = (coord_z > 0) ? myrank - (px * py) : MPI_PROC_NULL;
     *neighbor_front = (coord_z < pz - 1) ? myrank + (px * py) : MPI_PROC_NULL;
}

/* Stage outgoing boundary data into contiguous memory buffers
 * Here we fill the outgoing buffers with the required data present in
 * data buffer and we use the IDX macro to get the required index
*/
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

/* Integrate incoming contiguous network buffers into local ghost cells
 * There exist for each rank some "ghost cells" that store the incoming data from the halo
 * exchange and this data is then integrated with the present data in this function,
 * There are check that the neighbours are not MPI_PROC_NULL and then the recv buffers are
 * integrated into the data_buffers
*/
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

/* Compute Subdomain with CACHE BLOCKING (Tiled Loops) for maximum CPU efficiency.
 * Here we check for each subdomain the local isocount and then we also accumulate
 * all the values in the read_buiffers. 
 * Since this is one of the most computation intensive works we are doing in this code
 * we have optimized this using cache efficiency to compute such that the data fits 
 * inside the cache
*/
void compute_subdomain_region(
    double *read_buffer, double *write_buffer,
    int z_start, int z_end, int y_start, int y_end, int x_start, int x_end,
    int nx, int ny, int nz,
    int halo_depth, int d,
    int neighbor_left, int neighbor_right, int neighbor_bottom,
    int neighbor_top, int neighbor_back, int neighbor_front)
{
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

                                   // --- Variable D-Point Stencil ---
                                   double accumulation = read_buffer[flat_idx];
                                   int m = 1; // actual neighbour count for divide-by-m at global boundaries

                                   for (int step = 1; step <= halo_depth; step++)
                                   {
                                        if (x - step >= halo_depth || neighbor_left != MPI_PROC_NULL)
                                        { 
                                             accumulation += read_buffer[IDX(z, y, x - step, nx, ny, halo_depth)]; 
                                             m++; 
                                        }
                                        if (x + step <= halo_depth+nx - 1 || neighbor_right != MPI_PROC_NULL) 
                                        { 
                                             accumulation += read_buffer[IDX(z, y, x + step, nx, ny, halo_depth)]; 
                                             m++; 
                                        }
                                        if (y - step >= halo_depth || neighbor_bottom != MPI_PROC_NULL) 
                                        { 
                                             accumulation += read_buffer[IDX(z, y - step, x, nx, ny, halo_depth)]; 
                                             m++; 
                                        }
                                        if (y + step <= halo_depth+ny - 1 || neighbor_top != MPI_PROC_NULL) 
                                        { 
                                             accumulation += read_buffer[IDX(z, y + step, x, nx, ny, halo_depth)]; 
                                             m++; 
                                        }
                                        if (z - step >= halo_depth || neighbor_back != MPI_PROC_NULL) 
                                        { 
                                             accumulation += read_buffer[IDX(z - step, y, x, nx, ny, halo_depth)]; 
                                             m++; 
                                        }
                                        if (z + step <= halo_depth+nz - 1 || neighbor_front  != MPI_PROC_NULL) 
                                        { 
                                             accumulation += read_buffer[IDX(z + step, y, x, nx, ny, halo_depth)]; 
                                             m++; 
                                        }
                                   }

                                   write_buffer[flat_idx] = accumulation / (double)m;
                              }
                         }
                    }
               }
          }
     }
}

/* Count isovalue crossings in a region of a fully-computed buffer using the same
 * cache optimized way. This is computed on write_buffer after the halo exchange is done
 * so ghost cells hold neighbour values. This causes double counting at the boundaries.
 * For each grid point we check 3 forward edges (+x, +y, +z). An edge crossing
 * occurs when one endpoint is strictly below the isovalue and the other is at or
 * above it, i.e. (v - isovalue) * (neighbour - isovalue) < 0.
 * Checking only forward directions avoids double counting edges within a process.
*/
long long count_isovalues_region(
    double *data,
    int z_start, int z_end, int y_start, int y_end, int x_start, int x_end,
    int nx, int ny, int nz, int halo_depth, double isovalue)
{
     long long regional_iso_count = 0;

     for (int block_z = z_start; block_z <= z_end; block_z += CACHE_BLOCK_SIZE)
     {
          int limit_z = (block_z + CACHE_BLOCK_SIZE - 1 < z_end) ? block_z + CACHE_BLOCK_SIZE - 1 : z_end;

          for (int block_y = y_start; block_y <= y_end; block_y += CACHE_BLOCK_SIZE)
          {
               int limit_y = (block_y + CACHE_BLOCK_SIZE - 1 < y_end) ? block_y + CACHE_BLOCK_SIZE - 1 : y_end;

               for (int block_x = x_start; block_x <= x_end; block_x += CACHE_BLOCK_SIZE)
               {
                    int limit_x = (block_x + CACHE_BLOCK_SIZE - 1 < x_end) ? block_x + CACHE_BLOCK_SIZE - 1 : x_end;

                    for (int z = block_z; z <= limit_z; z++)
                    {
                         for (int y = block_y; y <= limit_y; y++)
                         {
                              for (int x = block_x; x <= limit_x; x++)
                              {
                                   double v = data[IDX(z, y, x, nx, ny, halo_depth)];
                                   double diff = v - isovalue;

                                   /* +x edge: at x = x_end the ghost cell holds the neighbour's
                                    * value (filled by the write_buffer halo exchange in step 6) */
                                   if (diff * (data[IDX(z, y, x + 1, nx, ny, halo_depth)] - isovalue) < 0)
                                        regional_iso_count++;

                                   /* +y edge */
                                   if (diff * (data[IDX(z, y + 1, x, nx, ny, halo_depth)] - isovalue) < 0)
                                        regional_iso_count++;

                                   /* +z edge */
                                   if (diff * (data[IDX(z + 1, y, x, nx, ny, halo_depth)] - isovalue) < 0)
                                        regional_iso_count++;
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
     int myrank, num_processes;
     MPI_Init(&argc, &argv);
     MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
     MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

     if (argc != 13)
     {
          if (myrank == 0)
               printf("Usage: %s d ppn px py pz nx ny nz T seed F isovalue\n", argv[0]);
          MPI_Finalize();
          return 1;
     }

     // input parsing
     int d = atoi(argv[1]); // d point stencil
     int ppn = atoi(argv[2]); // process per node (not required in the code)
     int px = atoi(argv[3]); // length of x axis process per rank
     int py = atoi(argv[4]); // length of y axis process per rank
     int pz = atoi(argv[5]); // length of z axis process per rank
     int nx = atoi(argv[6]); // length of x axis process globally
     int ny = atoi(argv[7]); // length of y axis process globally
     int nz = atoi(argv[8]); // length of z axis process globally
     int T = atoi(argv[9]); // Time steps
     int seed = atoi(argv[10]); // random seed for initial values
     int F = atoi(argv[11]); // number of fields
     double isovalue = atof(argv[12]); // isovalue

     if ((d - 1) % 6 != 0)
     {
          if (myrank == 0)
               printf("Error: Invalid stencil point configuration.\n");
          MPI_Finalize();
          return 1;
     }
     int halo_depth = (d - 1) / 6; // Since halo can be divided into 6 parts UP DOWN FRONT BACK LEFT RIGHT

     if (num_processes != px * py * pz)
     {
          if (myrank == 0)
               printf("Error: Process topology mismatch.\n");
          MPI_Finalize();
          return 1;
     }

     int neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front;
     find_process_neighbors(myrank, px, py, pz, &neighbor_left, &neighbor_right,
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
               
               read_buffer[i][flat_idx] = (double)rand() * (myrank + 1) / (110426.0 + i + j);
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
     long long *global_results = (myrank == 0) ? (long long *)malloc(T * F * sizeof(long long)) : NULL;

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
               compute_subdomain_region(read_buffer[i], write_buffer[i], I_L, I_Rz, I_L, I_Ry, I_L, I_Rx, nx, ny, nz, halo_depth, d, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front);

               // 3. WAIT FOR NETWORK COMPLETION
               MPI_Waitall(active_requests, network_requests, MPI_STATUSES_IGNORE);

               // 4. INTEGRATE RECEIVED HALOS
               integrate_incoming_halos(read_buffer[i], nx, ny, nz, halo_depth, recv_l, recv_r, recv_b, recv_t, recv_bk, recv_fr, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front);

               // 5. COMPUTE BOUNDARY SHELLS (Using exact non-overlapping geometrical regions)
               compute_subdomain_region(read_buffer[i], write_buffer[i], D_L, I_L - 1, D_L, D_Ry, D_L, D_Rx, nx, ny, nz, halo_depth, d, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front);   // Z-Bot
               compute_subdomain_region(read_buffer[i], write_buffer[i], I_Rz + 1, D_Rz, D_L, D_Ry, D_L, D_Rx, nx, ny, nz, halo_depth, d, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front); // Z-Top
               compute_subdomain_region(read_buffer[i], write_buffer[i], I_L, I_Rz, D_L, I_L - 1, D_L, D_Rx, nx, ny, nz, halo_depth, d, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front);   // Y-Bot
               compute_subdomain_region(read_buffer[i], write_buffer[i], I_L, I_Rz, I_Ry + 1, D_Ry, D_L, D_Rx, nx, ny, nz, halo_depth, d, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front); // Y-Top
               compute_subdomain_region(read_buffer[i], write_buffer[i], I_L, I_Rz, I_L, I_Ry, D_L, I_L - 1, nx, ny, nz, halo_depth, d, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front);   // X-Left
               compute_subdomain_region(read_buffer[i], write_buffer[i], I_L, I_Rz, I_L, I_Ry, I_Rx + 1, D_Rx, nx, ny, nz, halo_depth, d, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front); // X-Right

               /* 6. HALO EXCHANGE on write_buffer so boundary-spanning cells use real
                * neighbour values, without this the ghost cells were just using 
                * 0 (default because of calloc), now they use the actuall boundary values
                */
               MPI_Request count_requests[12];
               int count_active = 0;
               stage_outgoing_halos(write_buffer[i], nx, ny, nz, halo_depth, send_l, send_r, send_b, send_t, send_bk, send_fr);
               if (neighbor_left != MPI_PROC_NULL)
               {
                    MPI_Isend(send_l, halo_vol_x, MPI_DOUBLE, neighbor_left, 6, MPI_COMM_WORLD, &count_requests[count_active++]);
                    MPI_Irecv(recv_l, halo_vol_x, MPI_DOUBLE, neighbor_left, 7, MPI_COMM_WORLD, &count_requests[count_active++]);
               }
               if (neighbor_right != MPI_PROC_NULL)
               {
                    MPI_Isend(send_r, halo_vol_x, MPI_DOUBLE, neighbor_right, 7, MPI_COMM_WORLD, &count_requests[count_active++]);
                    MPI_Irecv(recv_r, halo_vol_x, MPI_DOUBLE, neighbor_right, 6, MPI_COMM_WORLD, &count_requests[count_active++]);
               }
               if (neighbor_bottom != MPI_PROC_NULL)
               {
                    MPI_Isend(send_b, halo_vol_y, MPI_DOUBLE, neighbor_bottom, 8, MPI_COMM_WORLD, &count_requests[count_active++]);
                    MPI_Irecv(recv_b, halo_vol_y, MPI_DOUBLE, neighbor_bottom, 9, MPI_COMM_WORLD, &count_requests[count_active++]);
               }
               if (neighbor_top != MPI_PROC_NULL)
               {
                    MPI_Isend(send_t, halo_vol_y, MPI_DOUBLE, neighbor_top, 9, MPI_COMM_WORLD, &count_requests[count_active++]);
                    MPI_Irecv(recv_t, halo_vol_y, MPI_DOUBLE, neighbor_top, 8, MPI_COMM_WORLD, &count_requests[count_active++]);
               }
               if (neighbor_back != MPI_PROC_NULL)
               {
                    MPI_Isend(send_bk, halo_vol_z, MPI_DOUBLE, neighbor_back, 10, MPI_COMM_WORLD, &count_requests[count_active++]);
                    MPI_Irecv(recv_bk, halo_vol_z, MPI_DOUBLE, neighbor_back, 11, MPI_COMM_WORLD, &count_requests[count_active++]);
               }
               if (neighbor_front != MPI_PROC_NULL)
               {
                    MPI_Isend(send_fr, halo_vol_z, MPI_DOUBLE, neighbor_front, 11, MPI_COMM_WORLD, &count_requests[count_active++]);
                    MPI_Irecv(recv_fr, halo_vol_z, MPI_DOUBLE, neighbor_front, 10, MPI_COMM_WORLD, &count_requests[count_active++]);
               }
               MPI_Waitall(count_active, count_requests, MPI_STATUSES_IGNORE);
               integrate_incoming_halos(write_buffer[i], nx, ny, nz, halo_depth, recv_l, recv_r, recv_b, recv_t, recv_bk, recv_fr, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front);

               /* 7. COUNT ISOVALUES on post-stencil write_buffer
                * Ghost cells are now filled, so boundary cells are double counted
                */
               local_results[t * F + i] = count_isovalues_region(write_buffer[i], D_L, D_Rz, D_L, D_Ry, D_L, D_Rx, nx, ny, nz, halo_depth, isovalue);

               // Pointer Swap
               double *temp_ptr = read_buffer[i];
               read_buffer[i] = write_buffer[i];
               write_buffer[i] = temp_ptr;
          }
     }

     MPI_Reduce(local_results, global_results, T * F, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
     MPI_Barrier(MPI_COMM_WORLD);
     double clock_end = MPI_Wtime();
	 
	 // Print all the values required
     if (myrank == 0)
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
	 
	 // Free all the allocated memory
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

     if (myrank == 0)
     {
          free(global_results); // Special buffer only present in case of rank 0
     }
          
     MPI_Finalize();
     return 0;
}
