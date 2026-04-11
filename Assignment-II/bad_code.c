#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define IDX(x, y, z) (((x) + hd) * NY * NZ + ((y) + hd) * NZ + ((z) + hd))

void allocalldata(double ***data, int F, int NX, int NY, int NZ) {
    *data = (double **)malloc(F * sizeof(double *));
    for (int f = 0; f < F; f++) {
        (*data)[f] = (double *)calloc(NX * NY * NZ, sizeof(double));
    }
}

void freealldata(double **data, int F) {
    for (int f = 0; f < F; f++) {
        free(data[f]);
    }
    free(data);
}

void freebuffers(double *buf_xm, double *buf_xp, double *buf_ym, double *buf_yp, double *buf_zm, double *buf_zp) {
    free(buf_xm); 
    free(buf_xp);
    free(buf_ym); 
    free(buf_yp);
    free(buf_zm); 
    free(buf_zp);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc != 12) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s d px py pz nx ny nz T seed F isovalue\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int d = atoi(argv[1]);
    int px = atoi(argv[2]);
    int py = atoi(argv[3]);
    int pz = atoi(argv[4]);
    int nx = atoi(argv[5]);
    int ny = atoi(argv[6]);
    int nz = atoi(argv[7]);
    int T = atoi(argv[8]);
    int seed = atoi(argv[9]);
    int F = atoi(argv[10]);
    double isovalue = atof(argv[11]);

    int hd = (d - 1) / 6;

    int ix = rank / (py * pz);
    int iy = (rank / pz) % py;
    int iz = rank % pz;

    int nbr_xm = (ix > 0) ? rank - py * pz : -1; // West
    int nbr_xp = (ix < px-1) ? rank + py * pz : -1; // East
    int nbr_ym = (iy > 0) ? rank - pz : -1; // South
    int nbr_yp = (iy < py-1) ? rank + pz : -1; // North
    int nbr_zm = (iz > 0) ? rank - 1 : -1; // Back
    int nbr_zp = (iz < pz-1) ? rank + 1 : -1; // Front

    int NX = nx + 2 * hd;
    int NY = ny + 2 * hd;
    int NZ = nz + 2 * hd;
    int arrSize = nx * ny * nz;

    double **data, **newdata;
    allocalldata(&data, F, NX, NY, NZ);
    allocalldata(&newdata, F, NX, NY, NZ);

    // Data Initialization
    srand(seed);
    for (int i = 0; i < F; i++) {
        for (int j = 0; j < arrSize; j++) {
            int jx = j / (ny * nz);
            int jy = (j / nz) % ny;
            int jz = j % nz;
            data[i][IDX(jx, jy, jz)] = (double)rand() * (rank + 1) / (110426.0 + i + j);
        }
    }

    int sz_x = hd * ny * nz;
    int sz_y = nx * hd * nz;
    int sz_z = nx * ny * hd;

    // Send/receive buffers for each of the 6 faces.
    double *sbuf_xm = (double *)malloc(sz_x * sizeof(double));
    double *sbuf_xp = (double *)malloc(sz_x * sizeof(double));
    double *sbuf_ym = (double *)malloc(sz_y * sizeof(double));
    double *sbuf_yp = (double *)malloc(sz_y * sizeof(double));
    double *sbuf_zm = (double *)malloc(sz_z * sizeof(double));
    double *sbuf_zp = (double *)malloc(sz_z * sizeof(double));
    double *rbuf_xm = (double *)malloc(sz_x * sizeof(double));
    double *rbuf_xp = (double *)malloc(sz_x * sizeof(double));
    double *rbuf_ym = (double *)malloc(sz_y * sizeof(double));
    double *rbuf_yp = (double *)malloc(sz_y * sizeof(double));
    double *rbuf_zm = (double *)malloc(sz_z * sizeof(double));
    double *rbuf_zp = (double *)malloc(sz_z * sizeof(double));

    // Per-field isovalue counts
    long long *lcnt = (long long *)malloc(F * sizeof(long long));
    long long *gcnt = (long long *)malloc(F * sizeof(long long));

    double t_start = MPI_Wtime();

    for (int t = 0; t < T; t++) {
        for (int f = 0; f < F; f++) {
            double *df = data[f];
            MPI_Request req[12];
            int nr = 0;

            // Post receives
            if (nbr_xm >= 0) MPI_Irecv(rbuf_xm, sz_x, MPI_DOUBLE, nbr_xm, 1, MPI_COMM_WORLD, &req[nr++]);
            if (nbr_xp >= 0) MPI_Irecv(rbuf_xp, sz_x, MPI_DOUBLE, nbr_xp, 1, MPI_COMM_WORLD, &req[nr++]);
            if (nbr_ym >= 0) MPI_Irecv(rbuf_ym, sz_y, MPI_DOUBLE, nbr_ym, 2, MPI_COMM_WORLD, &req[nr++]);
            if (nbr_yp >= 0) MPI_Irecv(rbuf_yp, sz_y, MPI_DOUBLE, nbr_yp, 2, MPI_COMM_WORLD, &req[nr++]);
            if (nbr_zm >= 0) MPI_Irecv(rbuf_zm, sz_z, MPI_DOUBLE, nbr_zm, 3, MPI_COMM_WORLD, &req[nr++]);
            if (nbr_zp >= 0) MPI_Irecv(rbuf_zp, sz_z, MPI_DOUBLE, nbr_zp, 3, MPI_COMM_WORLD, &req[nr++]);

            if (nbr_xm >= 0) {
                int k = 0;
                for (int x = 0; x < hd; x++)
                    for (int y = 0; y < ny; y++)
                        for (int z = 0; z < nz; z++)
                            sbuf_xm[k++] = df[IDX(x, y, z)];
                MPI_Isend(sbuf_xm, sz_x, MPI_DOUBLE, nbr_xm, 1, MPI_COMM_WORLD, &req[nr++]);
            }
            if (nbr_xp >= 0) {
                int k = 0;
                for (int x = nx - hd; x < nx; x++)
                    for (int y = 0; y < ny; y++)
                        for (int z = 0; z < nz; z++)
                            sbuf_xp[k++] = df[IDX(x, y, z)];
                MPI_Isend(sbuf_xp, sz_x, MPI_DOUBLE, nbr_xp, 1, MPI_COMM_WORLD, &req[nr++]);
            }
            if (nbr_ym >= 0) {
                int k = 0;
                for (int x = 0; x < nx; x++)
                    for (int y = 0; y < hd; y++)
                        for (int z = 0; z < nz; z++)
                            sbuf_ym[k++] = df[IDX(x, y, z)];
                MPI_Isend(sbuf_ym, sz_y, MPI_DOUBLE, nbr_ym, 2, MPI_COMM_WORLD, &req[nr++]);
            }
            if (nbr_yp >= 0) {
                int k = 0;
                for (int x = 0; x < nx; x++)
                    for (int y = ny - hd; y < ny; y++)
                        for (int z = 0; z < nz; z++)
                            sbuf_yp[k++] = df[IDX(x, y, z)];
                MPI_Isend(sbuf_yp, sz_y, MPI_DOUBLE, nbr_yp, 2, MPI_COMM_WORLD, &req[nr++]);
            }
            if (nbr_zm >= 0) {
                int k = 0;
                for (int x = 0; x < nx; x++)
                    for (int y = 0; y < ny; y++)
                        for (int z = 0; z < hd; z++)
                            sbuf_zm[k++] = df[IDX(x, y, z)];
                MPI_Isend(sbuf_zm, sz_z, MPI_DOUBLE, nbr_zm, 3, MPI_COMM_WORLD, &req[nr++]);
            }
            if (nbr_zp >= 0) {
                int k = 0;
                for (int x = 0; x < nx; x++)
                    for (int y = 0; y < ny; y++)
                        for (int z = nz - hd; z < nz; z++)
                            sbuf_zp[k++] = df[IDX(x, y, z)];
                MPI_Isend(sbuf_zp, sz_z, MPI_DOUBLE, nbr_zp, 3, MPI_COMM_WORLD, &req[nr++]);
            }

            MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);

            if (nbr_xm >= 0) {
                int k = 0;
                for (int x = -hd; x < 0; x++)
                    for (int y = 0; y < ny; y++)
                        for (int z = 0; z < nz; z++)
                            df[IDX(x, y, z)] = rbuf_xm[k++];
            }
            if (nbr_xp >= 0) {
                int k = 0;
                for (int x = nx; x < nx + hd; x++)
                    for (int y = 0; y < ny; y++)
                        for (int z = 0; z < nz; z++)
                            df[IDX(x, y, z)] = rbuf_xp[k++];
            }
            if (nbr_ym >= 0) {
                int k = 0;
                for (int x = 0; x < nx; x++)
                    for (int y = -hd; y < 0; y++)
                        for (int z = 0; z < nz; z++)
                            df[IDX(x, y, z)] = rbuf_ym[k++];
            }
            if (nbr_yp >= 0) {
                int k = 0;
                for (int x = 0; x < nx; x++)
                    for (int y = ny; y < ny + hd; y++)
                        for (int z = 0; z < nz; z++)
                            df[IDX(x, y, z)] = rbuf_yp[k++];
            }
            if (nbr_zm >= 0) {
                int k = 0;
                for (int x = 0; x < nx; x++)
                    for (int y = 0; y < ny; y++)
                        for (int z = -hd; z < 0; z++)
                            df[IDX(x, y, z)] = rbuf_zm[k++];
            }
            if (nbr_zp >= 0) {
                int k = 0;
                for (int x = 0; x < nx; x++)
                    for (int y = 0; y < ny; y++)
                        for (int z = nz; z < nz + hd; z++)
                            df[IDX(x, y, z)] = rbuf_zp[k++];
            }
        }

        for (int f = 0; f < F; f++) {
            double *df  = data[f];
            double *ndf = newdata[f];
            lcnt[f] = 0;

            for (int x = 0; x < nx; x++) {
                for (int y = 0; y < ny; y++) {
                    for (int z = 0; z < nz; z++) {
                        double sum = df[IDX(x, y, z)];
                        for (int k = 1; k <= hd; k++) {
                            sum += df[IDX(x + k, y, z)] + df[IDX(x - k, y, z)];
                            sum += df[IDX(x, y + k, z)] + df[IDX(x, y - k, z)];
                            sum += df[IDX(x, y, z + k)] + df[IDX(x, y, z - k)];
                        }
                        double val = sum / d;
                        ndf[IDX(x, y, z)] = val;
                        if (val == isovalue)
                            lcnt[f]++;
                    }
                }
            }
        }

        double **tmp = data; data = newdata; newdata = tmp;

        
        MPI_Reduce(lcnt, gcnt, F, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int f = 0; f < F; f++) {
                if (f) printf(" ");
                printf("%lld", gcnt[f]);
            }
            printf("\n");
        }

    } /* end time-step loop */

    double t_end   = MPI_Wtime();
    double local_t = t_end - t_start;
    double max_t;
    MPI_Reduce(&local_t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("%lf\n", max_t);


    freealldata(data, F);
    freealldata(newdata, F);
    freebuffers(sbuf_xm, sbuf_xp, sbuf_ym, sbuf_yp, sbuf_zm, sbuf_zp);
    freebuffers(rbuf_xm, rbuf_xp, rbuf_ym, rbuf_yp, rbuf_zm, rbuf_zp);
    free(lcnt);
    free(gcnt);

    MPI_Finalize();
    return 0;
}
