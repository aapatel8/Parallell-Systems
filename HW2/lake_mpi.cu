/******************************************************************************
* FILE: 
* DESCRIPTION:
*
* GROUP INFO:
 pranjan            Pritesh Ranjan
 kmishra            Kushagra Mishra
 aapatel8           Akshit Patel
* 
* 
******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "mpilake.h"

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0
#define ROOT 0

#define TAG_PEB_LOCS 1
#define TAG_PEB_VALS 2

#define MAX_PSZ 10

void pick_pebble_locations(int *loc, double *p, int pn, int n);
double f(double p, double t);
void init(double *u, double *pebbles, int n);
void init_pebbles(int *loc, double *vals, double *pebs, int pn, int n);
void pick_pebble_locations(int *loc, double *p, int pn, int n);

int main(int argc, char *argv[]) {
    if(argc != 5)
        {
            printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
            return 0;
        }
    int i;
    int rank, size;
    int     npoints   = atoi(argv[1]);  // 128
    int     npebs     = atoi(argv[2]);  // 5
    double  end_time  = (double)atof(argv[3]);  //1.0
    int     nthreads  = atoi(argv[4]);  // 8
    int     tot_area = npoints * npoints;
    double *u_i0, *u_i1;
    double *u_gpu, *pebs;
    double h;
    
    /*
        Allocate memory for lake and pebbles. 
    */
    u_i0 = (double *)malloc (sizeof(double) * tot_area);
    u_i1 = (double *)malloc (sizeof(double) * tot_area);
    pebs = (double *)malloc (sizeof(double) * tot_area);
    
    u_gpu = (double *)malloc (sizeof(double) * tot_area);
    
    if (u_i0 == NULL || u_i1 == NULL || pebs == NULL || u_gpu == NULL){
        printf("STAGE 1: Memory Allocation failed\n");
        return -1;
    }
    
    double elapsed_gpu;
    struct timeval gpu_start, gpu_end;
    
    /* Initialize pebble locations and values */
    int *loc = (int*) malloc(npebs*sizeof(int));
    double *peb_vals = (double*) malloc(npebs*sizeof(double));
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == ROOT) {
        pick_pebble_locations(loc, peb_vals, npebs, npoints);
        //TODO: this loc and peb_vals have to be send by root.
        for(i=1; i< size; i++){
            MPI_Send(loc, npebs, MPI_INT, i, TAG_PEB_LOCS, MPI_COMM_WORLD);
            MPI_Send(peb_vals, npebs, MPI_DOUBLE, i, TAG_PEB_VALS, MPI_COMM_WORLD);
        }
    } else {
        // TODO: Receive the pebbles location and value from ROOT.
        MPI_Recv(loc, npebs, MPI_INT, ROOT, TAG_PEB_LOCS, MPI_COMM_WORLD);
        MPI_Recv(peb_vals, npebs, MPI_DOUBLE, ROOT, TAG_PEB_VALS, MPI_COMM_WORLD);
    }
    
    init_pebbles(loc, peb_vals, npebs, npoints);
    init(u_i0, pebs, npoints);
    init(u_i1, pebs, npoints);
    
    print_heatmap("lake_i_mpi.dat", pebs, npoints, h);
    
    // ALL processes should sync here.
    run_gpu(u_gpu, u_i0, u_i1, pebs, npoints, h, end_time, nthreads); 

    gettimeofday(&gpu_start, NULL);
    gettimeofday(&gpu_end, NULL);
    elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
                gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
    printf("GPU took %f seconds\n", elapsed_gpu);
    //print_heatmap("lake_f_mpi.dat", u_gpu, npoints, h);
    free(u_i0);
    free(u_i1);
    free(pebs);
    free(u_gpu);
    MPI_Finalize();
}

void init_pebbles(int *loc, double *vals, double *pebs, int pn, int n) {
    memset(pebs, 0, sizeof(double) *n *n);
    int k=0;
    for(k=0; k<pn; k++) {
        pebs[loc[k]] = vals[k];
    }
}

void pick_pebble_locations(int *loc, double *p, int pn, int n) {
    /* Picks pn points out of n points at random and initializes them with 
    random values between 0 - MAX_PSZ 
    loc and p arrays should have capacity of pn elements.
    loc and p arrays shall be sent by ROOT process to each other process.
    */
    int i, j, k, idx;
    int sz;
    if (loc == NULL || p == NULL){
        printf("Location or pebbles array is NULL");
        return
    }
    
    srand(time(NULL));
    memset(p, 0, sizeof(double) * pn);
    for (k=0; k<pn; k++) {
        i = rand() % (n-4) + 2;
        j = rand() % (n-4) + 2;
        sz = rand() % MAX_PSZ;
        idx = j+i*n;
        loc[k] = idx;
        p[k] = (double) sz;
    }
}

void init(double *u, double *pebbles, int n) {
    int i, j, idx;

    for(i = 0; i < n ; i++)
    {
        for(j = 0; j < n ; j++)
        {
            idx = j + i * n;
            u[idx] = f(pebbles[idx], 0.0);  // equivalent to "u[idx] = pebbles[idx]"
        }
    }
}

double f(double p, double t) {
    return -expf(-TSCALE * t) * p;
}
