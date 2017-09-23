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
#include <mpi.h>
#include "mpilake.h"

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10

/* 13 point stencil evolve function */
void evolve13pt (double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);

void init_pebbles(double *p, int pn, int n);

void pick_pebble_locations(int *loc, double *p, int pn, int n);
double f(double p, double t);
void init(double *u, double *pebbles, int n);
void init_pebbles_gpu_mpi(int *loc, double *vals, double *pebs, int pn, int n);
void print_heatmap(const char *filename, double *u, int n, double h);

int main(int argc, char *argv[]) {
    if(argc != 5) {
            printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
            return 0;
        }
    int i;
    /* MPI related variables */
    int rank, size;
    MPI_Status status;

    int     npoints   = atoi(argv[1]);  // 128
    int     npebs     = atoi(argv[2]);  // 5
    double  end_time  = (double)atof(argv[3]);  //1.0
    int     nthreads  = atoi(argv[4]);  // 8
    int     tot_area = npoints * npoints;
    double *u_i0, *u_i1;
    double *u_cpu, *u_gpu, *pebs;
    double h;
    double elapsed_cpu, elapsed_gpu;
    struct timeval cpu_start, cpu_end, gpu_start, gpu_end;
    h = (XMAX - XMIN)/npoints; 
    /*
        Allocate memory for lake and pebbles. 
    */
    u_i0 = (double *)malloc (sizeof(double) * tot_area);
    u_i1 = (double *)malloc (sizeof(double) * tot_area);
    pebs = (double *)malloc (sizeof(double) * tot_area);
    
    u_cpu = (double*)malloc(sizeof(double) * tot_area);
    u_gpu = (double *)malloc (sizeof(double) * tot_area);
    
    /* Initialize pebble locations and values */
    int *loc = (int*) malloc(npebs*sizeof(int));
    double *peb_vals = (double*) malloc(npebs*sizeof(double));

    if (loc== NULL || peb_vals == NULL || u_i0 == NULL || u_i1 == NULL || pebs == NULL || u_gpu == NULL){
        printf("STAGE 1: Memory Allocation failed\n");
        return -1;
    }
        
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if ( ROOT == rank) {
        // Root picks the pebble locations and the value at those locations.
        pick_pebble_locations(loc, peb_vals, npebs, npoints);
        //Root sends array loc and peb_vals to every other process.
        for(i=1; i< size; i++){
            MPI_Send(loc, npebs, MPI_INT, i, TAG_PEB_LOCS, MPI_COMM_WORLD);
            MPI_Send(peb_vals, npebs, MPI_DOUBLE, i, TAG_PEB_VALS, MPI_COMM_WORLD);
        }
    } else {
        // Every non ROOT process receives the pebbles location and value from ROOT.
        MPI_Recv(loc, npebs, MPI_INT, ROOT, TAG_PEB_LOCS, MPI_COMM_WORLD, &status);
        MPI_Recv(peb_vals, npebs, MPI_DOUBLE, ROOT, TAG_PEB_VALS, MPI_COMM_WORLD, &status);  
    }
    // Every processes initializes the pebbles with the 'loc' and 'peb_vals' array.
    init_pebbles_gpu_mpi(loc, peb_vals, pebs, npebs, npoints);
    // Initialize initial verisons of the pond.
    init(u_i0, pebs, npoints);
    init(u_i1, pebs, npoints);
    if(ROOT == rank ){
        // The CPU serial version is run at the ROOT.
        print_heatmap("lake_i.dat", u_i0, npoints, h);
    
        gettimeofday(&cpu_start, NULL);
        run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);
        gettimeofday(&cpu_end, NULL);

        elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(
                    cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
        printf("CPU took %f seconds\n", elapsed_cpu);
        print_heatmap("lake_f.dat", u_cpu, npoints, h);
    }
    // ALL processes should sync here.
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&gpu_start, NULL);
    // Every process executes on one quadrant of GPU.
    run_gpu_mpi(u_gpu, u_i0, u_i1, pebs, npoints, h, end_time, nthreads, rank, size); 
    gettimeofday(&gpu_end, NULL);
    elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
                gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
    printf("GPU on rank %d, took %f seconds\n", rank, elapsed_gpu);
    char filename[20];
    sprintf(filename, "lake_f_%d.dat",rank);
    print_heatmap(filename, u_gpu, npoints, h);
    free(u_i0);
    free(u_i1);
    free(pebs);
    free(u_cpu);
    free(u_gpu);
    free(loc);
    free(peb_vals);
    MPI_Finalize();
}

void init_pebbles(double *p, int pn, int n)
{  /* Picks pn points at random and initializes them with random value between 0 - MAX_PSZ   */
    int i, j, k, idx;
    int sz;

    srand( time(NULL) );
    memset(p, 0, sizeof(double) * n * n);

    for( k = 0; k < pn ; k++ )
    {
        i = rand() % (n - 4) + 2;
        j = rand() % (n - 4) + 2;
        sz = rand() % MAX_PSZ;
        idx = j + i * n;
        p[idx] = (double) sz;
    }
}

void init_pebbles_gpu_mpi(int *loc, double *vals, double *pebs, int pn, int n) {
    // Initialize the pebbles array with pebble locations and values.
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
    loc and p arrays shall be sent by ROOT process to every other process.
    */
    int i, j, k, idx;
    int sz;
    if (loc == NULL || p == NULL){
        printf("Location or pebbles array is NULL");
        return;
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

void print_heatmap(const char *filename, double *u, int n, double h) {
    int i, j, idx;

    FILE *fp = fopen(filename, "w");  

    for( i = 0; i < n; i++ )
    {
        for( j = 0; j < n; j++ )
        {
            idx = j + i * n;
            fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
        }
    }

    fclose(fp);
} 

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time) {
    /* The CPU version */
    double *un, *uc, *uo, *tmp;
    double t, dt;

    un = (double*)malloc(sizeof(double) * n * n);
    uc = (double*)malloc(sizeof(double) * n * n);
    uo = (double*)malloc(sizeof(double) * n * n);

    memcpy(uo, u0, sizeof(double) * n * n);
    memcpy(uc, u1, sizeof(double) * n * n);

    t = 0.;
    dt = h / 2.;

    while(1)
    {
        evolve13pt (un, uc, uo, pebbles, n, h, dt, t);
        tmp = uo;
        uo = uc;
        uc = un;
        un = tmp;
        if(!tpdt(&t,dt,end_time)) break;
    }

    memcpy(u, uc, sizeof(double) * n * n);
}

void evolve13pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
    int i, j, idx;

    for( i = 0; i < n; i++)
    {
        for( j = 0; j < n; j++)
        {
            idx = j + i * n;

            if( i <= 1 || i >= n - 2 || j <= 1 || j >= n - 2 )
            {
                un[idx] = 0.;
            }
            else
            {
                un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) * (( WEST(idx) + 
                            EAST(idx) + NORTH(idx,n) + SOUTH(idx,n) + 0.25*(NORTHWEST(idx,n) + 
                                NORTHEAST(idx,n) + SOUTHWEST(idx,n) + SOUTHEAST(idx,n)) + 
                            0.125*(WESTWEST(idx) + EASTEAST(idx) + NORTHNORTH(idx,n) +
                                SOUTHSOUTH(idx,n)) - 6 * uc[idx])/(h * h) + f(pebbles[idx],t));

            }
        }
    }
}

int tpdt(double *t, double dt, double tf)
{
    if((*t) + dt > tf) return 0;
    (*t) = (*t) + dt;
    return 1;
}

