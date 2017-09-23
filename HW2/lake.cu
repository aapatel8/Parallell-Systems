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
#include "lake.h"

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10


void init(double *u, double *pebbles, int n);
void evolve13pt (double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap(const char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);


int main(int argc, char *argv[])
{

    if(argc != 5)
    {
        printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
        return 0;
    }

    int     npoints   = atoi(argv[1]);  // 128
    int     npebs     = atoi(argv[2]);  // 5
    double  end_time  = (double)atof(argv[3]);  //1.0
    int     nthreads  = atoi(argv[4]);  // 8
    int 	  narea	    = npoints * npoints;  // 128 * 128

    double *u_i0, *u_i1;
    double *u_cpu, *u_gpu, *pebs;
    double h;

    double elapsed_cpu, elapsed_gpu;
    struct timeval cpu_start, cpu_end, gpu_start, gpu_end;

    u_i0 = (double*)malloc(sizeof(double) * narea);
    u_i1 = (double*)malloc(sizeof(double) * narea);
    pebs = (double*)malloc(sizeof(double) * narea);

    u_cpu = (double*)malloc(sizeof(double) * narea);
    u_gpu = (double*)malloc(sizeof(double) * narea);

    printf("Running %s with (%d x %d) grid, until %f, with %d threads\n", argv[0], npoints, npoints, end_time, nthreads);

    h = (XMAX - XMIN)/npoints;

    init_pebbles(pebs, npebs, npoints);
    init(u_i0, pebs, npoints);
    init(u_i1, pebs, npoints);


    print_heatmap("lake_i.dat", u_i0, npoints, h);

    gettimeofday(&cpu_start, NULL);
    run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);
    gettimeofday(&cpu_end, NULL);

    elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(
                cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
    printf("CPU took %f seconds\n", elapsed_cpu);

    gettimeofday(&gpu_start, NULL);
    run_gpu(u_gpu, u_i0, u_i1, pebs, npoints, h, end_time, nthreads);
    gettimeofday(&gpu_end, NULL);
    elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
                gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
    printf("GPU took %f seconds\n", elapsed_gpu);

    print_heatmap("lake_f_cpu.dat", u_cpu, npoints, h);
    print_heatmap("lake_f.dat", u_gpu, npoints, h);
    
    free(u_i0);
    free(u_i1);
    free(pebs);
    free(u_cpu);
    free(u_gpu);

    return 1;
}

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time)
{
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
/*
        memcpy(uo, uc, sizeof(double) * n * n);
        memcpy(uc, un, sizeof(double) * n * n);
*/
        if(!tpdt(&t,dt,end_time)) break;
    }

    memcpy(u, uc, sizeof(double) * n * n);
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

double f(double p, double t)
{
    return -expf(-TSCALE * t) * p;
}

int tpdt(double *t, double dt, double tf)
{
    if((*t) + dt > tf) return 0;
    (*t) = (*t) + dt;
    return 1;
}

void init(double *u, double *pebbles, int n)
{
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


void print_heatmap(const char *filename, double *u, int n, double h)
{
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
