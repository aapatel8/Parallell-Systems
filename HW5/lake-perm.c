#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "jemalloc/jemalloc.h"

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0
#define TRUE 1
#define FALSE 0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

#define DEFAULT_NPEBS           4
#define DEFAULT_NPOINTS         128 
#define DEFAULT_END_TIME        1.0

// Lake functions
void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);
void init_pebbles(double *p, int pn, int n);
double f(double p, double t);
int tpdt(double *t, double dt, double end_time);
void init(double *u, double *pebbles, int n);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
void print_heatmap(const char *filename, double *u, int n, double h);

/* (u_cpu, u_i0, u_i1, pebs, npoints, h, end_time
 * &cpu_start &cpu_end
 * */

PERM double *u_i0;
PERM double *u_i1;
PERM double *u_cpu;
PERM double *pebs;

PERM double elapsed_cpu; 
PERM double t;

#define BACK_FILE "/tmp/app.back" /* Note: different backup and mmap files */
#define MMAP_FILE "/tmp/app.mmap"
#define MMAP_SIZE ((size_t)1 << 30)

int main(int argc, char *argv[])
{
    int     npoints   = DEFAULT_NPOINTS;
    int     npebs     = DEFAULT_NPEBS;
    double  end_time  = DEFAULT_END_TIME;
    int     narea     = npoints * npoints;

    const double h = (XMAX - XMIN)/npoints;

    //double elapsed_cpu;
    int do_restore;
    struct timeval cpu_start, cpu_end;
    do_restore = argc > 1 && strcmp("-r", argv[1]) == 0;
    char *mode = (do_restore) ? "r+" : "w+";
    perm(PERM_START, PERM_SIZE);
    mopen(MMAP_FILE, mode, MMAP_SIZE);
    bopen(BACK_FILE, mode);
    if(do_restore == TRUE){
        restore();   
    } else{
        u_i0 = (double*)malloc(sizeof(double) * narea);
        u_i1 = (double*)malloc(sizeof(double) * narea);
        pebs = (double*)malloc(sizeof(double) * narea);
        u_cpu = (double*)malloc(sizeof(double) * narea);

        printf("Running %s with (%d x %d) grid, until %f\n", argv[0], npoints, npoints, end_time);

        init_pebbles(pebs, npebs, npoints);
        init(u_i0, pebs, npoints);
        init(u_i1, pebs, npoints);
        memset(u_cpu, 0, sizeof(double) *narea);

        t = 0.;
        mflush();
        backup();
    }

  //  gettimeofday(&cpu_start, NULL);
    run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);
   // gettimeofday(&cpu_end, NULL);
/*
    elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(
                cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
 */   printf("Execution took %f seconds\n", elapsed_cpu);

    print_heatmap("lake_f.dat", u_cpu, npoints, h);

    free(u_i0);
    free(u_i1);
    free(pebs);
    free(u_cpu);

    mclose();
    bclose();
    return 0;
}

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time)
{
    double *un, *uc, *uo, *temp;
    double  dt;
    uo = u0;
    uc = u1;
    un = u;
    dt = h / 2.;
    struct timeval a_time, b_time;
    struct timeval *a, *b, *tt;
    a = &a_time;
    b = &b_time;
    gettimeofday(a, NULL);
    while(1)
    {
        printf("Timestep %f\n",t);
        evolve(un, uc, uo, pebbles, n, h, dt, t);
        temp = uo;
        uo = uc;
        uc = un;
        un = temp;
        gettimeofday(b, NULL);
        elapsed_cpu += (b->tv_sec + b->tv_usec/1000000.0) - (a->tv_sec + a->tv_usec/1000000.0);
        tt = a;
        a = b;
        b = tt;
        backup();
        if(!tpdt(&t,dt,end_time)) break;
    }
    if(u != uc){
        memcpy(u, uc, sizeof(double) * n * n);
    }
}

void init_pebbles(double *p, int pn, int n)
{
    int i, j, k, idx;
    int sz;

    srand( 10 );
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
            u[idx] = f(pebbles[idx], 0.0);
        }
    }
}

void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
    int i, j, idx;

    for( i = 0; i < n; i++)
    {
        for( j = 0; j < n; j++)
        {
            idx = j + i * n;

            if( i == 0 || i == n - 1 || j == 0 || j == n - 1)
            {
                un[idx] = 0.;
            }
            else
            {
                un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + 
                            uc[idx + n] + uc[idx - n] - 4 * uc[idx])/(h * h) + f(pebbles[idx],t));
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
