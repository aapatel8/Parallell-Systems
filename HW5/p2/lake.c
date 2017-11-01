#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

#define DEFAULT_NPEBS           4
#define DEFAULT_NPOINTS         512
#define DEFAULT_END_TIME        2.0

// Lake functions
void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);
void init_pebbles(double *p, int pn, int n);
double f(double p, double t);
int tpdt(double *t, double dt, double end_time);
void init(double *u, double *pebbles, int n);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
void print_heatmap(const char *filename, double *u, int n, double h);


int main(int argc, char *argv[])
{
  int     npoints   = DEFAULT_NPOINTS;
  int     npebs     = DEFAULT_NPEBS;
  double  end_time  = DEFAULT_END_TIME;
  int     narea     = npoints * npoints;

  double *u_i0, *u_i1;
  double *u_cpu, *pebs;
  double h;

  double elapsed_cpu;
  struct timeval cpu_start, cpu_end;
  
  u_i0 = (double*)malloc(sizeof(double) * narea);
  u_i1 = (double*)malloc(sizeof(double) * narea);
  pebs = (double*)malloc(sizeof(double) * narea);
  u_cpu = (double*)malloc(sizeof(double) * narea);

  printf("Running %s with (%d x %d) grid, until %f\n", argv[0], npoints, npoints, end_time);

  h = (XMAX - XMIN)/npoints;

  init_pebbles(pebs, npebs, npoints);
  init(u_i0, pebs, npoints);
  init(u_i1, pebs, npoints);


  //print_heatmap("lake_i.dat", u_i0, npoints, h);

  gettimeofday(&cpu_start, NULL);
  run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);
  gettimeofday(&cpu_end, NULL);

  elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(
                  cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
  printf("Execution took %f seconds\n", elapsed_cpu);

  print_heatmap("lake_f.dat", u_cpu, npoints, h);

  free(u_i0);
  free(u_i1);
  free(pebs);
  free(u_cpu);

  return 0;
}

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time)
{
  double *un, *uc, *uo;
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
    printf("Timestep %f\n",t);
    evolve(un, uc, uo, pebbles, n, h, dt, t);

    memcpy(uo, uc, sizeof(double) * n * n);
    memcpy(uc, un, sizeof(double) * n * n);

    if(!tpdt(&t,dt,end_time)) break;
  }

  memcpy(u, un, sizeof(double) * n * n);

  free(un);
  free(uc);
  free(uo);
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
