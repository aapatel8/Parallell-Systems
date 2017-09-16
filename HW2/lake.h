/* Contains declarations used by both lake.cu and lakegpu.cu
*/

#ifndef _LAKE_H_
#define _LAKE_H_

#define TSCALE 1.0
#define VSQR 0.1

#define NORTH(idx,n) uc[idx - n]
#define SOUTH(idx,n) uc[idx + n]
#define EAST(idx) uc[idx+1]
#define WEST(idx) uc[idx-1]

#define NORTHNORTH(idx,n) uc[idx - 2*n]
#define SOUTHSOUTH(idx,n) uc[idx + 2*n]
#define EASTEAST(idx) uc[idx + 2]
#define WESTWEST(idx) uc[idx - 2]

#define NORTHEAST(idx,n) uc[idx-n+1]
#define NORTHWEST(idx,n) uc[idx-n-1]
#define SOUTHEAST(idx,n) uc[idx+n+1]
#define SOUTHWEST(idx,n) uc[idx+n-1]

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads);

#endif