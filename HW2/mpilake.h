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

#ifndef __LAKEMPI_H__
#define __LAKEMPI_H__

#define GPU_COUNT 4
#define MAX_THREADS 512
#define TSCALE 1.0
#define VSQR 0.1

#define DEBUG_2 0

#define ROOT 0
/* Communication tags */

#define TAG_PEB_LOCS 1
#define TAG_PEB_VALS 2

#define TAG_0_TO_1  3
#define TAG_0_TO_2  4

#define TAG_1_TO_0  5
#define TAG_1_TO_3  6

#define TAG_2_TO_0  7
#define TAG_2_TO_3  8

#define TAG_3_TO_1  9
#define TAG_3_TO_2  10

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

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads, int rank, int size);

#endif
