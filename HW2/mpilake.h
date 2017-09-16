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

#define TSCALE 1.0
#define VSQR 0.1

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads);

#endif
