#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include "lake.h"

#define __DEBUG

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}

__device__ int tpdt_2(double *t, double dt, double tf)
{
    if((*t) + dt > tf) return 0;
    (*t) = (*t) + dt;
    return 1;
}

__device__ double f(double p, double t)
{
    return -expf(-TSCALE * t) * p;
}

__global__ evolve13GPU(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t) {
    
    /*
    int a = gridDim.x;
    int b = gridDim.y;
    
    int c = threadIdx.x;
    int d = threadIdx.y;
    
    int e = blockDim.x;
    int f = blockDim.y;
    
    int g = blockIdx.x;
    int h = blockIdx.y;
    
    int my_index = (a*e*f*h) + (d*e*a) + (g*e) + c;
    */
    int idx = (gridDim.x * blockDim.x * blockDim.y * blockIdx.y) + 
              (threadIdx.y * blockDim.x * gridDim.x) +
              (blockIdx.x * blockDim.x) + threadIdx.x;
    int i = idx / n;
    int j = idx % n;
    
    if (i <= 1 || i >= n - 2 || j <= 1 || j >= n - 2) {
        un[idx] = 0.;
    } else {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) * (( WEST(idx) + 
                            EAST(idx) + NORTH(idx,n) + SOUTH(idx,n) + 0.25*(NORTHWEST(idx,n) + 
                                NORTHEAST(idx,n) + SOUTHWEST(idx,n) + SOUTHEAST(idx,n)) + 
                            0.125*(WESTWEST(idx) + EASTEAST(idx) + NORTHNORTH(idx,n) +
                                SOUTHSOUTH(idx,n)) - 6 * uc[idx])/(h * h) + f(pebbles[idx],t));
    }
}

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads)
{
	cudaEvent_t kstart, kstop;
	float ktime;
    int nblocks = n/nthreads;
    printf("\nGPU method called\n");
	/* HW2: Define your local variables here */

    double *un_d, *uc_d, *uo_d, *pebbles_d, *tmp;
    double t, dt;
    t = 0.;
    dt = h/2;
    
        /* Set up device timers */  
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* HW2: Add CUDA kernel call preperation code here */
    cudaMalloc( (void **)&un_d, (sizeof double) * n *n);
    cudaMalloc( (void **)&uc_d, (sizeof double) * n *n);
    cudaMalloc( (void **)&uo_d, (sizeof double) * n *n);
    cudaMalloc( (void **)&pebbles_d, (sizeof double) * n *n);
    
    cudaMemcpy(uo_d, u0, (sizeof double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(uc_d, u1, (sizeof double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(pebbles_d, pebbles, (sizeof double)*n*n, cudaMemcpyHostToDevice);
    
	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

	/* HW2: Add main lake simulation loop here */
	while(1) {
        evolve13GPU<<<dim3(nblocks, nblocks,1), dim3(nthreads, nthreads, 1)>>>(un_d, uc_d, uo_d, pebbles_d,n, h, dt, t);
        tmp = uo_d;
        uo_d = uc_d;
        uc_d = un_d;
        un_d = tmp;
        if(!tpdt_2(&t, dt, end_time)) break;
    }
    cudaMemcpy(u, uc_d, (sizeof double)*n*n, cudaMemcpyDeviceToHost);
    
        /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* HW2: Add post CUDA kernel call processing and cleanup here */

    cudaFree(un_d);
    cudaFree(uc_d);
    cudaFree(uo_d);
    cudaFree(pebbles_d);

	/* timer cleanup */    
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
    printf("\nGPU method Exited\n");

}
