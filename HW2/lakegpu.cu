#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

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

__device__ double f_2(double p, double t)
{
    return -expf(-TSCALE * t) * p;
}

__global__ void evolve13GPU(double *un, double *uc, double *uo, double *pebbles, double *h, double *dt, double *t) {
    int a = blockDim.x;
    int b = blockDim.y;
    int x = gridDim.x;
    int y = gridDim.y;
    int c = threadIdx.x;
    int d = threadIdx.y;
    int e = blockIdx.x;
    int f = blockIdx.y;
    int j = (a * e) + c;
    int i = (b * f) + d;
    
    int n = x * a;
    //int idx = (a*x) *(f*b) + (a*x)*d + a*e + c;
    // or 
    int idx = j + i * n;
    int north = idx - n;
    int south = idx + n;
    int east = idx + 1;
    int west = idx -1;
    int northnorth = idx - 2*n;
    int southsouth = idx + 2*n;
    int easteast   = idx + 2;
    int westwest   = idx - 1;
    
    int northeast = idx-n + 1;
    int northwest = idx-n-1;
    int southeast = idx+n+1;
    int southwest = idx+n-1;
    
    if (i <= 1 || i >= n-2 || j <= 1 || j >= n-1){
        un[idx] = 0;
    } else {
        un[idx] = 2*uc[idx] - uo[idx]  + VSQR *(dt * dt) * (( uc[west] + 
                            uc[east] + uc[north] + uc[south] + 0.25*(uc[northwest] +
                            uc[northeast] + uc[southwest] + uc[southeast]) + 
                            0.125*(uc[westwest] + uc[easteast] + uc[northnorth] +
                            uc[southsouth]) - 6 * uc[idx])/(h * h) + f_2(pebbles[idx],t));
    }
}

__global__ void copyPointersAround(double *un, double *uc, double *uo){
    double *temp;
    temp = uo;
    uo = uc;
    uc = un;
    un = temp;
}

int tpdt_2(double *t, double dt, double tf)
{
    if((*t) + dt > tf) return 0;
    (*t) = (*t) + dt;
    return 1;
}


void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads)
{
	cudaEvent_t kstart, kstop;
	float ktime;
    int nblocks = npoints/nthreads; 
    
    if (0 != (npoints % nthreads)){
        printf("\nInvalid Input(npoints= %d, nthreads= %d): npoints should be perfectly divisible by nthreads ", npoints, nthreads);
        return -1;
    }
    
    printf("\nGPU method called\n");
	/* HW2: Define your local variables here */
    double *un_d, *uc_d, *uo_d, *pebbles_d;
    double *h_d, *dt_d, *t_d;

    double t_h, dt_h;
    
    t_h = 0;
    dt_h = h/2;
    

    
    /* Set up device timers */  
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* HW2: Add CUDA kernel call preperation code here */
    
    cudaMalloc( (void **) &un_d, sizeof(double) * n * n);
    cudaMalloc( (void **) &uc_d, sizeof(double) * n * n);
    cudaMalloc( (void **) &uo_d, sizeof(double) * n * n);
    cudaMalloc( (void **) &pebbles_d, sizeof(double) * n * n);
    cudaMalloc( (void **) &h_d, sizeof(double) *1);
    cudaMalloc( (void **) &dt_d, sizeof(double) * 1);
    cudaMalloc( (void **) &t_d, sizeof(double) * 1);
    
    cudaMemcpy(uo_d, u0, n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(uc_d, u1, n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(pebbles_d, pebbles, n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(h_d, h, 1, cudaMemcpyHostToDevice);
    cudaMemcpy(dt_d, dt_h, 1, cudaMemcpyHostToDevice);
    cudaMemcpy(t_d, t_h, 1, cudaMemcpyHostToDevice);
    
	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

	/* HW2: Add main lake simulation loop here */
    while(1){
        // Call evolve kernel
        evolve13GPU<<<dim3(nblocks, nblocks), dim3(nthreads, nthreads) >>>(un_d, uc_d, uo_d, pebbles_d, h_d, dt_d, t_d);
        // call the kernel to Copy pointers around.
        copyPointersAround<<<1,1>>>(un_d, uc_d, uo_d);
        if(!tpdt_2(&t, dt, end_time)) break;
    }
    
    /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* HW2: Add post CUDA kernel call processing and cleanup here */
    // Copy the final lake state from device memory to host memory.
    cudaMemcpy(u, un_d, n*n, cudaMemcpyDeviceToHost);
	/* timer cleanup */
    printf("\nGPU method called\n");

	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
    cudaFree(un_d);
    cudaFree(uc_d);
    cudaFree(uo_d);
}
