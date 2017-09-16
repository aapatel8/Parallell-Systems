#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define __DEBUG
#define TSCALE 1.0
#define VSQR 0.1

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

__global__ void evolve13GPU(double *un, double *uc, double *uo, double *pebbles, int n, double *h, double *dt, double *t) {
  int idx = (blockIdx.x * gridDim.x + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y;
  int i = idx / n;
  int j = idx % n;
  if(!(i == 0 || i == n - 1 || j == 0 || j == n - 1))
    un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + uc[idx + n] + uc[idx - n] + 0.25*(uc[idx + n - 1] + uc[idx + n + 1] + uc[idx - n - 1] + uc[idx - n + 1])- 5 * uc[idx])/(h * h) + f_CUDA(pebbles[idx],t));
  else un[idx] = 0.;

}

__global__ void copyPointersAround(double *un, double *uc, double *uo){
    double *temp;
    temp = uo;
    uo = uc;
    uc = un;
    un = temp;
}

__global__ void copyLakes(double *uo_d, double *uc_d, double *un_d){
        int idx;
        idx = (blockDim.x * gridDim.x * blockIdx.y * blockDim.y) + (blockDim.x * gridDim.x * threadIdx.y ) + blockDim.x * blockIdx.x + threadIdx.x;
        
        uo_d[idx] = uc_d[idx];
        uc_d[idx] = un_d[idx];
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
    int nblocks = n/nthreads; 
    
    if (0 != (n% nthreads)){
        printf("\nInvalid Input(npoints= %d, nthreads= %d): npoints should be divisible by nthreads ", n, nthreads);
        return ;
    }
    
    printf("\nGPU method called\n");
	/* HW2: Define your local variables here */
    double *un_d, *uc_d, *uo_d, *pebbles_d, *tmp;
    //double h_d, dt_d, t_d;

    double t, dt;
    
    t = 0;
    dt = h/2;
    

    
    /* Set up device timers */  
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* HW2: Add CUDA kernel call preperation code here */
    
    cudaMalloc( (void **) &un_d, sizeof(double) * n * n);
    cudaMalloc( (void **) &uc_d, sizeof(double) * n * n);
    cudaMalloc( (void **) &uo_d, sizeof(double) * n * n);
    cudaMalloc( (void **) &pebbles_d, sizeof(double) * n * n);
//    cudaMalloc( (void **) &h_d, sizeof(double) *1);
  //  cudaMalloc( (void **) &dt_d, sizeof(double) * 1);
    //cudaMalloc( (void **) &t_d, sizeof(double) * 1);
    
    cudaMemcpy(uo_d, u0, n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(uc_d, u1, n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(pebbles_d, pebbles, n*n, cudaMemcpyHostToDevice);
    //cudaMemcpy(h_d, &h, 1, cudaMemcpyHostToDevice);
    //cudaMemcpy(dt_d, &dt_h, 1, cudaMemcpyHostToDevice);
    //cudaMemcpy(t_d, &t_h, 1, cudaMemcpyHostToDevice);
    
	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

	/* HW2: Add main lake simulation loop here */
    while(1){
        // Call evolve kernel
        evolve13GPU<<<dim3(nblocks, nblocks,1), dim3(nthreads, nthreads,1) >>>(un_d, uc_d, uo_d, pebbles_d,n, h, dt, t);
        // call the kernel to Copy pointers around.
        //copyPointersAround<<<1,1>>>(un_d, uc_d, uo_d);
        //copyLakes<<<dim3(nblocks, nblocks), dim3(nthreads, nthreads)>>>(uo_d, uc_d, un_d);
        temp = uo;
        uo = uc;
        uc = un;
        un = temp;
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
    cudaFree(pebbles_d);
    //cudaFree(h_d);
    //cudaFree(dt_d);
    //cudaFree(t_d);
}
