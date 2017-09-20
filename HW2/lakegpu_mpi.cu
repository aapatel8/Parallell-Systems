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
#include <cuda_runtime.h>
#include <time.h>
#include "mpi.h"

#include "mpilake.h"
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
inline void __cudaSafeCall( cudaError err, const char *file, const int line ) {
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

inline void __cudaCheckError( const char *file, const int line ) {
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

void copy_from_buffer_to_lake(double *lake, double *buffer, int incr);
void copy_from_lake_to_buffer(double *buffer, double *lake, int incr);

int tpdt_2(double *t, double dt, double tf) {
    if((*t) + dt > tf) return 0;
    (*t) = (*t) + dt;
    return 1;
}

__device__ double f_2(double p, double t) {
    return -expf(-TSCALE * t) * p;
}

__global__ void evolve13_gpu_MPI(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t, int rank){
    int block_idx_x = gridDim.x*(rank%2) + blockIdx.x;
    int block_idx_y = gridDim.y*(rank/2) + blockIdx.y;
    
    int griddim_x = 2 * gridDim.x;
    
    int idx = (griddim_x * blockDim.x * blockDim.y * block_idx_y) + 
              (threadIdx.y * blockDim.x * griddim_x) +
              (block_idx_x * blockDim.x) + threadIdx.x;
              
    int i = idx / n;
    int j = idx % n;
    
    if (i <= 1 || i >= n - 2 || j <= 1 || j >= n - 2) {
        un[idx] = 0.;
    } else {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) * (( WEST(idx) + 
                EAST(idx) + NORTH(idx,n) + SOUTH(idx,n) + 0.25*(NORTHWEST(idx,n) + 
                NORTHEAST(idx,n) + SOUTHWEST(idx,n) + SOUTHEAST(idx,n)) + 
                0.125*(WESTWEST(idx) + EASTEAST(idx) + NORTHNORTH(idx,n) +
                SOUTHSOUTH(idx,n)) - 6 * uc[idx])/(h * h) + f_2(pebbles[idx],t));
    }
}

void run_gpu_mpi (double *u, double *u0, double *u1, double *pebbles, int npoints, double h, double end_time, int nthreads, int rank, int size) {

    if ((npoints % GPU_COUNT != 0) || (npoints % nthreads != 0) || (nthreads * nthreads > MAX_THREADS)) {
        printf("Invalid arguments: Can't launch on GPU");
        return;
    }
	/* Divide the (npoints * npoints) sized lake into 4 smaller square grids. */
    int npts = npoints/2;   // = 64
    int nblocks = npts/nthreads;   // (nblocks * nblocks) blocks per gpu. nblocks = 8
    
    cudaEvent_t kstart, kstop;
	float ktime;
    printf("\nGPU method called on GPU # %d out of %d GPUs\n", rank, size);
	/* HW2: Define your local variables here */
    double *un_d, *uc_d, *uo_d, *pebbles_d, *tmp;
    double t, dt;
    double *recv_a, *recv_b, *send_c, *send_d;
    double *source, *dest, *loc;
    size_t tot_area = sizeof(double)*npoints*npoints;
    size_t side_len = sizeof(double)*npoints;
    t = 0.;
    dt = h/2;
    
    /* Set up device timers */  
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));
    
    MPI_Request send_reqs[2], recv_reqs[2];
    MPI_Status  send_stats[2], recv_stats[2];
    
    /* Buffers to receive/send data from/to neighbors */
    recv_a = (double*)malloc(side_len);
    recv_b = (double*)malloc(side_len);
    send_c = (double*)malloc(side_len);
    send_d = (double*)malloc(side_len);

	/* HW2: Add CUDA kernel call preperation code here */
    cudaMalloc( (void **)&un_d, tot_area);
    cudaMalloc( (void **)&uc_d, tot_area);
    cudaMalloc( (void **)&uo_d, tot_area);
    cudaMalloc( (void **)&pebbles_d, tot_area);
    
    cudaMemcpy(uo_d, u0, tot_area, cudaMemcpyHostToDevice);
    cudaMemcpy(uc_d, u1, tot_area, cudaMemcpyHostToDevice);
    cudaMemcpy(pebbles_d, pebbles, tot_area, cudaMemcpyHostToDevice);
    
	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));
    printf("rank %d, Memory Allocated. Entering While Loop\n",rank);

	/* HW2: Add main lake simulation loop here */
	while(1){
        // TODO: Evolve Kernel call here
        printf(".");
        evolve13_gpu_MPI<<<dim3(nblocks, nblocks,1), dim3(nthreads, nthreads, 1)>>>(un_d, uc_d, uo_d, pebbles_d, npoints, h, dt, t, rank);
        cudaMemcpy(u, un_d, tot_area, cudaMemcpyDeviceToHost);
        if(!tpdt_2(&t, dt, end_time)) break;
        
        if (ROOT == rank) {
            MPI_Irecv(recv_a, npoints, MPI_DOUBLE, 2, TAG_2_TO_0, MPI_COMM_WORLD, &recv_reqs[0]);
            MPI_Irecv(recv_b, npoints, MPI_DOUBLE, 1, TAG_1_TO_0, MPI_COMM_WORLD, &recv_reqs[1]);
            
            source = &u[npts-2];
            copy_from_lake_to_buffer(send_d, source, npoints);
            MPI_Isend(send_d, npoints, MPI_DOUBLE, 1, TAG_0_TO_1, MPI_COMM_WORLD, &send_reqs[0]);
            
            loc = &u[npoints * (npts-2)];
            memcpy(send_c, loc, sizeof(double) * npts);
            memcpy(&send_c[npts], &loc[npoints], sizeof(double) *npts);
            MPI_Isend(send_c, npoints, MPI_DOUBLE, 2, TAG_0_TO_2, MPI_COMM_WORLD, &send_reqs[1]);
            if(MPI_SUCCESS != MPI_Waitall(2, recv_reqs, recv_stats)) {
                printf("STAGE 2, rank %d : Receive Waitall Failed \n", rank);
                exit(-1) ;
            }
            
            dest = &u[npts];
            copy_from_buffer_to_lake(dest, recv_b, npoints);
            
            dest = &u[npoints*npts];
            memcpy(dest, recv_a, sizeof(double)*npts);
            dest += npoints;
            memcpy(dest, &recv_a[npts], sizeof(double)*npts);
           /* if(MPI_SUCCESS != MPI_Waitall(2, send_reqs, send_stats)) {
                printf("STAGE 2: Send Waitall Failed\n");
                exit(-1);
            }
            */
        } else if(1 == rank) {
            MPI_Irecv(recv_a, npoints, MPI_DOUBLE, 3, TAG_3_TO_1, MPI_COMM_WORLD, &recv_reqs[0]);
            MPI_Irecv(recv_b, npoints, MPI_DOUBLE, 0, TAG_0_TO_1, MPI_COMM_WORLD, &recv_reqs[1]);
            
            source = &u[npts];
            copy_from_lake_to_buffer(send_d, source, npoints);
            MPI_Isend(send_d, npoints, MPI_DOUBLE, 0, TAG_1_TO_0, MPI_COMM_WORLD, &send_reqs[0]);
            
            loc = &u[npoints*(npts-2) + npts];
            memcpy(send_c, loc, sizeof(double)*npts);
            memcpy(&send_c[npts], &loc[npoints], sizeof(double)*npts);
            MPI_Isend(send_c, npoints, MPI_DOUBLE, 3, TAG_1_TO_3, MPI_COMM_WORLD, &send_reqs[1]);
            if(MPI_SUCCESS != MPI_Waitall(2, recv_reqs, recv_stats)){
                printf("STAGE 2, rank %d : Receive Waitall Failed \n", rank);
                exit(-1) ;
            }
            dest = &u[npts-2];
            copy_from_buffer_to_lake(dest, recv_b, npoints);
            
            dest = &u[npoints*npts+npts];
            memcpy(dest, recv_a, sizeof(double)*npts);
            dest+= npoints;
            memcpy(dest, &recv_a[npts], sizeof(double)*npts);
        } else if(2 == rank) {
            MPI_Irecv(recv_a, npoints, MPI_DOUBLE, 0, TAG_0_TO_2, MPI_COMM_WORLD, &recv_reqs[0]);
            MPI_Irecv(recv_b, npoints, MPI_DOUBLE, 3, TAG_3_TO_2, MPI_COMM_WORLD, &recv_reqs[1]);
            
            source = &u[npoints*npts + npts-2];
            copy_from_lake_to_buffer(send_d, source, npoints);
            MPI_Isend(send_d, npoints, MPI_DOUBLE, 3, TAG_2_TO_3, MPI_COMM_WORLD, &send_reqs[0]);
            
            loc = &u[npoints*npts];
            memcpy(send_c, loc, sizeof(double)*npts);
            memcpy(&send_c[npts], &loc[npoints], sizeof(double)*npts);
            MPI_Isend(send_c, npoints, MPI_DOUBLE, 0, TAG_2_TO_0, MPI_COMM_WORLD, &send_reqs[1]);
            if(MPI_SUCCESS != MPI_Waitall(2, recv_reqs, recv_stats)){
                printf("STAGE 2, rank %d : Receive Waitall Failed \n", rank);
                exit(-1) ;
            }
            
            dest = &u[npoints*npts+npts];
            copy_from_buffer_to_lake(dest, recv_b, npoints);
            
            dest = &u[npoints*(npts-2)];
            memcpy(dest, recv_a, sizeof(double)*npts);
            dest+= npoints;
            memcpy(dest, &recv_a[npts], sizeof(double)*npts);   
        } else if(3 == rank) {
            MPI_Irecv(recv_a, npoints, MPI_DOUBLE, 1, TAG_1_TO_3, MPI_COMM_WORLD, &recv_reqs[0]);
            MPI_Irecv(recv_b, npoints, MPI_DOUBLE, 2, TAG_2_TO_3, MPI_COMM_WORLD, &recv_reqs[1]);
            
            source = &u[npoints*npts+npts];
            copy_from_lake_to_buffer(send_d, source, npoints);
            MPI_Isend(send_d, npoints, MPI_DOUBLE, 2, TAG_3_TO_2, MPI_COMM_WORLD, &send_reqs[0]);
            
            loc = &u[npoints*npts+npts];
            memcpy(send_c, loc, sizeof(double)*npts);
            memcpy(&send_c[npts], &loc[npoints], sizeof(double)*npts);
            MPI_Isend(send_c, npoints, MPI_DOUBLE, 1, TAG_3_TO_1, MPI_COMM_WORLD, &send_reqs[1]);
            
            if(MPI_SUCCESS != MPI_Waitall(2, recv_reqs, recv_stats)) {
                printf("STAGE 2, rank %d : Receive Waitall Failed \n", rank);
                exit(-1) ;
            }
            dest = &u[npoints*npts+npts-2];
            copy_from_buffer_to_lake(dest, recv_b, npoints);
            
            dest = &u[npoints*(npts-2) + npts];
            memcpy(dest, recv_a, sizeof(double)*npts);
            dest+= npoints;
            memcpy(dest, &recv_a[npts], sizeof(double)*npoints);            
        }
        MPI_Barrier(MPI_COMM_WORLD);
        //printf("Rank %d Exited Barrier.\n", rank);
        cudaMemcpy(un_d, u, tot_area, cudaMemcpyHostToDevice);
        
        tmp = uo_d;
        uo_d = uc_d;
        uc_d = un_d;
        un_d = tmp;
    }
        /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* HW2: Add post CUDA kernel call processing and cleanup here */

	/* timer cleanup */
    printf("\nGPU method Done\n");

	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}

void copy_from_lake_to_buffer(double *buffer, double *lake, int incr){
    int k =0;
    double *ptr = lake;
    for(k=0; k<incr; k+=2) {
        buffer[k] = ptr[0];
        buffer[k+1] = ptr[1];
        ptr += incr;
    }
}

void copy_from_buffer_to_lake(double *lake, double *buffer, int incr){
    int k =0;
    double *ptr = lake;
    for(k=0; k<incr; k+=2) {
        ptr[0] = buffer[k];
        ptr[1] = buffer[k+1];
        ptr += incr;
    }
}
