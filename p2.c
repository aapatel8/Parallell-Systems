#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include "mpi.h"

// The number of grid points
#define NGRID 20
// The first and last grid point
#define XI -1.0
#define XF 1.5
#define EPSILON 0.005
#define DEGREE 3

#define SEND_TO_SUCC 1
#define RECV_FROM_PRED 1
#define SEND_TO_PRED 0
#define RECV_FROM_SUCC 0
#define ROOT 0
double fn(double x) {
    return pow(x,3) - pow(x,2) - x + 1;
    //return pow(x,2);
}

double dfn(double x) {
    return 3 * pow(x,2) - 2 * x - 1;
    //return 2 * x;
}

int main(int argc, char *argv[]) {
    // loop index, process identity, number of processes.
    int i, j, rank, size, succ, pred;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    MPI_Status status;
    succ = (rank+1) % size;
    pred = (rank-1 + size) % size;
    // domain array and step size
    double x[NGRID +2], dx;
    int n_ngrid;  // Number of grid points allocated to the process.
    int start_x, end_x; // start and end index for x values for this process.
    double block_size = NGRID / size;  // Number of grid points in a block
    
    // Last process is responsible for the extra grid points, which will 
    // not be more than size extra grid points.
    if (rank == size-1) { 
        n_ngrid = (NGRID / size) + (NGRID % size);
        start_x = rank * block_size;
        end_x = start_x + n_ngrid;

    } else {
        n_ngrid = (NGRID / size);
        start_x = rank * block_size;
        end_x = start_x + n_ngrid;
    }

    double  local_min_max[DEGREE-1];
    int min_max_count = 0;
    // The x axis width of decomposed grid.
    double step_size = (double)(XF - XI)/(double)NGRID; 
    double *y, *dy, *err;
    double *glo_err= NULL;
    //TODO: Create a different communicator

    printf("\nRank= %d, size= %d, start_x= %d, end_x= %d, n_ngrid=%d, step_size=%f ", rank, size, start_x, end_x, n_ngrid, step_size);
    
    for (i=1; i <=NGRID ; i++) {
        x[i] = XI + (XF - XI) * (double)(i-1)/(double)(NGRID-1);
    }

    dx = x[2] - x[1];
    x[0] = x[1] - dx;  
    x[NGRID+1] = x[NGRID] + dx; 
    printf("succ= %d, pred= %d,  dx= %f",succ, pred, dx);
    y = (double *) malloc((n_ngrid +2) * sizeof(double));
    dy = (double *) malloc(n_ngrid * sizeof(double));
    err = (double *) malloc(n_ngrid * sizeof(double));
    glo_err = (double *)malloc((NGRID +2) * sizeof(double));
    if (glo_err == NULL) {
        printf("\n Memory Allocation Failed");
    }
    for (i=start_x, j=1; i< end_x; i++, j++) {
        y[j] = fn(x[i]);
    }
    //TODO: send y[1] to predecessor
    //      send y[n_ngrid] to successor
    //      Receive in y[0] from predeessor
    //      Receive in y[n_ngrid+1] from successor.
    //

    if(rank == ROOT) { // Root has no predecessor
        // ROOT sends to rank 1, with tag SEND_TO_SUCC and receives with tag RECV_FROM_SUCC 
        y[0] = fn(x[0]-dx);
        MPI_Send(&y[n_ngrid], 1, MPI_DOUBLE, succ, SEND_TO_SUCC, MPI_COMM_WORLD);
        MPI_Recv(&y[n_ngrid+1], 1, MPI_DOUBLE, succ, RECV_FROM_SUCC, MPI_COMM_WORLD, &status);
        for(j=1; j <= n_ngrid; j++) {  // y[0] is irrelevant in this case
            dy[j-1] = (y[j+1] - y[j-1]) / (2 * dx);
        }
    } else if (rank == size-1) { // Last process has no successor
        y[n_ngrid+1] = fn(x[NGRID]); 
        MPI_Recv(&y[0], 1, MPI_DOUBLE, pred, RECV_FROM_PRED, MPI_COMM_WORLD, &status);
        MPI_Send(&y[1], 1, MPI_DOUBLE, pred, SEND_TO_PRED, MPI_COMM_WORLD);
        for(j=1; j<= n_ngrid; j++) {  // y[n_ngrid+1] is irrelevant in this case
            dy[j-1] = (y[j+1] - y[j-1]) / (2 * dx);
        }
    } else {
        if (rank %2 != 0) { // Odd ranked processes
            MPI_Send(&y[n_ngrid], 1, MPI_DOUBLE, succ, SEND_TO_SUCC, MPI_COMM_WORLD);
            MPI_Recv(&y[0], 1, MPI_DOUBLE, pred, RECV_FROM_PRED, MPI_COMM_WORLD, &status);

            MPI_Send(&y[1], 1, MPI_DOUBLE, pred, SEND_TO_PRED, MPI_COMM_WORLD);
            MPI_Recv(&y[n_ngrid+1], 1, MPI_DOUBLE, succ, RECV_FROM_SUCC, MPI_COMM_WORLD, &status);
        } else { // Even ranked processes
        
            MPI_Recv(&y[0], 1, MPI_DOUBLE, pred, RECV_FROM_PRED, MPI_COMM_WORLD, &status);
            MPI_Send(&y[n_ngrid], 1, MPI_DOUBLE, succ, SEND_TO_SUCC, MPI_COMM_WORLD);

            MPI_Recv(&y[n_ngrid+1], 1, MPI_DOUBLE, succ, RECV_FROM_SUCC, MPI_COMM_WORLD, &status);
            MPI_Send(&y[1], 1, MPI_DOUBLE, pred, SEND_TO_PRED, MPI_COMM_WORLD);
        }
        for(j=1; j<=n_ngrid; j++) {
            dy[j-1] = (y[j+1] - y[j-1]) / (2*dx);
        }
    }
    printf("\n%d No deadlock occured\n", rank); 
    
    for(i=start_x, j=0; i < end_x; i++, j++) {
        err[j] = fabs(dy[j] - dfn(x[i]));
        if(fabs(dy[j]) < EPSILON) {
            local_min_max[min_max_count++] = x[i];
           printf("\nMIN/MAX found at x= %f , dy= %f ",x[i], dy[j]);
        }
        printf("%d %d %d %8f, %8f, %8f, %8f, %8f, %e\n", rank, i, j, x[i], fn(x[i]), y[j], dfn(x[i]), dy[j], err[j]);
        //printf("%d %d %d %8f %10e\n",rank, i, j, x[i], err[j]);
    }
    //TODO: Send this error vector to ROOT
    int st=0;
    for(i=0; i< n_ngrid; i++) {
        //printf(" err[%d] = %e ",i, err[i]);
    }
    st = MPI_Gather(err, n_ngrid, MPI_DOUBLE, glo_err, NGRID+2, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    for(i=0; i< NGRID; i++) {
        //printf("  %d  ",i);
        if (fabs(glo_err[i]) < EPSILON)
            ;
            //printf("\nMIM/MAXXX at x= %f, dy= %f", x[i], glo_err[i]);
    }
    if (st == MPI_SUCCESS) {
        printf("\nResult of Gather %d",st);
    }else printf("\nError in MPI_GATHER %d\n",st);
    printf("\n%d %d %d %d %d",MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_BUFFER);
    MPI_Finalize();
    return 0;
}
