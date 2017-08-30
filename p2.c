#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include "mpi.h"

// The number of grid points
#define NGRID 1000
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
#define DEBUG 0
#define VERBOSE 1

double fn(double x) {
    return pow(x,3) - pow(x,2) - x + 1;
    //return pow(x,2);
}

double dfn(double x) {
    return 3 * pow(x,2) - 2 * x - 1;
    //return 2 * x;
}

void create_new_communicator(int *rank, int *size, MPI_Comm *new_comm) {
    MPI_Group new_group, orig_group;
    MPI_Comm_size(MPI_COMM_WORLD, size);
    int *new_ranks, i;
    new_ranks = (int*)malloc((*size) * sizeof(int));
    for (i=0; i< *size; i++) {
        new_ranks[i] = i;
    }
    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
    MPI_Group_incl(orig_group, *size, new_ranks, &new_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_group, new_comm);

    MPI_Comm_rank(*new_comm, rank);
    MPI_Comm_size(*new_comm, size);
    if (new_ranks)
        free(new_ranks);

}

void get_x_axis_limits(int rank, int size, int *n_ngrid, int *start_x, int *end_x, int *succ, int *pred) {
    /* Utility Function to calculate the decomposed grid's limits on X axis.
    
    Last process is responsible for the extra grid points, which will 
    not be more than size extra grid points.
    */
    if (rank == size-1) {
        *n_ngrid = (NGRID/size) + (NGRID % size);
        *start_x = rank * (NGRID/size);
        *end_x = *start_x + *n_ngrid;
    } else {
        *n_ngrid = (NGRID/size);
        *start_x = rank * (NGRID/size);
        *end_x = *start_x + *n_ngrid;
    }
    *succ = (rank+1) % size;
    *pred = (rank-1 + size) % size;
}

double create_x_axis_grid_points(double *x) {
    int i=1;
    double dx;
    
    for(i=1; i <= NGRID; i++) {
        x[i] = XI + (XF - XI) * (double)(i-1)/(double)(NGRID-1);
    }
    dx = x[2] - x[1];
    x[0] = x[1] - dx;
    x[NGRID+1] = x[NGRID] + dx;
    return dx;
}

void blocking_and_manual_reduce(int rank, int size, MPI_Comm new_comm) {
    int i, j, succ, pred;
    double x[NGRID +2], dx;
    int n_ngrid;  // Number of grid points allocated to the process.
    int start_x, end_x; // start and end index for x values for this process.
    int block_size = NGRID / size;  // Number of grid points in a block
    
    get_x_axis_limits(rank, size, &n_ngrid, &start_x, &end_x, &succ, &pred);
    dx = create_x_axis_grid_points(x);
    if(VERBOSE) printf("\nRank= %d, size= %d, start_x= %d, end_x= %d, "
                 "n_ngrid=%d, succ= %d, pred= %d,  dx= %f ",rank, size, 
                  start_x, end_x, n_ngrid,succ, pred, dx);
    
    double  local_min_max[DEGREE-1];
    int min_max_count = 0;
    double *y=NULL, *dy=NULL, *err=NULL, *glo_err= NULL;
    double err_sum, std_dev, err_avg, *glo_min_max=NULL;


    y = (double *) malloc((n_ngrid +2) * sizeof(double));
    dy = (double *) malloc(n_ngrid * sizeof(double));
    err = (double *) malloc(n_ngrid * sizeof(double));
   
    if (y == NULL || dy == NULL || err == NULL) {
        printf("\n Memory Allocation Failed");
        exit(-1);
    }
    for (i=start_x, j=1; i< end_x; i++, j++) {
        y[j] = fn(x[i]);
    }
    
    /*
    send y[1] to predecessor
    send y[n_ngrid] to successor
    Receive in y[0] from predeessor
    Receive in y[n_ngrid+1] from successor.
    */
    if(rank == ROOT) { // Root has no predecessor
        glo_err = (double *)malloc(NGRID  * sizeof(double));
        glo_min_max = (double *)malloc(((DEGREE-1)*size) * sizeof(double));

        // ROOT sends to rank 1, with tag SEND_TO_SUCC and receives with tag RECV_FROM_SUCC 
        y[0] = fn(x[0]-dx);
        MPI_Send(&y[n_ngrid], 1, MPI_DOUBLE, succ, SEND_TO_SUCC, new_comm);
        MPI_Recv(&y[n_ngrid+1], 1, MPI_DOUBLE, succ, RECV_FROM_SUCC, new_comm, &status);
        for(j=1; j <= n_ngrid; j++) {  // y[0] is irrelevant in this case
            dy[j-1] = (y[j+1] - y[j-1]) / (2 * dx);
        }
    } else if (rank == size-1) { // Last process has no successor
        y[n_ngrid+1] = fn(x[NGRID]); 
        MPI_Recv(&y[0], 1, MPI_DOUBLE, pred, RECV_FROM_PRED, new_comm, &status);
        MPI_Send(&y[1], 1, MPI_DOUBLE, pred, SEND_TO_PRED, new_comm);
        for(j=1; j<= n_ngrid; j++) {  // y[n_ngrid+1] is irrelevant in this case
            dy[j-1] = (y[j+1] - y[j-1]) / (2 * dx);
        }
    } else {
        if (rank %2 != 0) { // Odd ranked processes
            MPI_Send(&y[n_ngrid], 1, MPI_DOUBLE, succ, SEND_TO_SUCC, new_comm);
            MPI_Recv(&y[0], 1, MPI_DOUBLE, pred, RECV_FROM_PRED, new_comm, &status);

            MPI_Send(&y[1], 1, MPI_DOUBLE, pred, SEND_TO_PRED, new_comm);
            MPI_Recv(&y[n_ngrid+1], 1, MPI_DOUBLE, succ, RECV_FROM_SUCC, new_comm, &status);
        } else { // Even ranked processes
        
            MPI_Recv(&y[0], 1, MPI_DOUBLE, pred, RECV_FROM_PRED, new_comm, &status);
            MPI_Send(&y[n_ngrid], 1, MPI_DOUBLE, succ, SEND_TO_SUCC, new_comm);

            MPI_Recv(&y[n_ngrid+1], 1, MPI_DOUBLE, succ, RECV_FROM_SUCC, new_comm, &status);
            MPI_Send(&y[1], 1, MPI_DOUBLE, pred, SEND_TO_PRED, new_comm);
        }
        for(j=1; j<=n_ngrid; j++) {
            dy[j-1] = (y[j+1] - y[j-1]) / (2*dx);
        }
    }
    
    if(VERBOSE) printf("\n%d No deadlock occured\n", rank); 
    for(i=start_x, j=0; j < n_ngrid; i++, j++) {
        err[j] = fabs(dy[j] - dfn(x[i]));
        
        if(fabs(dy[j]) < EPSILON) {
            local_min_max[min_max_count++] = x[i];
            if(VERBOSE) printf("\nProcess %d found MIN/MAX at x= %f , dy= %f ",rank, x[i], dy[j]);
        }
    }
    
    for(j=min_max_count; j<DEGREE-1; j++){
            local_min_max[j] = INT_MAX;
    }

    int *rcounts=NULL, *displs=NULL;
    rcounts = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));

    for(i=0; i< size-1; i++) {
        rcounts[i] =  block_size;
        displs[i] = i * block_size;
    }
    rcounts[size-1] = block_size + (NGRID % size);
    displs[size-1] = (size-1)*block_size;

    MPI_Gatherv(err, n_ngrid , MPI_DOUBLE, glo_err, rcounts, displs, MPI_DOUBLE, ROOT, new_comm);
    MPI_Gather(local_min_max, DEGREE-1, MPI_DOUBLE, glo_min_max, DEGREE-1, MPI_DOUBLE, ROOT, new_comm);

    if (rank == ROOT) {
        err_sum = std_dev = 0;
        for (i=0; i< NGRID; i++) {
             if(DEBUG) printf("\ni=%d , x= %f, err= %e", i, x[i], glo_err[i]);
             err_sum += glo_err[i];
        }
        err_avg = err_sum / (double)NGRID;
        for (i=0; i< NGRID; i++) {
            std_dev += pow(glo_err[i] - err_avg, 2);
        }
        std_dev = sqrt(std_dev/(double)NGRID);
        if(VERBOSE) printf("\nErr_sum = %e, Err_avg = %e, std_dev= %e",err_sum, err_avg, std_dev);
    
        for(i=0; i< (DEGREE-1)*size; i++)
            if(glo_min_max[i] != INT_MAX) {
               if(VERBOSE)printf("\n(%f, %f)",glo_min_max[i], fn(glo_min_max[i]));
            }
    }
    if(y) free(y);
    if(dy) free(dy);
    if(err) free(err);
    if(displs) free(displs);
    if(glo_err) free(glo_err);
    if(rcounts) free(rcounts);
    if(glo_min_max) free(glo_min_max);
}

void blocking_and_MPI_reduce() {
}

void non_blocking_and_manual_reduce() {
}

void non_blocking_and_MPI_reduce() {
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Comm  new_comm;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    create_new_communicator(&rank, &size, &new_comm);
    blocking_and_manual_reduce(rank, size, new_comm);
   
    MPI_Finalize();
    return 0;
}
