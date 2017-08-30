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

int main(int argc, char *argv[]) {
    // loop index, process identity, number of processes.
    int i, j, rank, size, succ, pred;

    MPI_Comm  new_comm;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    create_new_communicator(&rank, &size, &new_comm);
    
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
    double *y, *dy, *err;

    double *glo_err= NULL;
    printf("\nRank= %d, size= %d, start_x= %d, end_x= %d, n_ngrid=%d, ", rank, size, start_x, end_x, n_ngrid );
   
    // Create the grid points on X axis
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
   
    if (y == NULL || dy == NULL || err == NULL) {
        printf("\n Memory Allocation Failed");
        exit(-1);
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
        glo_err = (double *)malloc(NGRID  * sizeof(double));
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
    printf("\n%d No deadlock occured\n", rank); 
    
    for(i=start_x, j=0; j < n_ngrid; i++, j++) {
        err[j] = fabs(dy[j] - dfn(x[i]));
        
        if(fabs(dy[j]) < EPSILON) {
            local_min_max[min_max_count++] = x[i];
           printf("\nProcess %d found MIN/MAX at x= %f , dy= %f ",rank, x[i], dy[j]);
        }
        //printf("%d %d %d %8f, %8f, %8f, %8f, %8f, %e\n", rank, i, j, x[i], fn(x[i]), y[j], dfn(x[i]), dy[j], err[j]);
        //printf("%d %d %d %8f %10e\n",rank, i, j, x[i], err[j]);
    }
    for(j=min_max_count; j<DEGREE-1; j++){
            local_min_max[j] = INT_MAX;
    }
    //TODO: Send this error vector to ROOT
    int st=0, *rcounts, *displs;
    rcounts = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));

    for(i=0; i< size-1; i++) {
        rcounts[i] =  block_size;
        displs[i] = i * block_size;
    }
    rcounts[size-1] = block_size + (NGRID % size);
    displs[size-1] = (size-1)*block_size;

    st = MPI_Gatherv(err, n_ngrid , MPI_DOUBLE, glo_err, rcounts, displs, MPI_DOUBLE, ROOT, new_comm);
    double err_sum, std_dev, err_avg;
    if (rank == ROOT)
    {   err_sum = std_dev = 0;
        for (i=0; i< NGRID; i++) {
             //printf("\ni=%d , x= %f, err= %e", i, x[i], glo_err[i]);
             err_sum += glo_err[i];
        }
        err_avg = err_sum / (double)NGRID;
        for (i=0; i< NGRID; i++) {
            std_dev += pow(glo_err[i] - err_avg, 2);
        }
        std_dev = sqrt(std_dev/(double)NGRID);
        printf("\nErr_sum = %e, Err_avg = %e, std_dev= %e",err_sum, err_avg, std_dev);
    }
    double * glo_min_max=NULL;
    if(rank == ROOT) {
        glo_min_max = (double *)malloc(((DEGREE-1)*size) * sizeof(double));
    }
    st = MPI_Gather(local_min_max, DEGREE-1, MPI_DOUBLE, glo_min_max, DEGREE-1, MPI_DOUBLE, ROOT, new_comm);
    if (rank == ROOT) {
        for(i=0; i< (DEGREE-1)*size; i++)
            if(glo_min_max[i] != INT_MAX) {
               printf("\n(%f, %f)",glo_min_max[i], fn(glo_min_max[i]));
            }
    }
    if (st != MPI_SUCCESS) {
        printf("\nError in MPI_GATHER %d\n",st);
        exit(0);
    }
    
    MPI_Finalize();
    
    if(y) free(y);
    if(dy) free(dy);
    if(err) free(err);
    if(glo_err) free(glo_err);
    return 0;
}
