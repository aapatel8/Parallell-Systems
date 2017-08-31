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
#define VERBOSE 0

double fn(double x) {
    return pow(x,3) - pow(x,2) - x + 1;
    //return pow(x,2);
}

double dfn(double x) {
    return 3 * pow(x,2) - 2 * x - 1;
    //return 2 * x;
}

typedef struct {
    double dy, xi;
    } dy_x;

void min_max(dy_x *inbuf, dy_x *outbuf, int *len, MPI_Datatype type) {
    int i, j=0;
    dy_x *temp = (dy_x *) malloc((*len) *sizeof(dy_x));
    for(i=0; i <*len; i++) {
        if (fabs(inbuf[i].dy) < EPSILON) {
            temp[j].dy = inbuf[i].dy;
            temp[j++].xi = inbuf[i].xi;
        }
    }
    for(i = 0; i < *len; i++) {
        if (fabs(outbuf[i].dy) < EPSILON) {
            temp[j].dy = outbuf[i].dy;
            temp[j++].xi = outbuf[i].xi;
        }
    }
    for(i=0; i < j; i++) {
        outbuf[i].dy = temp[i].dy;
        outbuf[i].xi = temp[i].xi;
    }
    for(; i< *len; i++) {
        outbuf[i].dy = INT_MAX;
        outbuf[i].xi = INT_MAX;
    }
    if (temp) free(temp);
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
    if(VERBOSE) printf("\nRank= %d, size= %d, start_x= %d, end_x= %d, "
             "n_ngrid=%d, succ= %d, pred= %d",rank, size, 
              *start_x, *end_x, *n_ngrid, *succ, *pred);

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

void print_error_data(char *filename, int np, double avgerr, double stdd, 
                      double *x, double *err, double *min_max_array, 
                      int min_max_len)
{
  int   i;
  FILE *fp = fopen(filename, "w");

  fprintf(fp, "%e\n%e\n", avgerr, stdd);
  if(VERBOSE) printf("\nErr_avg = %e, std_dev= %e", avgerr, stdd);

  for(i = 0; i<min_max_len; i++)
  {
	if (min_max_array[i] != INT_MAX) {
		fprintf(fp, "(%f, %f)\n", min_max_array[i], fn(min_max_array[i]));
    }
  }
  
  for(i = 0; i < np; i++)
  {
	fprintf(fp, "%f %e \n", x[i], err[i]);
  }
  fclose(fp);
}

void print_error_data_dydx(char *filename, int np, double avgerr, double stdd,
                          double *x, double *err, dy_x * dydx_arr, int len) {
    int i;
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%e\n%e\n", avgerr, stdd);
    if (VERBOSE) printf("\nErr_avg = %e, std_dev= %e", avgerr, stdd);

    for(i=0; i<len; i++)
    {
        if(dydx_arr[i].dy != INT_MAX) {
            fprintf(fp, "(%f, %f)\n", dydx_arr[i].xi, fn(dydx_arr[i].xi));
        }
    }

    for(i=0; i< np; i++)
    {
        fprintf(fp, "%f %e \n",x[i], err[i]);
    }
    fclose(fp);
}
/* The calling function should call free on y, dy and err buffers */ 
void calculate_y_axis_values(double *x, double *y, int start_x, int end_x) {
    if (y == NULL) {
        printf("\n Memory Allocation Failed");
        exit(-1);
    }
    int i, j;
    for (i=start_x, j=1; i< end_x; i++, j++) {
        y[j] = fn(x[i]);
    }
}

void non_blocking_transfer_boundary_values(int rank, int size, int n_ngrid, 
                        int pred, int succ, double *x, double *y, double *dy, 
                        MPI_Comm new_comm) {
    int i, j;
    double dx = x[2]-x[1];
    
    MPI_Request *reqs = NULL; 
    MPI_Status *stats = NULL;
    
    if (rank == ROOT) {
        y[0] = fn(x[0]-dx);
        reqs = (MPI_Request *) malloc(2*sizeof(MPI_Request));
        stats = (MPI_Status *) malloc(2*sizeof(MPI_Status));
        MPI_Irecv(&y[n_ngrid+1], 1, MPI_DOUBLE, succ, RECV_FROM_SUCC, new_comm, &reqs[0]);
        MPI_Isend(&y[n_ngrid], 1, MPI_DOUBLE, succ, SEND_TO_SUCC, new_comm, &reqs[1]);
        MPI_Waitall(2, reqs, stats);
    } else if (rank == size-1) {
        y[n_ngrid+1] = fn(x[NGRID]);
        reqs = (MPI_Request *) malloc(2*sizeof(MPI_Request));
        stats = (MPI_Status *) malloc(2*sizeof(MPI_Status));
        MPI_Irecv(&y[0], 1, MPI_DOUBLE, pred, RECV_FROM_PRED, new_comm, &reqs[0]);
        MPI_Isend(&y[1], 1, MPI_DOUBLE, pred, SEND_TO_PRED, new_comm, &reqs[1]);
        MPI_Waitall(2, reqs, stats);
    } else {
        reqs = (MPI_Request *) malloc(4*sizeof(MPI_Request));
        stats = (MPI_Status *) malloc(4*sizeof(MPI_Status));
        MPI_Irecv(&y[0], 1, MPI_DOUBLE, pred, RECV_FROM_PRED, new_comm, &reqs[0]);
        MPI_Irecv(&y[n_ngrid+1], 1, MPI_DOUBLE, succ, RECV_FROM_SUCC, new_comm, &reqs[1]);

        MPI_Isend(&y[n_ngrid], 1, MPI_DOUBLE, succ, SEND_TO_SUCC, new_comm, &reqs[2]); 
        MPI_Isend(&y[1], 1, MPI_DOUBLE, pred, SEND_TO_PRED, new_comm, &reqs[3]);
        MPI_Waitall(4, reqs, stats);
        }
    for(j=1; j<=n_ngrid; j++) {
        dy[j-1] = (y[j+1] - y[j-1]) / (2*dx);
    }
    if (reqs) free(reqs);
    if (stats) free(stats);
}

void blocking_transfer_boundary_values(int rank, int size, int n_ngrid, 
                        int pred, int succ, double *x, double *y, double *dy, 
                        MPI_Comm new_comm) {
    int i, j;
    double dx = x[2] - x[1];
    MPI_Status status;
    /*
    send y[1] to predecessor
    send y[n_ngrid] to successor
    Receive in y[0] from predeessor
    Receive in y[n_ngrid+1] from successor.
    */

    if(rank == ROOT) { // Root has no predecessor
        // ROOT sends to rank 1, with tag SEND_TO_SUCC and receives with tag RECV_FROM_SUCC 
        y[0] = fn(x[0]-dx);
        MPI_Send(&y[n_ngrid], 1, MPI_DOUBLE, succ, SEND_TO_SUCC, new_comm);
        MPI_Recv(&y[n_ngrid+1], 1, MPI_DOUBLE, succ, RECV_FROM_SUCC, new_comm, &status);
    } else if (rank == size-1) { // Last process has no successor
        y[n_ngrid+1] = fn(x[NGRID]); 
        MPI_Recv(&y[0], 1, MPI_DOUBLE, pred, RECV_FROM_PRED, new_comm, &status);
        MPI_Send(&y[1], 1, MPI_DOUBLE, pred, SEND_TO_PRED, new_comm);
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
    }
    for(j=1; j<=n_ngrid; j++) {
        dy[j-1] = (y[j+1] - y[j-1]) / (2*dx);
    }
}


void calculate_finite_differencing_error(int start_x, int n_ngrid, double *err, double *dy, double *x, double *local_min_max) {
    int min_max_count = 0, i, j;
    for(i=start_x, j=0; j < n_ngrid; i++, j++) {
        err[j] = fabs(dy[j] - dfn(x[i]));
        
        if(fabs(dy[j]) < EPSILON) {
            local_min_max[min_max_count++] = x[i];
            if(VERBOSE) printf("\nProcess found MIN/MAX at x= %f , dy= %f ", x[i], dy[j]);
        }
    }
    for(j=min_max_count; j<DEGREE-1; j++){
        local_min_max[j] = INT_MAX;
    }
}

void gather_err_vector(int rank, int size, int n_ngrid, double *err, double *glo_err, MPI_Comm new_comm) {
    int i, block_size = NGRID / size;  // Number of grid points in a block
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
    
    if(displs) free(displs);
    if(rcounts) free(rcounts);
}

void calculate_std_deviation(double *std_dev, double *err_avg, double *glo_err) {
    int i;
    double err_sum;
    err_sum = *std_dev = 0;
    for (i=0; i< NGRID; i++) {
        err_sum += glo_err[i];
    }
    *err_avg = err_sum / (double)NGRID;
    for (i=0; i< NGRID; i++) {
        *std_dev += pow(glo_err[i] - *err_avg, 2);
    }
    *std_dev = sqrt(*std_dev/(double)NGRID);
}

void non_blocking_and_manual_reduce(int rank, int size, MPI_Comm new_comm) {
    int i, j, succ, pred;
    double x[NGRID +2], dx;
    int n_ngrid;  // Number of grid points allocated to the process.
    int start_x, end_x; // start and end index for x values for this process.

    double  local_min_max[DEGREE-1];
    double *y=NULL, *dy=NULL, *err=NULL, *glo_err= NULL;
    double std_dev, err_avg, *glo_min_max=NULL;
    
    get_x_axis_limits(rank, size, &n_ngrid, &start_x, &end_x, &succ, &pred);
    dx = create_x_axis_grid_points(x);    

    y = (double *) malloc((n_ngrid +2) * sizeof(double));
    dy = (double *) malloc(n_ngrid * sizeof(double));
    err = (double *) malloc(n_ngrid * sizeof(double));
   
    calculate_y_axis_values(x, y, start_x, end_x);
    non_blocking_transfer_boundary_values(rank, size, n_ngrid, pred, succ, x, y, dy, new_comm);
    if (VERBOSE) printf("\n Boundary values transfer success\n");
    calculate_finite_differencing_error(start_x, n_ngrid, err, dy, x, local_min_max);
    if (rank == ROOT){
        glo_min_max = (double *)malloc(((DEGREE-1)*size) * sizeof(double));
        glo_err = (double *)malloc(NGRID  * sizeof(double));
    }
    MPI_Gather(local_min_max, DEGREE-1, MPI_DOUBLE, glo_min_max, DEGREE-1, MPI_DOUBLE, ROOT, new_comm);
    gather_err_vector(rank, size, n_ngrid, err, glo_err, new_comm);
    if (rank == ROOT) {
        calculate_std_deviation(&std_dev, &err_avg, glo_err);
        print_error_data("err.dat", NGRID, err_avg, std_dev, &x[1], glo_err, glo_min_max, (DEGREE-1)*size);
    }
    if(y) free(y);
    if(dy) free(dy);
    if(err) free(err);
    if(glo_err) free(glo_err);
    if(glo_min_max) free(glo_min_max);
}

void non_blocking_and_MPI_reduce(int rank, int size, MPI_Comm new_comm) {
    int i, j, succ, pred;
    double x[NGRID +2], dx;
    int n_ngrid;  // Number of grid points allocated to the process.
    int start_x, end_x; // start and end index for x values for this process.

    double  local_min_max[DEGREE-1];
    double *y=NULL, *dy=NULL, *err=NULL, *glo_err= NULL;
    
    double std_dev, err_avg; 
    dy_x * dyxi = NULL, *glo_dyxi=NULL;

    MPI_Op my_op;
    
    get_x_axis_limits(rank, size, &n_ngrid, &start_x, &end_x, &succ, &pred);
    dx = create_x_axis_grid_points(x);    

    int xlen = NGRID/size + NGRID % size;
    y = (double *) malloc((n_ngrid +2) * sizeof(double));
    dy = (double *) malloc(xlen * sizeof(double));
    err = (double *) malloc(n_ngrid * sizeof(double));
    dyxi = (dy_x*)malloc(xlen * sizeof(dy_x)); 

    calculate_y_axis_values(x, y, start_x, end_x);
    for(i=n_ngrid; i < xlen; i++)
        dy[i] = INT_MAX;
    
    non_blocking_transfer_boundary_values(rank, size, n_ngrid, pred, succ, x, y, dy, new_comm);
    for(i=start_x, j=0; j< xlen; i++, j++) {
        dyxi[j].dy = dy[j];
        dyxi[j].xi = x[i];
    }

    if (VERBOSE) printf("\n Boundary values transfer success\n");
    calculate_finite_differencing_error(start_x, n_ngrid, err, dy, x, local_min_max);
    if (rank == ROOT) {
        glo_dyxi = (dy_x*)malloc(xlen * sizeof(dy_x));
        glo_err = (double *) malloc(NGRID * sizeof(double));
    }
    MPI_Datatype dydx_type, oldtype[1];
    int blockcount[1];
    MPI_Aint offset[1];

    offset[0] = 0;
    oldtype[0] = MPI_DOUBLE;
    blockcount[0] = 2;
    MPI_Type_struct(1, blockcount, offset, oldtype, &dydx_type);
    MPI_Type_commit(&dydx_type);

    MPI_Op_create((MPI_User_function*)min_max, 0, &my_op);
    if(DEBUG) printf("\n%d, MPI OPeration created",rank);
    MPI_Reduce(dyxi, glo_dyxi, xlen, dydx_type, my_op, ROOT, new_comm);
    if(DEBUG) printf("\n%d MPI_reduce Success", rank);
    gather_err_vector(rank, size, n_ngrid, err, glo_err, new_comm);
    if(DEBUG) printf("\n%d Err vector gathered", rank);
    if (rank == ROOT) {
        calculate_std_deviation(&std_dev, &err_avg, glo_err);
        if(DEBUG) printf("\nCalculated Standard deviation");
        //print_error_data("err2.dat", NGRID, err_avg, std_dev, &x[1], glo_err, glo_min_max, xlen);
        print_error_data_dydx("err3.dat", NGRID, err_avg, std_dev, &x[1], glo_err, glo_dyxi, xlen);
        if(DEBUG) printf("\n Printed err value to file");
   }
    if(y) free(y);
    if(dy) free(dy);
    if(err) free(err);
    if(glo_err) free(glo_err);
    if(dyxi) free(dyxi);
    if(glo_dyxi) free(glo_dyxi);
    MPI_Op_free(&my_op);
}

void blocking_and_MPI_reduce(int rank, int size, MPI_Comm new_comm) {
    int i, j, succ, pred;
    double x[NGRID +2], dx;
    int n_ngrid;  // Number of grid points allocated to the process.
    int start_x, end_x; // start and end index for x values for this process.

    double  local_min_max[DEGREE-1];
    double *y=NULL, *dy=NULL, *err=NULL, *glo_err= NULL;
    double std_dev, err_avg; 
    dy_x * dyxi = NULL, *glo_dyxi=NULL;

    MPI_Op my_op;
    
    get_x_axis_limits(rank, size, &n_ngrid, &start_x, &end_x, &succ, &pred);
    dx = create_x_axis_grid_points(x);    
    
    int xlen = NGRID/size + NGRID % size;
    y = (double *) malloc((n_ngrid +2) * sizeof(double));
    dy = (double *) malloc(xlen * sizeof(double));
    err = (double *) malloc(n_ngrid * sizeof(double));
    dyxi = (dy_x*)malloc(xlen * sizeof(dy_x)); 

    calculate_y_axis_values(x, y, start_x, end_x);
    for(i=n_ngrid; i < xlen; i++)
        dy[i] = INT_MAX;
    blocking_transfer_boundary_values(rank, size, n_ngrid, pred, succ, x, y, dy, new_comm);
    
    // Fill in dy and xi values in dyxi
    for(i=start_x, j=0; j< xlen; i++, j++) {
        dyxi[j].dy = dy[j];
        dyxi[j].xi = x[i];
    }
    if(VERBOSE) printf("\n%d No deadlock occured\n", rank); 
    calculate_finite_differencing_error(start_x, n_ngrid, err, dy, x, local_min_max);
    if (rank == ROOT) {
        glo_dyxi = (dy_x*)malloc(xlen * sizeof(dy_x));
        glo_err = (double *) malloc(NGRID * sizeof(double));
    }
    MPI_Datatype dydx_type, oldtype[1];
    int blockcount[1];
    MPI_Aint offset[1];

    offset[0] = 0;
    oldtype[0] = MPI_DOUBLE;
    blockcount[0] = 2;

    MPI_Type_struct(1, blockcount, offset, oldtype, &dydx_type);
    MPI_Type_commit(&dydx_type);

    MPI_Op_create((MPI_User_function*)min_max, 0, &my_op);
    if(DEBUG) printf("\n%d, MPI OPeration created",rank);
    MPI_Reduce(dyxi, glo_dyxi, xlen, dydx_type, my_op, ROOT, new_comm);
    if(DEBUG) printf("\n%d MPI_reduce Success", rank);
    gather_err_vector(rank, size, n_ngrid, err, glo_err, new_comm);
    if(DEBUG) printf("\n%d Err vector gathered", rank);
    if (rank == ROOT) {
        calculate_std_deviation(&std_dev, &err_avg, glo_err);
        if(DEBUG) printf("\nCalculated Standard deviation");
        //print_error_data("err2.dat", NGRID, err_avg, std_dev, &x[1], glo_err, glo_min_max, xlen);
        print_error_data_dydx("err2.dat", NGRID, err_avg, std_dev, &x[1], glo_err, glo_dyxi, xlen);
        if(DEBUG) printf("\n Printed err value to file");
   }
    if(y) free(y);
    if(dy) free(dy);
    if(err) free(err);
    if(glo_err) free(glo_err);
    if(dyxi) free(dyxi);
    if(glo_dyxi) free(glo_dyxi);
    MPI_Op_free(&my_op);
}

void blocking_and_manual_reduce(int rank, int size, MPI_Comm new_comm) {
    int i, j, succ, pred;
    double x[NGRID +2], dx;
    int n_ngrid;  // Number of grid points allocated to the process.
    int start_x, end_x; // start and end index for x values for this process.

    double  local_min_max[DEGREE-1];
    double *y=NULL, *dy=NULL, *err=NULL, *glo_err= NULL;
    double std_dev, err_avg, *glo_min_max=NULL;
    
    get_x_axis_limits(rank, size, &n_ngrid, &start_x, &end_x, &succ, &pred);
    dx = create_x_axis_grid_points(x);    

    y = (double *) malloc((n_ngrid +2) * sizeof(double));
    dy = (double *) malloc(n_ngrid * sizeof(double));
    err = (double *) malloc(n_ngrid * sizeof(double));
   
    calculate_y_axis_values(x, y, start_x, end_x);
    blocking_transfer_boundary_values(rank, size, n_ngrid, pred, succ, x, y, dy, new_comm);
    if(VERBOSE) printf("\n%d No deadlock occured\n", rank); 
    calculate_finite_differencing_error(start_x, n_ngrid, err, dy, x, local_min_max);
    
    if (rank == ROOT){
        glo_min_max = (double *)malloc(((DEGREE-1)*size) * sizeof(double));
        glo_err = (double *)malloc(NGRID  * sizeof(double));
    }
    
    MPI_Gather(local_min_max, DEGREE-1, MPI_DOUBLE, glo_min_max, DEGREE-1, MPI_DOUBLE, ROOT, new_comm);

    gather_err_vector(rank, size, n_ngrid, err, glo_err, new_comm);

    if (rank == ROOT) {
        calculate_std_deviation(&std_dev, &err_avg, glo_err);
        print_error_data("err1.dat", NGRID, err_avg, std_dev, &x[1], glo_err, glo_min_max, (DEGREE-1)*size);
    }
    if(y) free(y);
    if(dy) free(dy);
    if(err) free(err);
    if(glo_err) free(glo_err);
    if(glo_min_max) free(glo_min_max);
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Comm  new_comm;
    MPI_Status status;
    double time2=0, time1=0;

    MPI_Init(&argc, &argv);
    create_new_communicator(&rank, &size, &new_comm);
    if (rank ==ROOT) {
        printf("\n NGRID= %d, EPSILON= %f, Size = %d",NGRID, EPSILON, size);
    }
    MPI_Barrier(new_comm);
    time1 = MPI_Wtime();
    blocking_and_manual_reduce(rank, size, new_comm);
    time2 = MPI_Wtime();
    printf("\nBlocking with Manual Reduce. rank= %d duration= %e",rank, time2-time1);
    MPI_Barrier(new_comm);
    time1 = MPI_Wtime();
    non_blocking_and_manual_reduce(rank, size, new_comm);
    time2 = MPI_Wtime();
    printf("\nNon-blocking with Manual Reduce, rank= %d, duration = %e", rank, time2-time1);
    MPI_Barrier(new_comm);
    time1 = MPI_Wtime();
    blocking_and_MPI_reduce(rank, size, new_comm);
    time2 = MPI_Wtime();
    printf("\nBLocking with MPI Reduce, rank= %d, duration= %e",rank, time2-time1);
    MPI_Barrier(new_comm);
    time1 = MPI_Wtime();
    non_blocking_and_MPI_reduce(rank, size, new_comm);
    time2 = MPI_Wtime();
    printf("\nNon - BLocking with MPI Reduce, rank= %d, duration= %e",rank, time2-time1);
    
    MPI_Finalize();
    return 0;
}
