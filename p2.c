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

double fn(double x) {
    return pow(x,3) - pow(x,2) - x + 1;
}

double dfn(double x) {
    return 3 * pow(x,2) - 2 * x - 1;
}

int main(int argc, char *argv[]) {
    // loop index, process identity, number of processes.
    int i, rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // domain array and step size
    // TODO: NGRID is for the whole range or for each decomposed grid?
    double *x, dx;
    int n_ngrid;

    if (rank == size-1) { // Last process is responsible for the extra grid points
        n_ngrid = (NGRID / size) + (NGRID % size);
    } else {
        n_ngrid = (NGRID / size);
    }
    
    x = (double *) malloc(sizeof(double) * (2+n_ngrid));
    // The x axis width of decomposed grid.
    float step_size = (XF - XI) /(float)NGRID; // TODO: Waiting for answer on forum
    float block_size = step_size * NGRID / size;
    float *y, *dy;

    //TODO: Create a different communicator

    printf("\n I am rank= %d amongst %d processes\n", rank, size);
    
    for (i=1; i<= n_ngrid; i++) {
        x[i] = XI + (rank * block_size) + (step_size * i);
    }

    dx = x[2] - x[1];
    x[0] = x[1] - dx;
    x[n_ngrid+1] = x[n_ngrid] + dx;

    // TODO: The size of allocated array might also change 
    y = (double *) malloc((n_ngrid +2) * sizeof(double));
    dy = (double *) malloc((n_ngrid +2) * sizeof(double));
    
    y[0] = fn(x[0]);  // TODO: Send this to predecessor.
    y[n_ngrid+1] = fn(x[n_ngrid+1]);  // Send this to successor.

    float y_pred , y_succ; // Values received from predecessor and successor.

    for (i=1; i<= n_ngrid; i++) {
        y[i] = fn(x[i]);
    }

    
    MPI_Finalize();
    return 0;
}
