#include <stdlib.h>
#include "mpi.h"
#include <stdio.h>
#define NGRIDS 8

void  min_max(double *inbuf, double *outbuf, int *len, MPI_Datatype type){
    int i=0;
    for(i=0; i< *len; i++) {
        if (inbuf[i] > 10)
            outbuf[i] = inbuf[i];
    }
}

int main(int argc, char *argv[])  {
    int i, j, rank, size ;

    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int xlen = 8;
    double *dy = (double *)malloc(xlen * sizeof(double));

    double *coll = NULL;
    if (rank == 0)
        coll = (double *)malloc(xlen * sizeof(double));

    for(i=0; i<xlen; i++) {
        //coll[i] = 1; 
        
        dy[i] = 2;
    
    }
    /*if(rank == 2)
        dy[4]= 55;
    if (rank == 1)
        dy[4] = 23;
    */
    dy[rank] = 99;
    MPI_Op op;
    MPI_Op_create((MPI_User_function*)min_max, 0, &op);
    MPI_Reduce(dy, coll, xlen, MPI_DOUBLE, op, 0, MPI_COMM_WORLD);
    
    if(rank== 0) {
        for(i=0; i< xlen; i++){
            printf("\ncoll[%d]= %f, dy[%d]= %f",i, coll[i], i, dy[i]);
            
            }
    }

    MPI_Finalize();
    return 0;
}
