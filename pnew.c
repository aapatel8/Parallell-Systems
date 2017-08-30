#include <stdlib.h>
#include "mpi.h"
#include <stdio.h>
#define NPROCS 8

int main(int argc, char *argv[])  {
    int i, j, rank, size, succ, pred;
    int new_rank, new_size;

    MPI_Group orig_group, new_group;
    MPI_Comm  new_comm;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *new_ranks;
    new_ranks = (int*)malloc(size * sizeof(int));
    for (i=0; i< size; i++)
        new_ranks[i] = i;

    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
    MPI_Group_incl(orig_group, size, new_ranks, &new_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_size);

    printf("\n rank = %d, size = %d, new rank= %d, size= %d",rank, size, new_rank, new_size);

    succ = (rank+1) % size;
    pred = (rank-1 + size) % size;

    MPI_Finalize();
    return 0;
}
