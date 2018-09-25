//mpicc exec.c -o exec

//mpirun -machinefile maquinas.in -np NUMEROPROCESSOS a.out

#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv){
    int rank, token = 0, size;
    int tag=0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0) {
        //for(int i=1;i<size;i++) {
            rank++;
            printf("0 enviando 20 para %d\n", rank);
            MPI_Send(&rank,1,MPI_INT,rank,tag,
            MPI_COMM_WORLD);
        //}
    }
    else{
        printf("%i esta esperando\n", rank);
        MPI_Recv(&token,1,MPI_INT,rank-1,tag,MPI_COMM_WORLD,&status);
        printf("Message received: %i\n", rank);
        if (rank == size-1){
            rank = 0;
        }else{
            rank = ++token;
        }
        MPI_Send(&rank, 1, MPI_INT, rank, tag, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
