#include <stdio.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char **argv){
	int rank,size,tag=0;
	MPI_Status status;
    MPI_Datatype novo;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Type_vector(size,1,size, MPI_INT, &novo);//AQUI!!!!!!!!
    MPI_Type_commit(&novo);
	if(rank == 0) {
        int matrix[size];
        for(int i = 0; i < size; i++){
            MPI_Recv(&matrix[i*size], size, MPI_INT, i+1, tag, MPI_COMM_WORLD, &status);
        }
	} else {
        int vetor[size];
        for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++){
                if(i == j)
                    vetor[i] = (j * 10 + i) % (rank * 2 + 1);
            }
        }
        MPI_Send(&vetor, 1, novo, 0, tag, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}
