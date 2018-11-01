#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>


double wtime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec / 1000000.0;
}

int main(int argc, char **argv)
{
	double start_time, end_time;
	int size = 2048;
	int filhos = 8;
	int rank, tag = 0;
	int chunk;
	int s = 0;
	float **A, **B, **G;
	MPI_Status status;
	MPI_Datatype vecDiag;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Allocates for A,B and G
	A = malloc(sizeof(float*)*size);
	B = malloc(sizeof(float*)*size);
	G = malloc(sizeof(float*)*size);
	for(int i = 0; i < size; i++){
		A[i] = malloc(sizeof(float)*size);
		B[i] = malloc(sizeof(float)*size);
		G[i] = malloc(sizeof(float)*size);
		for(int j = 0; j < size; j++){
			A[i][j] = B[i][j] = G[i][j] = 1;
		}
	}

	if (rank == 0)
	{
		// Impede Matriz menor do que 8
		if(size < filhos){
			printf("Coloque a dimensao matricial maior do que 8 (qtd filhos)\n");
			return 0;
		}

		// Logica da droga do chunk
		chunk = size/filhos;


		start_time = wtime();
		// Master distributes to its sons
		for(int i = 1; i <= filhos; i++) {
			MPI_Send(A, size*size, MPI_FLOAT, i, tag, MPI_COMM_WORLD);
			MPI_Send(B, size*size, MPI_FLOAT, i, tag, MPI_COMM_WORLD);
			MPI_Send(&chunk, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
		}



		float **T = malloc(sizeof(float*)*chunk);
		for(int i = 0; i < size; i++){
			T[i] = malloc(sizeof(float)*size);
		}

		for (int i = 1; i <= filhos; i++){
					MPI_Recv(&T, size*chunk, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
					for(int i = 0; i < i*chunk + chunk; i++){
						for(int j = 0; j < size; j++){
							G[i][j] = T[i][j];
						}
					}
		}
		/*
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				printf("|%d|", G[i][j]);
				fflush(stdout);
			}
			printf("\n");
			fflush(stdout);
		}*/
		end_time = wtime();
		printf("%f\n",end_time - start_time);

	}
	else
	{
		// Son receives A and B from master
		MPI_Recv(&A[rank-1][0], size*size, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
		MPI_Recv(&B[rank-1][0], size*size, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
		MPI_Recv(&chunk, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

		float **T = malloc(sizeof(float*)*chunk);
		for(int i = 0; i < size; i++){
			T[i] = malloc(sizeof(float)*size);
		}

		int newPos = rank*chunk + chunk;
		for(int i = rank*chunk; i < newPos; i++){
			for (int j = 0; j < size; j++) {
				for (int k = 0; k < size; k++){
					s += A[i][k] * B[k][j];
				}
				T[i][j] = s;
				s = 0;
			}
		}

		printf("\033[38;5;192mpassou %i\033[0m\n", rank);

		MPI_Send(T, size*chunk, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}