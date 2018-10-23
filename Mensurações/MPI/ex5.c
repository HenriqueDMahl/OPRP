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
	int rank, tag = 0;
	int s = 0;
	int A[size][size];
	int B[size][size];
	int G[size][size];
	MPI_Status status;
	MPI_Datatype vecDiag;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);



	if (rank == 0)
	{
		//Init A & B
		for(int i=0; i < size; i++){
			for(int j=0; j < size; j++){
				A[i][j] = 1;
				B[i][j] = 1;
			}
		}
		start_time = wtime();
		// Master distributes to its sons
		for(int i=1; i < size; i++) {
			MPI_Send(A, size*size, MPI_INT, i, tag, MPI_COMM_WORLD);
			MPI_Send(B, size*size, MPI_INT, i, tag, MPI_COMM_WORLD);
		}

		// Master processes
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++){
				s += A[0][k] * B[k][j];
			}
			G[0][j] = s;
			s = 0;
		}

		for (int i = 1; i < size; i++)
			MPI_Recv(&G[i][0], size, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
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
		MPI_Recv(&A, size*size, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
		MPI_Recv(&B, size*size, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

		int T[size];
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++){
				s += A[rank][k] * B[k][j];
			}
			T[j] = s;
			s = 0;
		}

		MPI_Send(T, size, MPI_INT, 0, tag, MPI_COMM_WORLD);

	}
	MPI_Finalize();
	return 0;
}
