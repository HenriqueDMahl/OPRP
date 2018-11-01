#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

double wtime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec / 1000000.0;

}

// Works only for matricial dimensions which results in no resty situations

int main(int argc, char **argv)
{
	int rank, tag = 0, sons = 0;
	int s = 0;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &sons);

	// Matrix dimension
	int size = 2048;
	// Matrices
	  int **A = malloc(sizeof(int*)*size);
	    int **B = malloc(sizeof(int*)*size);
	    int **G = malloc(sizeof(int*)*size);

	    int * block1 = malloc(sizeof(int) * size * size);
	    int * block2 = malloc(sizeof(int) * size * size);
	    int * block3 = malloc(sizeof(int) * size * size);

	    for (int i = 0; i < size; i++)
	    {
	        A[i] = &(block1[i * size]);
	        B[i] = &(block2[i * size]);
	        G[i] = &(block3[i * size]);

	        for (int j = 0; j < size; j++)
	        {
	            A[i][j] = 0;
	            B[i][j] = 0;
	            G[i][j] = 0;
	        }
	    }

	//int A[size][size], B[size][size], G[size][size];// good for up to 128

	/*
	    for (int i; i < size; i++)
	    {
	        A[i] = malloc(sizeof(int) * size);
	        B[i] = malloc(sizeof(int) * size);
	        G[i] = malloc(sizeof(int) * size);
	    }
	*/

	int chunk = 0;
	double start_time, end_time;

	chunk = size / sons;

	if (rank == 0)
	{
		//Init A & B
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				A[i][j] = 1;
				B[i][j] = 1;
			}
		}


//		printf("chunk: %d\n", chunk);

		start_time = wtime();
		// Master distributes to its sons
		for (int i = 1; i < sons; i++)
		{
			MPI_Send(&A[0][0], size * size, MPI_INT, i, tag, MPI_COMM_WORLD);
			MPI_Send(&B[0][0], size * size, MPI_INT, i, tag, MPI_COMM_WORLD);
			//MPI_Send(&chunk, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
		}


		// Master processes
		for (int i = 0; i < chunk * rank + chunk; i++)
		{
			for (int j = 0; j < size; j++)
			{
				for (int k = 0; k < size; k++)
				{
					s += A[i][k] * B[k][j];
				}
				G[i][j] = s;
				s = 0;
			}
		}

		for (int i = 1; i < sons; i++)
		{
//			for (int c = 0; c < chunk; c++)
//			{
				MPI_Recv(G[i * chunk], size * chunk, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
				//printf("recebi\n");
//			}
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
		        }
		        */
		end_time = wtime();
		printf("%lf\n", end_time - start_time);

	}
	else
	{
		// Son receives A and B from master
		MPI_Recv(&A[0][0], size * size, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
		//printf("recebeu 1\n");
		MPI_Recv(&B[0][0], size * size, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
		//printf("recebeu 2\n");
		//MPI_Recv(&chunk, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

		s = 0;
		//int T[chunk][size];
		int **T = malloc(sizeof(int *) * chunk);
		int *data = malloc(sizeof(int) * chunk * size);
		for (int i = 0; i < chunk; i++) {
			T[i] = &data[i*size];
		}
		int aux = 0;
		for (int i = rank * chunk; i < rank * chunk + chunk; i++)
		{
			for (int j = 0; j < size; j++)
			{
				for (int k = 0; k < size; k++)
				{
					s += A[i][k] * B[k][i];
				}
				T[aux][j] = s;
				//      printf("%d ", T[i][j]);
				s = 0;
			}
			aux++;
			//  printf("\n");
		}
		MPI_Send(&T[0][0], chunk * size, MPI_INT, 0, tag, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}
