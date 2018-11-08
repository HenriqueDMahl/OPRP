#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define TAM 2048
#define BLOCK 32
#define THREADS_PER_BLOCK 512

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
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &sons);
	//float *d_a, *d_b, *d_g;
	//int size = TAM * TAM * sizeof( float );

	// Matrices
	  	float **A = (float **)malloc(sizeof(float*)*TAM);
	    float **B = (float **)malloc(sizeof(float*)*TAM);
	    float **G = (float **)malloc(sizeof(float*)*TAM);

	    float * block1 = (float *)malloc(sizeof(float) * TAM * TAM);
	    float * block2 = (float *)malloc(sizeof(float) * TAM * TAM);
	    float * block3 = (float *)malloc(sizeof(float) * TAM * TAM);

	    for (int i = 0; i < TAM; i++)
	    {
	        A[i] = &(block1[i * TAM]);
	        B[i] = &(block2[i * TAM]);
	        G[i] = &(block3[i * TAM]);

	        for (int j = 0; j < TAM; j++)
	        {
	            A[i][j] = 1;
	            B[i][j] = 1;
	            G[i][j] = 0;
	        }
	    }
		/*cudaMalloc( (void **) &d_a, size );
		cudaMalloc( (void **) &d_b, size );
		cudaMalloc( (void **) &d_g, size );*/

	int chunk = 0;
	double start_time, end_time;

	chunk = TAM / sons;

	if (rank == 0)
	{
		//Init A & B
		/*for (int i = 0; i < TAM; i++)
		{
			for (int j = 0; j < TAM; j++)
			{
				A[i][j] = 1;
				B[i][j] = 1;
			}
		}*/


//		printf("chunk: %d\n", chunk);
		printf("PAI Executa\n");
		fflush(stdout);
		start_time = wtime();
		// Master distributes to its sons
		printf("ta ok?\n");
		fflush(stdout);
		for (int i = 1; i < sons; i++)
		{
			MPI_Send(&A[0][0], TAM * TAM, MPI_FLOAT, i, tag, MPI_COMM_WORLD);
			MPI_Send(&B[0][0], TAM * TAM, MPI_FLOAT, i, tag, MPI_COMM_WORLD);
			printf("Mandou para filho %d\n", i);
			fflush(stdout);
			//MPI_Send(&chunk, 1, MPI_FLOAT, i, tag, MPI_COMM_WORLD);
		}

		G[0] = call_me_maybe(A,B,chunk,rank);
		/*cudaMemcpy( d_a, A[0], size, cudaMemcpyHostToDevice );
		cudaMemcpy( d_b, B[0], size, cudaMemcpyHostToDevice );

		dim3 dimBlock (BLOCK,BLOCK);
		dim3 dimGrid ((TAM+BLOCK-1)/(float)BLOCK,(TAM+BLOCK-1)/(float)BLOCK);
		mult<<< dimBlock,dimGrid >>> ( d_a, d_b, d_g, 0, chunk );

		cudaMemcpy( G[0], d_g, chunk, cudaMemcpyDeviceToHost );*/
		/*
		// Master processes
		for (int i = 0; i < chunk * rank + chunk; i++)
		{
			for (int j = 0; j < TAM; j++)
			{
				for (int k = 0; k < TAM; k++)
				{
					s += A[i][k] * B[k][j];
				}
				G[i][j] = s;
				s = 0;
			}
		}*/

		for (int i = 1; i < sons; i++)
		{
//			for (int c = 0; c < chunk; c++)
//			{
				MPI_Recv(G[i * chunk], TAM * chunk, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
				printf("recebeu do filho %d\n", i);
				fflush(stdout);
				//printf("recebi\n");
//			}
		}
		end_time = wtime();
		printf("%lf\n", end_time - start_time);
		fflush(stdout);

		if(G[0][0] == 2048.00){
			printf("OK!\n");
			fflush(stdout);
		}else{
			printf("Feels bad! = %f\n", G[0][0]);
			fflush(stdout);
		}

	}
	else
	{
		// Son receives A and B from master
		MPI_Recv(&A[0][0], TAM * TAM, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
		//printf("recebeu 1\n");
		MPI_Recv(&B[0][0], TAM * TAM, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
		//printf("recebeu 2\n");
		//MPI_Recv(&chunk, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
		printf("Filho %d recebeu do pai\n", rank);
		fflush(stdout);

		//int T[chunk][size];
		float **T = (float **)malloc(sizeof(float *) * chunk);
		float *data = (float *)malloc(sizeof(float) * chunk * TAM);
		for (int i = 0; i < chunk; i++) {
			T[i] = &data[i*TAM];
		}

		T = call_me_maybe(A,B,chunk,rank);

		/*cudaMemcpy( d_a, A[0], size, cudaMemcpyHostToDevice );
		cudaMemcpy( d_b, B[0], size, cudaMemcpyHostToDevice );


		dim3 dimBlock (BLOCK,BLOCK);
	    dim3 dimGrid ((TAM+BLOCK-1)/(float)BLOCK,(TAM+BLOCK-1)/(float)BLOCK);
		mult<<< dimBlock,dimGrid >>> ( d_a, d_b, d_g, chunk*rank, chunk*rank + chunk );

		cudaMemcpy( T[0], d_g, size, cudaMemcpyDeviceToHost );*/
		/*
		for (int i = rank * chunk; i < rank * chunk + chunk; i++)
		{
			for (int j = 0; j < TAM; j++)
			{
				for (int k = 0; k < TAM; k++)
				{
					s += A[i][k] * B[k][i];
				}
				T[aux][j] = s;
				//      printf("%d ", T[i][j]);
				s = 0;
			}
			aux++;
			//  printf("\n");
		}*/
		MPI_Send(&T[0][0], chunk * TAM, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
		printf("Filho %d enviou para o pai\n", rank);
		fflush(stdout);
	}
	MPI_Finalize();
	return 0;
}
