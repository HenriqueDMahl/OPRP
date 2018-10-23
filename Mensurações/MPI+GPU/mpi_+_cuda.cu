#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>

#define TAM 3
#define BLOCK 32
#define THREADS_PER_BLOCK 512

__global__ void soma(float *a, float *b, float *c){
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	// Identifies the index of the operation(vector)
	int id = TAM*x + y;
	// Performs the sum
	if(id < TAM*TAM)
		c[id] = a[id] + b[id];
}

__global__ void mult(float *a, float *b, float *c){
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if( x < TAM && y < TAM)
    {
        for(int i = 0; i < TAM; i++)
        {
            sum += a[x * TAM + y] * b[y * TAM + x];
        }
        c[y * TAM + x] = sum;
    }
}

/*
float* newMatrix(int flag){
	float *m = (float *) malloc(sizeof(float)*TAM*TAM);
	for(int i = 0; i < TAM*TAM; i++)
		if(!flag)
			m[i] = 0.0;
		else
			m[i] = 1.0;
	return m;

}

void printMatrix(float *m){
	int cont = 0;
	for(int i = 0; i< TAM*TAM; i++){
		if(cont == TAM){
			cont = 0;
			printf("\n");
		}
		if(cont != TAM){
			printf("%.2f ", m[i]);
			cont++;
		}
	}
	printf("\n");
}*/

double wtime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec / 1000000.0;
}

int main(int argc, char **argv)
{
	double start_time, end_time;
	int size = 4;
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
        /*
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++){
				s += A[0][k] * B[k][j];
			}
			G[0][j] = s;
			s = 0;
		}*/

		for (int i = 1; i < size; i++)
			MPI_Recv(&G[i], size, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
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

        float *d_a, *d_b, *d_c;
        cudaMalloc( (void **) &d_a, size );
    	cudaMalloc( (void **) &d_b, size );
    	cudaMalloc( (void **) &d_c, size );

    	cudaEvent_t start, stop;

    	cudaEventCreate(&start);

    	cudaEventCreate(&stop);

    	cudaEventRecord(start);

        cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
    	cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

    	//soma<<< (N + ceil(THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c );
    	dim3 dimBlock (BLOCK,BLOCK);
        dim3 dimGrid ((TAM+BLOCK-1)/(float)BLOCK,(TAM+BLOCK-1)/(float)BLOCK);
    	mult<<< dimBlock,dimGrid >>> ( d_a, d_b, d_c );

    	cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );

    	cudaEventRecord(stop);

    	cudaEventSynchronize(stop);

    	float milliseconds = 0;

    	cudaEventElapsedTime(&milliseconds, start, stop);

    	printf("%lf\n", milliseconds);

        free(a);
    	free(b);
    	free(c);
    	cudaFree( d_a );
    	cudaFree( d_b );
    	cudaFree( d_c );

		/*int T[size];
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++){
				s += A[rank][k] * B[k][j];
			}
			T[j] = s;
			s = 0;
		}*/


		MPI_Send(T, size, MPI_INT, 0, tag, MPI_COMM_WORLD);

	}
	MPI_Finalize();
	return 0;
}
