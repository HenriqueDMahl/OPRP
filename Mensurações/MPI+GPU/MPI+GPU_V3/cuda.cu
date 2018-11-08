#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TAM 2048
#define BLOCK 32
#define THREADS_PER_BLOCK 512

__global__ void __mult__(float *a, float *b, float *c, int *inicio, int *chunk, float *d_sum){
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if( x < (*inicio) * (*chunk) + (*chunk) && y < (*inicio) * (*chunk) + (*chunk))
    {
        for(int i = (*inicio) * (*chunk); i < (*inicio) * (*chunk) + (*chunk); i++)
        {
            sum += a[x * (*inicio) * (*chunk) + y] * b[y * (*inicio) * (*chunk) + x];
        }
        c[y * (*chunk) + x] = sum;
				d_sum = &sum;
    }
}

extern "C" float ** call_me_maybe(float **A,float**B,int chunk, int rank){
	float *d_a, *d_b, *d_g;
	int *d_rank, *d_chunk;
	float *d_sum;
	int size = TAM * TAM * sizeof( float );
	int sum;
	float **T = (float **)malloc(sizeof(float *) * chunk);
	float *data = (float *)malloc(sizeof(float) * chunk * TAM);
	for (int i = 0; i < chunk; i++) {
		T[i] = &data[i*TAM];
	}

	printf("A[0][0] = %f do %d\n", A[0][0],rank);
	fflush(stdout);
	printf("B[0][0] = %f do %d\n", B[0][0],rank);
	fflush(stdout);

	printf("Alocando %d\n", rank);
	fflush(stdout);
	cudaMalloc( (void **) &d_a, size );
	cudaMalloc( (void **) &d_b, size );
	cudaMalloc( (void **) &d_g, size );
	cudaMalloc( (void **) &d_sum, sizeof(float) );
	cudaMalloc( (void **) &d_rank, sizeof( int ) );
	cudaMalloc( (void **) &d_chunk, sizeof( int ) );
	printf("Copiando para GPU %d\n", rank);
	fflush(stdout);
	cudaMemcpy( d_a, A[0], size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, B[0], size, cudaMemcpyHostToDevice );
	printf("Copiou Matrizes para GPU %d\n", rank);
	cudaMemcpy( d_rank, &rank, sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( d_chunk, &chunk, sizeof( int ), cudaMemcpyHostToDevice );
	printf("Mandando para funcao %d\n", rank);
	fflush(stdout);
	dim3 dimBlock (BLOCK,BLOCK);
	dim3 dimGrid ((TAM+BLOCK-1)/(float)BLOCK,(TAM+BLOCK-1)/(float)BLOCK);
	__mult__<<< dimBlock,dimGrid >>> ( d_a, d_b, d_g, d_rank, d_chunk, d_sum );
	printf("Funcao executou %d\n", rank);
	fflush(stdout);

	printf("Copiando para CPU %d\n", rank);
	fflush(stdout);
	cudaMemcpy( &sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost );
	printf("Retorno SUM = %f do %d\n",T[0][0],rank);
	fflush(stdout);
	cudaMemcpy( T[0], d_g, size, cudaMemcpyDeviceToHost );
	printf("Retorno e %f do %d\n",T[0][0],rank);
	fflush(stdout);
	printf("Retornando %d\n",rank);
	fflush(stdout);

	return T;
}
