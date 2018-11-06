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

__global__ void __mult__(float *a, float *b, float *c, int *rank, int *chunk){
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if( x < TAM && y < TAM)
    {
        for(int i = (*rank) * (*chunk); i < (*rank) * (*chunk) + (*chunk); i++)
        {
            sum += a[x * (*chunk) + y] * b[y * (*chunk) + x];
        }
        c[y * (*chunk) + x] = sum;
    }
}

extern "C" float ** call_me_maybe(float **A,float**B, int chunk, int rank){
	float *d_a, *d_b, *d_g;
	int *d_chunk, *d_rank;
	int size = TAM * TAM * sizeof( float );
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
	cudaMalloc( (void **) &d_g, TAM*chunk );
	cudaMalloc( (void) &d_chunk, sizeof(int) );
	cudaMalloc( (void) &d_rank, sizeof(int) );
	printf("Copiando para GPU %d\n", rank);
	fflush(stdout);
	cudaMemcpy( d_a, A[0], size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, B[0], size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_chunk, chunk, chunk * sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_rank, rank, rank * sizeof(int), cudaMemcpyHostToDevice );
	printf("Mandando para funcao %d\n", rank);
	fflush(stdout);
	dim3 dimBlock (BLOCK,BLOCK);
	dim3 dimGrid ((TAM+BLOCK-1)/(float)BLOCK,(TAM+BLOCK-1)/(float)BLOCK);
	__mult__<<< dimBlock,dimGrid >>> ( d_a, d_b, d_g, d_rank, d_chunk );
	printf("Funcao executou %d\n", rank);
	fflush(stdout);

	printf("Copiando para CPU %d\n", rank);
	fflush(stdout);
	cudaMemcpy( T[0], d_g, TAM*chunk, cudaMemcpyDeviceToHost );
	printf("Retorno e %f do %d\n",T[0][0],rank);
	fflush(stdout);
	printf("Retornando %d\n",rank);
	fflush(stdout);

	return T;
}
