#include <stdio.h>
#include <math.h>
#define N 1024// 1024 = 32 * 32
#define TAM 3
#define THREADS_PER_BLOCK 512

__global__ void soma(float *a, float *b, float *c){
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	// Identifies the index of the operation(vector)
	int id = TAM*y + x;
	// Performs the sum
	if(id < TAM*TAM)
		c[id] = a[id] + b[id];
}

__global__ void vector_add(int *a, int *b, int *c) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < N) c[index] = a[index] + b[index];
}

int * newMatrix(int flag){
	int *m = (int *) malloc(sizeof(int)*TAM*TAM);
	for(int i = 0; i < TAM*TAM; i++)
		if(!flag)
			m[i] = 0;
		else
			m[i] = 1;
	return m;

}

void printMatrix(int *m){
	int cont = 0;
	for(int i = 0; i< TAM*TAM; i++){
		if(cont == TAM){
			cont = 0;
			printf("\n");
		}
		if(cont != TAM){
			printf("%d ", m[i]);
			cont++;
		}
	}
	printf("\n");
}

int main() {
	int *a, *b, *c;
	float *d_a, *d_b, *d_c;
	int size = N * sizeof( int );
	cudaMalloc( (void **) &d_a, size );
	cudaMalloc( (void **) &d_b, size );
	cudaMalloc( (void **) &d_c, size );

	a = newMatrix(1);
	b = newMatrix(1);
	c = newMatrix(0);

	//printMatrix(a);

	cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

	//soma<<< (N + ceil(THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c );
	dim3 dimBlock (TAM + (int)ceil(TAM*TAM/N));
    dim3 dimGrid (32,32);
	soma<<< dimBlock,dimGrid >>> ( d_a, d_b, d_c );
	cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );
	printMatrix(c);
	//printf( "c[0] = %d\n",c[0] );
	//printf( "c[%d] = %d\n",N-1, c[N-1] );
	//printf("d_a %f\n", *d_a);
	free(a);
	free(b);
	free(c);
	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );
	return 0;
} /* end main */
