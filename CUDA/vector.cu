#include <stdio.h>
#include <math.h>
/**

Warp corta de 32/16 as threads, exemplo 1024 threads será dividida em pedaços de 32/16 de execução paralela efetiva.

**/
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
__global__ void vector_add(int *a, int *b, int *c) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < N) c[index] = a[index] + b[index];
}*/

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
}

int main() {
	float *a, *b, *c;
	float *d_a, *d_b, *d_c;
	int size = TAM * TAM * sizeof( float );
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
	dim3 dimBlock (BLOCK,BLOCK);
    dim3 dimGrid ((TAM+BLOCK-1)/(float)BLOCK,(TAM+BLOCK-1)/(float)BLOCK);
	mult<<< dimBlock,dimGrid >>> ( d_a, d_b, d_c );
	cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );
	printf("Matriz A\n");
	printMatrix(a);
	printf("Matriz B\n");
	printMatrix(b);
	printf("Matriz C\n");
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
