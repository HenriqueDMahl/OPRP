#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include "matrix.h"

double wtime() {

	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec / 1000000.0;
}

int main(int argc, char **argv) {

	double start_time, end_time;
	int nrows, ncols;

	if ((argc != 3)) {
		printf("Uso: %s <rows> <cols>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	nrows = atoi(argv[1]);
	ncols = atoi(argv[2]);

	start_time = wtime();

	matrix_t *A = matrix_create_block(nrows,ncols);
	matrix_t *B = matrix_create_block(nrows,ncols);
	matrix_t *R = matrix_create_block(nrows,ncols);

	A->data[0][0] = 1;
	A->data[0][1] = 2;
	A->data[1][0] = 1;
	A->data[1][1] = 2;
	B->data[0][0] = 1;
	B->data[0][1] = 2;
	B->data[1][0] = 1;
	B->data[1][1] = 2;

//	matrix_fill(A, 2);
//	matrix_fill(B, 2);

	//   R = matrix_multiply(A,B,matrix_create_block);
	//   R = matrix_sum(A,B,matrix_create_block);
	//   R = matrix_inversion(A,matrix_create_block); //ARRUMAR
	//   R = matrix_transpose(A,matrix_create_block);

	int det = 0, ig = 0;
//	matrix_print(R);
//	det = matrix_determinant(A, matrix_create_block);

//	ig = matrix_equal(A,B);

	printf("%d \n", det);
	printf("%d \n", ig);

	matrix_destroy_block(A);
	matrix_destroy_block(B);
	matrix_destroy_block(R);

	end_time = wtime();


	printf("%d %d %f\n", nrows, ncols, end_time - start_time);
	fflush(stdout);

	return EXIT_SUCCESS;
}
