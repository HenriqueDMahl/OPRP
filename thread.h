#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

typedef struct {
	int id;
	matrix_t *A;
	matrix_t *B;
	matrix_t *R;
} DadosThread;

matrix_t* threaded_matrix_multiply(matrix_t *A, matrix_t *B, matrix_t *R);

void* call_threaded_matrix_multiply(void *arg);

matrix_t* threaded_matrix_sum(matrix_t *A, matrix_t *B, matrix_t *R);

void* call_threaded_matrix_sum(void *arg);

matrix_t* threaded_matrix_inversion(matrix_t *A, matrix_t *(*p) (int, int), void (*p2) (matrix_t*));

matrix_t *threaded_matrix_transpose(matrix_t *A, matrix_t *R);

void* call_threaded_matrix_transpose(void *arg);

int threaded_matrix_determinant(matrix_t* matrix, matrix_t *(*p) (int,int));

void* call_threaded_matrix_determinant(void *arg);
