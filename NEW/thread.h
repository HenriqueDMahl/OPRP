#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

typedef struct {
	int id;
	int dataBlock_rows;
	int dataBlock_cols;
	int lastThread;
	int flagRows;
	int flagCols;
	int debug;
	double* det;
	int* equal;
	matrix_t *A;
	matrix_t *B;
	matrix_t *R;
	matrix_t *Rt;
} DadosThread;

matrix_t* threaded_matrix_multiply(matrix_t *A, matrix_t *B, matrix_t *R, int debug);

void* call_threaded_matrix_multiply(void *arg);

matrix_t* threaded_matrix_sum(int id, matrix_t *A, matrix_t *B, matrix_t *R, int debug, int dataBlock_rows, int lastThread, int flagRows);

void* call_threaded_matrix_sum(void *arg);

matrix_t* threaded_matrix_inversion(matrix_t *A, matrix_t *(*p) (int, int), void (*p2) (matrix_t*), int debug);

void* call_threaded_matrix_inversion(void *arg);

matrix_t *threaded_matrix_transpose(matrix_t *A, matrix_t *Rt, int debug);

void* call_threaded_matrix_transpose(void *arg);

double threaded_matrix_determinant(matrix_t* matrix, matrix_t *(*p) (int,int), int debug);

void* call_threaded_matrix_determinant(void *arg);

int threaded_matrix_equality(matrix_t *A, matrix_t *B, int debug);

void* call_threaded_matrix_equal(void *arg);
