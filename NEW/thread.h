#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

typedef struct {
	int id;
	int dataBlock_rows;
	int lastThread;
	int flagRows;
	int debug;
	int *equal;
	int resto;
	double* det;
	double* detU;
	double* detL;
	matrix_t *A;
	matrix_t *B;
	matrix_t *R;
	matrix_t *L;
	matrix_t *U;
	matrix_t *Rt;
	matrix_t *(*p) (int,int);
	void (*p2) (matrix_t*);
} DadosThread;

void* threaded_matrix_multiply(void *arg);

void* threaded_matrix_sum(void *arg);

void* threaded_matrix_inversion(void *arg);

void* threaded_matrix_transpose(void *arg);

void* threaded_matrix_determinant(void *arg);

void* threaded_matrix_equal(void *arg);
