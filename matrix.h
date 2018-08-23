#ifndef __MATRIX_H
#define __MATRIX_H

#include <stdlib.h>

#define random() ((rand() ^ rand()) / (RAND_MAX + 1.0))

typedef struct {
      double **data;
      int rows;
      int cols;
} matrix_t;

/*
 * All functions must return a new matriz (when need)
 */

matrix_t *matrix_create_pointers(int rows, int cols);

matrix_t *matrix_create_block(int rows, int cols);

void matrix_destroy_pointers(matrix_t *m);

void matrix_destroy_block(matrix_t *m);

void matrix_randfill(matrix_t *m);

void matrix_fill(matrix_t *m, double val);

matrix_t *matrix_multiply(matrix_t *A, matrix_t *B, matrix_t *(*p) (int, int));

matrix_t *matrix_sum(matrix_t *A, matrix_t *B, matrix_t *(*p) (int, int));

matrix_t *matrix_inversion(matrix_t *A, matrix_t *(*p) (int, int));

matrix_t *matrix_transpose(matrix_t *A, matrix_t *(*p) (int, int));

int matrix_determinant(matrix_t* matrix, matrix_t *(*p) (int,int));

void matrix_print(matrix_t *m);

int matrix_equal(matrix_t*A, matrix_t *B);

#endif
