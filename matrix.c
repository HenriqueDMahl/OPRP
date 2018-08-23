#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "matrix.h"
#include <math.h>

matrix_t *matrix_create_block(int rows, int cols) {
	matrix_t* matrix = NULL;

	matrix = malloc(sizeof(matrix_t));
	double* bloco = malloc(sizeof(double) * rows * cols);
	matrix->data = malloc(sizeof(double*) * rows);
	matrix->rows = rows;
	matrix->cols = cols;
	for(int i = 0; i <= rows; i++) {
		matrix->data[i] = bloco+i*cols;
	}

	return matrix;
}

matrix_t *matrix_create_pointers(int rows, int cols) {
	matrix_t* matrix = NULL;

	matrix = malloc(sizeof(matrix_t));
	matrix->data = malloc(sizeof(double)*rows);
	matrix->rows = rows;
	matrix->cols = cols;

	for(int i = 0; i <= rows; i++) {
		matrix->data[i] = malloc(sizeof(double)*cols);
	}
	return matrix;
}

void matrix_destroy_pointers(matrix_t *m) {
	for(int i = 0; i <= sizeof(m->data[0]); i++) {
		for(int j = 0; j <= sizeof(m->data[0][0]); j++) {
			free(&m->data[i][j]);
		}
	}
	free(m->data);
	free(m);
}

void matrix_destroy_block(matrix_t *m) {
	free(m->data[0]);
	free(m->data);
	free(m);
}

void matrix_randfill(matrix_t *m) {
	int i, j;

	for (i = 0; i < m->rows; i++) {
		for (j = 0; j < m->cols; j++) {
			m->data[i][j] = random();
		}
	}
}

void matrix_fill(matrix_t *m, double val) {
	int i, j;

	for (i = 0; i < m->rows; i++) {
		for (j = 0; j < m->cols; j++) {
			m->data[i][j] = val;
		}
	}
}

matrix_t *matrix_multiply(matrix_t *A, matrix_t *B, matrix_t *(*p) (int, int)) {
	matrix_t *g = p(A->rows,B->cols);
	int s = 0;
	for (int i = 0; i < A->rows; i++) {
		for (int j = 0; j < B->cols; j++) {
			for (int k = 0; k < A->cols; k++){
				s += A->data[i][k] * B->data[k][j];
			}
			g->data[i][j] = s;
			s = 0;
		}
	}
	return g;
}

void matrix_print(matrix_t *m) {
	int i, j;

	for (i = 0; i < m->rows; i++) {
		for (j = 0; j < m->cols; j++) {
			printf("%.17f ", m->data[i][j]);
		}
		printf("\n");
	}
	fflush(stdout);
}

int matrix_equal(matrix_t*A, matrix_t *B) {
	int i, j;

	if (A->rows != B->rows || A->cols != B->cols)
		return 0;

	for (i = 0; i < A->rows; i++) {
		for (j = 0; j < A->cols; j++) {
			if (A->data[i][j] != B->data[i][j])
				return 0;
		}
	}
	return 1;
}

matrix_t *matrix_sum(matrix_t *A, matrix_t *B, matrix_t *(*p) (int, int)) {
	matrix_t *g = p(A->rows,B->cols);

	for (int i = 0; i < A->rows; i++) {
		for (int j = 0; j < B->cols; j++) {
			g->data[i][j] =  A->data[i][j] + B->data[i][j];
		}
	}

	return g;
}

matrix_t *matrix_inversion(matrix_t *A, matrix_t *(*p) (int, int)) {
	matrix_t *g = p(A->rows,A->cols);
  int det = matrix_determinant(A,p);
	if(det == 0)
		return g = NULL;
	else{
		for (int i = 0; i < A->rows; i++) {
			for (int j = 0; j < A->cols; j++) {
				g->data[i][j] =  A->data[i][j] * aux->data[i][j];
			}
		}
	}

	return g;
}

matrix_t *matrix_transpose(matrix_t *A, matrix_t *(*p) (int, int)) {
	matrix_t *g = p(A->rows,A->cols);

	for (int i = 0; i < A->rows; i++) {
		for (int j = 0; j < A->cols; j++) {
			g->data[i][j] =  A->data[j][i];
		}
	}

	return g;
}

int matrix_determinant(matrix_t* matrix, matrix_t *(*p) (int,int)) {
	if(matrix->rows == 1){//aims to find the 1x1 matrix case
		return matrix->data[0][0];
	} else{
		int det = 0;
		int i, row, col, j_aux, i_aux;

		//Chooses first line to calc cofactors
		for(i = 0; i < matrix->rows; i++) {
			//Ignores 0s because it is the multiplication identity
			if (matrix->data[0][i] != 0) {
				matrix_t* g = p(matrix->rows-1,matrix->cols-1);
				g->rows = matrix->rows-1;
				g->rows = matrix->rows-1;
				i_aux = 0;
				j_aux = 0;
				//Gen matrices to calc cofactors
				for(row = 1; row < matrix->rows; row++) {
					for(col = 0; col < matrix->rows; col++){
						if(col != i){
							g->data[i_aux][j_aux] = matrix->data[row][col];
							j_aux++;
						}
					}
					i_aux++;
					j_aux = 0;
				}
				double factor = (i % 2 == 0)? matrix->data[0][i] : -matrix->data[0][i];
				det = det + factor * matrix_determinant(g, p);
			}
		}
		return det;
	}
}
