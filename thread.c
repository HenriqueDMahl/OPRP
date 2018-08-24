#include "thread.h"
#include "matrix.h"

matrix_t *threaded_matrix_multiply(matrix_t *A, matrix_t *B, matrix_t *R)
{
	double s = 0;
	for (int i = 0; i < A->rows; i++) {
		for (int j = 0; j < B->cols; j++) {
			for (int k = 0; k < A->cols; k++){
				s += A->data[i][k] * B->data[k][j];
			}
			R->data[i][j] = s;
			s = 0;
		}
	}
	return R;
}

void* call_threaded_matrix_multiply(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;
	
	argRef->R = threaded_matrix_multiply(argRef->A, argRef->B, argRef->R);

	return NULL;
}

matrix_t *threaded_matrix_sum(matrix_t *A, matrix_t *B, matrix_t *R)
{
	for (int i = 0; i < A->rows; i++) {
		for (int j = 0; j < B->cols; j++) {
			R->data[i][j] =  A->data[i][j] + B->data[i][j];
		}
	}

	return R;
}

void* call_threaded_matrix_sum(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;
	
	argRef->R = threaded_matrix_sum(argRef->A, argRef->B, argRef->R);

	return NULL;
}

matrix_t *threaded_matrix_inversion(matrix_t *A, matrix_t *(*p) (int, int), void (*p2) (matrix_t*))
{//Applies the Gauss-Jordan matrix reduction

	matrix_t* G2 = NULL;

	//Conditions to exist inverse of A
	// 1) A is a square matrix
	// 2) A has determinant and it's not equal to 0
	int det = matrix_determinant(A, matrix_create_block);

	if(A->rows == A-> cols && det != 0)
	{//A has inverse
		// Defines a matrix twice as long as A in terms of rows and columns
		matrix_t *G  = p(A->rows*2,A->cols*2);
		G2 = p(A->rows,A->cols);
		// Auxiliary storage
		double aux = 0;

		// Copy A to G
		for (int i = 0; i < A->rows; i++)
			for (int j = 0; j < A->rows; j++)
				G->data[i][j] = A->data[i][j];

		// Initializes G|I
		for (int i = 0; i < A->rows; i++)
			for (int j = 0; j < 2 * A->rows; j++)
				if (j == (i + A->rows))
					G->data[i][j] = 1;

		// Partial pivoting
		for (int i = A->rows-1; i > 0; i--)
		{
			if (G->data[i - 1][0] == G->data[i][0])
				for (int j = 0; j < A->rows * 2; j++)
				{
					aux = G->data[i][j];
					G->data[i][j] = G->data[i - 1][j];
					G->data[i - 1][j] = aux;
				}
		}

		// Reducing to diagonal matrix
		for (int i = 0; i < A->rows; i++)
		{
			for (int j = 0; j < A->rows * 2; j++)
				if (j != i)
				{
					aux = G->data[j][i] / G->data[i][i];
					for (int k = 0; k < A->rows * 2; k++)
					{
						G->data[j][k] -= G->data[i][k] * aux;
					}
				}
		}

		// Reducing to unit matrix
		for (int i = 0; i < A->rows; i++)
		{
			aux = G->data[i][i];
			for (int j = 0; j < A->rows * 2; j++)
				G->data[i][j] = G->data[i][j] / aux;
		}

		// Copying G to G2
		for (int i = 0; i < A->rows; i++) {
			for (int j = 0; j < A->rows; j++) {
				G2->data[i][j] = G->data[i][j + A->rows];
			}
		}

		// Dispose of the G matrix
		p2(G);

		// Return inverse of matrix A
		return G2;
	}
	// Return NULL matrix_t*
	return G2;
}

void* call_threaded_matrix_inversion(void *arg)
{
	return NULL;
}

matrix_t *threaded_matrix_transpose(matrix_t *A, matrix_t *R)
{
	for (int i = 0; i < A->rows; i++) {
		for (int j = 0; j < A->cols; j++) {
			R->data[i][j] =  A->data[j][i];
		}
	}

	return R;
}

void* call_threaded_matrix_transpose(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;
	
	argRef->R = threaded_matrix_transpose(argRef->A, argRef->R);

	return NULL;
}

double threaded_matrix_determinant(matrix_t* matrix, matrix_t *(*p) (int,int))
{
	if(matrix->rows == 1){//aims to find the 1x1 matrix case
		return matrix->data[0][0];
	} else{
		double det = 0;
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

void* call_threaded_matrix_determinant(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;
	
	argRef->det = threaded_matrix_determinant(argRef->A, matrix_create_block);

	return NULL;
}
