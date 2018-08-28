#include "thread.h"
#include "matrix.h"

// Review how to split tasks between threads
matrix_t *threaded_matrix_multiply(matrix_t *A, matrix_t *B, matrix_t *R, int debug)
{
	double s = 0;
	for (int i = 0; i < A->rows; i++)
	{
		for (int j = 0; j < B->cols; j++)
		{
			for (int k = 0; k < A->cols; k++)
			{
				if(debug) printf("%lf ", A->data[i][k] * B->data[k][j]);
				s += A->data[i][k] * B->data[k][j];
			}
			R->data[i][j] = s;
			s = 0;
			if(debug) printf("\n");
		}
	}
	return R;
}

void* call_threaded_matrix_multiply(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;

	argRef->R = threaded_matrix_multiply(argRef->A, argRef->B, argRef->R, argRef->debug);

	pthread_exit(NULL);
}

// Review how to split tasks between threads
matrix_t* threaded_matrix_sum(int id, matrix_t *A, matrix_t *B, matrix_t *R, int debug, int dataBlock_rows, int lastThread, int flagRows)
{
	printf("ID %d\n", id);
	if(id == lastThread){
		if(flagRows) dataBlock_rows = dataBlock_rows + (A->rows - id*dataBlock_rows);
	}

	for (int i = id*dataBlock_rows; i < (id*dataBlock_rows)+dataBlock_rows; i++)
	{
		for (int j = 0; j < A->cols; j++)
		{
			if(debug) printf("%lf ", A->data[i][j] + B->data[i][j]);
			R->data[i][j] =  A->data[i][j] + B->data[i][j];
		}
		if(debug) printf("\n");
	}
	// Return R as the sum of A and B
	return NULL;
}

void* call_threaded_matrix_sum(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;

	argRef->R = threaded_matrix_sum(argRef->id, argRef->A, argRef->B, argRef->R, argRef->debug, argRef->dataBlock_rows, argRef->lastThread, argRef->flagRows);

	pthread_exit(NULL);
}

// Implement control for debug
// Review how to split tasks between threads
matrix_t *threaded_matrix_inversion(matrix_t *A, matrix_t *(*p) (int, int), void (*p2) (matrix_t*), int debug)
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
		for (int i = 0; i < A->rows; i++)
		{
			for (int j = 0; j < A->rows; j++)
			{
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

// Review how to split tasks between threads
void* call_threaded_matrix_inversion(void *arg)
{
	pthread_exit(NULL);
}

// Review how to split tasks between threads
matrix_t *threaded_matrix_transpose(matrix_t *A, matrix_t *Rt, int debug)
{
	for (int i = 0; i < A->rows; i++)
	{
		for (int j = 0; j < A->cols; j++)
		{
			if(debug) printf("%lf ", A->data[i][j]);
			Rt->data[j][i] = A->data[i][j];
		}
		if(debug) printf("\n");
	}

	return Rt;
}

void* call_threaded_matrix_transpose(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;

	argRef->Rt = threaded_matrix_transpose(argRef->A, argRef->Rt, argRef->debug);

	pthread_exit(NULL);
}

// Implement control for debug
// Review how to split tasks between threads
// Review how to split chunks between threads
double threaded_matrix_determinant(matrix_t* matrix, matrix_t *(*p) (int,int), int debug)
{
	if(matrix->rows == 1)
	{//aims to find the 1x1 matrix case
		return matrix->data[0][0];
	} else
	{
		double det = 0;
		int i, row, col, j_aux, i_aux;

		//Chooses first line to calc cofactors
		for(i = 0; i < matrix->rows; i++)
		{
			//Ignores 0s because it is the multiplication identity
			if (matrix->data[0][i] != 0)
			{
				matrix_t* g = p(matrix->rows-1,matrix->cols-1);
				g->rows = matrix->rows-1;
				g->rows = matrix->rows-1;
				// Location auxiliaries
				i_aux = 0;
				j_aux = 0;
				//Gen matrices to calc cofactors
				for(row = 1; row < matrix->rows; row++)
				{
					for(col = 0; col < matrix->rows; col++)
					{
						if(col != i)
						{
							g->data[i_aux][j_aux] = matrix->data[row][col];
							j_aux++;
						}
					}
					i_aux++;
					j_aux = 0;
				}
				// Sign control
				double factor = (i % 2 == 0)? matrix->data[0][i] : -matrix->data[0][i];
				det += factor * matrix_determinant(g, p);
			}
		}
		return det;
	}
}

void* call_threaded_matrix_determinant(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;

	*argRef->det = threaded_matrix_determinant(argRef->A, matrix_create_block, argRef->debug);
	printf("%lf \n", *argRef->det);

	pthread_exit(NULL);
}

// Review how to split tasks between threads
int threaded_matrix_equal(matrix_t*A, matrix_t *B, int debug)
{// 0 = True and 1 = False
	int i, j;

	if (A->rows != B->rows || A->cols != B->cols)
		return 0;

	for (i = 0; i < A->rows; i++)
	{
		for (j = 0; j < A->cols; j++)
		{
			if (A->data[i][j] != B->data[i][j])
			{
				if(debug) printf("0 ");
				return 0;
			}
		}
		if(debug) printf("\n");
	}
	return 1;
}

void* call_threaded_matrix_equal(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;

	*argRef->equal = threaded_matrix_equal(argRef->A, argRef->B, argRef->debug);

	pthread_exit(NULL);
}
