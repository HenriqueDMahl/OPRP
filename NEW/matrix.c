#include "matrix.h"

matrix_t *matrix_create_block(int rows, int cols)
{
	matrix_t* G = NULL;

	G = malloc(sizeof(matrix_t));
	double* bloco = malloc(sizeof(double) * rows * cols);
	G->data = malloc(sizeof(double*) * rows);
	G->rows = rows;
	G->cols = cols;
	for(int i = 0; i <= rows; i++)
	{
		G->data[i] = bloco+i*cols;
	}

	return G;
}

matrix_t *matrix_create_pointers(int rows, int cols)
{
	matrix_t* G = NULL;

	G = malloc(sizeof(matrix_t));
	G->data = malloc(sizeof(double)*rows);
	G->rows = rows;
	G->cols = cols;

	for(int i = 0; i <= rows; i++)
	{
		G->data[i] = malloc(sizeof(double)*cols);
	}

	return G;
}

matrix_t *matrix_create_block_init(int rows, int cols)
{// Initializes a block with 0s
	matrix_t* G = NULL;

	G = calloc(1,sizeof(matrix_t));
	double* bloco = calloc(rows*cols,sizeof(double));
	G->data = calloc(rows,sizeof(double*));
	G->rows = rows;
	G->cols = cols;
	for(int i = 0; i <= rows; i++)
	{
		G->data[i] = bloco+i*cols;
	}

	return G;
}

matrix_t *matrix_create_pointers_init(int rows, int cols)
{// Initializes pointers block with 0s
	matrix_t* G = NULL;

	G = calloc(1,sizeof(matrix_t));
	G->data = calloc(rows,sizeof(double));
	G->rows = rows;
	G->cols = cols;

	for(int i = 0; i <= rows; i++)
	{
		G->data[i] = calloc(cols,sizeof(double));
	}

	return G;
}

void matrix_destroy_pointers(matrix_t *m)
{
	for(int i = 0; i <= sizeof(m->data[0]); i++)
	{
		for(int j = 0; j <= sizeof(m->data[0][0]); j++)
		{
			free(&m->data[i][j]);
		}
	}
	free(m->data);
	free(m);
}

void matrix_destroy_block(matrix_t *m)
{
	free(m->data[0]);
	free(m->data);
	free(m);
}

void matrix_randfill(matrix_t *m)
{
	int i, j;
    
	for(i = 0; i < m->rows; i++)
	{
		for(j = 0; j < m->cols; j++)
		{
			m->data[i][j] = random();
		}
	}
}

void matrix_fill(matrix_t *m, double val)
{
	int i, j;

	for(i = 0; i < m->rows; i++)
	{
		for(j = 0; j < m->cols; j++)
		{
			m->data[i][j] = val;
		}
	}
}

matrix_t *matrix_multiply(matrix_t *A, matrix_t *B, matrix_t *(*p) (int, int))
{
	matrix_t *G = p(A->rows,B->cols);

	double s = 0;
	for(int i = 0; i < A->rows; i++)
	{
		for(int j = 0; j < B->cols; j++)
		{
			for(int k = 0; k < A->cols; k++)
			{
				s += A->data[i][k] * B->data[k][j];
			}
			G->data[i][j] = s;
			s = 0;
		}
	}

	return G;
}

void matrix_print(matrix_t *m)
{
	int i, j;

	for(i = 0; i < m->rows; i++)
	{
		for(j = 0; j < m->cols; j++)
		{
			printf("%.17f ", m->data[i][j]);
		}
		printf("\n");
	}
	fflush(stdout);
}

int matrix_equal(matrix_t*A, matrix_t *B)
{
	int i, j;

	if (A->rows != B->rows || A->cols != B->cols)
		return 0;

	for(i = 0; i < A->rows; i++)
	{
		for(j = 0; j < A->cols; j++)
		{
			if (A->data[i][j] != B->data[i][j])
				return 0;
		}
	}
	return 1;
}

matrix_t *matrix_sum(matrix_t *A, matrix_t *B, matrix_t *(*p) (int, int))
{
	matrix_t *G = p(A->rows,B->cols);

	if(A->rows == B->rows && A->cols == B->cols)
	{// Matrix A dimension equality to B is required
		for(int i = 0; i < A->rows; i++)
		{
			for(int j = 0; j < B->cols; j++)
			{
				G->data[i][j] =  A->data[i][j] + B->data[i][j];
			}
		}

		return G;
	} else
	{// Teturn as NULL
		return G;
	}
}

matrix_t *matrix_inversion(matrix_t *A, matrix_t *(*p) (int, int), void (*p2) (matrix_t*))
{//Applies the Gauss-Jordan matrix reduction
	matrix_t* G2 = NULL;

	//Conditions to exist inverse of A
	// 1) A is a square matrix
	// 2) A has determinant and it's not equal to 0
	double det = matrix_determinant(A->rows, A->data);

	if(A->rows == A-> cols && det != 0)
	{//A has inverse
		// Defines a matrix twice as long as A in terms of rows and columns
		matrix_t *G  = p(A->rows*2,A->cols*2);
		G2 = p(A->rows,A->cols);
		// Auxiliary storage
		double aux = 0;

		// Copy A to G
		for(int i = 0; i < A->rows; i++)
			for(int j = 0; j < A->rows; j++)
				G->data[i][j] = A->data[i][j];

		// Initializes G|I
		for(int i = 0; i < A->rows; i++)
			for(int j = 0; j < 2 * A->rows; j++)
				if(j == (i + A->rows))
					G->data[i][j] = 1;

		// Partial pivoting
		for(int i = A->rows-1; i > 0; i--)
		{
			if(G->data[i - 1][0] == G->data[i][0])
				for(int j = 0; j < A->rows * 2; j++)
				{
					aux = G->data[i][j];
					G->data[i][j] = G->data[i - 1][j];
					G->data[i - 1][j] = aux;
				}
		}

		// Reducing to diagonal matrix
		for(int i = 0; i < A->rows; i++)
		{
			for(int j = 0; j < A->rows * 2; j++)
				if (j != i)
				{
					aux = G->data[j][i] / G->data[i][i];
					for(int k = 0; k < A->rows * 2; k++)
					{
						G->data[j][k] -= G->data[i][k] * aux;
					}
				}
		}

		// Reducing to unit matrix
		for(int i = 0; i < A->rows; i++)
		{
			aux = G->data[i][i];
			for(int j = 0; j < A->rows * 2; j++)
				G->data[i][j] = G->data[i][j] / aux;
		}

		// Copying G to G2
		for(int i = 0; i < A->rows; i++)
		{
			for(int j = 0; j < A->rows; j++)
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

matrix_t *matrix_transpose(matrix_t *A, matrix_t *(*p) (int, int))
{
	matrix_t *G = p(A->rows,A->cols);

	for(int i = 0; i < A->rows; i++)
	{
		for(int j = 0; j < A->cols; j++)
		{
			G->data[i][j] = A->data[j][i];
		}
		printf("\n");
	}

	return G;
}

// returns pivot column, or mx_sz if there is no pivot
size_t get_pivot(size_t row, size_t mx_sz, double **mx)
{
	size_t j = 0;

	while(j < mx_sz && fabs(mx[row][j]) < DELTA)
	{
		++j;
	}

	return j;
}

// Determinant iterative for NxN
double matrix_determinant(size_t mx_sz, double **A)
{
	size_t pivot1, pivot2;
	size_t row;
	double factor;
	bool finished = false;
	double result = 1.0;

	while(!finished)
	{
		finished = true;
		row = 1;
		while(row < mx_sz)
		{
			// determinant is zero if there is a zero row
			if((pivot1 = get_pivot(row - 1, mx_sz, A)) == mx_sz || (pivot2 = get_pivot(row, mx_sz, A)) == mx_sz)
			{
				return 0.0;
			}
			if(pivot1 == pivot2)
			{
				factor = -A[row][pivot1] / A[row - 1][pivot1];
				// Reduce matrix mx
				for (size_t j = 0; j < mx_sz; j++) A[row][j] += (factor * A[row-1][j]);
				finished = false;
			}else if(pivot2 < pivot1)
			{
				// Interchange matrix mx
				double temp;
				for (size_t j = 0; j < mx_sz; j++)
				{
					temp = A[row-1][j];
					A[row-1][j] = A[row][j];
					A[row][j] = temp;
				}
				result = -result;
				finished = false;
			}
			++row;
		}
	}
	
	// Obtains the determinant itself
	for(size_t j = 0; j < mx_sz; j++){
		result *= A[j][j];
	}

	return result;
}
