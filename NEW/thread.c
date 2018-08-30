#include "thread.h"
#include "matrix.h"

// Review how to split tasks between threads
void* threaded_matrix_multiply(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;

	int newPos = 0;

	if(argRef->id == argRef->lastThread){
		if(argRef->flagRows) {
			newPos = (argRef->id*argRef->dataBlock_rows)+argRef->dataBlock_rows + argRef->resto; // INDICA QUE O NUMERO DE THREADS C	 IMPAR
		}else{
			newPos = (argRef->id*argRef->dataBlock_rows)+argRef->dataBlock_rows; // INDICA QUE O NUMERO DE THREADS C	 PAR
		}
	}else{
		newPos = (argRef->id*argRef->dataBlock_rows)+argRef->dataBlock_rows;
	}

	double s = 0;
	for (int i = argRef->id*argRef->dataBlock_rows; i < newPos; i++)
	{
		for (int j = 0; j < argRef->B->cols; j++)
		{
			for (int k = 0; k < argRef->A->cols; k++)
			{
				if(argRef->debug) printf("ID:%d %lf ", argRef->id, argRef->A->data[i][k] * argRef->B->data[k][j]);
				s += argRef->A->data[i][k] * argRef->B->data[k][j];
			}
			argRef->R->data[i][j] = s;
			s = 0;
			if(argRef->debug) printf("\n");
		}
	}

	pthread_exit(NULL);
}

// Split as row vectors tasks per threads
void* threaded_matrix_sum(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;

	int newPos = 0;

	if(argRef->id == argRef->lastThread){
		if(argRef->flagRows) {
			newPos = (argRef->id*argRef->dataBlock_rows)+argRef->dataBlock_rows + argRef->resto; // INDICA QUE O NUMERO DE THREADS C	 IMPAR
		}else{
			newPos = (argRef->id*argRef->dataBlock_rows)+argRef->dataBlock_rows; // INDICA QUE O NUMERO DE THREADS C	 PAR
		}
	}else{
		newPos = (argRef->id*argRef->dataBlock_rows)+argRef->dataBlock_rows;
	}

	for (int i = argRef->id*argRef->dataBlock_rows; i < newPos; i++)
	{
		for (int j = 0; j < argRef->A->cols; j++)
		{
			if(argRef->debug) printf("ID:%d %lf ", argRef->id, argRef->A->data[i][j] + argRef->B->data[i][j]);
			argRef->R->data[i][j] =  argRef->A->data[i][j] + argRef->B->data[i][j];
		}
		if(argRef->debug) printf("\n");
	}

	pthread_exit(NULL);
}

// Implement control for debug
// Review how to split tasks between threads
void* threaded_matrix_inversion(void *arg)
{//Applies the Gauss-Jordan matrix reduction

	DadosThread *argRef = (DadosThread *) arg;
//	matrix_t* (*p)(int, int) = &matrix_create_block;
//	void (*p2)(matrix_t*) = &matrix_destroy_block;

	matrix_t* G2 = NULL;

	//Conditions to exist inverse of A
	// 1) A is a square matrix
	// 2) A has determinant and it's not equal to 0
	double det = matrix_determinant(argRef->A, matrix_create_block);

	if(argRef->A->rows == argRef->A->cols && det != 0)
	{//A has inverse
		// Defines a matrix twice as long as A in terms of rows and columns
		matrix_t *G  = argRef->p(argRef->A->rows*2,argRef->A->cols*2);
		G2 = argRef->p(argRef->A->rows,argRef->A->cols);
		// Auxiliary storage
		double aux = 0;

		// Copy A to G
		for (int i = 0; i < argRef->A->rows; i++)
			for (int j = 0; j < argRef->A->rows; j++)
				G->data[i][j] = argRef->A->data[i][j];

		// Initializes G|I
		for (int i = 0; i < argRef->A->rows; i++)
			for (int j = 0; j < 2 * argRef->A->rows; j++)
				if (j == (i + argRef->A->rows))
					G->data[i][j] = 1;

		// Partial pivoting
		for (int i = argRef->A->rows-1; i > 0; i--)
		{
			if (G->data[i - 1][0] == G->data[i][0])
				for (int j = 0; j < argRef->A->rows * 2; j++)
				{
					aux = G->data[i][j];
					G->data[i][j] = G->data[i - 1][j];
					G->data[i - 1][j] = aux;
				}
		}

		// Reducing to diagonal matrix
		for (int i = 0; i < argRef->A->rows; i++)
		{
			for (int j = 0; j < argRef->A->rows * 2; j++)
				if (j != i)
				{
					aux = G->data[j][i] / G->data[i][i];
					for (int k = 0; k < argRef->A->rows * 2; k++)
					{
						G->data[j][k] -= G->data[i][k] * aux;
					}
				}
		}

		// Reducing to unit matrix
		for (int i = 0; i < argRef->A->rows; i++)
		{
			aux = G->data[i][i];
			for (int j = 0; j < argRef->A->rows * 2; j++)
				G->data[i][j] = G->data[i][j] / aux;
		}

		// Copying G to G2
		for (int i = 0; i < argRef->A->rows; i++)
		{
			for (int j = 0; j < argRef->A->rows; j++)
			{
				G2->data[i][j] = G->data[i][j + argRef->A->rows];
			}
		}

		// Dispose of the G matrix
		argRef->p2(G);

		// Return inverse of matrix A
		return G2;
	}
	// Return NULL matrix_t*
	return G2;


	pthread_exit(NULL);
}

// Review how to split tasks between threads
void* threaded_matrix_transpose(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;

	int newPos = 0;

	if(argRef->id == argRef->lastThread){
		if(argRef->flagRows) {
			newPos = (argRef->id*argRef->dataBlock_rows)+argRef->dataBlock_rows + argRef->resto; // INDICA QUE O NUMERO DE THREADS C	 IMPAR
		}else{
			newPos = (argRef->id*argRef->dataBlock_rows)+argRef->dataBlock_rows; // INDICA QUE O NUMERO DE THREADS C	 PAR
		}
	}else{
		newPos = (argRef->id*argRef->dataBlock_rows)+argRef->dataBlock_rows;
	}

	for (int i = argRef->id*argRef->dataBlock_rows; i < newPos; i++)
	{
		for (int j = 0; j < argRef->A->cols; j++)
		{
			if(argRef->debug) printf("ID:%d %lf ",argRef->id, argRef->A->data[i][j]);
			argRef->Rt->data[j][i] = argRef->A->data[i][j];
		}
		if(argRef->debug) printf("\n");
	}

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
void* threaded_matrix_equal(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;

		int newPos = 0;

		if(argRef->id == argRef->lastThread){
			if(argRef->flagRows) {
				newPos = (argRef->id*argRef->dataBlock_rows)+argRef->dataBlock_rows + argRef->resto; // INDICA QUE O NUMERO DE THREADS C	 IMPAR
			}else{
				newPos = (argRef->id*argRef->dataBlock_rows)+argRef->dataBlock_rows; // INDICA QUE O NUMERO DE THREADS C	 PAR
			}
		}else{
			newPos = (argRef->id*argRef->dataBlock_rows)+argRef->dataBlock_rows;
		}

	for (int i = argRef->id*argRef->dataBlock_rows; i < newPos; i++)
	{
		for (int j = 0; j < argRef->A->cols; j++)
		{
		    if(*argRef->equal == 0) break; //Verifica se alguma thread já marcou o equal = 0, se sim cai fora
			if (argRef->A->data[i][j] != argRef->B->data[i][j])
			{
				if(argRef->debug) printf("ID:%d 0 ",argRef->id);
				*(argRef->equal) = 0;
				break;
			}else{
				if(argRef->debug) printf("ID:%d 1 ",argRef->id);
	            *(argRef->equal) = 1;
			}
		}
		if(argRef->debug) printf("\n");
	}

	pthread_exit(NULL);
}
