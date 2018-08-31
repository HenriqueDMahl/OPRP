#include "thread.h"
#include "matrix.h"

// Split as row vectors tasks per threads
void* threaded_matrix_multiply(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;
	int newPos = 0;

	if(argRef->id == argRef->lastThread)
	{
		if(argRef->flagRows)
		{// Indicates odd thread amount
			newPos = argRef->id * argRef->dataBlock_rows + argRef->dataBlock_rows + argRef->resto; // INDICA QUE O NUMERO DE THREADS C	 IMPAR
		}else
		{// Indicates even thread amount
			newPos = argRef->id * argRef->dataBlock_rows + argRef->dataBlock_rows;
		}
	}else
	{
		newPos = argRef->id * argRef->dataBlock_rows + argRef->dataBlock_rows;
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

	if(argRef->id == argRef->lastThread)
	{
		if(argRef->flagRows)
		{
			newPos = argRef->id * argRef->dataBlock_rows + argRef->dataBlock_rows + argRef->resto;
		}else
		{
			newPos = argRef->id * argRef->dataBlock_rows + argRef->dataBlock_rows;
		}
	}else
	{
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

// Split as row vectors tasks per threads
void* threaded_matrix_inversion(void *arg)
{//Applies the Gauss-Jordan matrix reduction
}

// Split as row vectors tasks per threads
void* threaded_matrix_transpose(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;
	int newPos = 0;

	if(argRef->id == argRef->lastThread)
	{
		if(argRef->flagRows)
		{
			newPos = argRef->id*argRef->dataBlock_rows + argRef->dataBlock_rows + argRef->resto;
		}else
		{
			newPos = argRef->id*argRef->dataBlock_rows + argRef->dataBlock_rows;
		}
	}else
	{
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
// Split as row vectors tasks per threads
double threaded_matrix_determinant(matrix_t* matrix, matrix_t *(*p) (int,int), int debug)
{
}

void* call_threaded_matrix_determinant(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;

	*argRef->det = threaded_matrix_determinant(argRef->A, matrix_create_block, argRef->debug);
	printf("%lf \n", *argRef->det);


	pthread_exit(NULL);
}


// Split as row vectors tasks per threads
void* threaded_matrix_equal(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;
	int newPos = 0;

	if(argRef->id == argRef->lastThread)
	{
		if(argRef->flagRows)
		{
			newPos = argRef->id * argRef->dataBlock_rows + argRef->dataBlock_rows + argRef->resto;
		}else
		{
			newPos = argRef->id * argRef->dataBlock_rows + argRef->dataBlock_rows;
		}
	}else
	{
		newPos = argRef->id * argRef->dataBlock_rows + argRef->dataBlock_rows;
	}

	for (int i = argRef->id*argRef->dataBlock_rows; i < newPos; i++)
	{
		for (int j = 0; j < argRef->A->cols; j++)
		{
			if(*argRef->equal == 0) break;// Ensures that if a thread already marked it as different then it won't go any further processing for others
			if (argRef->A->data[i][j] != argRef->B->data[i][j])
			{
				if(argRef->debug) printf("ID:%d 0 ",argRef->id);
				*(argRef->equal) = 0;
				break;
			}else
			{
				if(argRef->debug) printf("ID:%d 1 ",argRef->id);
				*(argRef->equal) = 1;
			}
		}
		if(argRef->debug) printf("\n");
	}

	pthread_exit(NULL);
}
