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

void* threaded_matrix_determinant(void *arg)
{
	DadosThread *argRef = (DadosThread *) arg;
	int newPos = 0;
    
    matrix_t *L = matrix_create_block(argRef->A->rows,argRef->A->rows);
    matrix_t *U = matrix_create_block(argRef->A->rows,argRef->A->rows);
	
    int i = 0, j = 0, k = 0;
	
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
	
	//-------------------------------------------------------------
	for (i = argRef->id*argRef->dataBlock_rows; i < newPos; i++)
    {
        for (j = 0; j < argRef->A->rows; j++)
        {
            if (j < i)
                L->data[j][i] = 0;
            else
            {
                L->data[j][i] = argRef->A->data[j][i];
                for (k = 0; k < i; k++)
                {
                    L->data[j][i] = L->data[j][i] - L->data[j][k] * U->data[k][i];
                }
            }
        }
        for (j = 0; j < argRef->A->rows; j++)
        {
            if (j < i)
                U->data[i][j] = 0;
            else if (j == i)
                U->data[i][j] = 1;
            else
            {
                U->data[i][j] = argRef->A->data[i][j] / L->data[i][i];
                for (k = 0; k < i; k++)
                {
                    if(argRef->debug) printf("ID:%d %lf ",argRef->id, argRef->A->data[i][j]);
                    U->data[i][j] = U->data[i][j] - ((L->data[i][k] * U->data[k][j]) / L->data[i][i]);
                }
            }
        }
    }
    
    for (i = argRef->id*argRef->dataBlock_rows; i < newPos; i++)
    { 
        *(argRef->detU) *= U->data[i][i];
        *(argRef->detL) *= L->data[i][i];
    }
    printf("ID:%d DETU = %lf\n",argRef->id,*(argRef->detU));
    printf("ID:%d DETL = %lf\n",argRef->id,*(argRef->detL));


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
