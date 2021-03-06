#include "main.h"
#include "matrix.h"
#include "thread.h"

double wtime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec / 1000000.0;
}

int main(int argc, char **argv)
{
	double start_time, end_time, timeThreaded, timeSequential;
	int nrows1, ncols1, nrows2, ncols2, nthreads = 0, calcSum, calcMul, calcInv, calcTra, calcDet, calcEqu;
	int equal=1, freeLaterA = 0, freeLaterB = 0, freeLaterR = 0, freeLaterRt = 0;
	int execThr = 0, execSeq = 0, execBoth, mode2Debug;
	double det = 0;
	DadosThread *dt = NULL;
	pthread_t *threads = NULL;

	if ((argc != 14))
	{
		printf("Uso: %s <rows1> <cols1> <rows2> <cols2> <nthreads> <m_sum>\n"
		"<m_multi> <m_inver> <m_trans> <m_deter> <m_equal> <execMode> <mode2Debug>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	nrows1 = atoi(argv[1]);
	ncols1 = atoi(argv[2]);
	nrows2 = atoi(argv[3]);
	ncols2 = atoi(argv[4]);
	nthreads = atoi(argv[5]);
	calcSum = atoi(argv[6]);
	calcMul = atoi(argv[7]);
	calcInv = atoi(argv[8]);
	calcTra = atoi(argv[9]);
	calcDet = atoi(argv[10]);
	calcEqu = atoi(argv[11]);
	execBoth = atoi(argv[12]);
	mode2Debug = atoi(argv[13]);

	if(nthreads > 8)
	{
		printf("Max number of threads supported is 8, your value was changed\n");
		nthreads = MaxThread;
	}else if(nthreads > nrows1)
	{// Prevents nthreads being higher than A matrix row
			nthreads = nrows1;
	}else if(nthreads > nrows2)
	{// Prevents nthreads being higher than B matrix row
		nthreads = nrows2;
	}

	if(!(dt = (DadosThread *) malloc(sizeof(DadosThread) * nthreads)))
	{
		printf("Failed to allocate memory\n");
		exit(EXIT_FAILURE);
	}

	if(!(threads = (pthread_t *) malloc(sizeof(pthread_t) * nthreads)))
	{
		printf("Failed to allocate memory\n");
		exit(EXIT_FAILURE);
	}

	/* execBoth modes:
	 * 0 enables the threadned execution only
	 * 1 enables the sequential execution only
	 * 2 enables the execution of the two previous modes
	 */
	switch(execBoth)
	{// Cases
		case 0:
			execThr = 1;
			execSeq = 0;
			break;
		case 1:
			execThr = 0;
			execSeq = 1;
			mode2Debug = 0;// Debug is overwritten
			break;
		case 2:
			execThr = 1;
			execSeq = 1;
			break;
	}

	if(execThr)
	{
		// Threaded matrix functions
		start_time = wtime();

		// Matrices initialization
		matrix_t *A = matrix_create_block(nrows1,ncols1); freeLaterA = 1;
		matrix_t *B = matrix_create_block(nrows2,ncols2); freeLaterB = 1;
		// If A and B are multiplicative then R shall support its result
		matrix_t *R =  (ncols1 == nrows2)? matrix_create_block(nrows1,ncols2) : matrix_create_block(nrows1,ncols1); freeLaterR = 1;
		// If A has nrows != ncols then
		matrix_t *Rt = (ncols1 == nrows1)? matrix_create_block(nrows1,ncols1) : matrix_create_block(ncols1,nrows1); freeLaterRt = 1;

		// Fill randomically the matrices
	//	matrix_randfill(A);
	//	matrix_randfill(B);

		A->data[0][0] = 1;
        A->data[0][1] = 2;
        A->data[0][2] = 2;
        A->data[0][3] = 2;
        
        A->data[1][0] = 1;
        A->data[1][1] = 4;
        A->data[1][2] = 1;
        A->data[1][3] = 2;
        
        A->data[2][0] = 1;
        A->data[2][1] = 4;
        A->data[2][2] = 5;
        A->data[2][3] = 1;
        
        A->data[3][0] = 2;
        A->data[3][1] = 3;
        A->data[3][2] = 4;
        A->data[3][3] = 5;
/*
		B->data[0][0] = 2;
		B->data[0][1] = 1;
		B->data[0][2] = 2;

		B->data[1][0] = 1;
		B->data[1][1] = 2;
		B->data[1][2] = 1;

		B->data[2][0] = 1;
		B->data[2][1] = 9;
		B->data[2][2] = 2;*/


		if(mode2Debug)
		{
			printf("Matrix A\n");
			matrix_print(A);

			printf("\nMatrix B\n");
			matrix_print(B);
			printf("\n");

			printf("Thread number = %d\n", nthreads);
		}
		// Threaded function execution control
		if(calcSum)
		{//Threaded matrix sum

			if(A->rows == B->rows && A->cols == B->cols)
			{// Determine the datablock to be processed
				int dataBlock_rows = A->rows/nthreads;
				int flagRows = 0;

				int resto = A->rows % nthreads;

				if(resto){
					flagRows = 1;
				}

				// Starting threads for threaded matrix sum
				for (int i = 0; i < nthreads; i++)
				{
					dt[i].id             = i;
					dt[i].dataBlock_rows = dataBlock_rows;
					dt[i].lastThread     = nthreads-1;
					dt[i].flagRows       = flagRows;
					dt[i].debug          = mode2Debug;
					dt[i].resto          = resto;
					dt[i].A              = A;
					dt[i].B              = B;
					dt[i].R              = R;
					dt[i].det            = 0;

					pthread_create(&threads[i], NULL, threaded_matrix_sum, (void *) (dt + i));
				}

				// Killing called threads for threaded matrix sum
				for (int i = 0; i < nthreads; i++)
				{
					pthread_join(threads[i], NULL);
				}
				free(dt);
				free(threads);

				printf("\nSum matrix R\n");
				matrix_print(R);
			}
		}

		if(calcMul)
		{//Threaded matrix multiplication
			// Starting threads for threaded matrix multiplication
			int dataBlock_rows = A->rows/nthreads;
			int flagRows = 0;

			int resto = A->rows % nthreads;

			if(resto){
				flagRows = 1;
			}
			for (int i = 0; i < nthreads; i++)
			{
				dt[i].id             = i;
				dt[i].dataBlock_rows = dataBlock_rows;
				dt[i].lastThread     = nthreads-1;
				dt[i].flagRows       = flagRows;
				dt[i].debug          = mode2Debug;
				dt[i].resto          = resto;
				dt[i].A              = A;
				dt[i].B              = B;
				dt[i].R              = R;
				dt[i].det            = 0;
				pthread_create(&threads[i], NULL, threaded_matrix_multiply, (void *) (dt + i));
			}

			// Killing called threads for threaded matrix sum
			for (int i = 0; i < nthreads; i++)
			{
				pthread_join(threads[i], NULL);
			}
			free(dt);
			free(threads);

			printf("\nProduct matrix R\n");
			matrix_print(R);
		}

		if(calcInv)
		{//Threaded matrix inversion
			// Starting threads for threaded matrix inversion
			for (int i = 0; i < nthreads; i++)
			{
				dt[i].id    = i;
				dt[i].A     = A;
				dt[i].B     = B;
				dt[i].R     = R;
				dt[i].debug = mode2Debug;
				dt[i].p     = &matrix_create_block;
				dt[i].p2    = &matrix_destroy_block;
				//TODO implement it (depends upon determinant)
				pthread_create(&threads[i], NULL, threaded_matrix_inversion, (void *) (dt + i));
			}

			// Killing called threads for threaded matrix inversion
			for (int i = 0; i < nthreads; i++)
			{
				pthread_join(threads[i], NULL);
			}
			free(dt);
			free(threads);

			printf("\nInverse matrix R\n");
			matrix_print(R);
		}

		if(calcTra)
		{//Threaded matrix transpose
			// Starting threads for threaded matrix transpose
			int dataBlock_rows = A->rows/nthreads;
			int flagRows = 0;

			int resto = A->rows % nthreads;

			if(resto){
				flagRows = 1;
			}

			for (int i = 0; i < nthreads; i++)
			{
				dt[i].id             = i;
				dt[i].dataBlock_rows = dataBlock_rows;
				dt[i].lastThread     = nthreads-1;
				dt[i].flagRows       = flagRows;
				dt[i].debug          = mode2Debug;
				dt[i].resto          = resto;
				dt[i].A              = A;
				dt[i].Rt             = Rt;
				dt[i].det            = 0;
				pthread_create(&threads[i], NULL, threaded_matrix_transpose, (void *) (dt + i));
			}

			// Killing called threads for threaded matrix transpose
			for (int i = 0; i < nthreads; i++)
			{
				pthread_join(threads[i], NULL);
			}
			free(dt);
			free(threads);

			printf("\nTranspose matrix Rt\n");
			matrix_print(Rt);
		}

		if(calcDet)
		{//Threaded matrix determinant
			// Starting threads for threaded matrix determinant
		    
			int dataBlock_rows = A->rows/nthreads;
			int flagRows = 0;

			int resto = A->rows % nthreads;

			if(resto){
				flagRows = 1;
			}
			
			double detU = 1;
			double detL = 1;
			det = 0;
			
			for (int i = 0; i < nthreads; i++)
			{
				dt[i].id             = i;
				dt[i].A              = A;
				dt[i].dataBlock_rows = dataBlock_rows;
				dt[i].lastThread     = nthreads-1;
				dt[i].flagRows       = flagRows;
				dt[i].resto          = resto;
				dt[i].det            = &det;
				dt[i].detU           = &detU;
				dt[i].detL           = &detL;
				dt[i].debug          = mode2Debug;
				//TODO how to implement the split of chunks to the threads
				pthread_create(&threads[i], NULL, threaded_matrix_determinant, (void *) (dt + i));
			}
			
			
			// Killing called threads for threaded matrix determinant
			for (int i = 0; i < nthreads; i++)
			{
				pthread_join(threads[i], NULL);
			}
			
			free(dt);
			free(threads);

			
			
			matrix_print(A);
			printf("DET-L = %lf\n", detL);
			printf("DET-U = %lf\n", detU);
			printf("\nDeterminant result: %lf\n", det);
		}

		if(calcEqu && A->rows == B->rows && A->cols == A->rows)
		{//Threaded matrix equality
			// Starting threads for threaded matrix equality
			equal = 1;
			int dataBlock_rows = A->rows/nthreads;
			int flagRows = 0;

			int resto = A->rows % nthreads;

			if(resto){
				flagRows = 1;
			}

			for (int i = 0; i < nthreads; i++)
			{
				dt[i].id             = i;
				dt[i].dataBlock_rows = dataBlock_rows;
				dt[i].lastThread     = nthreads-1;
				dt[i].flagRows       = flagRows;
				dt[i].debug          = mode2Debug;
				dt[i].resto          = resto;
				dt[i].A              = A;
				dt[i].B              = B;
				dt[i].R              = R;
				dt[i].equal          = &equal;
				dt[i].det            = 0;
				pthread_create(&threads[i], NULL, threaded_matrix_equal, (void *) (dt + i));
			}

			// Killing called threads for threaded matrix equality
			for (int i = 0; i < nthreads; i++)
			{
				pthread_join(threads[i], NULL);
			}
			free(dt);
			free(threads);

			(equal)? printf("\nMatrices are equal\n") : printf("\nMatrices are not equal\n");
		}

		// Free used matrices
		if (freeLaterA) matrix_destroy_block(A);
		if (freeLaterB) matrix_destroy_block(B);
		if (freeLaterR) matrix_destroy_block(R);
		if (freeLaterRt) matrix_destroy_block(Rt);

		end_time = wtime();
		timeThreaded = end_time-start_time;
		printf("Time taken by threaded functions executed: %lf\n", timeThreaded);
	}

	if(execSeq)
	{
		if(execThr && mode2Debug)
			printf("_____________________________________________\n");

		/* Sequential matrix functions */
		start_time = wtime();

		// Matrices initialization
		matrix_t *A = matrix_create_block(nrows1,ncols1); freeLaterA = 1;
		matrix_t *B = matrix_create_block(nrows2,ncols2); freeLaterB = 1;
		// If A and B are multiplicative then R shall support its result
		matrix_t *R = (ncols1 == nrows2)? matrix_create_block(nrows1,ncols2) : matrix_create_block(nrows1,ncols1); freeLaterR = 1;
		// If A has nrows != ncols then
		matrix_t *Rt = (ncols1 == nrows1)? matrix_create_block(nrows1,ncols1) : matrix_create_block(ncols1,nrows1); freeLaterRt = 1;

		//matrix_randfill(A);
		//matrix_randfill(B);
		
		A->data[0][0] = 1;
        A->data[0][1] = 2;
        A->data[0][2] = 2;
        A->data[0][3] = 2;
        
        A->data[1][0] = 1;
        A->data[1][1] = 4;
        A->data[1][2] = 1;
        A->data[1][3] = 2;
        
        A->data[2][0] = 1;
        A->data[2][1] = 4;
        A->data[2][2] = 5;
        A->data[2][3] = 1;
        
        A->data[3][0] = 2;
        A->data[3][1] = 3;
        A->data[3][2] = 4;
        A->data[3][3] = 5;
/*
		B->data[2][2] = 2;*/
		if(!execThr)
		{
			printf("Matrix A\n");
			matrix_print(A);

			printf("\nMatrix B\n");
			matrix_print(B);
			printf("\n");
		}

		// Sequential function execution control

		if(calcSum)
		{//Sequential matrix sum
			R = matrix_sum(A,B,matrix_create_block);

			printf("\nSum matrix R\n");
			matrix_print(R);
		}

		if(calcMul)
		{//Sequential matrix multiplication
			R = matrix_multiply(A,B,matrix_create_block);

			printf("\nProduct matrix R\n");
			matrix_print(R);
		}

		// There only exists inverse for symmetric matrices
		if(calcInv && A->rows == B->rows && A->cols == A->rows)
		{//Sequential matrix inversion
			R = matrix_inversion(A, matrix_create_block_init, matrix_destroy_block);

			printf("\nInverse matrix R\n");
			matrix_print(R);
		}

		if(calcTra)
		{//Sequential matrix transpose
			Rt = matrix_transpose(A,matrix_create_block);

			printf("\nTranspose matrix Rt\n");
			matrix_print(Rt);
		}

		// There only exists determinant for symmetric matrices
		if(calcDet && A->rows == A->cols)
		{//Sequential matrix determinant
		//	det = matrix_determinant2(A, matrix_create_block_init, matrix_destroy_block);

		det = matrix_determinant(A->rows, A->data);

			printf("\nDeterminant result: %lf\n", det);
		}

		if(calcEqu)
		{//Sequential matrix equality
			(matrix_equal(A,B))? printf("\nMatrices are equal\n") : printf("\nMatrices are not equal\n");
		}

		// Free used matrices
		if (freeLaterA)  matrix_destroy_block(A);
		if (freeLaterB)  matrix_destroy_block(B);
		if (freeLaterR)  matrix_destroy_block(R);
		if (freeLaterRt) matrix_destroy_block(Rt);

		end_time = wtime();
		timeSequential = end_time-start_time;
		printf("Time taken by sequential functions executed: %lf\n", timeSequential);
	}

	if(execBoth == 2)
	{
		printf("\n\nTime taken by:"
		"\nThreaded functions executed: %lf"
		"\nSequential functions executed: %lf\n", timeThreaded, timeSequential);
	}

	fflush(stdout);
	return EXIT_SUCCESS;
}
