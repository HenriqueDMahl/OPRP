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
	double start_time, end_time;
	int nrows, ncols;
	int nthreads = 0;
	DadosThread *dt = NULL;
	pthread_t *threads = NULL;
	
	if ((argc != 4))
	{
		printf("Uso: %s <rows> <cols> <threads>\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	
	nrows = atoi(argv[1]);
	ncols = atoi(argv[2]);
	nthreads = atoi(argv[3]);

	if (!(dt = (DadosThread *) malloc(sizeof(DadosThread) * nthreads)))
	{
		printf("Failed to allocate memory\n");
		exit(EXIT_FAILURE);
	}

	if (!(threads = (pthread_t *) malloc(sizeof(pthread_t) * nthreads)))
	{
		printf("Failed to allocate memory\n");
		exit(EXIT_FAILURE);
	}

	start_time = wtime();

	matrix_t *A = matrix_create_block(nrows,ncols);
	matrix_t *B = matrix_create_block(nrows,ncols);
	matrix_t *R = matrix_create_block(nrows,ncols);
	double *det = malloc(sizeof(double));

	//TODO fix the random number gerator, it gens the same values for both matrices
//	matrix_randfill(A);
	matrix_randfill(B);

	A->data[0][0] = 2;
	A->data[0][1] = 1;
//	A->data[0][2] = 2;

	A->data[1][0] = 1;
	A->data[1][1] = 2;
//	A->data[1][2] = 1;

//	A->data[2][0] = 1;
//	A->data[2][1] = 9;
//	A->data[2][2] = 2;

	printf("Matrix A\n");
	matrix_print(A);

	printf("\n\nMatrix B\n");
	matrix_print(B);
	printf("\n\n");

	// Starting threads for threaded matrix sum
	for (int i = 0; i < nthreads; i++)
	{
		dt[i].id = i;
		dt[i].A = A;
		dt[i].B = B;
		dt[i].R = R;
		dt[i].det = *det;
		pthread_create(&threads[i], NULL, call_threaded_matrix_determinant, (void *) (dt + i));
	}

	// Killing called threads for threaded matrix sum
	for (int i = 0; i < nthreads; i++)
	{
		pthread_join(threads[i], NULL);
	}
	free(dt);
	free(threads);

/*
	// Starting threads for threaded matrix inversion
	for (int i = 0; i < nthreads; i++)
	{
		dt[i].id = i;
		dt[i].A = A;
		dt[i].B = B;
		dt[i].R = R;
		//TODO implement and find where is missing a definition to "call_threaded_matrix_inversion"
//		pthread_create(&threads[i], NULL, call_threaded_matrix_inversion, (void *) (dt + i));
	}

	// Killing called threads for threaded matrix inversion
	for (int i = 0; i < nthreads; i++)
	{
		pthread_join(threads[i], NULL);
	}
	free(dt);
	free(threads);

	// Starting threads for threaded matrix transpose
	for (int i = 0; i < nthreads; i++)
	{
		dt[i].id = i;
		dt[i].A = A;
		dt[i].B = B;
		dt[i].R = R;
		//TODO implement
//		pthread_create(&threads[i], NULL, call_threaded_matrix_transpose, (void *) (dt + i));
	}

	// Killing called threads for threaded matrix transpose
	for (int i = 0; i < nthreads; i++)
	{
		pthread_join(threads[i], NULL);
	}
	free(dt);
	free(threads);

	// Starting threads for threaded matrix determinant
	for (int i = 0; i < nthreads; i++)
	{
		dt[i].id = i;
		dt[i].A = A;
		dt[i].B = B;
		dt[i].R = R;
		//TODO implement
//		pthread_create(&threads[i], NULL, call_threaded_matrix_sum, (void *) (dt + i));
	}

	// Killing called threads for threaded matrix determinant
	for (int i = 0; i < nthreads; i++)
	{
		pthread_join(threads[i], NULL);
	}
	free(dt);
	free(threads);
*/
	printf("Matrix R\n");
	matrix_print(R);
	printf("\n\n");
	printf("Determinante\n");
	printf("%lf",det);

//	int det = 0, ig = 0;

//	ig = matrix_equal(A,B);

//	printf("%d \n", det);
//	printf("%d \n", ig);

	matrix_destroy_block(A);
	matrix_destroy_block(B);
	matrix_destroy_block(R);

	end_time = wtime();

//	printf("%d %d %f\n", nrows, ncols, end_time - start_time);
	fflush(stdout);

	return EXIT_SUCCESS;
}
