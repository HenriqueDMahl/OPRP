#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define NRA 2048                 /* number of rows in matrix A */
#define NCA 2048                 /* number of columns in matrix A */
#define NCB 2048                  /* number of columns in matrix B */

double wtime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec / 1000000.0;
}

int main (int argc, char *argv[])
{
	double start_time, end_time;
	int  i, j, k;

	float** a = malloc(sizeof(float*)*NRA);

	for(i = 0; i < NRA; i++){
		a[i] = malloc(sizeof(float)*NRA);
	}


	float** b = malloc(sizeof(float*)*NRA);
	for(i = 0; i < NRA; i++){
		b[i] = malloc(sizeof(float)*NRA);
	}


	float** c = malloc(sizeof(float*)*NRA);
	for(i = 0; i < NRA; i++){
		c[i] = malloc(sizeof(float)*NRA);
	}

	/*
	double a[NRA][NCA],           // matrix A to be multiplied /
	       b[NCA][NCB],           // matrix B to be multiplied /
	       c[NRA][NCB];           // result matrix C //
     */

    start_time = wtime();

	#pragma omp parallel private(i, j, k) shared(a, b, c)
	{
		omp_set_nested(1);
		#pragma omp for schedule(dynamic) nowait
		for (i = 0; i < NRA; i++)
			for (j = 0; j < NCA; j++){
				a[i][j] = 1;
				b[i][j] = 1;
			}

		omp_set_nested(1);
		#pragma omp for schedule(dynamic) nowait
		for (i = 0; i < NRA; i++)
			for (j = 0; j < NCB; j++)
				c[i][j] = 0;

		omp_set_nested(1);
		#pragma omp for schedule(dynamic) nowait
		for (i = 0; i < NRA; i++)
			for (j = 0; j < NCB; j++)
				for (k = 0; k < NCA; k++)
					c[i][j] += a[i][k] * b[k][j];

	}
	end_time = wtime();
	printf("%f\n",end_time - start_time);
	/*
	printf("******************************************************\n");
	printf("Result Matrix:\n");

	for (i = 0; i < NRA; i++)
		for (j = 0; j < NCB; j++)
			printf("%6.2f   ", c[i][j]);
	printf("\n");

	printf("******************************************************\n");
	printf ("Done.\n");*/
}
