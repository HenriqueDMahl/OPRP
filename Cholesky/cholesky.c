/*****************************************************
 * Site: https://rosettacode.org/wiki/Cholesky_decomposition
 *****************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

void rand_fill(double** x, int n){
  int i, j;
  #pragma omp parallel private(i,j) shared(x)
  {
      for(i = 0; i < n; i++)
        for(j = 0; j < n; j++){
          x[i][j] = (i+1)*(j+1);
        }
      for(i = 0; i < n; i++)
        x[i][i] *= 100;
  }
}

void cholesky(double** A, int n) {
  int i,j,k;
  double s;
  #pragma omp parallel private(i,j,k) shared(A) num_threads(8)
  {
     // #pragma omp for
      for (i = 0; i < n; i++)
        for (j = 0; j < (i + 1); j++) {
          s = 0;
          #pragma omp for reduction(+: s)
          for (k = 0; k < j; k++)
            s += A[i][k] * A[j][k];
          A[i][j] = (i == j) ? sqrt(A[i][i] - s) : (1.0 / A[j][j] * (A[i][j] - s));
        }
  }
}

void show_matrix(double** A, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
        printf("%2.5f ", A[i][j]);
    printf("\n");
  }
}

void mat_zero(double** x, int n) {
  int i, j;
  #pragma omp parallel for private(i,j) schedule(dynamic) collapse(2)
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      x[i][j] = 0.0;
    }
  }
}

double** mat_new(int n) {
  int i;
  double** x = malloc(sizeof(double*) * n);
  assert(x != NULL);

  for(i = 0; i < n; i++){
    x[i]  = malloc(sizeof(double) * n);
    assert(x[i] != NULL);
  }

  mat_zero(x,n);

  return x;
}

void mat_gen(double** s, int n) {
  int i,j;
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      scanf("%lf",&s[i][j]);
    }
  }
}

void mat_del(double** x) {
  free(x[0]);
  free(x);
}

int main() {
  int n;
  fscanf(stdin,"%d",&n);
  double** A = mat_new(n);
  rand_fill(A, n);
  //#pragma omp barrier
  //mat_gen(A, n);

  cholesky(A, n);
  //show_matrix(A, n);

  mat_del(A);

  return 0;
}
