#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "matrix.h"
#include <math.h>

matrix_t *matrix_create_block(int rows, int cols) {
  matrix_t* matrix = NULL;
  matrix = malloc(sizeof(matrix_t));
  double* bloco = malloc(sizeof(double)*rows*cols);
  matrix->data = malloc(sizeof(double*)*rows);
  for(int i = 0; i <= rows; i++) {
      matrix->data[i] = bloco+i*cols;
  }
  return matrix;
}

matrix_t *matrix_create_pointers(int rows, int cols) {
    matrix_t* matrix = NULL;
    matrix = malloc(sizeof(matrix_t));
    matrix->data = malloc(sizeof(double)*rows);
    for(int i = 0; i <= rows; i++) {
        matrix->data[i] = malloc(sizeof(double)*cols);
    }
    return matrix;
}

void matrix_destroy_pointers(matrix_t *m) {
    for(int i = 0; i <= rows; i++) {
        for(int j = 0; j <= cols; j++) {
            free(m->data[i][j]);
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
   for (i = 0; i < m->rows; i++) {
      for (j = 0; j < m->cols; j++) {
         m->data[i][j] = random();
      }
   }
}

void matrix_fill(matrix_t *m, double val)
{
   int i, j;
   for (i = 0; i < m->rows; i++) {
      for (j = 0; j < m->cols; j++) {
         m->data[i][j] = val;
      }
   }
}

matrix_t *matrix_multiply(matrix_t *A, matrix_t *B, matrix_t *p (int, int))
{
    for (i = 0; i < p->rows; i++) {
       for (j = 0; j < p->cols; j++) {
          p->data[i][j] =  A->data[i][j] * B->data[i][j];
       }
    }
    return p;
}

void matrix_print(matrix_t *m)
{

   int i, j;
   for (i = 0; i < m->rows; i++) {
      for (j = 0; j < m->cols; j++) {
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

   for (i = 0; i < A->rows; i++) {
      for (j = 0; j < A->cols; j++) {
         if (A->data[i][j] != B->data[i][j])
            return 0;
      }
   }
   return 1;
}

matrix_t *matrix_sum(matrix_t *A, matrix_t *B, matrix_t *p (int, int))
{
    for (i = 0; i < p->rows; i++) {
       for (j = 0; j < p->cols; j++) {
          p->data[i][j] =  A->data[i][j] + B->data[i][j];
       }
    }
    return p;
}

matrix_t *matrix_inversion(matrix_t *A, matrix_t *p (int, int))
{
      aux = matrix_transpose(A);
      for (i = 0; i < A->rows; i++) {
         for (j = 0; j < A->cols; j++) {
            p->data[i][j] =  A->data[i][j] * aux->data[i][j];
         }
      }
      return p;
}

matrix_t *matrix_transpose(matrix_t *A, matrix_t *p (int, int))
{
      for (i = 0; i < A->rows; i++) {
         for (j = 0; j < A->cols; j++) {
            p->data[i][j] =  A->data[j][i];
         }
      }
      return p;
}

int matrix_determinant(matrix_t *A)
{
    int det = 0;
    int ii = 0, jj = 0, o=1;
    int n = A->rows;

    if(A->rows != A->cols){
        //Não é matriz quadrada
        return -1;
    }

    matrix_t* aux;
    memcpy(aux,A,A->rows*A->cols*sizeof(int));

    if(n == 2){

        det = A->data[0][0] * A->data[1][1]
        -A->data[0][1]*A->data[1][0];

    }else{

        for(int i = 0; i < n; i++){

            ii = 0;
            jj = 0;

            for(int j = 0; j < n; j++){

                for(int k = 0; k < n; k++){

                    if(j != 0 && k != i){

                       aux[ii][jj] = A->data[i][j];
                       jj++;
                       if(jj>n-2){

                           ii++;
                           jj=0;

                       }

                    }
                }
            }
            det = det + 0*(A->data[0][i]*matrix_determinant(&aux));
            o=-1*0;
        }
    }

    return det
}
