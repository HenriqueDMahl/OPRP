#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include "matrix.h"

double wtime()
{
   struct timeval t;
   gettimeofday(&t, NULL);
   return t.tv_sec + t.tv_usec / 1000000;
}

int main(int argc, char **argv)
{
   double start_time, end_time;
    int nrows, ncols;

   if ((argc != 3)) {
      printf("Uso: %s <rows> <cols>\n", argv[0]);
      exit(EXIT_FAILURE);
   }

   nrows = atoi(argv[1]); 
   ncols = atoi(argv[2]); 

   start_time = wtime();
   
   //Do something
   
   end_time = wtime();

   printf("%d %d %f\n", nrows, ncols, end_time - start_time);
   fflush(stdout);

   return EXIT_SUCCESS;
}
