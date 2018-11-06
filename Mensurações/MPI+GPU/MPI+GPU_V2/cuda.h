#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

__global__ void mult(float *a, float *b, float *c, int inicio, int chunk);
