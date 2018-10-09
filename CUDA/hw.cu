#include <stdio.h>

__device__ const char *STR = "HELLO WORLD!";
const char STR_LENGHT = 12;

__global__ void hello()
{
    printf("%c\n", STR[threadIdx.x % STR_LENGHT]);
}

int main()
{
    int num_threads = STR_LENGHT;
    int num_blocks = 2;
    dim3 dimBlock (16,16);
    dim3 dimGrid (32,32);
    hello<<<1,12    >>>();
    cudaDeviceSynchronize();

    return 0;
}
