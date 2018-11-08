all:
	@mpicc -c mpi.c -o main.o
	@nvcc -arch=sm_20 -c cuda.cu -o multiply.o
	@mpicc main.o multiply.o -lcudart -L/apps/CUDA/cuda-5.0/lib64/ -o program
	@mpiexec -np 2 ./program

clean:
	@rm -rf medidas_cuda_mpi;
	@rm -rf a.out;
