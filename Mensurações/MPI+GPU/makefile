all:
	#@$(MAKE) clean;
	@nvcc mpi_+_cuda.cu -I/usr/mpi/gcc/openmpi-1.4.6/include -L/usr/mpi/gcc/openmpi-1.4.6/lib64 -lmpi -lm;
	#nvcc mpi_+_cuda.cu -lmpi -L/usr/lib64 -L/usr/bin -lutil
	mpirun -machinefile machinefile.in -np 8 a.out >> medidas_cuda_mpi;
	#@mpirun -np 4 ./app;

clean:
	@rm -rf medidas_cuda_mpi;
	@rm -rf a.out;
