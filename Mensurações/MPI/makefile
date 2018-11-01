all:
	@mpicc ex5.c;
	@mpirun -machinefile machinefile.in -np 8 a.out >> medidas_mpi;
	#@mpirun -np 8 ./app >> medidas_mpi;

clean:
	@rm -rf medidas_mpi;
	@rm -rf a.out;
