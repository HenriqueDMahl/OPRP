all:
	@mpicc mpi_example.c -o app;
	@mpirun -machinefile machinefile.in -np 10 app;

clear:
	@rm -rf app;
