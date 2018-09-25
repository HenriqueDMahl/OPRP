all:
	@mpicc exerc.c -o app;
	@mpirun -machinefile machinefile.in -np 10 app;

clear:
	@rm -rf app;
