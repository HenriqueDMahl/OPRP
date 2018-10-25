all:
	@$(MAKE) clean;
	@mpicc ex5.c -o app;
	@mpirun -machinefile machinefile.in app;
	#@mpirun -np 16 ./app;

clean:
	@rm medidas_mpi;
	@rm -rf app;
