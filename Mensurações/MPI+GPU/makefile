all:
	@$(MAKE) clean;
	@mpicc ex5.c -o app;
	#@mpirun -machinefile machinefile.in -np 2 app;
	@mpirun -np 4 ./app;

clean:
	@rm -rf app;
