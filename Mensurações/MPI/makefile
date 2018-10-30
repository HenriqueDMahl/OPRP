all:
	@$(MAKE) clean;
	@mpicc ex5.c -o app;
	@mpirun -machinefile machinefile.in -np 8 app;
	#@mpirun -np 16 ./app;

clean:
	@rm -rf app;
