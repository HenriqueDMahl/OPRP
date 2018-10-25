CC=gcc
CCFLAGS=-Wall -fopenmp -o
TARGET=example4
OTHER_TARGETS=example4

all:
	@$(MAKE) clean;
	@$(CC) example4.c $(CCFLAGS) $(TARGET);
	@./example4 >> medidas_opm_2048;

clean:
	@rm medidas_opm_2048;
	@rm -f $(TARGET) $(OTHER_TARGETS);
