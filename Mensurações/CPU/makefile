CC=gcc
CCFLAGS=-Wall -fopenmp -o
TARGET=example4
OTHER_TARGETS=example4

all:
	@$(MAKE) clean;
	@$(CC) example4.c $(CCFLAGS) $(TARGET)
	@./example4

clean:
	@rm -f $(TARGET) $(OTHER_TARGETS);
