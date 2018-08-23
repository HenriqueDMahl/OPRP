CC=gcc
CCFLAGS=-Wall -pg -o
TARGET=main
OTHER_TARGETS=matrix

all:
	@$(clean)
	@$(CC) $(CCFLAGS) $(TARGET) main.c matrix.c matrix.h;
	@./main 2 2;
	@gprof -a main gmon.out > saida.txt;

clean:
	@rm -f $(TARGET) $(OTHER_TARGETS);
