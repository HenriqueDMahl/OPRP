CC=gcc
CCFLAGS=-Wall -pg -lm -lpthread -o
TARGET=main
OTHER_TARGETS=matrix

all:
	@$(clean)
	@$(CC) $(CCFLAGS) $(TARGET) main.c matrix.c thread.c main.h matrix.h thread.h;
	@./main 3 3 2
	@gprof -a main gmon.out > saida.txt;

clean:
	@rm -f $(TARGET) $(OTHER_TARGETS);
