CC=gcc
CCFLAGS=-Wall -O3 -pthread -lm -pg  -o
TARGET=main
OTHER_TARGETS=matrix

# Matrix A dimensions
Arows=3
Acols=3

# Matrix B dimensions-pthread
Brows=3
Bcols=3

# Active threads
nThreads=2

# Program execution mode
# 0 executes only threaded
# 1 executes only sequential
# 2 executes both
Mode=0

# Only when Mode=0 or 2 is selected (refers to the prints of threads' actions)
# Activate debug mode
Mode2Debug=1

# Function execution control
# Select only one at a time
M_Summm=0
M_Multi=0
M_Inver=0
M_Trasp=0
M_Deter=0
M_Equal=1

all:
	@$(clean) clear;
	@$(CC) $(CCFLAGS) $(TARGET) main.c matrix.c thread.c main.h matrix.h thread.h;
	@./main $(Arows) $(Acols) $(Brows) $(Bcols) $(nThreads) $(M_Summm) $(M_Multi) $(M_Inver) $(M_Trasp) $(M_Deter) $(M_Equal) $(Mode) $(Mode2Debug);
	@gprof -a main gmon.out > saida.txt;

clean:
	@rm -f $(TARGET) $(OTHER_TARGETS);
