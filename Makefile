all: lfm pre test

lfm: lfm.cpp lfm.hpp
	g++ -g -O3 -fopenmp -o lfm lfm.cpp
pre: pre.cpp
	g++ -g -O3 -o pre pre.cpp
test: test.cpp
	g++ -g -O3 -o test test.cpp
clean:
	rm -f lfm pre test
