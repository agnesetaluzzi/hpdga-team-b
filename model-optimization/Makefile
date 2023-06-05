CXX=nvcc
CXXFLAGS= -O3 -std=c++11

LDFLAGS=-lm -lstdc++

CXXFILES=src/gcn.cpp src/optim.cpp src/module.cu src/variable.cpp src/parser.cpp src/rand.cpp src/timer.cpp src/kernels.cu
HFILES=include/gcn.h include/optim.h include/module.h include/variable.h include/sparse.h include/parser.h include/rand.h include/timer.h include/kernels.cuh

all: gcn-seq

gcn-seq: src/main.cpp $(CXXFILES) $(HFILES)
	mkdir -p exec
	$(CXX) $(CXXFLAGS) -o exec/gcn-seq $(CXXFILES) src/main.cpp $(LDFLAGS)

clean:
	rm exec/*

