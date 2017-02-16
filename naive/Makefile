include ../make.common
BIN=x.qrnn
NVCC=nvcc
NVOPTS=-O3 $(ARCH) -DDEBUG

$(BIN): kernel.o
	$(NVCC) $(NVOPTS) -o $(BIN) kernel.o

kernel.o: kernel.cu
	$(NVCC) $(NVOPTS) -c kernel.cu 

clean:
	rm -rf kernel.o $(BIN)
