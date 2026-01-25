CUDA_PATH ?= /usr/local/cuda

CXX = g++
NVCC = nvcc

CXXFLAGS = -O2 `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

TARGET = gpu_image_processing

SRC = main.cu gpu_kernels.cu cpu_filters.cpp

all:
	$(NVCC) -O2 $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
