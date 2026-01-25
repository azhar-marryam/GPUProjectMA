#include <cuda_runtime.h>
#include "gpu_kernels.h"

#define BLOCK 16

__global__ void grayscaleKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char b = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char r = input[idx + 2];

        output[y * width + x] =
            (unsigned char)(0.299f * r +
                             0.587f * g +
                             0.114f * b);
    }
}

void gpu_grayscale(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid(
        (width + BLOCK - 1) / BLOCK,
        (height + BLOCK - 1) / BLOCK
    );

    grayscaleKernel<<<grid, block>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
