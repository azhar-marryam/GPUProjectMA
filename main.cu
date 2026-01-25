#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "gpu_kernels.h"
#include "timer.h"

namespace fs = std::filesystem;

void cpu_grayscale(const cv::Mat&, cv::Mat&);

int main() {
    std::string path = "data/images";

    Timer cpuTimer, gpuTimer;
    double cpuTime = 0.0;
    double gpuTime = 0.0;

    int count = 0;

    for (const auto& entry : fs::directory_iterator(path)) {
        cv::Mat img = cv::imread(entry.path().string());

        if (img.empty()) continue;

        count++;

        // CPU
        cv::Mat cpuOut;
        cpuTimer.start();
        cpu_grayscale(img, cpuOut);
        cpuTime += cpuTimer.stop();

        // GPU
        unsigned char *d_input, *d_output;
        size_t imgSize = img.rows * img.cols * 3;
        size_t graySize = img.rows * img.cols;

        cudaMalloc(&d_input, imgSize);
        cudaMalloc(&d_output, graySize);

        gpuTimer.start();
        cudaMemcpy(d_input, img.data, imgSize, cudaMemcpyHostToDevice);
        gpu_grayscale(d_input, d_output, img.cols, img.rows);
        cudaMemcpy(cpuOut.data, d_output, graySize, cudaMemcpyDeviceToHost);
        gpuTime += gpuTimer.stop();

        cudaFree(d_input);
        cudaFree(d_output);
    }

    std::cout << "Processed images: " << count << "\n";
    std::cout << "CPU time (ms): " << cpuTime << "\n";
    std::cout << "GPU time (ms): " << gpuTime << "\n";
    std::cout << "Speedup: " << cpuTime / gpuTime << "x\n";

    return 0;
}
