"""
Real-Time GPU-Accelerated Image Processing Suite
Authors: Marryam Azhar, Asfa Toor
Course: IT00CG19 GPU Programming 2025
Åbo Akademi University, Turku

Revised to implement custom CUDA kernels via CuPy RawKernel and
ElementwiseKernel, demonstrating low-level GPU programming:
  - Manual thread/block/grid configuration
  - Explicit memory management and transfers
  - Custom kernel implementations for every operation
  - Shared memory tiling for Gaussian blur
  - GPU execution timing with CUDA events
"""

import numpy as np
import cupy as cp
import cv2
import time
import os
from pathlib import Path
import json
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
import psutil


# ==============================================================================
# CUSTOM CUDA KERNELS
# Each kernel is written in CUDA C and compiled at runtime via RawKernel.
# This demonstrates understanding of the CUDA execution model:
#   - Each thread computes one output pixel (or one channel of one pixel)
#   - Threads are grouped into 16x16 blocks
#   - A grid of blocks covers the entire image
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. GRAYSCALE KERNEL (ElementwiseKernel)
#    One thread per pixel. Applies the luminosity formula:
#    gray = 0.299*R + 0.587*G + 0.114*B
#    Input image is passed as three separate channel arrays.
# ------------------------------------------------------------------------------
_grayscale_kernel = cp.ElementwiseKernel(
    in_params='T r, T g, T b',
    out_params='T gray',
    operation='gray = (T)(0.299f * (float)r + 0.587f * (float)g + 0.114f * (float)b)',
    name='grayscale_luminosity'
)

# ------------------------------------------------------------------------------
# 2. GAUSSIAN BLUR KERNEL (RawKernel with shared memory tiling)
#    Uses a pre-computed Gaussian kernel passed as a device array.
#    Shared memory tile: each block loads a (TILE + 2*half) x (TILE + 2*half)
#    patch into shared memory, avoiding redundant global memory reads.
#    One thread per output pixel. Boundary pixels are handled by clamping.
# ------------------------------------------------------------------------------
_gaussian_blur_raw = cp.RawKernel(r'''
extern "C" __global__
void gaussian_blur(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const float* __restrict__ kernel,
    int width, int height, int ksize)
{
    // Shared memory tile: TILE_SIZE + 2*half border for halo
    // Maximum supported ksize = 15 (half = 7), TILE_SIZE = 16
    const int TILE = 16;
    const int half = ksize / 2;

    __shared__ float tile[30][30];  // TILE(16) + 2*max_half(7) = 30

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global coordinates of this thread's output pixel
    int out_x = blockIdx.x * TILE + tx;
    int out_y = blockIdx.y * TILE + ty;

    // Shared memory coordinates (offset by halo)
    int sm_x = tx + half;
    int sm_y = ty + half;

    // Load the central pixel into shared memory
    int cx = min(max(out_x, 0), width  - 1);
    int cy = min(max(out_y, 0), height - 1);
    tile[sm_y][sm_x] = (float)input[cy * width + cx];

    // Load halo pixels (top / bottom / left / right edges of the tile)
    // Threads in the first `half` rows/cols also load the border region
    if (tx < half) {
        int lx = min(max(out_x - half, 0), width - 1);
        tile[sm_y][tx] = (float)input[cy * width + lx];

        int rx = min(out_x + TILE, width - 1);
        tile[sm_y][tx + TILE + half] = (float)input[cy * width + rx];
    }
    if (ty < half) {
        int ty_ = min(max(out_y - half, 0), height - 1);
        tile[ty][sm_x] = (float)input[ty_ * width + cx];

        int by = min(out_y + TILE, height - 1);
        tile[ty + TILE + half][sm_x] = (float)input[by * width + cx];
    }
    // Corner halo
    if (tx < half && ty < half) {
        int lx = min(max(out_x - half, 0), width  - 1);
        int uy = min(max(out_y - half, 0), height - 1);
        int rx = min(out_x + TILE, width  - 1);
        int by = min(out_y + TILE, height - 1);
        tile[ty][tx]                         = (float)input[uy * width + lx];
        tile[ty][tx + TILE + half]           = (float)input[uy * width + rx];
        tile[ty + TILE + half][tx]           = (float)input[by * width + lx];
        tile[ty + TILE + half][tx + TILE + half] = (float)input[by * width + rx];
    }

    __syncthreads();

    // Only compute output for valid pixels
    if (out_x >= width || out_y >= height) return;

    float sum = 0.0f;
    for (int ky = 0; ky < ksize; ky++) {
        for (int kx = 0; kx < ksize; kx++) {
            sum += tile[sm_y + ky - half][sm_x + kx - half]
                 * kernel[ky * ksize + kx];
        }
    }
    output[out_y * width + out_x] = (unsigned char)min(max(sum, 0.0f), 255.0f);
}
''', 'gaussian_blur')

# ------------------------------------------------------------------------------
# 3. SOBEL EDGE DETECTION KERNEL (RawKernel)
#    One thread per pixel. Computes Gx and Gy in a single pass, then
#    outputs gradient magnitude = sqrt(Gx^2 + Gy^2).
#    Border pixels are set to 0 (no valid neighbourhood).
# ------------------------------------------------------------------------------
_sobel_raw = cp.RawKernel(r'''
extern "C" __global__
void sobel_edge(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Border pixels: no full 3x3 neighbourhood
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    // Sobel Gx kernel:  [-1  0  1]   Gy kernel:  [-1 -2 -1]
    //                   [-2  0  2]               [ 0  0  0]
    //                   [-1  0  1]               [ 1  2  1]
    float gx =
        -1.0f * input[(y-1)*width + (x-1)] + 1.0f * input[(y-1)*width + (x+1)]
        -2.0f * input[ y   *width + (x-1)] + 2.0f * input[ y   *width + (x+1)]
        -1.0f * input[(y+1)*width + (x-1)] + 1.0f * input[(y+1)*width + (x+1)];

    float gy =
        -1.0f * input[(y-1)*width + (x-1)] - 2.0f * input[(y-1)*width + x] - 1.0f * input[(y-1)*width + (x+1)]
        +1.0f * input[(y+1)*width + (x-1)] + 2.0f * input[(y+1)*width + x] + 1.0f * input[(y+1)*width + (x+1)];

    float mag = sqrtf(gx * gx + gy * gy);
    output[y * width + x] = (unsigned char)min(mag, 255.0f);
}
''', 'sobel_edge')

# ------------------------------------------------------------------------------
# 4. SEPIA FILTER KERNEL (RawKernel)
#    One thread per pixel. Applies the 3x3 sepia colour-matrix transformation
#    to each BGR pixel in a single kernel launch (no reshape, no cp.dot).
#    Output channels:
#      out_R = 0.393*R + 0.769*G + 0.189*B
#      out_G = 0.349*R + 0.686*G + 0.168*B
#      out_B = 0.272*R + 0.534*G + 0.131*B
#    Note: input is stored in BGR order (OpenCV convention).
# ------------------------------------------------------------------------------
_sepia_raw = cp.RawKernel(r'''
extern "C" __global__
void sepia_filter(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    // OpenCV stores as BGR
    float b = (float)input[idx + 0];
    float g = (float)input[idx + 1];
    float r = (float)input[idx + 2];

    float out_r = 0.393f * r + 0.769f * g + 0.189f * b;
    float out_g = 0.349f * r + 0.686f * g + 0.168f * b;
    float out_b = 0.272f * r + 0.534f * g + 0.131f * b;

    output[idx + 0] = (unsigned char)min(out_b, 255.0f);
    output[idx + 1] = (unsigned char)min(out_g, 255.0f);
    output[idx + 2] = (unsigned char)min(out_r, 255.0f);
}
''', 'sepia_filter')

# ------------------------------------------------------------------------------
# 5. SHARPEN KERNEL (RawKernel)
#    One thread per pixel per channel. Applies the 3x3 unsharp-mask kernel:
#         [ 0  -1   0]
#         [-1   5  -1]
#         [ 0  -1   0]
#    Border pixels are clamped to the nearest valid pixel (edge padding).
#    Processes all three colour channels in a single kernel, avoiding the
#    Python-level channel loop present in the original code.
# ------------------------------------------------------------------------------
_sharpen_raw = cp.RawKernel(r'''
extern "C" __global__
void sharpen(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Clamp-to-edge neighbours
    int xm = max(x - 1, 0),       xp = min(x + 1, width  - 1);
    int ym = max(y - 1, 0),       yp = min(y + 1, height - 1);

    for (int c = 0; c < 3; c++) {
        float val =
             5.0f * (float)input[(y  * width + x ) * 3 + c]
            -1.0f * (float)input[(ym * width + x ) * 3 + c]
            -1.0f * (float)input[(yp * width + x ) * 3 + c]
            -1.0f * (float)input[(y  * width + xm) * 3 + c]
            -1.0f * (float)input[(y  * width + xp) * 3 + c];

        output[(y * width + x) * 3 + c] =
            (unsigned char)min(max(val, 0.0f), 255.0f);
    }
}
''', 'sharpen')

# ------------------------------------------------------------------------------
# 6. BRIGHTNESS / CONTRAST KERNEL (ElementwiseKernel)
#    One thread per element (flattened pixel array).
#    output = clamp(input * contrast + brightness, 0, 255)
# ------------------------------------------------------------------------------
_brightness_contrast_kernel = cp.ElementwiseKernel(
    in_params='T x, float32 contrast, float32 brightness',
    out_params='T y',
    operation='''
        float v = (float)x * contrast + brightness;
        v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
        y = (T)v;
    ''',
    name='brightness_contrast'
)


# ==============================================================================
# HELPER UTILITIES
# ==============================================================================

def _make_gaussian_kernel(ksize: int, sigma: float) -> cp.ndarray:
    """
    Build a normalised 2-D Gaussian kernel on the GPU.
    The kernel is separable, but we store it as a flat ksize*ksize array
    to keep the indexing in the CUDA code simple.
    """
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    return cp.asarray(k.flatten())


def _grid_block_2d(width: int, height: int, tile: int = 16):
    """Return (grid, block) tuples for a 2-D kernel covering width x height."""
    block = (tile, tile, 1)
    grid  = ((width  + tile - 1) // tile,
             (height + tile - 1) // tile,
             1)
    return grid, block


# ==============================================================================
# DATACLASS
# ==============================================================================

@dataclass
class PerformanceMetrics:
    """Store performance metrics for analysis"""
    operation:  str
    cpu_time:   float   # milliseconds
    gpu_time:   float   # milliseconds
    speedup:    float
    image_size: Tuple[int, int]
    memory_used: float  # MB


# ==============================================================================
# MAIN PROCESSOR CLASS
# ==============================================================================

class GPUImageProcessor:
    """
    GPU-accelerated image processing system.
    Every GPU operation is implemented as a custom CUDA kernel
    (RawKernel or ElementwiseKernel) — no high-level CuPy/scipy calls.
    """

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.mempool = cp.get_default_memory_pool()

    def measure_memory(self) -> float:
        """GPU memory in use (MB)."""
        return self.mempool.used_bytes() / (1024 ** 2)

    # ------------------------------------------------------------------
    # GPU OPERATIONS (custom kernels)
    # ------------------------------------------------------------------

    @staticmethod
    def grayscale_gpu(image: cp.ndarray) -> cp.ndarray:
        """
        Grayscale via ElementwiseKernel.
        Each CUDA thread processes one pixel: gray = 0.299R + 0.587G + 0.114B.
        Input must be H x W x 3 (BGR). Returns H x W uint8 array.
        """
        if len(image.shape) == 2:
            return image
        # Split channels (contiguous views — no copy)
        b = cp.ascontiguousarray(image[:, :, 0])
        g = cp.ascontiguousarray(image[:, :, 1])
        r = cp.ascontiguousarray(image[:, :, 2])
        return _grayscale_kernel(r, g, b)

    @staticmethod
    def grayscale_cpu(image: np.ndarray) -> np.ndarray:
        """CPU reference: luminosity grayscale."""
        if len(image.shape) == 2:
            return image
        return np.dot(image[..., :3].astype(np.float32),
                      np.array([0.114, 0.587, 0.299], dtype=np.float32)).astype(np.uint8)

    # ------------------------------------------------------------------

    @staticmethod
    def gaussian_blur_gpu(image: cp.ndarray,
                          ksize: int = 5,
                          sigma: float = 1.0) -> cp.ndarray:
        """
        Gaussian blur via shared-memory tiled RawKernel.
        Processes each channel independently; one thread = one output pixel.

        Thread/block layout:
          block = (16, 16)
          grid  = (ceil(W/16), ceil(H/16))

        Shared memory tile size: (16 + 2*half) x (16 + 2*half)
        Maximum supported ksize = 15 (constrained by shared mem declaration).
        """
        assert ksize % 2 == 1 and ksize <= 15, "ksize must be odd and <= 15"
        kern = _make_gaussian_kernel(ksize, sigma)

        def _blur_channel(ch: cp.ndarray) -> cp.ndarray:
            h, w = ch.shape
            src = cp.ascontiguousarray(ch.astype(cp.uint8))
            dst = cp.zeros((h, w), dtype=cp.uint8)
            grid, block = _grid_block_2d(w, h)
            _gaussian_blur_raw(grid, block, (src, dst, kern,
                                             np.int32(w), np.int32(h),
                                             np.int32(ksize)))
            cp.cuda.Stream.null.synchronize()
            return dst

        if len(image.shape) == 3:
            channels = [_blur_channel(image[:, :, c]) for c in range(image.shape[2])]
            return cp.stack(channels, axis=2)
        else:
            return _blur_channel(image)

    @staticmethod
    def gaussian_blur_cpu(image: np.ndarray,
                          ksize: int = 5,
                          sigma: float = 1.0) -> np.ndarray:
        """CPU reference using OpenCV."""
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # ------------------------------------------------------------------

    @staticmethod
    def sobel_edge_detection_gpu(image: cp.ndarray) -> cp.ndarray:
        """
        Sobel edge detection via RawKernel.
        Input is converted to float32 grayscale first (using the grayscale
        ElementwiseKernel), then one thread computes gradient magnitude for
        each pixel in a single pass (both Gx and Gy in the same thread).

        Thread/block layout:
          block = (16, 16)
          grid  = (ceil(W/16), ceil(H/16))
        """
        # Convert to grayscale float32 on the GPU
        if len(image.shape) == 3:
            gray = GPUImageProcessor.grayscale_gpu(image).astype(cp.float32)
        else:
            gray = image.astype(cp.float32)

        h, w = gray.shape
        src = cp.ascontiguousarray(gray)
        dst = cp.zeros((h, w), dtype=cp.uint8)

        grid, block = _grid_block_2d(w, h)
        _sobel_raw(grid, block, (src, dst, np.int32(w), np.int32(h)))
        cp.cuda.Stream.null.synchronize()
        return dst

    @staticmethod
    def sobel_edge_detection_cpu(image: np.ndarray) -> np.ndarray:
        """CPU reference using OpenCV Sobel."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.clip(np.sqrt(gx ** 2 + gy ** 2), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------

    @staticmethod
    def sepia_filter_gpu(image: cp.ndarray) -> cp.ndarray:
        """
        Sepia tone via RawKernel.
        Each thread processes one pixel: reads BGR, applies the 3x3 sepia
        matrix, writes clamped BGR output. No reshape or cp.dot involved.

        Thread/block layout:
          block = (16, 16)
          grid  = (ceil(W/16), ceil(H/16))
        """
        if len(image.shape) == 2:
            image = cp.stack([image] * 3, axis=2)

        h, w = image.shape[:2]
        src = cp.ascontiguousarray(image)
        dst = cp.zeros_like(src)

        grid, block = _grid_block_2d(w, h)
        _sepia_raw(grid, block, (src, dst, np.int32(w), np.int32(h)))
        cp.cuda.Stream.null.synchronize()
        return dst

    @staticmethod
    def sepia_filter_cpu(image: np.ndarray) -> np.ndarray:
        """CPU reference sepia."""
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        mat = np.array([[0.131, 0.168, 0.189],
                        [0.534, 0.686, 0.769],
                        [0.272, 0.349, 0.393]], dtype=np.float32)
        h, w, c = image.shape
        result = image.reshape(-1, c).astype(np.float32) @ mat.T
        return np.clip(result, 0, 255).astype(np.uint8).reshape(h, w, c)

    # ------------------------------------------------------------------

    @staticmethod
    def sharpen_gpu(image: cp.ndarray) -> cp.ndarray:
        """
        Unsharp-mask sharpening via RawKernel.
        One thread handles all three channels of one output pixel, applying
        the 3x3 kernel [0,-1,0 / -1,5,-1 / 0,-1,0] with clamp-to-edge padding.
        Replaces the original Python-level pixel loops (which were not GPU
        parallel and ran slower than CPU).

        Thread/block layout:
          block = (16, 16)
          grid  = (ceil(W/16), ceil(H/16))
        """
        if len(image.shape) == 2:
            image = cp.stack([image] * 3, axis=2)

        h, w = image.shape[:2]
        src = cp.ascontiguousarray(image)
        dst = cp.zeros_like(src)

        grid, block = _grid_block_2d(w, h)
        _sharpen_raw(grid, block, (src, dst, np.int32(w), np.int32(h)))
        cp.cuda.Stream.null.synchronize()
        return dst

    @staticmethod
    def sharpen_cpu(image: np.ndarray) -> np.ndarray:
        """CPU reference sharpening via OpenCV filter2D."""
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        return np.clip(cv2.filter2D(image.astype(np.float32), -1, kernel),
                       0, 255).astype(np.uint8)

    # ------------------------------------------------------------------

    @staticmethod
    def brightness_contrast_gpu(image: cp.ndarray,
                                 brightness: float = 0.0,
                                 contrast: float = 1.0) -> cp.ndarray:
        """
        Brightness/contrast adjustment via ElementwiseKernel.
        output[i] = clamp(input[i] * contrast + brightness, 0, 255)
        One CUDA thread per element (all channels flattened).
        """
        src = image.astype(cp.uint8)
        contrast_gpu   = cp.float32(contrast)
        brightness_gpu = cp.float32(brightness)
        return _brightness_contrast_kernel(src, contrast_gpu, brightness_gpu)

    @staticmethod
    def brightness_contrast_cpu(image: np.ndarray,
                                 brightness: float = 0.0,
                                 contrast: float = 1.0) -> np.ndarray:
        """CPU reference brightness/contrast."""
        result = image.astype(np.float32) * contrast + brightness
        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # BATCH PROCESSING
    # ------------------------------------------------------------------

    def process_batch_gpu(self, images: List[np.ndarray],
                          operation: str) -> List[np.ndarray]:
        """
        Batch process multiple images on GPU using custom kernels.
        Images are transferred to device memory one at a time, processed,
        and the result is transferred back.
        """
        results = []
        for img in images:
            gpu_img = cp.asarray(img)

            if operation == 'grayscale':
                result = self.grayscale_gpu(gpu_img)
            elif operation == 'blur':
                result = self.gaussian_blur_gpu(gpu_img)
            elif operation == 'edge':
                result = self.sobel_edge_detection_gpu(gpu_img)
            elif operation == 'sepia':
                result = self.sepia_filter_gpu(gpu_img)
            elif operation == 'sharpen':
                result = self.sharpen_gpu(gpu_img)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            results.append(cp.asnumpy(result))
        return results

    # ------------------------------------------------------------------
    # BENCHMARKING
    # ------------------------------------------------------------------

    def benchmark_operation(self, image: np.ndarray,
                             operation: str,
                             iterations: int = 10) -> PerformanceMetrics:
        """
        CPU vs GPU benchmark.
        GPU timing uses CUDA events (cp.cuda.Event) for accurate measurement,
        avoiding Python-level timing overhead.
        A 3-iteration warm-up ensures JIT compilation is not included.
        """
        print(f"\nBenchmarking: {operation}  |  image: {image.shape}")

        gpu_image = cp.asarray(image)

        def _run_gpu(img):
            if operation == 'grayscale':
                return self.grayscale_gpu(img)
            elif operation == 'blur':
                return self.gaussian_blur_gpu(img)
            elif operation == 'edge':
                return self.sobel_edge_detection_gpu(img)
            elif operation == 'sepia':
                return self.sepia_filter_gpu(img)
            elif operation == 'sharpen':
                return self.sharpen_gpu(img)

        def _run_cpu(img):
            if operation == 'grayscale':
                return self.grayscale_cpu(img)
            elif operation == 'blur':
                return self.gaussian_blur_cpu(img)
            elif operation == 'edge':
                return self.sobel_edge_detection_cpu(img)
            elif operation == 'sepia':
                return self.sepia_filter_cpu(img)
            elif operation == 'sharpen':
                return self.sharpen_cpu(img)

        # Warm-up (also triggers RawKernel JIT compilation)
        for _ in range(3):
            _run_gpu(gpu_image)
            _run_cpu(image)
        cp.cuda.Stream.null.synchronize()

        # GPU timing with CUDA events
        gpu_times = []
        for _ in range(iterations):
            start_ev = cp.cuda.Event()
            end_ev   = cp.cuda.Event()
            start_ev.record()
            _run_gpu(gpu_image)
            end_ev.record()
            end_ev.synchronize()
            gpu_times.append(cp.cuda.get_elapsed_time(start_ev, end_ev))  # ms

        # CPU timing
        cpu_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _run_cpu(image)
            cpu_times.append((time.perf_counter() - t0) * 1000)  # ms

        gpu_time_ms = float(np.mean(gpu_times))
        cpu_time_ms = float(np.mean(cpu_times))
        speedup     = cpu_time_ms / gpu_time_ms
        memory_used = self.measure_memory()

        metrics = PerformanceMetrics(
            operation=operation,
            cpu_time=cpu_time_ms,
            gpu_time=gpu_time_ms,
            speedup=speedup,
            image_size=image.shape[:2],
            memory_used=memory_used,
        )
        self.metrics.append(metrics)

        print(f"  CPU: {cpu_time_ms:.2f} ms  |  GPU: {gpu_time_ms:.2f} ms  "
              f"|  Speedup: {speedup:.2f}x  |  GPU Mem: {memory_used:.2f} MB")
        return metrics

    # ------------------------------------------------------------------
    # REPORTING
    # ------------------------------------------------------------------

    def generate_performance_report(self, output_dir: str = "results"):
        """Generate performance plots and JSON metrics file."""
        Path(output_dir).mkdir(exist_ok=True)

        operations = [m.operation  for m in self.metrics]
        cpu_times  = [m.cpu_time   for m in self.metrics]
        gpu_times  = [m.gpu_time   for m in self.metrics]
        speedups   = [m.speedup    for m in self.metrics]
        memories   = [m.memory_used for m in self.metrics]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        x = np.arange(len(operations))
        w = 0.35

        # Plot 1: Execution time comparison
        axes[0, 0].bar(x - w/2, cpu_times, w, label='CPU', color='coral')
        axes[0, 0].bar(x + w/2, gpu_times, w, label='GPU (custom kernel)', color='skyblue')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(operations, rotation=45)
        axes[0, 0].set_xlabel('Operation')
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].set_title('CPU vs GPU Execution Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Speedup
        axes[0, 1].bar(x, speedups, color='green', alpha=0.7)
        axes[0, 1].axhline(y=1, color='r', linestyle='--', label='Baseline (1x)')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(operations, rotation=45)
        axes[0, 1].set_xlabel('Operation')
        axes[0, 1].set_ylabel('Speedup (x)')
        axes[0, 1].set_title('GPU Speedup over CPU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Memory
        axes[1, 0].plot(x, memories, marker='o', linewidth=2,
                        markersize=8, color='purple')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(operations, rotation=45)
        axes[1, 0].set_xlabel('Operation')
        axes[1, 0].set_ylabel('GPU Memory (MB)')
        axes[1, 0].set_title('GPU Memory Usage')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Throughput
        throughput = [1000.0 / t for t in gpu_times]
        axes[1, 1].bar(x, throughput, color='orange', alpha=0.7)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(operations, rotation=45)
        axes[1, 1].set_xlabel('Operation')
        axes[1, 1].set_ylabel('Operations / Second')
        axes[1, 1].set_title('GPU Kernel Throughput')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_analysis.png", dpi=300,
                    bbox_inches='tight')
        plt.close()

        metrics_dict = {
            'summary': {
                'average_speedup': float(np.mean(speedups)),
                'max_speedup':     float(np.max(speedups)),
                'min_speedup':     float(np.min(speedups)),
                'total_operations': len(self.metrics),
            },
            'detailed_metrics': [
                {
                    'operation':    m.operation,
                    'cpu_time_ms':  m.cpu_time,
                    'gpu_time_ms':  m.gpu_time,
                    'speedup':      m.speedup,
                    'image_size':   list(m.image_size),
                    'memory_mb':    m.memory_used,
                }
                for m in self.metrics
            ],
        }
        with open(f"{output_dir}/metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"\nPerformance report saved to {output_dir}/")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 65)
    print("GPU-Accelerated Image Processing Suite")
    print("Åbo Akademi University — IT00CG19 GPU Programming 2025")
    print("Custom CUDA Kernels (RawKernel / ElementwiseKernel)")
    print("=" * 65)

    # Print GPU info
    device = cp.cuda.Device(0)
    props  = cp.cuda.runtime.getDeviceProperties(device.id)
    print(f"\nGPU: {props['name'].decode()}")
    print(f"Compute capability: {props['major']}.{props['minor']}")
    print(f"Total memory: {props['totalGlobalMem'] / 2**30:.1f} GB")

    processor  = GPUImageProcessor()
    output_dir = "results"
    Path(output_dir).mkdir(exist_ok=True)

    # Load dataset
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"\nError: '{dataset_dir}' not found — run generate_dataset.py first.")
        return

    image_files = (list(Path(dataset_dir).glob("*.jpg")) +
                   list(Path(dataset_dir).glob("*.png")))
    if not image_files:
        print("No images found in dataset directory.")
        return

    print(f"\nFound {len(image_files)} images.")

    # Pick the largest image for benchmarking — GPU advantage only becomes
    # clear on large images where parallelism outweighs kernel launch overhead.
    def _image_pixels(p):
        img = cv2.imread(str(p))
        return img.shape[0] * img.shape[1] if img is not None else 0

    largest_file = max(image_files, key=_image_pixels)
    test_image   = cv2.imread(str(largest_file))
    print(f"Using largest image for benchmarks: {largest_file.name}  {test_image.shape}")

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PERFORMANCE BENCHMARKING  (custom CUDA kernels vs CPU)")
    print("=" * 65)

    for op in ['grayscale', 'blur', 'edge', 'sepia', 'sharpen']:
        processor.benchmark_operation(test_image, op, iterations=10)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("BATCH PROCESSING  (edge detection on up to 5 images)")
    print("=" * 65)

    batch_images = [cv2.imread(str(p)) for p in image_files[:5]
                    if cv2.imread(str(p)) is not None]
    t0 = time.perf_counter()
    batch_results = processor.process_batch_gpu(batch_images, 'edge')
    batch_ms = (time.perf_counter() - t0) * 1000

    print(f"Processed {len(batch_images)} images in {batch_ms:.1f} ms  "
          f"({batch_ms / len(batch_images):.1f} ms/image)")

    for idx, result in enumerate(batch_results):
        cv2.imwrite(f"{output_dir}/batch_edge_{idx}.jpg", result)

    # ------------------------------------------------------------------
    # Visual filter comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("GENERATING VISUAL COMPARISONS")
    print("=" * 65)

    gpu_img = cp.asarray(test_image)

    gray_r   = cp.asnumpy(processor.grayscale_gpu(gpu_img))
    blur_r   = cp.asnumpy(processor.gaussian_blur_gpu(gpu_img))
    edge_r   = cp.asnumpy(processor.sobel_edge_detection_gpu(gpu_img))
    sepia_r  = cp.asnumpy(processor.sepia_filter_gpu(gpu_img))
    sharp_r  = cp.asnumpy(processor.sharpen_gpu(gpu_img))
    bright_r = cp.asnumpy(processor.brightness_contrast_gpu(
                            gpu_img, brightness=30, contrast=1.2))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    def _show(ax, img, title, cmap=None):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    _show(axes[0, 0], cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB), 'Original')
    _show(axes[0, 1], gray_r,                                       'Grayscale\n(ElementwiseKernel)', cmap='gray')
    _show(axes[0, 2], cv2.cvtColor(blur_r, cv2.COLOR_BGR2RGB),      'Gaussian Blur\n(RawKernel + shared mem)')
    _show(axes[0, 3], edge_r,                                        'Sobel Edges\n(RawKernel)', cmap='gray')
    _show(axes[1, 0], cv2.cvtColor(sepia_r,  cv2.COLOR_BGR2RGB),    'Sepia\n(RawKernel)')
    _show(axes[1, 1], cv2.cvtColor(sharp_r,  cv2.COLOR_BGR2RGB),    'Sharpen\n(RawKernel)')
    _show(axes[1, 2], cv2.cvtColor(bright_r, cv2.COLOR_BGR2RGB),    'Brightness/Contrast\n(ElementwiseKernel)')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/filter_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved filter_comparison.png")

    # ------------------------------------------------------------------
    # Performance report
    # ------------------------------------------------------------------
    processor.generate_performance_report(output_dir)

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)
    print(f"Results in: {output_dir}/")
    print("  performance_analysis.png")
    print("  filter_comparison.png")
    print("  metrics.json")
    print("  batch_edge_*.jpg")


if __name__ == "__main__":
    main()