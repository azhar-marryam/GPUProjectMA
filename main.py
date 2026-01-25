"""
Real-Time GPU-Accelerated Image Processing Suite
Authors: Marryam Azhar, Asfa Toor
Course: IT00CG19 GPU Programming 2025
Åbo Akademi University
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


@dataclass
class PerformanceMetrics:
    """Store performance metrics for analysis"""
    operation: str
    cpu_time: float
    gpu_time: float
    speedup: float
    image_size: Tuple[int, int]
    memory_used: float


class GPUImageProcessor:
    """High-performance GPU-accelerated image processing system"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.mempool = cp.get_default_memory_pool()
        
    def measure_memory(self) -> float:
        """Measure GPU memory usage in MB"""
        return self.mempool.used_bytes() / (1024 ** 2)
    
    # ==================== GPU KERNELS ====================
    
    @staticmethod
    def grayscale_gpu(image: cp.ndarray) -> cp.ndarray:
        """
        GPU kernel for grayscale conversion using luminosity method
        Formula: 0.299*R + 0.587*G + 0.114*B
        """
        if len(image.shape) == 2:
            return image
        return cp.dot(image[..., :3], cp.array([0.299, 0.587, 0.114]))
    
    @staticmethod
    def grayscale_cpu(image: np.ndarray) -> np.ndarray:
        """CPU version for comparison"""
        if len(image.shape) == 2:
            return image
        return np.dot(image[..., :3], np.array([0.299, 0.587, 0.114]))
    
    @staticmethod
    def gaussian_blur_gpu(image: cp.ndarray, kernel_size: int = 5, 
                         sigma: float = 1.0) -> cp.ndarray:
        """
        GPU-accelerated Gaussian blur using separable convolution
        Optimized with memory coalescing
        """
        # Generate 1D Gaussian kernel
        k = kernel_size // 2
        x = cp.arange(-k, k + 1)
        kernel_1d = cp.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d /= kernel_1d.sum()
        
        # Separable convolution: horizontal then vertical
        # This reduces complexity from O(n²) to O(2n)
        if len(image.shape) == 3:
            result = cp.zeros_like(image, dtype=cp.float32)
            for channel in range(image.shape[2]):
                temp = cp.convolve(image[:, :, channel].ravel(), 
                                  kernel_1d, mode='same')
                temp = temp.reshape(image.shape[:2])
                result[:, :, channel] = cp.convolve(temp.T.ravel(), 
                                                    kernel_1d, mode='same').reshape(image.shape[:2]).T
        else:
            temp = cp.convolve(image.ravel(), kernel_1d, mode='same')
            temp = temp.reshape(image.shape)
            result = cp.convolve(temp.T.ravel(), kernel_1d, mode='same').reshape(image.shape).T
        
        return cp.clip(result, 0, 255).astype(cp.uint8)
    
    @staticmethod
    def gaussian_blur_cpu(image: np.ndarray, kernel_size: int = 5, 
                         sigma: float = 1.0) -> np.ndarray:
        """CPU version using OpenCV for fair comparison"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def sobel_edge_detection_gpu(image: cp.ndarray) -> cp.ndarray:
        """
        GPU-accelerated Sobel edge detection
        Uses parallel gradient computation in X and Y directions
        """
        # Sobel kernels
        sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
        sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = GPUImageProcessor.grayscale_gpu(image)
        else:
            gray = image.astype(cp.float32)
        
        # Pad image
        padded = cp.pad(gray, 1, mode='edge')
        
        # Compute gradients
        h, w = gray.shape
        grad_x = cp.zeros_like(gray)
        grad_y = cp.zeros_like(gray)
        
        for i in range(h):
            for j in range(w):
                region = padded[i:i+3, j:j+3]
                grad_x[i, j] = cp.sum(region * sobel_x)
                grad_y[i, j] = cp.sum(region * sobel_y)
        
        # Compute magnitude
        magnitude = cp.sqrt(grad_x**2 + grad_y**2)
        magnitude = cp.clip(magnitude, 0, 255)
        
        return magnitude.astype(cp.uint8)
    
    @staticmethod
    def sobel_edge_detection_cpu(image: np.ndarray) -> np.ndarray:
        """CPU version for comparison"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.clip(magnitude, 0, 255)
        
        return magnitude.astype(np.uint8)
    
    @staticmethod
    def sepia_filter_gpu(image: cp.ndarray) -> cp.ndarray:
        """
        GPU-accelerated sepia tone filter
        Matrix transformation optimized for parallel execution
        """
        if len(image.shape) == 2:
            image = cp.stack([image] * 3, axis=2)
        
        sepia_matrix = cp.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        # Reshape for matrix multiplication
        h, w, c = image.shape
        flat_image = image.reshape(-1, c).astype(cp.float32)
        
        # Parallel matrix multiplication
        result = cp.dot(flat_image, sepia_matrix.T)
        result = cp.clip(result, 0, 255)
        
        return result.reshape(h, w, c).astype(cp.uint8)
    
    @staticmethod
    def sepia_filter_cpu(image: np.ndarray) -> np.ndarray:
        """CPU version for comparison"""
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        h, w, c = image.shape
        flat_image = image.reshape(-1, c).astype(np.float32)
        result = np.dot(flat_image, sepia_matrix.T)
        result = np.clip(result, 0, 255)
        
        return result.reshape(h, w, c).astype(np.uint8)
    
    @staticmethod
    def sharpen_gpu(image: cp.ndarray) -> cp.ndarray:
        """GPU-accelerated sharpening filter"""
        kernel = cp.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=cp.float32)
        
        if len(image.shape) == 3:
            result = cp.zeros_like(image, dtype=cp.float32)
            padded = cp.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
            
            h, w, c = image.shape
            for channel in range(c):
                for i in range(h):
                    for j in range(w):
                        region = padded[i:i+3, j:j+3, channel]
                        result[i, j, channel] = cp.sum(region * kernel)
        else:
            result = cp.zeros_like(image, dtype=cp.float32)
            padded = cp.pad(image, 1, mode='edge')
            h, w = image.shape
            
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+3, j:j+3]
                    result[i, j] = cp.sum(region * kernel)
        
        return cp.clip(result, 0, 255).astype(cp.uint8)
    
    @staticmethod
    def brightness_contrast_gpu(image: cp.ndarray, brightness: float = 0, 
                               contrast: float = 1.0) -> cp.ndarray:
        """
        GPU-accelerated brightness and contrast adjustment
        Parallel per-pixel operation
        """
        result = image.astype(cp.float32)
        result = result * contrast + brightness
        return cp.clip(result, 0, 255).astype(cp.uint8)
    
    # ==================== BATCH PROCESSING ====================
    
    def process_batch_gpu(self, images: List[np.ndarray], 
                         operation: str) -> List[np.ndarray]:
        """
        Batch process multiple images on GPU
        Optimized with stream processing
        """
        results = []
        gpu_images = [cp.asarray(img) for img in images]
        
        for gpu_img in gpu_images:
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
    
    # ==================== PERFORMANCE BENCHMARKING ====================
    
    def benchmark_operation(self, image: np.ndarray, operation: str, 
                          iterations: int = 10) -> PerformanceMetrics:
        """
        Comprehensive performance benchmarking
        Measures CPU vs GPU performance with warm-up
        """
        print(f"\nBenchmarking {operation}...")
        print(f"Image size: {image.shape}")
        
        # Warm-up runs
        gpu_image = cp.asarray(image)
        for _ in range(3):
            if operation == 'grayscale':
                _ = self.grayscale_gpu(gpu_image)
                _ = self.grayscale_cpu(image)
            elif operation == 'blur':
                _ = self.gaussian_blur_gpu(gpu_image)
                _ = self.gaussian_blur_cpu(image)
            elif operation == 'edge':
                _ = self.sobel_edge_detection_gpu(gpu_image)
                _ = self.sobel_edge_detection_cpu(image)
            elif operation == 'sepia':
                _ = self.sepia_filter_gpu(gpu_image)
                _ = self.sepia_filter_cpu(image)
        
        cp.cuda.Stream.null.synchronize()
        
        # GPU timing
        gpu_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            
            if operation == 'grayscale':
                result_gpu = self.grayscale_gpu(gpu_image)
            elif operation == 'blur':
                result_gpu = self.gaussian_blur_gpu(gpu_image)
            elif operation == 'edge':
                result_gpu = self.sobel_edge_detection_gpu(gpu_image)
            elif operation == 'sepia':
                result_gpu = self.sepia_filter_gpu(gpu_image)
            
            cp.cuda.Stream.null.synchronize()
            gpu_times.append(time.perf_counter() - start)
        
        gpu_time = np.mean(gpu_times)
        
        # CPU timing
        cpu_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            
            if operation == 'grayscale':
                result_cpu = self.grayscale_cpu(image)
            elif operation == 'blur':
                result_cpu = self.gaussian_blur_cpu(image)
            elif operation == 'edge':
                result_cpu = self.sobel_edge_detection_cpu(image)
            elif operation == 'sepia':
                result_cpu = self.sepia_filter_cpu(image)
            
            cpu_times.append(time.perf_counter() - start)
        
        cpu_time = np.mean(cpu_times)
        speedup = cpu_time / gpu_time
        memory_used = self.measure_memory()
        
        metrics = PerformanceMetrics(
            operation=operation,
            cpu_time=cpu_time * 1000,  # Convert to ms
            gpu_time=gpu_time * 1000,
            speedup=speedup,
            image_size=image.shape[:2],
            memory_used=memory_used
        )
        
        self.metrics.append(metrics)
        
        print(f"CPU Time: {metrics.cpu_time:.2f} ms")
        print(f"GPU Time: {metrics.gpu_time:.2f} ms")
        print(f"Speedup: {metrics.speedup:.2f}x")
        print(f"GPU Memory: {metrics.memory_used:.2f} MB")
        
        return metrics
    
    def generate_performance_report(self, output_dir: str = "results"):
        """Generate comprehensive performance analysis report"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create performance plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        operations = [m.operation for m in self.metrics]
        cpu_times = [m.cpu_time for m in self.metrics]
        gpu_times = [m.gpu_time for m in self.metrics]
        speedups = [m.speedup for m in self.metrics]
        
        # Plot 1: Execution Time Comparison
        x = np.arange(len(operations))
        width = 0.35
        axes[0, 0].bar(x - width/2, cpu_times, width, label='CPU', color='coral')
        axes[0, 0].bar(x + width/2, gpu_times, width, label='GPU', color='skyblue')
        axes[0, 0].set_xlabel('Operation')
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].set_title('CPU vs GPU Execution Time')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(operations, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Speedup Factor
        axes[0, 1].bar(operations, speedups, color='green', alpha=0.7)
        axes[0, 1].axhline(y=1, color='r', linestyle='--', label='No speedup')
        axes[0, 1].set_xlabel('Operation')
        axes[0, 1].set_ylabel('Speedup Factor')
        axes[0, 1].set_title('GPU Speedup over CPU')
        axes[0, 1].set_xticklabels(operations, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Memory Usage
        memory_usage = [m.memory_used for m in self.metrics]
        axes[1, 0].plot(operations, memory_usage, marker='o', linewidth=2, 
                       markersize=8, color='purple')
        axes[1, 0].set_xlabel('Operation')
        axes[1, 0].set_ylabel('GPU Memory (MB)')
        axes[1, 0].set_title('GPU Memory Usage')
        axes[1, 0].set_xticklabels(operations, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Efficiency (Operations per second)
        efficiency = [1000 / m.gpu_time for m in self.metrics]
        axes[1, 1].bar(operations, efficiency, color='orange', alpha=0.7)
        axes[1, 1].set_xlabel('Operation')
        axes[1, 1].set_ylabel('Operations/Second')
        axes[1, 1].set_title('GPU Processing Throughput')
        axes[1, 1].set_xticklabels(operations, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_analysis.png", dpi=300, 
                   bbox_inches='tight')
        plt.close()
        
        # Save metrics to JSON
        metrics_dict = {
            'summary': {
                'average_speedup': np.mean(speedups),
                'max_speedup': np.max(speedups),
                'min_speedup': np.min(speedups),
                'total_operations': len(self.metrics)
            },
            'detailed_metrics': [
                {
                    'operation': m.operation,
                    'cpu_time_ms': m.cpu_time,
                    'gpu_time_ms': m.gpu_time,
                    'speedup': m.speedup,
                    'image_size': m.image_size,
                    'memory_mb': m.memory_used
                }
                for m in self.metrics
            ]
        }
        
        with open(f"{output_dir}/metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"\nPerformance report saved to {output_dir}/")


def main():
    """Main execution pipeline"""
    print("=" * 60)
    print("GPU-Accelerated Image Processing Suite")
    print("Åbo Akademi University - IT00CG19 GPU Programming 2025")
    print("=" * 60)
    
    # Initialize processor
    processor = GPUImageProcessor()
    
    # Create output directories
    output_dir = "results"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load test images
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"\nError: Dataset directory '{dataset_dir}' not found!")
        print("Please run the dataset generation script first.")
        return
    
    image_files = list(Path(dataset_dir).glob("*.jpg")) + \
                  list(Path(dataset_dir).glob("*.png"))
    
    if not image_files:
        print("No images found in dataset directory!")
        return
    
    print(f"\nFound {len(image_files)} images for processing")
    
    # Test with different image sizes
    test_image = cv2.imread(str(image_files[0]))
    
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKING")
    print("=" * 60)
    
    # Benchmark all operations
    operations = ['grayscale', 'blur', 'edge', 'sepia']
    
    for operation in operations:
        processor.benchmark_operation(test_image, operation, iterations=10)
    
    # Batch processing demonstration
    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMONSTRATION")
    print("=" * 60)
    
    batch_images = [cv2.imread(str(img)) for img in image_files[:5]]
    
    start_time = time.time()
    batch_results = processor.process_batch_gpu(batch_images, 'edge')
    batch_time = time.time() - start_time
    
    print(f"Processed {len(batch_images)} images in {batch_time:.3f} seconds")
    print(f"Average time per image: {batch_time/len(batch_images)*1000:.2f} ms")
    
    # Save batch results
    for idx, result in enumerate(batch_results):
        cv2.imwrite(f"{output_dir}/batch_edge_{idx}.jpg", result)
    
    # Generate visual comparisons
    print("\n" + "=" * 60)
    print("GENERATING VISUAL COMPARISONS")
    print("=" * 60)
    
    test_img_gpu = cp.asarray(test_image)
    
    # Process with all filters
    gray_result = cp.asnumpy(processor.grayscale_gpu(test_img_gpu))
    blur_result = cp.asnumpy(processor.gaussian_blur_gpu(test_img_gpu))
    edge_result = cp.asnumpy(processor.sobel_edge_detection_gpu(test_img_gpu))
    sepia_result = cp.asnumpy(processor.sepia_filter_gpu(test_img_gpu))
    sharp_result = cp.asnumpy(processor.sharpen_gpu(test_img_gpu))
    bright_result = cp.asnumpy(processor.brightness_contrast_gpu(test_img_gpu, 
                                                                 brightness=30, 
                                                                 contrast=1.2))
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray_result, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(blur_result, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Gaussian Blur')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(edge_result, cmap='gray')
    axes[0, 3].set_title('Edge Detection')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(sepia_result, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Sepia Tone')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(sharp_result, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Sharpened')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(bright_result, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Brightness/Contrast')
    axes[1, 2].axis('off')
    
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/filter_comparison.png", dpi=300, 
               bbox_inches='tight')
    plt.close()
    
    # Generate performance report
    processor.generate_performance_report(output_dir)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")
    print("Generated files:")
    print("  - performance_analysis.png")
    print("  - filter_comparison.png")
    print("  - metrics.json")
    print("  - batch_edge_*.jpg")


if __name__ == "__main__":
    main()