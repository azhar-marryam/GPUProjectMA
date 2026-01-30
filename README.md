# GPUProjectMA
# GPU-Accelerated Image Processing Suite - Setup Guide

**Authors:** Marryam Azhar (2502069), Asfa Toor (2401097)  
**Course:** IT00CG19 GPU Programming 2025  
**Institution:** Åbo Akademi University

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Dataset Generation](#dataset-generation)
4. [Running the Application](#running-the-application)
5. [Understanding the Output](#understanding-the-output)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements
- **GPU:** NVIDIA GPU with CUDA support (Compute Capability 6.0+)
  - Recommended: RTX 2060 or higher
  - Minimum: GTX 1050 or higher
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 2GB free space for dataset and results

### Software Requirements
- **Operating System:** Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python:** 3.8 or higher --> we used python 3.9
- **CUDA Toolkit:** 12.x (will be installed with CuPy)
- **CUDA**: 12.1.1
- **GCC**: 10.4.0
- **NVIDIA Driver:** Latest version (520.x or higher)

---

## Installation Steps

### Step 1: Verify NVIDIA GPU and Driver

**Windows:**
```bash
nvidia-smi
```

**Linux:**
```bash
nvidia-smi
lspci | grep -i nvidia
```

You should see your GPU information and driver version.
Requesting the GPU:

module load gcc/10.4.0 cuda/12.1.1

### Step 2: Create Python Virtual Environment

**Windows:**
```bash
# Navigate to your project directory
cd path/to/your/project

# Create virtual environment
python -m venv gpu_env

# Activate virtual environment
gpu_env\Scripts\activate
```

**Linux/macOS:**
```bash
# Navigate to your project directory
cd path/to/your/project

# Create virtual environment
module load python-data/3.9
python3 -m venv gpu_env

# Activate virtual environment
source gpu_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

**Note:** CuPy installation may take 5-10 minutes as it needs to compile CUDA kernels.

### Step 4: Verify CuPy Installation

```bash
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount(), 'GPU(s) detected'); print('CuPy version:', cp.__version__)"
This should show GPU detected something like below:
```

Expected output:
```
1 GPU(s) detected
CuPy version: 12.x.x
```

---

## Dataset Generation

### Step 1: Generate Dataset

Run the dataset generation script:

```bash
python generate_dataset.py
```

This will:
- Create a `dataset/` directory
- Generate synthetic images of various sizes (512x512 to 4K)
- Attempt to download sample images from Lorem Picsum
- Display dataset statistics

Expected output:
```
============================================================
Dataset Generation for GPU Image Processing
============================================================
Generating synthetic test images...
Created: dataset/synthetic_gradient_small_512x512.jpg
Created: dataset/synthetic_checkerboard_small_512x512.jpg
...
Total images: 24
Total dataset size: 156.45 MB
```

### Step 2: Verify Dataset

Check that the `dataset/` folder contains multiple .jpg files:

```bash
# Windows
dir dataset

# Linux/macOS
ls -lh dataset/
```

---

## Running the Application

### Complete Execution

```bash
python main.py
```

### Expected Execution Flow

The application will:

1. **Initialize GPU** (2-3 seconds)
2. **Load Images** from dataset
3. **Run Performance Benchmarks** for each operation:
   - Grayscale conversion
   - Gaussian blur
   - Edge detection (Sobel)
   - Sepia filter
4. **Batch Processing Demonstration**
5. **Generate Visual Comparisons**
6. **Create Performance Report**

### Sample Output

```
============================================================
GPU-Accelerated Image Processing Suite
Åbo Akademi University - IT00CG19 GPU Programming 2025
============================================================

Found 24 images for processing

============================================================
PERFORMANCE BENCHMARKING
============================================================

Benchmarking grayscale...
Image size: (2048, 2048, 3)
CPU Time: 45.23 ms
GPU Time: 2.15 ms
Speedup: 21.03x
GPU Memory: 48.50 MB

Benchmarking blur...
Image size: (2048, 2048, 3)
CPU Time: 156.78 ms
GPU Time: 4.32 ms
Speedup: 36.29x
GPU Memory: 62.10 MB

...

============================================================
BATCH PROCESSING DEMONSTRATION
============================================================
Processed 5 images in 0.245 seconds
Average time per image: 49.00 ms

============================================================
PROCESSING COMPLETE
============================================================
Results saved to: results/
Generated files:
  - performance_analysis.png
  - filter_comparison.png
  - metrics.json
  - batch_edge_*.jpg
```

---

## Understanding the Output

### Generated Files

After execution, you'll find these files in the `results/` directory:

1. **performance_analysis.png**
   - 4-panel visualization showing:
     - CPU vs GPU execution times
     - Speedup factors
     - Memory usage
     - Processing throughput

2. **filter_comparison.png**
   - Visual comparison of all filters applied to a test image
   - Shows: Original, Grayscale, Blur, Edge Detection, Sepia, Sharpen, Brightness/Contrast

3. **metrics.json**
   - Detailed performance metrics in JSON format
   - Contains:
     - Summary statistics (average/max/min speedup)
     - Per-operation metrics (times, speedup, memory)

4. **batch_edge_*.jpg**
   - Results of batch edge detection processing
   - Demonstrates parallel batch processing capability

### Key Metrics to Present

From `metrics.json`, highlight:

- **Average Speedup:** Typically 20-45x
- **Peak Performance:** Which operation achieved highest speedup
- **Memory Efficiency:** GPU memory usage per operation
- **Throughput:** Operations per second

---

## Troubleshooting

### Issue: "No CUDA-capable device detected"

**Solution:**
1. Verify GPU is NVIDIA: `nvidia-smi`
2. Update NVIDIA drivers
3. Reinstall CuPy: `pip uninstall cupy-cuda12x && pip install cupy-cuda12x`

### Issue: "Out of memory" error

**Solution:**
1. Close other GPU-intensive applications
2. Reduce image sizes in dataset
3. Process fewer images in batch mode

### Issue: CuPy installation fails

**Solution:**
1. Ensure CUDA Toolkit is installed
2. Try pip installation with verbose flag: `pip install cupy-cuda12x -v`
3. Alternative: Use conda: `conda install -c conda-forge cupy`

### Issue: Import errors for cv2 or matplotlib

**Solution:**
```bash
pip install --upgrade opencv-python matplotlib
```

### Issue: Dataset generation fails

**Solution:**
- Internet required only for downloading sample images
- Synthetic images are always generated (work offline)
- Verify write permissions in project directory

---

## Performance Optimization Tips

### For Best Results:

1. **Close Background Applications**
   - Close browsers, games, other GPU-intensive apps
   - Free up GPU memory

2. **Use Larger Images**
   - GPU advantage increases with image size
   - Test with 4K images for maximum speedup

3. **Batch Processing**
   - Process multiple images together
   - Better GPU utilization

4. **Monitor GPU Usage**
   ```bash
   # Keep this running in another terminal
   nvidia-smi -l 1
   ```

---

## Quick Reference Commands

```bash
# Setup (one-time)
python -m venv gpu_env
source gpu_env/bin/activate  # or gpu_env\Scripts\activate on Windows
pip install -r requirements.txt

# Generate dataset
python generate_dataset.py

# Run application
python main.py

# Check results
ls results/  # or dir results on Windows

# Deactivate environment when done
deactivate
```

---

## Expected Timeline

- **Setup:** 10-15 minutes (including CuPy installation)
- **Dataset Generation:** 1-2 minutes
- **Main Execution:** 3-5 minutes (depending on dataset size)
- **Total:** ~20 minutes for complete first run

---

## Support

For issues specific to:
- **CuPy:** https://docs.cupy.dev/
- **CUDA:** https://docs.nvidia.com/cuda/
- **Course-related:** Contact course instructor

---

**Good luck with your presentation!**
