"""
Dataset Generator for GPU Image Processing
Creates diverse test images of various sizes
"""

import numpy as np
import cv2
from pathlib import Path
import requests
from io import BytesIO
from PIL import Image


def create_synthetic_images(output_dir: str):
    """Create synthetic test images of various sizes"""
    Path(output_dir).mkdir(exist_ok=True)
    
    sizes = [
        (512, 512),      # Small
        (1024, 1024),    # Medium
        (2048, 2048),    # Large
        (4096, 4096),    # Very Large
        (1920, 1080),    # HD
        (3840, 2160),    # 4K
    ]
    
    patterns = {
        'gradient': lambda h, w: create_gradient(h, w),
        'checkerboard': lambda h, w: create_checkerboard(h, w),
        'noise': lambda h, w: create_noise(h, w),
        'circles': lambda h, w: create_circles(h, w),
        'complex': lambda h, w: create_complex_pattern(h, w),
    }
    
    print("Generating synthetic test images...")
    
    for size_name, (h, w) in [
        ('small', sizes[0]),
        ('medium', sizes[1]),
        ('large', sizes[2]),
        ('hd', sizes[4])
    ]:
        for pattern_name, pattern_func in patterns.items():
            img = pattern_func(h, w)
            filename = f"{output_dir}/synthetic_{pattern_name}_{size_name}_{h}x{w}.jpg"
            cv2.imwrite(filename, img)
            print(f"Created: {filename}")


def create_gradient(h: int, w: int) -> np.ndarray:
    """Create gradient pattern"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Horizontal gradient
    for i in range(w):
        img[:, i, 0] = int(255 * i / w)  # Red channel
    
    # Vertical gradient
    for i in range(h):
        img[i, :, 1] = int(255 * i / h)  # Green channel
    
    # Diagonal gradient
    for i in range(h):
        for j in range(w):
            img[i, j, 2] = int(255 * (i + j) / (h + w))  # Blue channel
    
    return img


def create_checkerboard(h: int, w: int, square_size: int = 64) -> np.ndarray:
    """Create checkerboard pattern"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(0, h, square_size):
        for j in range(0, w, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                img[i:min(i+square_size, h), j:min(j+square_size, w)] = [255, 255, 255]
            else:
                img[i:min(i+square_size, h), j:min(j+square_size, w)] = [0, 0, 0]
    
    return img


def create_noise(h: int, w: int) -> np.ndarray:
    """Create random noise pattern"""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def create_circles(h: int, w: int) -> np.ndarray:
    """Create pattern with circles"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Background gradient
    for i in range(h):
        img[i, :] = [int(128 + 127 * np.sin(2 * np.pi * i / h)), 100, 150]
    
    # Draw circles
    num_circles = min(20, h // 100)
    for _ in range(num_circles):
        center = (np.random.randint(0, w), np.random.randint(0, h))
        radius = np.random.randint(20, min(h, w) // 10)
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.circle(img, center, radius, color, -1)
    
    return img


def create_complex_pattern(h: int, w: int) -> np.ndarray:
    """Create complex pattern with multiple features"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create base pattern
    for i in range(h):
        for j in range(w):
            r = int(127 + 127 * np.sin(2 * np.pi * i / 100))
            g = int(127 + 127 * np.cos(2 * np.pi * j / 100))
            b = int(127 + 127 * np.sin(2 * np.pi * (i + j) / 200))
            img[i, j] = [r, g, b]
    
    # Add geometric shapes
    cv2.rectangle(img, (w//4, h//4), (3*w//4, 3*h//4), (255, 0, 0), 5)
    cv2.circle(img, (w//2, h//2), min(h, w)//4, (0, 255, 0), 5)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'GPU Test', (w//4, h//2), font, 2, (255, 255, 255), 3)
    
    return img


def download_sample_images(output_dir: str):
    """
    Download sample images from placeholder service
    Using Lorem Picsum for copyright-free images
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\nDownloading sample images from Lorem Picsum...")
    
    # Different sizes to test
    sizes = [
        (640, 480),
        (1280, 720),
        (1920, 1080),
        (2560, 1440),
    ]
    
    for idx, (w, h) in enumerate(sizes):
        try:
            # Lorem Picsum provides random images
            url = f"https://picsum.photos/{w}/{h}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img_array = np.array(img)
                
                # Convert RGB to BGR for OpenCV
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                filename = f"{output_dir}/downloaded_{w}x{h}_{idx}.jpg"
                cv2.imwrite(filename, img_array)
                print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Failed to download {w}x{h}: {e}")
    
    print("Note: If downloads fail, synthetic images will be sufficient for testing.")


def main():
    """Main dataset generation"""
    dataset_dir = "dataset"
    
    print("=" * 60)
    print("Dataset Generation for GPU Image Processing")
    print("=" * 60)
    
    # Create synthetic images (always works, no internet needed)
    create_synthetic_images(dataset_dir)
    
    # Try to download real images (optional)
    try:
        download_sample_images(dataset_dir)
    except Exception as e:
        print(f"\nCould not download images: {e}")
        print("Using synthetic images only.")
    
    # Verify dataset
    image_files = list(Path(dataset_dir).glob("*.jpg")) + \
                  list(Path(dataset_dir).glob("*.png"))
    
    print("\n" + "=" * 60)
    print(f"Dataset created successfully!")
    print(f"Total images: {len(image_files)}")
    print(f"Location: {dataset_dir}/")
    print("=" * 60)
    
    # Show dataset statistics
    total_size = 0
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            size_mb = img.nbytes / (1024 * 1024)
            total_size += size_mb
            print(f"{img_file.name}: {img.shape} ({size_mb:.2f} MB)")
    
    print(f"\nTotal dataset size: {total_size:.2f} MB")


if __name__ == "__main__":
    main()