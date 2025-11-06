"""
Week 1 - Days 5-7: Image Processing Fundamentals with OpenCV
Reading, writing, manipulating images, color spaces, transformations, filtering
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def create_sample_image() -> np.ndarray:
    """Create a sample image for demonstration."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (0, 255, 0), -1)
    cv2.circle(img, (150, 150), 50, (255, 0, 0), -1)
    cv2.putText(img, 'OpenCV', (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2)
    return img


def demonstrate_image_io():
    """Demonstrate reading and writing images."""
    print("=" * 60)
    print("Image I/O Operations")
    print("=" * 60)
    
    # Create sample image
    img = create_sample_image()
    
    # Create output directory
    output_dir = Path('week1/day5-7/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    cv2.imwrite(str(output_dir / 'sample_image.png'), img)
    print(f"\n✓ Sample image created and saved to {output_dir / 'sample_image.png'}")
    
    # Read image
    loaded_img = cv2.imread(str(output_dir / 'sample_image.png'))
    print(f"✓ Image loaded. Shape: {loaded_img.shape}")
    print(f"  Height: {loaded_img.shape[0]}, Width: {loaded_img.shape[1]}, Channels: {loaded_img.shape[2]}")
    
    return img


def demonstrate_color_spaces(img: np.ndarray):
    """Demonstrate different color spaces."""
    print("\n" + "=" * 60)
    print("Color Space Conversions")
    print("=" * 60)
    
    output_dir = Path('week1/day5-7/output')
    
    # Convert BGR to different color spaces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Save converted images
    cv2.imwrite(str(output_dir / 'gray.png'), gray)
    cv2.imwrite(str(output_dir / 'hsv.png'), hsv)
    cv2.imwrite(str(output_dir / 'lab.png'), lab)
    
    print("\n✓ Color space conversions:")
    print(f"  - Grayscale: {gray.shape}")
    print(f"  - HSV: {hsv.shape}")
    print(f"  - LAB: {lab.shape}")
    
    # Display comparison
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original (BGR)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    # HSV to RGB: HSV -> BGR -> RGB
    hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    axes[1, 0].imshow(cv2.cvtColor(hsv_bgr, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('HSV')
    axes[1, 0].axis('off')
    
    # LAB to RGB: LAB -> BGR -> RGB
    lab_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    axes[1, 1].imshow(cv2.cvtColor(lab_bgr, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('LAB')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'color_spaces.png')
    print(f"✓ Color space comparison saved to {output_dir / 'color_spaces.png'}")


def demonstrate_transformations(img: np.ndarray):
    """Demonstrate image transformations."""
    print("\n" + "=" * 60)
    print("Image Transformations")
    print("=" * 60)
    
    output_dir = Path('week1/day5-7/output')
    height, width = img.shape[:2]
    
    # Scaling
    scaled = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(output_dir / 'scaled_2x.png'), scaled)
    
    # Rotation
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
    cv2.imwrite(str(output_dir / 'rotated_45deg.png'), rotated)
    
    # Translation
    translation_matrix = np.float32([[1, 0, 50], [0, 1, 50]])
    translated = cv2.warpAffine(img, translation_matrix, (width, height))
    cv2.imwrite(str(output_dir / 'translated.png'), translated)
    
    # Flipping
    flipped_horizontal = cv2.flip(img, 1)
    flipped_vertical = cv2.flip(img, 0)
    cv2.imwrite(str(output_dir / 'flipped_horizontal.png'), flipped_horizontal)
    cv2.imwrite(str(output_dir / 'flipped_vertical.png'), flipped_vertical)
    
    print("\n✓ Transformations applied:")
    print("  - Scaling (2x)")
    print("  - Rotation (45°)")
    print("  - Translation")
    print("  - Flipping (horizontal & vertical)")


def demonstrate_filtering(img: np.ndarray):
    """Demonstrate image filtering operations."""
    print("\n" + "=" * 60)
    print("Image Filtering")
    print("=" * 60)
    
    output_dir = Path('week1/day5-7/output')
    
    # Convert to grayscale for some filters
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    cv2.imwrite(str(output_dir / 'gaussian_blur.png'), blurred)
    
    # Median blur
    median = cv2.medianBlur(img, 5)
    cv2.imwrite(str(output_dir / 'median_blur.png'), median)
    
    # Bilateral filter (edge-preserving)
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite(str(output_dir / 'bilateral_filter.png'), bilateral)
    
    print("\n✓ Filtering operations:")
    print("  - Gaussian blur")
    print("  - Median blur")
    print("  - Bilateral filter")


def demonstrate_edge_detection(img: np.ndarray):
    """Demonstrate edge detection techniques."""
    print("\n" + "=" * 60)
    print("Edge Detection")
    print("=" * 60)
    
    output_dir = Path('week1/day5-7/output')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Canny edge detection
    edges_canny = cv2.Canny(gray, 50, 150)
    cv2.imwrite(str(output_dir / 'edges_canny.png'), edges_canny)
    
    # Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    cv2.imwrite(str(output_dir / 'edges_sobel.png'), sobel)
    
    # Laplacian edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    cv2.imwrite(str(output_dir / 'edges_laplacian.png'), laplacian)
    
    print("\n✓ Edge detection methods:")
    print("  - Canny")
    print("  - Sobel")
    print("  - Laplacian")
    
    # Display comparison
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(edges_canny, cmap='gray')
    axes[0, 1].set_title('Canny')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(sobel, cmap='gray')
    axes[1, 0].set_title('Sobel')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(laplacian, cmap='gray')
    axes[1, 1].set_title('Laplacian')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'edge_detection_comparison.png')
    print(f"✓ Edge detection comparison saved to {output_dir / 'edge_detection_comparison.png'}")


def demonstrate_webcam_capture():
    """Demonstrate real-time webcam capture (if available)."""
    print("\n" + "=" * 60)
    print("Webcam Capture (Optional)")
    print("=" * 60)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("⚠ Webcam not available. Skipping webcam demo.")
        return
    
    print("Press 'q' to quit webcam capture")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply some processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Display
        cv2.imshow('Original', frame)
        cv2.imshow('Edges', edges)
        
        frame_count += 1
        if frame_count >= 100:  # Capture for ~3 seconds at 30fps
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Webcam capture demonstration completed")


if __name__ == "__main__":
    print("OpenCV Fundamentals Demonstration")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Create sample image
    img = demonstrate_image_io()
    
    # Run demonstrations
    demonstrate_color_spaces(img)
    demonstrate_transformations(img)
    demonstrate_filtering(img)
    demonstrate_edge_detection(img)
    
    # Optional: webcam capture
    # demonstrate_webcam_capture()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("Check the 'week1/day5-7/output' directory for generated images.")
    print("=" * 60)

