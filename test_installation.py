"""Test script to verify all installations"""

print("=" * 60)
print("Kurulum Testi")
print("=" * 60)

# Test OpenCV
try:
    import cv2
    print(f"✓ OpenCV kurulu! Versiyon: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV kurulu değil: {e}")

# Test NumPy
try:
    import numpy as np
    print(f"✓ NumPy kurulu! Versiyon: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy kurulu değil: {e}")

# Test Matplotlib
try:
    import matplotlib
    print(f"✓ Matplotlib kurulu! Versiyon: {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib kurulu değil: {e}")

# Test PyTorch
try:
    import torch
    print(f"✓ PyTorch kurulu! Versiyon: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch kurulu değil: {e}")

# Test Torchvision
try:
    import torchvision
    print(f"✓ Torchvision kurulu! Versiyon: {torchvision.__version__}")
except ImportError as e:
    print(f"✗ Torchvision kurulu değil: {e}")

print("\n" + "=" * 60)
print("Test tamamlandı!")
print("=" * 60)

