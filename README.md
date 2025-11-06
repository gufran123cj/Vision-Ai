# Program for Vision AI

A comprehensive 4-week program to master Computer Vision using PyTorch, OpenCV, and modern deep learning techniques.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Learning Objectives

By the end of this program, you will be able to:

- Understand the fundamentals of deep learning and neural networks
- Build, train, and evaluate Convolutional Neural Networks (CNNs) using Python and PyTorch
- Implement common computer vision tasks: image classification, object detection, and segmentation
- Apply transfer learning with pre-trained models to solve custom problems efficiently
- Develop and containerize a real-time vision application

## Core Technologies

- **Primary Language**: Python
- **Key Libraries**: PyTorch, OpenCV, NumPy, Matplotlib
- **Tools**: Jupyter Notebooks, Git, Docker

## Project Structure

```
vision-ai-training/
├── week1/                          # AI & Deep Learning Foundations
│   ├── day1-2/                     # From ML to Deep Learning
│   │   └── neural_networks_basics.py
│   ├── day3-4/                     # Introduction to PyTorch
│   │   └── pytorch_introduction.py
│   └── day5-7/                     # Image Processing with OpenCV
│       ├── opencv_fundamentals.py
│       └── output/                 # Generated images
├── week2/                          # Convolutional Neural Networks
│   ├── day1-3/                     # Understanding CNNs
│   │   └── cnn_from_scratch.py
│   ├── day4-5/                     # Training, Evaluation & Optimization
│   │   └── training_optimization.py
│   └── day6-7/                     # Transfer Learning
│       └── transfer_learning.py
├── week3/                          # Advanced Vision Tasks
│   ├── day1-3/                     # Object Detection (YOLO)
│   │   └── object_detection_yolo.py
│   └── day4-7/                     # Image Segmentation
│       └── image_segmentation.py
├── week4/                          # Capstone Project & Deployment
│   ├── README.md                   # Week 4 project documentation
│   ├── PROJECT_STATUS.md           # Project status report
│   └── YOLO-Safety-Equipment-Detection-main/
│       ├── capstone_project.py     # Real-time PPE detection application
│       ├── train_model.py          # Model training script
│       ├── test_model.py           # Model testing script (interactive menu)
│       ├── quick_test.py           # Quick inference test
│       ├── data/                   # Safety equipment dataset
│       │   ├── train/              # Training images and labels
│       │   ├── valid/              # Validation images and labels
│       │   ├── test/               # Test images and labels
│       │   └── data.yaml           # Dataset configuration
│       ├── runs/                   # Training and detection outputs
│       └── results/                # Training results and metrics
├── data/                           # Training datasets (MNIST, CIFAR-10)
│   ├── MNIST/
│   └── cifar-10-batches-py/
├── models/                         # Extra custom trained models and datasets
│   ├── Hard Hat Workers.v14-raw_headclassonly.yolov8/
│   └── safety-vest.v1i.yolov8/
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container configuration
├── docker-compose.yml              # Docker Compose configuration
├── SETUP.md                        # Detailed setup guide
├── DATASET_GUIDE.md                # Dataset usage guide
├── PROJECT_STRUCTURE.md            # Detailed project structure
├── LICENSE                         # MIT License
├── CONTRIBUTING.md                 # Contribution guidelines
├── GITHUB_SETUP.md                 # GitHub setup guide
└── README.md                       # This file
```

## Quick Start

### Detailed Setup Guide

**For beginners**: Read the [`SETUP.md`](SETUP.md) file.

### Quick Setup

#### 1. Download the Project
```bash
git clone <repository-url>
cd vision-ai-training
```

#### 2. Python 3.12 Installation (Important!)

**Python 3.12 is required** because PyTorch CUDA support is available for Python 3.8-3.12.

**Windows:**
- Python 3.12 Download: https://www.python.org/downloads/release/python-31212/
- During installation, check the **"Add Python 3.12 to PATH"** option

**Verify:**
```powershell
py -3.12 --version
```

#### 3. Create Virtual Environment

**Windows:**
```powershell
py -3.12 -m venv venv312
venv312\Scripts\activate
```

**Linux/Mac:**
```bash
python3.12 -m venv venv312
source venv312/bin/activate
```

#### 4. GPU Support (Recommended)

**PyTorch Installation with CUDA:**
```powershell
# First, uninstall existing PyTorch (if any)
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**GPU Check:**
```python
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

#### 5. Install Packages
```bash
pip install -r requirements.txt
```

#### 6. Test Installation
```bash
py test_installation.py
```

#### 7. Model Training (Optional)

**Interactive Mode:**
```bash
py train_models.py
```

**Quick Training with GPU:**
```bash
py train_models.py --train-all --epochs 50 --device cuda
```

**Details:** [`DATASET_GUIDE.md`](DATASET_GUIDE.md)

#### 8. Run the Project
```bash
# Interactive mode (with menu)
cd week4/YOLO-Safety-Equipment-Detection-main
python capstone_project.py

# With webcam
python capstone_project.py --mode webcam

# With video
python capstone_project.py --mode video --input path/to/video.mp4
```

### Installation with Docker

#### Prerequisites

1. **Docker Desktop** must be installed
   - Windows: [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
   - After installation, restart your computer

2. **NVIDIA GPU** (optional but recommended):
   - NVIDIA Container Toolkit should be installed
   - Enable WSL Integration in Docker Desktop: Settings > Resources > WSL Integration

#### Step-by-Step Docker Setup

**Step 1: Navigate to project directory**
```powershell
cd "path/to/Vision-Ai"
```

**Step 2: Build Docker image**
```powershell
docker-compose build
```
This process will:
- Download CUDA 11.8 supported base image
- Install Python 3.12
- Install all dependencies (PyTorch, OpenCV, etc.)
- Copy project files
- May take 5-10 minutes (depending on internet speed)

**Step 3: Start the container**
```powershell
docker-compose up -d
```
The `-d` parameter runs it in the background.

**Step 4: Verify container is running**
```powershell
docker-compose ps
```
Should show status as "Up".

**Step 5: Connect to container (interactive mode)**
```powershell
docker-compose exec vision-ai-app bash
```
Or:
```powershell
docker exec -it vision-ai-training bash
```

**Step 6: Verify installation inside container**
```bash
# Check Python version
python --version
# Should show: Python 3.12.x

# Check CUDA/GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Check PyTorch version
python -c "import torch; print('PyTorch:', torch.__version__)"

# Check OpenCV
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

#### Using the Project in Docker

**Model Training:**
```bash
# Navigate to week 4 project
cd week4/YOLO-Safety-Equipment-Detection-main

# Train model
python train_model.py
```

**Model Testing:**
```bash
# Quick test
python quick_test.py

# Interactive test menu
python test_model.py

# Real-time detection with video
python capstone_project.py --mode video --input /app/samplevideo.mp4
```

#### Useful Docker Commands

```powershell
# Stop container
docker-compose down

# Stop container and remove volumes (careful!)
docker-compose down -v

# View container logs
docker-compose logs -f

# Restart container
docker-compose restart

# Check container status
docker-compose ps
```

#### Important Notes

1. **Volume Mounts (Data Persistence):**
   - `./data`, `./models`, `./runs` folders are shared with host
   - Training results and models will persist

2. **Webcam Access (Windows):**
   - `/dev/video0` in `docker-compose.yml` is for Linux only
   - On Windows, webcam access may require additional configuration
   - Alternative: Use video files instead

3. **GPU Check:**
   ```bash
   # Inside container
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

For more details: [`SETUP.md`](SETUP.md)

## Weekly Schedule

### Week 1: AI & Deep Learning Foundations
- **Days 1-2**: From ML to Deep Learning (Neural networks, backpropagation, gradient descent)
- **Days 3-4**: Introduction to PyTorch (Tensors, autograd, building models)
- **Days 5-7**: Image Processing Fundamentals with OpenCV

### Week 2: Convolutional Neural Networks (CNNs)
- **Days 1-3**: Understanding CNNs (Convolution, pooling, padding, stride)
- **Days 4-5**: Training, Evaluation & Optimization (Loss functions, optimizers, data augmentation)
- **Days 6-7**: Transfer Learning with Pre-trained Models (ResNet, VGG, MobileNet)

### Week 3: Advanced Vision Tasks
- **Days 1-3**: Object Detection (YOLO, bounding boxes, IoU, NMS)
- **Days 4-7**: Image Segmentation (Semantic vs instance segmentation, U-Net)

### Week 4: Capstone Project & Deployment
- **Capstone Project**: Real-time Safety Equipment Detection (Hard Hat & Safety Vest)
- Custom model training with GPU support
- Interactive mode for easy usage
- Containerize and deploy the application

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Jupyter Notebooks Guide](https://jupyter-notebook.readthedocs.io/)

## 🚀 Next Steps in Your AI Journey

After completing this 4-week program, you can explore advanced topics to further enhance your computer vision skills:

### Generative AI
- **Stable Diffusion**: Explore text-to-image generation and synthetic data creation for dataset augmentation
- **GANs (Generative Adversarial Networks)**: Generate realistic safety equipment images to improve model training with limited data

### Vision Transformers (ViT)
- **Transformer-based Detection**: Experiment with Vision Transformers for object detection tasks
- **Attention Mechanisms**: Understand how self-attention improves vision model performance
- **Hybrid Architectures**: Combine CNNs with Transformers for enhanced accuracy

### Advanced Topics
- **Multi-modal AI**: Combine vision with audio, text, or sensor data for enhanced safety monitoring
- **Edge Deployment**: Optimize models for deployment on edge devices (Jetson Nano, Raspberry Pi)
- **Real-time Optimization**: Explore model quantization, pruning, and distillation for faster inference

For more details on future enhancements, see [`week4/README.md`](week4/README.md) and [`week4/PROJECT_STATUS.md`](week4/PROJECT_STATUS.md).

## License

This training program is for educational purposes.

