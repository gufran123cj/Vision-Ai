# Week 4: Real-Time Personal Protective Equipment (PPE) Detector

## 📋 Project Overview

This capstone project implements a real-time system that detects whether a person in a video feed is wearing:
- **Hard Hat (Helmet)**
- **Safety Vest (Jacket)**

The system uses a custom-trained YOLOv8 model fine-tuned on a safety equipment dataset.

## 🎯 Project Requirements

### Days 1-4: Project Development

- ✅ **Task 1: Data Collection & Preparation** - Open-source dataset of people with and without PPE
- ✅ **Task 2: Model Training** - Fine-tuned pre-trained YOLO model on custom dataset
- ✅ **Task 3: Application Building** - Python script using OpenCV for real-time video feed, frame-by-frame inference, and bounding boxes

### Days 5-6: Optimization & Deployment

- ✅ **Task 4: Containerization** - Dockerfile to containerize application and dependencies
- ✅ **Task 5: Deployment** - Deploy containerized application locally

### Day 7: Review and Future Steps

- 📝 **Documentation** - Project documentation on GitHub
- 🔮 **Future Steps** - Explore Generative AI (Stable Diffusion, GANs) or Vision Transformers (ViT)

## 📁 Project Structure

```
week4/
├── YOLO-Safety-Equipment-Detection-main/
│   ├── data/
│   │   ├── train/          # Training images and labels
│   │   ├── valid/          # Validation images and labels
│   │   ├── test/           # Test images and labels
│   │   └── data.yaml       # Dataset configuration
│   ├── runs/
│   │   ├── train/          # Training results
│   │   └── detect/         # Inference results
│   ├── train_model.py      # Model training script
│   ├── test_model.py       # Model testing script
│   ├── quick_test.py       # Quick inference test
│   ├── capstone_project.py # Real-time PPE detection application
│   └── README.md           # Detailed project documentation
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA support (recommended)
- OpenCV
- Ultralytics YOLO

### Installation

1. **Install dependencies:**
   ```bash
   pip install ultralytics opencv-python torch torchvision torchaudio
   ```

2. **Navigate to project directory:**
   ```bash
   cd week4/YOLO-Safety-Equipment-Detection-main
   ```

### Step 1: Train the Model

Train a custom YOLO model on the safety equipment dataset:

```bash
python train_model.py
```

**Note:** The training script automatically uses GPU if available. Training typically takes 1-2 hours on a modern GPU.

**Training Output:**
- Model weights: `runs/train/safety_equipment/weights/best.pt`
- Training metrics: `runs/train/safety_equipment/results.png`
- Confusion matrix: `runs/train/safety_equipment/confusion_matrix.png`

### Step 2: Test the Model

Test the trained model on images or videos:

```bash
# Interactive testing menu
python test_model.py

# Quick test on a single image
python quick_test.py
```

### Step 3: Run Real-Time Detection

#### Webcam Mode (Real-Time)

```bash
python capstone_project.py --mode webcam --model runs/train/safety_equipment/weights/best.pt
```

**Controls:**
- Press `q` to quit

#### Video File Mode

```bash
python capstone_project.py --mode video --input path/to/video.mp4 --model runs/train/safety_equipment/weights/best.pt
```

**Save output video:**
```bash
python capstone_project.py --mode video --input input.mp4 --output output.mp4 --model runs/train/safety_equipment/weights/best.pt
```

#### Advanced Options

```bash
# Custom confidence threshold
python capstone_project.py --mode webcam --model runs/train/safety_equipment/weights/best.pt --conf 0.4

# Custom model path
python capstone_project.py --mode webcam --model path/to/custom/model.pt
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
# From project root
docker build -t ppe-detector .
```

### Run with Docker Compose

```bash
docker-compose up -d
```

### Run Container Manually

```bash
# Interactive mode
docker run -it --gpus all -v $(pwd):/app ppe-detector

# Run training inside container
docker run -it --gpus all -v $(pwd):/app ppe-detector python week4/YOLO-Safety-Equipment-Detection-main/train_model.py

# Run real-time detection (requires webcam access)
docker run -it --gpus all --device=/dev/video0 -v $(pwd):/app ppe-detector python week4/YOLO-Safety-Equipment-Detection-main/capstone_project.py --mode webcam
```

**Note:** Webcam access in Docker requires additional configuration on Windows/Mac.

## 📊 Model Performance

The trained model detects 5 classes:
- **Helmet** (Hard Hat)
- **Goggles**
- **Jacket** (Safety Vest)
- **Gloves**
- **Footwear**

**Key Metrics:**
- Model: YOLOv8s (Small)
- Input Size: 640x640
- Training Epochs: 25
- Batch Size: 24 (6GB GPU) / 32 (8GB+ GPU)

## 🎨 Features

### Real-Time Detection
- Frame-by-frame inference
- Low latency processing
- GPU acceleration support

### Visual Feedback
- Color-coded bounding boxes:
  - 🟢 Green: PPE detected
  - 🔴 Red: PPE missing
- Real-time compliance status
- Confidence scores

### Detection Modes
- **Webcam:** Real-time detection from camera feed
- **Video:** Process pre-recorded video files
- **Image:** Single image inference (via test_model.py)

## 🔧 Troubleshooting

### GPU Not Detected

**Problem:** Training uses CPU instead of GPU

**Solution:**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Webcam Not Working

**Problem:** Cannot access webcam

**Solution:**
- Check webcam permissions
- Try different camera index: `cv2.VideoCapture(1)`
- On Windows, ensure camera drivers are installed

### Model Not Found

**Problem:** `Model not found` error

**Solution:**
```bash
# Train the model first
python train_model.py

# Or use pre-trained model
python capstone_project.py --model yolov8s.pt
```

## 📈 Future Improvements

### Potential Enhancements
1. **Multi-person tracking** - Track multiple people simultaneously
2. **Alert system** - Sound/visual alerts for non-compliance
3. **Database logging** - Store detection history
4. **Web interface** - Flask/FastAPI web application
5. **Mobile deployment** - Optimize for mobile devices
6. **Edge deployment** - Deploy on edge devices (Jetson Nano, etc.)

### Advanced AI Exploration
- **Generative AI:** Explore Stable Diffusion for synthetic data generation
- **GANs:** Generate realistic safety equipment images for data augmentation
- **Vision Transformers (ViT):** Experiment with transformer-based detection
- **Multi-modal AI:** Combine vision with audio for enhanced safety monitoring

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📝 License

This project is part of the Vision AI Training Program.

## 👤 Author

Developed as part of the Four Week Training Program for Vision AI.

---

**Status:** ✅ Complete - All tasks implemented and tested

