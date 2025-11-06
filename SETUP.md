# Setup Guide - Vision AI Training Program

## Prerequisites

- **Python 3.12** (required for CUDA support - Python 3.14+ not supported yet)
- pip package manager
- **NVIDIA GPU** (optional, but highly recommended for model training)
- **CUDA-capable GPU** with NVIDIA drivers (optional, for GPU acceleration)
- Git (optional, for version control)
- Webcam (optional, for real-time demos)

## Installation Steps

### 1. Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd vision-ai-training

# Or simply download and extract the project
```

### 2. Python 3.12 Kurulumu

**ÖNEMLİ**: Python 3.12 gerekli çünkü PyTorch CUDA desteği için Python 3.8-3.12 destekleniyor.

**Windows:**
1. Python 3.12 İndir: https://www.python.org/downloads/release/python-31212/
2. Kurulum sırasında **"Add Python 3.12 to PATH"** seçeneğini işaretleyin
3. Kurulumu tamamlayın

**Kontrol:**
```powershell
py -3.12 --version
```

### 3. Create Virtual Environment

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

### 4. GPU Desteği Kurulumu (Önerilen)

**CUDA ile PyTorch Kurulumu:**
```bash
# Önce mevcut PyTorch'u kaldırın (eğer varsa)
pip uninstall torch torchvision torchaudio -y

# CUDA 11.8 ile PyTorch kurun
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**GPU Kontrolü:**
```python
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Not**: GPU yoksa veya kurmak istemiyorsanız, normal `pip install -r requirements.txt` komutu CPU versiyonu kuracaktır.

### 5. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Verify Installation

```bash
python -c "import torch; import cv2; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('OpenCV:', cv2.__version__)"
```

## Running the Examples

### Week 1 Examples

**Days 1-2: Neural Networks Basics**
```bash
python week1/day1-2/neural_networks_basics.py
```

**Days 3-4: PyTorch Introduction**
```bash
python week1/day3-4/pytorch_introduction.py
```

**Days 5-7: OpenCV Fundamentals**
```bash
python week1/day5-7/opencv_fundamentals.py
```

### Week 2 Examples

**Days 1-3: CNN from Scratch**
```bash
python week2/day1-3/cnn_from_scratch.py
```

**Days 4-5: Training & Optimization**
```bash
python week2/day4-5/training_optimization.py
```

**Days 6-7: Transfer Learning**
```bash
python week2/day6-7/transfer_learning.py
```

### Week 3 Examples

**Days 1-3: Object Detection (YOLO)**
```bash
python week3/day1-3/object_detection_yolo.py
```

**Days 4-7: Image Segmentation**
```bash
python week3/day4-7/image_segmentation.py
```

### Week 4: Capstone Project

**Model Eğitimi (GPU ile - Önerilen):**
```bash
# İnteraktif mod
py train_models.py

# Komut satırı ile
py train_models.py --train-all --epochs 50 --device cuda
```

**Real-time Vision Application:**
```bash
# İnteraktif mod
py week4/capstone_project.py

# Webcam ile
py week4/capstone_project.py --mode webcam --helmet-model models/helmet_model_trained.pt --vest-model models/vest_model_trained.pt

# Video ile
py week4/capstone_project.py --mode video --input video.mp4 --helmet-model models/helmet_model_trained.pt --vest-model models/vest_model_trained.pt

# Tek görüntü ile
py week4/capstone_project.py --mode image --input path/to/image.jpg --output result.jpg --helmet-model models/helmet_model_trained.pt --vest-model models/vest_model_trained.pt
```

**Detaylar için:** [`DATASET_GUIDE.md`](DATASET_GUIDE.md) ve [`week4/MODEL_SETUP.md`](week4/MODEL_SETUP.md)

## Troubleshooting

### CUDA/GPU Issues

**GPU Kurulumu:**

1. **NVIDIA Driver Kurulumu:**
   - NVIDIA Driver İndir: https://www.nvidia.com/Download/index.aspx
   - GPU modelinizi seçin ve driver'ı indirin
   - Kurulumu tamamlayın ve bilgisayarı yeniden başlatın
   - Kontrol: `nvidia-smi`

2. **PyTorch CUDA Kurulumu:**
   ```bash
   # Mevcut PyTorch'u kaldır
   pip uninstall torch torchvision torchaudio -y
   
   # CUDA 11.8 ile kur
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **GPU Testi:**
   ```python
   python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
   ```

**Sorun Giderme:**
- Python 3.12 kullandığınızdan emin olun (3.14+ CUDA desteklemiyor)
- NVIDIA driver'ın yüklü olduğunu kontrol edin (`nvidia-smi`)
- Virtual environment'ın aktif olduğundan emin olun
- PyTorch'un CUDA versiyonu kurulduğunu kontrol edin (`torch.cuda.is_available()`)

### OpenCV Issues

If OpenCV installation fails:
```bash
pip install opencv-python-headless
```

### Webcam Access Issues

**Linux:**
- Ensure user is in `video` group: `sudo usermod -a -G video $USER`
- Reboot or log out/in

**Windows:**
- Check device manager for camera permissions
- Ensure no other application is using the webcam

**Mac:**
- Grant camera permissions in System Preferences > Security & Privacy

## Docker Setup (Optional)

### Build Docker Image
```bash
docker build -t vision-ai-app .
```

### Run with Docker
```bash
docker-compose up
```

## Data Download

### Automatic Downloads

Some scripts will automatically download datasets:
- MNIST: Automatically downloaded by PyTorch
- CIFAR-10: Automatically downloaded by PyTorch
- YOLO models: Automatically downloaded on first use

### Custom Datasets

Projede iki özel dataset mevcuttur:
- **Hard Hat Workers** (`models/Hard Hat Workers.v14-raw_headclassonly.yolov8/`)
- **Safety Vest** (`models/safety-vest.v1i.yolov8/`)

Bu dataset'lerle custom model eğitebilirsiniz:
```bash
py train_models.py --train-all --epochs 50 --device cuda
```

**Detaylar:** [`DATASET_GUIDE.md`](DATASET_GUIDE.md)

## Next Steps

1. Start with Week 1 examples
2. Follow the weekly schedule
3. Experiment with different parameters
4. Try your own datasets
5. Build your capstone project

## Getting Help

- Check the README.md for project overview
- Review inline code comments
- Refer to official documentation:
  - [PyTorch Docs](https://pytorch.org/docs/)
  - [OpenCV Docs](https://docs.opencv.org/)
  - [YOLO Docs](https://docs.ultralytics.com/)

