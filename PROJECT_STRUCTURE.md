# Project Structure

```
vision-ai-training/
│
├── README.md                    # Main project documentation
├── SETUP.md                     # Detailed setup instructions
├── DATASET_GUIDE.md            # Dataset usage and training guide
├── PROJECT_STRUCTURE.md        # This file - detailed project structure
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── Dockerfile                   # Docker container configuration
├── docker-compose.yml           # Docker Compose configuration
├── train_models.py             # Model training script (GPU support)
├── test_models.py              # Model testing script
├── download_safety_models.py   # Helper script for downloading models
├── test_installation.py        # Installation verification script
│
├── week1/                       # Week 1: AI & Deep Learning Foundations
│   ├── day1-2/
│   │   └── neural_networks_basics.py      # Neural network from scratch
│   ├── day3-4/
│   │   └── pytorch_introduction.py        # PyTorch basics, tensors, autograd
│   └── day5-7/
│       └── opencv_fundamentals.py         # Image processing with OpenCV
│
├── week2/                       # Week 2: Convolutional Neural Networks
│   ├── day1-3/
│   │   └── cnn_from_scratch.py            # CNN implementation (MNIST/CIFAR-10)
│   ├── day4-5/
│   │   └── training_optimization.py       # Optimizers, LR scheduling, augmentation
│   └── day6-7/
│       └── transfer_learning.py           # Transfer learning with ResNet, VGG, etc.
│
├── week3/                       # Week 3: Advanced Vision Tasks
│   ├── day1-3/
│   │   └── object_detection_yolo.py       # YOLO object detection, IoU, NMS
│   └── day4-7/
│       └── image_segmentation.py          # Semantic & instance segmentation (U-Net)
│
├── week4/                       # Week 4: Capstone Project & Deployment
│   ├── capstone_project.py                # Real-time safety equipment detection
│   ├── README.md                          # Capstone project documentation
│   └── MODEL_SETUP.md                     # Model setup and usage guide
│
├── models/                      # Custom models and datasets
│   ├── Hard Hat Workers.v14-raw_headclassonly.yolov8/  # Hard hat dataset
│   │   ├── train/              # Training images and labels
│   │   ├── valid/              # Validation images and labels
│   │   ├── test/               # Test images and labels
│   │   └── data.yaml           # Dataset configuration
│   ├── safety-vest.v1i.yolov8/            # Safety vest dataset
│   │   ├── train/              # Training images and labels
│   │   ├── valid/              # Validation images and labels
│   │   ├── test/               # Test images and labels
│   │   └── data.yaml           # Dataset configuration
│   ├── helmet_model_trained.pt # Trained helmet model (after training)
│   └── vest_model_trained.pt   # Trained vest model (after training)
│
├── runs/                        # Training results (created automatically)
│   └── train/                   # Training outputs
│       ├── helmet_model/        # Helmet model training results
│       └── vest_model/          # Vest model training results
│
└── data/                        # Dataset storage (created automatically)
    ├── MNIST/                  # MNIST dataset (auto-downloaded)
    └── cifar-10-batches-py/   # CIFAR-10 dataset (auto-downloaded)
```

## File Descriptions

### Week 1
- **neural_networks_basics.py**: Implementation of a simple neural network from scratch to understand forward/backward pass, gradient descent
- **pytorch_introduction.py**: PyTorch tensors, autograd, building models with torch.nn
- **opencv_fundamentals.py**: Image I/O, color spaces, transformations, filtering, edge detection

### Week 2
- **cnn_from_scratch.py**: Building CNNs from scratch for MNIST and CIFAR-10 classification
- **training_optimization.py**: Comparing optimizers, learning rate scheduling, data augmentation, regularization
- **transfer_learning.py**: Using pretrained models (ResNet, VGG, MobileNet) as feature extractors or fine-tuning

### Week 3
- **object_detection_yolo.py**: YOLO object detection, IoU calculation, NMS, real-time webcam detection
- **image_segmentation.py**: U-Net for semantic segmentation, instance segmentation with YOLO

### Week 4
- **capstone_project.py**: Real-time safety equipment detection application (hard hat & safety vest)
- **README.md**: Capstone project documentation and usage instructions
- **MODEL_SETUP.md**: Guide for setting up and using custom models

### Training Scripts
- **train_models.py**: Interactive model training script with GPU support
  - Automatic GPU detection
  - Batch size optimization based on GPU memory
  - Interactive mode for easy usage
  - Command-line interface for automation
- **test_models.py**: Script to test and verify trained models
- **download_safety_models.py**: Helper script for downloading pre-trained models

## Output Directories

Each week creates output directories for storing:
- Training plots and visualizations
- Model predictions
- Processed images
- Statistics and metrics

Output locations:
- `week1/day5-7/output/` - OpenCV processed images
- `week2/*/output/` - CNN training results and visualizations
- `week3/*/output/` - Detection and segmentation results
- `week4/output/` - Capstone project outputs

## Data Directories

Datasets are automatically downloaded to:
- `./data/` - Main data directory
- `./data/MNIST/` - MNIST dataset
- `./data/cifar-10-batches-py/` - CIFAR-10 dataset

## Model Files

### Pre-trained Models (Auto-downloaded)
- YOLO models: Downloaded to current directory on first use (`yolov8n.pt`, `yolov8n-seg.pt`)
- PyTorch pretrained models: Cached in torch hub directory

### Custom Trained Models
After training with `train_models.py`, models are saved to:
- `models/helmet_model_trained.pt` - Trained hard hat detection model
- `models/vest_model_trained.pt` - Trained safety vest detection model

### Training Results
Training outputs are saved to `runs/train/`:
- `helmet_model/` - Helmet model training results
  - `weights/best.pt` - Best model weights
  - `results.csv` - Training metrics
  - `args.yaml` - Training configuration
  - Visualizations and plots
- `vest_model/` - Vest model training results (same structure)






