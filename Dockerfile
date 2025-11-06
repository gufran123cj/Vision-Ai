# Dockerfile for Vision AI Training Application with CUDA Support
# Base image with CUDA support for GPU acceleration
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# Create symlink for python command
RUN ln -s /usr/bin/python3.12 /usr/bin/python && \
    ln -s /usr/bin/python3.12 /usr/bin/python3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 11.8 support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY . .

# Create necessary directories for outputs, models, and data
RUN mkdir -p \
    data \
    output \
    models \
    runs/train \
    runs/detect \
    week1/day5-7/output \
    week2/day1-3/output \
    week2/day4-5/output \
    week2/day6-7/output \
    week3/day1-3/output \
    week3/day4-7/output \
    week4/output \
    week4/YOLO-Safety-Equipment-Detection-main/data/train/images \
    week4/YOLO-Safety-Equipment-Detection-main/data/valid/images \
    week4/YOLO-Safety-Equipment-Detection-main/data/test/images \
    week4/YOLO-Safety-Equipment-Detection-main/runs/train \
    week4/YOLO-Safety-Equipment-Detection-main/runs/detect \
    week4/YOLO-Safety-Equipment-Detection-main/results

# Set permissions
RUN chmod -R 755 /app

# Expose port (if using Flask/FastAPI server)
EXPOSE 5000

# Default command - interactive bash shell for week 4 scripts
# Users can run:
#   python week4/YOLO-Safety-Equipment-Detection-main/train_model.py
#   python week4/YOLO-Safety-Equipment-Detection-main/test_model.py
#   python week4/YOLO-Safety-Equipment-Detection-main/capstone_project.py --mode webcam
CMD ["/bin/bash"]
