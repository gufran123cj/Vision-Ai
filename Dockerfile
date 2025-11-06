# PyTorch 2.4 + CUDA 12.1 runtime (güncel ve yaygın)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Sistem bağımlılıkları (OpenCV/ffmpeg ve bazı GUI lib'leri)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 ffmpeg wget \
 && rm -rf /var/lib/apt/lists/*

# (Varsa) Python bağımlılıkları
# Not: requirements.txt yoksa bu bloğu atlayabilirsin.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# Projeyi kopyala
COPY . .

# İsteğe bağlı: proje içi importlar için
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Uygulama bir port dinlemiyorsa EXPOSE şart değil, ama kalsın
EXPOSE 5000

# Varsayılan: etkileşimli kabuk; compose ile exec yapıp komutları çalıştırırsın
CMD ["bash"]
