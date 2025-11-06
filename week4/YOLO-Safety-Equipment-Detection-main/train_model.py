"""
YOLO Safety Equipment Detection - Model Training Script
Bu script custom safety equipment modelini eğitir.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch

# Working directory ayarla
WORK_DIR = Path(__file__).parent
os.chdir(WORK_DIR)

print("=" * 60)
print("YOLO Safety Equipment Detection - Model Training")
print("=" * 60)
print(f"Working directory: {WORK_DIR}")

# Model ve data yolları
DATA_YAML = Path("data/data.yaml").absolute()

# Dataset kontrolü
if not DATA_YAML.exists():
    print(f"❌ Hata: data.yaml dosyası bulunamadı: {DATA_YAML}")
    exit(1)

# YAML dosyasını absolute path'lerle düzelt
with open(DATA_YAML, 'r') as f:
    yaml_data = yaml.safe_load(f)

# Path'leri absolute yap
data_dir = DATA_YAML.parent
yaml_data['train'] = str(data_dir / 'train' / 'images')
yaml_data['val'] = str(data_dir / 'valid' / 'images')

# Düzeltilmiş YAML'i geçici olarak kaydet
fixed_yaml = DATA_YAML.parent / 'data_fixed.yaml'
with open(fixed_yaml, 'w') as f:
    yaml.dump(yaml_data, f)

DATA_YAML = str(fixed_yaml)

print(f"\n📊 Dataset: {DATA_YAML}")
print("📦 Classes: Helmet, Goggles, Jacket, Gloves, Footwear")

# GPU kontrolü - HER ZAMAN GPU KULLAN
if not torch.cuda.is_available():
    print("\n❌ HATA: CUDA/GPU bulunamadı!")
    print("   GPU kullanılamıyor. Lütfen:")
    print("   1. NVIDIA GPU driver'ının kurulu olduğundan emin olun")
    print("   2. PyTorch CUDA versiyonunu kurun:")
    print("      pip uninstall torch torchvision torchaudio -y")
    print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    exit(1)

# GPU bilgilerini göster
device = 0  # GPU kullan (0 = ilk GPU)
gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

print(f"\n🎮 GPU Kullanılıyor: {gpu_name}")
print(f"   GPU Memory: {gpu_memory:.2f} GB")
print(f"   CUDA Version: {torch.version.cuda}")

# GPU memory'ye göre batch size ayarla
if gpu_memory >= 8:
    batch_size = 32  # 8GB+ GPU için
elif gpu_memory >= 6:
    batch_size = 24  # 6GB GPU için (RTX 2060 gibi)
else:
    batch_size = 16  # 4GB GPU için

print("\n🚀 Model eğitimi başlıyor...")
print(f"   Base model: yolov8s.pt")
print(f"   Device: GPU ({gpu_name})")
print(f"   Epochs: 25")
print(f"   Image size: 640")
print(f"   Batch size: {batch_size}")

try:
    # Pre-trained model yükle
    model = YOLO("yolov8s.pt")
    print("\n✅ Pre-trained model yüklendi")
    
    # Model eğit
    print("\n⏳ Eğitim başlıyor (bu işlem biraz zaman alabilir)...")
    results = model.train(
        data=DATA_YAML,
        epochs=25,
        imgsz=640,
        batch=batch_size,
        plots=True,
        project="runs/train",
        name="safety_equipment",
        save=True,
        device=device,  # GPU veya CPU
        workers=0  # Windows'ta multiprocessing sorunları için 0 (single-threaded)
    )
    
    print("\n" + "=" * 60)
    print("✅ Eğitim tamamlandı!")
    print(f"📁 Model kaydedildi: runs/train/safety_equipment/weights/best.pt")
    print(f"📊 Eğitim sonuçları: runs/train/safety_equipment/")
    print("\n💡 Artık bu modeli kullanarak inference yapabilirsiniz:")
    print("   python quick_test.py")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Hata: {e}")
    import traceback
    traceback.print_exc()

