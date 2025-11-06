"""
Quick Test Script - YOLO Safety Equipment Detection
Bu script otomatik olarak test inference yapar.
"""

import os
from pathlib import Path
from ultralytics import YOLO

# Working directory ayarla
WORK_DIR = Path(__file__).parent
os.chdir(WORK_DIR)

print("=" * 60)
print("YOLO Safety Equipment Detection - Quick Test")
print("=" * 60)
print(f"Working directory: {WORK_DIR}")

# Test görüntüsü
TEST_IMAGE = "data/test/images/1.jpeg"

if not Path(TEST_IMAGE).exists():
    print(f"❌ Test görüntüsü bulunamadı: {TEST_IMAGE}")
    exit(1)

print(f"\n📸 Test görüntüsü: {TEST_IMAGE}")
print("🔄 Model yükleniyor...")

# Önce custom model kontrol et, yoksa pre-trained kullan
CUSTOM_MODEL = "runs/train/safety_equipment/weights/best.pt"

try:
    if Path(CUSTOM_MODEL).exists():
        # Custom trained model kullan
        model = YOLO(CUSTOM_MODEL)
        print(f"✅ Custom model yüklendi: {CUSTOM_MODEL}")
        print("   Classes: Helmet, Goggles, Jacket, Gloves, Footwear")
    else:
        # Pre-trained model kullan (genel amaçlı)
        model = YOLO("yolov8s.pt")
        print("⚠️  Custom model bulunamadı, pre-trained model kullanılıyor (yolov8s.pt)")
        print("   ⚠️  Bu model safety equipment tespiti için eğitilmemiş!")
        print("   💡 Custom model eğitmek için: python train_model.py")
    
    print("\n🔍 Inference başlıyor...")
    print("   Confidence threshold: 0.25")
    
    # Inference yap
    results = model.predict(
        source=TEST_IMAGE,
        conf=0.25,
        save=True,
        project="runs/detect",
        name="quick_test"
    )
    
    print("\n✅ Inference tamamlandı!")
    print(f"📁 Sonuçlar kaydedildi: runs/detect/quick_test/")
    print("\n" + "=" * 60)
    
except Exception as e:
    print(f"\n❌ Hata: {e}")
    import traceback
    traceback.print_exc()

