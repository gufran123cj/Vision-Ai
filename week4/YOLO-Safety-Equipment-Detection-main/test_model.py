"""
YOLO Safety Equipment Detection - Model Testing Script
Eğitilmiş modeli test etmek için kapsamlı script.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml
import cv2

# Working directory ayarla
WORK_DIR = Path(__file__).parent
os.chdir(WORK_DIR)


def main():
    """Main function - Windows multiprocessing fix."""
    print("=" * 60)
    print("YOLO Safety Equipment Detection - Model Testing")
    print("=" * 60)
    print(f"Working directory: {WORK_DIR}")

    # Model path
    MODEL_PATH = "runs/train/safety_equipment/weights/best.pt"
    DATA_YAML = "data/data.yaml"

    # Model kontrolü
    if not Path(MODEL_PATH).exists():
        print(f"\n❌ Hata: Model dosyası bulunamadı: {MODEL_PATH}")
        print("   💡 Önce modeli eğitin: python train_model.py")
        exit(1)

    print(f"\n📦 Model: {MODEL_PATH}")

    # Kullanıcıya seçenek sun
    print("\nNe yapmak istersiniz?")
    print("1. Validation (Model performansını ölç)")
    print("2. Tek görüntü üzerinde test")
    print("3. Tüm test görüntüleri üzerinde inference")
    print("4. Video üzerinde test")
    print("5. Webcam ile real-time detection")

    choice = input("\nSeçiminiz (1-5): ").strip()

    try:
        # Model yükle
        model = YOLO(MODEL_PATH)
        print(f"\n✅ Model yüklendi!")
        
        # Model bilgilerini göster
        if hasattr(model, 'names'):
            print(f"   Classes: {list(model.names.values())}")
        
        if choice == "1":
            # Validation
            print("\n🔍 Validation başlıyor...")
            
            # YAML dosyasını düzelt (absolute path)
            data_yaml_path = Path(DATA_YAML).absolute()
            with open(data_yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            data_dir = data_yaml_path.parent
            yaml_data['train'] = str(data_dir / 'train' / 'images')
            yaml_data['val'] = str(data_dir / 'valid' / 'images')
            
            fixed_yaml = data_dir / 'data_fixed.yaml'
            with open(fixed_yaml, 'w') as f:
                yaml.dump(yaml_data, f)
            
            # Windows multiprocessing fix: workers=0
            results = model.val(data=str(fixed_yaml), workers=0)
            
            print("\n✅ Validation tamamlandı!")
            print(f"📊 Sonuçlar: runs/train/safety_equipment/")
        
        elif choice == "2":
            # Tek görüntü test
            image_path_input = input("\nGörüntü path'i girin (örn: data/test/images/1.jpeg): ").strip()
            
            # Tırnak işaretlerini temizle (kullanıcı path'i tırnak içinde verebilir)
            image_path = image_path_input.strip('"').strip("'").strip()
            
            # Path'i kontrol et (absolute veya relative)
            image_path_obj = Path(image_path)
            
            # Path bulunamazsa farklı yolları dene
            if not image_path_obj.exists():
                # Eğer absolute path değilse, relative path olarak dene
                if not image_path_obj.is_absolute():
                    image_path_obj = WORK_DIR / image_path
                    if not image_path_obj.exists():
                        # Kullanıcının girdiği path'i olduğu gibi dene
                        image_path_obj = Path(image_path_input)
                
                # Hala bulunamazsa
                if not image_path_obj.exists():
                    print(f"\n❌ Görüntü bulunamadı: {image_path_input}")
                    print(f"   Denenen path 1: {image_path}")
                    print(f"   Denenen path 2: {image_path_obj}")
                    print(f"\n💡 İpuçları:")
                    print(f"   - Path'i direkt yapıştırın (tırnak işareti olmadan)")
                    print(f"   - Windows path için: C:\\Users\\... şeklinde")
                    print(f"   - Veya dosyayı sürükle-bırak yaparak path'i alın")
                    exit(1)
            
            # Final path'i absolute yap
            image_path = str(image_path_obj.absolute())
            print(f"   ✅ Dosya bulundu: {image_path}")
            
            print(f"\n🔍 Inference başlıyor...")
            print(f"   Image: {image_path}")
            
            conf_threshold = float(input("Confidence threshold (varsayılan: 0.25): ").strip() or "0.25")
            
            results = model.predict(
                source=image_path,
                conf=conf_threshold,
                save=True,
                project="runs/detect",
                name="single_test"
            )
            
            print(f"\n✅ Inference tamamlandı!")
            print(f"📁 Sonuç kaydedildi: runs/detect/single_test/")
        
        elif choice == "3":
            # Tüm test görüntüleri
            test_images = "data/test/images"
            
            if not Path(test_images).exists():
                print(f"❌ Test görüntüleri klasörü bulunamadı: {test_images}")
                exit(1)
            
            print(f"\n🔍 Tüm test görüntüleri üzerinde inference başlıyor...")
            print(f"   Test images: {test_images}")
            
            conf_threshold = float(input("Confidence threshold (varsayılan: 0.25): ").strip() or "0.25")
            
            results = model.predict(
                source=test_images,
                conf=conf_threshold,
                save=True,
                project="runs/detect",
                name="batch_test"
            )
            
            print(f"\n✅ Inference tamamlandı!")
            print(f"📁 Sonuçlar kaydedildi: runs/detect/batch_test/")
            print(f"   Toplam {len(results)} görüntü işlendi")
        
        elif choice == "4":
            # Video test
            video_path = input("\nVideo path'i girin: ").strip()
            
            # Tırnak işaretlerini temizle (kullanıcı path'i tırnak içinde verebilir)
            video_path = video_path.strip('"').strip("'").strip()
            
            # Path'i kontrol et (absolute veya relative)
            video_path_obj = Path(video_path)
            
            if not video_path_obj.exists():
                # Absolute path denemesi
                if not video_path_obj.is_absolute():
                    # Relative path olarak dene
                    video_path_obj = WORK_DIR / video_path
                
                if not video_path_obj.exists():
                    print(f"❌ Video bulunamadı: {video_path}")
                    print(f"   Denenen path: {video_path_obj}")
                    exit(1)
            
            video_path = str(video_path_obj.absolute())
            
            print(f"\n🔍 Video üzerinde inference başlıyor...")
            print(f"   Video: {video_path}")
            
            conf_threshold = float(input("Confidence threshold (varsayılan: 0.25): ").strip() or "0.25")
            
            results = model.predict(
                source=video_path,
                conf=conf_threshold,
                save=True,
                project="runs/detect",
                name="video_test"
            )
            
            print(f"\n✅ Inference tamamlandı!")
            print(f"📁 Sonuç kaydedildi: runs/detect/video_test/")
        
        elif choice == "5":
            # Webcam real-time detection
            print("\n" + "=" * 60)
            print("Real-Time PPE Detection - Webcam Mode")
            print("Press 'q' to quit")
            print("=" * 60)
            
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Error: Could not open webcam")
                print("   💡 Webcam'in bağlı olduğundan ve başka bir uygulama tarafından kullanılmadığından emin olun")
                exit(1)
            
            # Set webcam properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            print("\n✅ Webcam opened successfully")
            print("   Processing frames...")
            print("   Press 'q' to quit")
            
            conf_threshold = float(input("\nConfidence threshold (varsayılan: 0.25): ").strip() or "0.25")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from webcam")
                    break
                
                # Run inference
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    verbose=False
                )
                
                # Get annotated frame
                annotated_frame = results[0].plot()
                
                # Display frame
                cv2.imshow('PPE Detector - Real-Time', annotated_frame)
                
                frame_count += 1
                
                # Print status every 30 frames
                if frame_count % 30 == 0:
                    result = results[0]
                    detected_classes = []
                    if len(result.boxes) > 0:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            confidence = float(box.conf[0])
                            detected_classes.append(f"{class_name} ({confidence:.2f})")
                    
                    if detected_classes:
                        print(f"Frame {frame_count}: Detected: {', '.join(detected_classes)}")
                    else:
                        print(f"Frame {frame_count}: No detections")
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n✅ Webcam processing completed")
            print(f"   Processed {frame_count} frames")
        
        else:
            print("❌ Geçersiz seçim!")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

