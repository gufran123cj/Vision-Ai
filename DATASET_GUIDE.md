# 📚 Dataset Kullanım Rehberi

Projenizde iki dataset bulunuyor:

## 📦 Dataset'ler

### 1. **Hard Hat Workers.v14-raw_headclassonly.yolov8**
- **Amaç**: Kask tespiti
- **Sınıf**: `head` (1 sınıf)
- **Açıklama**: Kask takılmamış kafaları tespit eder
- **Split**: 
  - Train: 4,916 görüntü
  - Valid: 1,413 görüntü
  - Test: 706 görüntü
- **Kaynak**: [Roboflow - Hard Hat Workers](https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/14)

**Not**: Bu dataset'te `head` sınıfı = kask yok demektir. Yani tespit edilen head = violation.

### 2. **safety-vest.v1i.yolov8**
- **Amaç**: Güvenlik ekipmanı tespiti
- **Sınıflar**: `Goggles`, `HardHat`, `Vest` (3 sınıf)
- **Açıklama**: Gözlük, kask ve yelek tespiti
- **Split**:
  - Train: 1,119 görüntü
  - Valid: 106 görüntü
  - Test: 51 görüntü
- **Kaynak**: [Roboflow - Safety Vest](https://universe.roboflow.com/na-u7nn0/safety-vest-i1uyc/dataset/1)

**Not**: Bu dataset'te `Vest` sınıfını kullanıyoruz.

---

## 🚀 Model Eğitimi

### Hızlı Başlangıç

#### 1. Her İki Modeli Eğit (Önerilen - GPU ile)
```bash
# İnteraktif mod
py train_models.py

# Komut satırı ile (GPU otomatik algılama)
py train_models.py --train-all --epochs 50 --device auto

# GPU ile zorla
py train_models.py --train-all --epochs 50 --device cuda

# CPU ile
py train_models.py --train-all --epochs 50 --device cpu
```

#### 2. Sadece Helmet Modeli
```bash
py train_models.py --train-helmet --epochs 50 --device cuda
```

#### 3. Sadece Vest Modeli
```bash
py train_models.py --train-vest --epochs 50 --device cuda
```

### Parametreler

```bash
py train_models.py \
  --train-all \                                    # Her iki modeli eğit
  --epochs 50 \                                    # Eğitim epoch sayısı
  --imgsz 640 \                                    # Görüntü boyutu
  --device cuda \                                  # GPU kullan (auto, cuda, veya cpu)
  --helmet-dataset "models/Hard Hat Workers.v14-raw_headclassonly.yolov8" \
  --vest-dataset "models/safety-vest.v1i.yolov8"
```

### GPU Desteği

**Otomatik GPU Algılama:**
- `--device auto` veya device belirtilmezse otomatik algılar
- GPU varsa GPU, yoksa CPU kullanır
- Batch size otomatik olarak GPU memory'ye göre ayarlanır:
  - 8GB+ GPU: batch size 32
  - 6GB GPU (RTX 2060): batch size 24
  - 4GB GPU: batch size 16
  - CPU: batch size 8

**Manuel Device Seçimi:**
- `--device cuda`: GPU kullan (GPU yoksa uyarı verir)
- `--device cpu`: CPU kullan
- `--device auto`: Otomatik algılama (varsayılan)

### Eğitim Süresi

**CPU ile:**
- **Helmet Model**: ~30-120 dakika (epoch sayısına göre)
- **Vest Model**: ~20-80 dakika (epoch sayısına göre)

**GPU ile (RTX 2060 6GB örneği):**
- **Helmet Model**: ~5-15 dakika (50 epochs)
- **Vest Model**: ~3-10 dakika (50 epochs)
- **Batch size**: Otomatik olarak 24 (GPU memory'ye göre)

**Not**: GPU kullanımı eğitim süresini önemli ölçüde azaltır (10-20x hızlanma).

---

## 📊 Model Kullanımı

### Eğitilmiş Modelleri Kullanma

Eğitim tamamlandıktan sonra modeller `models/` klasörüne kaydedilir:

- `models/helmet_model_trained.pt`
- `models/vest_model_trained.pt`

### Programı Çalıştırma

```bash
py week4\capstone_project.py \
  --helmet-model models\helmet_model_trained.pt \
  --vest-model models\vest_model_trained.pt \
  --mode video \
  --input video.mp4
```

---

## 🔍 Dataset Analizi

### Hard Hat Dataset Analizi

```python
# Dataset yapısı
data.yaml:
  - Sınıf: head
  - Amaç: Kask takılmamış kafaları tespit
  
# Mantık:
# - head tespit edildi = kask yok = violation
# - head tespit edilmedi = kask var = compliant
```

### Safety Vest Dataset Analizi

```python
# Dataset yapısı
data.yaml:
  - Sınıflar: Goggles, HardHat, Vest
  - Amaç: Güvenlik ekipmanı tespiti
  
# Kullanım:
# - Vest sınıfını tespit ediyoruz
# - Vest tespit edildi = yelek var = compliant
# - Vest tespit edilmedi = yelek yok = violation
```

---

## ⚙️ Model Eğitimi Detayları

### Helmet Model Eğitimi

1. **Dataset**: Hard Hat Workers v14
2. **Sınıf**: `head` (kask yok = head var)
3. **Mantık**: 
   - Head tespit edildi → Helmet YOK
   - Head tespit edilmedi → Helmet VAR

### Vest Model Eğitimi

1. **Dataset**: Safety Vest v1i
2. **Sınıf**: `Vest`
3. **Mantık**:
   - Vest tespit edildi → Yelek VAR
   - Vest tespit edilmedi → Yelek YOK

---

## 🎯 Önerilen Eğitim Parametreleri

### CPU için:
```bash
py train_models.py --train-all --epochs 30 --imgsz 640 --device cpu
```

### GPU için (Önerilen):
```bash
py train_models.py --train-all --epochs 100 --imgsz 640 --device cuda
```

### Hızlı Test için:
```bash
# GPU ile hızlı test (5-10 dakika)
py train_models.py --train-all --epochs 10 --imgsz 416 --device cuda

# CPU ile hızlı test (20-30 dakika)
py train_models.py --train-all --epochs 10 --imgsz 416 --device cpu
```

---

## 📝 Eğitim Sonuçları

Eğitim tamamlandıktan sonra:
- Modeller: `runs/train/helmet_model/weights/best.pt`
- Grafikler: `runs/train/helmet_model/results.png`
- Confusion matrix: `runs/train/helmet_model/confusion_matrix.png`

Modeller otomatik olarak `models/` klasörüne kopyalanır.

---

## ✅ Kontrol Listesi

- [ ] Dataset'ler models/ klasöründe
- [ ] train_models.py script'i hazır
- [ ] Eğitim başlatıldı
- [ ] Helmet modeli eğitildi
- [ ] Vest modeli eğitildi
- [ ] Modeller models/ klasörüne kopyalandı
- [ ] Program test edildi

---

## 🐛 Sorun Giderme

### Dataset yolu hatası
```
❌ Dataset yaml dosyası bulunamadı
```

**Çözüm**: Dataset yolunu kontrol edin:
```bash
--helmet-dataset "models/Hard Hat Workers.v14-raw_headclassonly.yolov8"
```

### Eğitim çok yavaş
**Çözüm**: 
- Epoch sayısını azaltın: `--epochs 20`
- Image size'ı küçültün: `--imgsz 416`
- GPU kullanın (varsa)

### Model bulunamıyor
**Çözüm**: 
- Eğitim tamamlandı mı kontrol edin
- `runs/train/` klasörünü kontrol edin
- Modeller otomatik kopyalanır, manuel kontrol edin

---

**Hazırsınız! 🎉**

Önce modelleri eğitin, sonra programı çalıştırın!


