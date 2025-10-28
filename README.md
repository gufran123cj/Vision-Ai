# 🧠 Vision AI - Week 1: From ML to DL

Bu hafta, klasik Makine Öğrenmesinden (ML) Derin Öğrenmeye (DL) geçiş sürecini teoriden pratiğe adım adım öğreneceksin.  
Her modül, teorik açıklamalar, uygulamalı kod örnekleri ve mini deneylerle desteklenmiştir.

---

## **01 — Theory: From ML to DL**

### 🎯 Amaç
Klasik ML yaklaşımlarının sınırlarını ve DL’in neden doğduğunu anlamak;  
Gradient Descent mantığını sezgisel olarak gözlemlemek.

### 📘 İçerik
- ML ve DL farkı
- Supervised / Unsupervised Learning
- Neural Networks: katman, nöron, aktivasyon fonksiyonları
- Backpropagation ve Gradient Descent
- Overfitting, learning rate ve loss dinamikleri

### 🧩 Görevler
1. **Not alma:** Her kavram için 3–5 maddelik özet + küçük şemalar.
2. **Mini deney:** `y ≈ w*x` modelinde MSE’yi minimize eden GD döngüsü.
3. **Karşılaştırma:** Aynı veriyi sklearn `LinearRegression` ile çöz, sonuç farklarını incele.

### ✅ Çıktı Dosyası
📄 `01_theory_from_ml_to_dl.ipynb`

---

## **02 — PyTorch Fundamentals**

### 🎯 Amaç
PyTorch kütüphanesinin temel taşlarını (Tensor, Autograd, Device) öğrenmek ve  
grad takibi mantığını anlamak.

### 📘 İçerik
- `torch.Tensor` oluşturma, shape ve dtype kontrolü
- Cihaz seçimi (`cpu` / `cuda`)
- `requires_grad`, `backward()`, `no_grad()`
- `view()` / `reshape()` farkı
- `matmul` ile matris çarpımı

### 🧩 Görevler
1. **Tensor 101:**
   - Rastgele tensor oluştur, dilimle, yeniden şekillendir.
   - `matmul` ile basit lineer işlem yap.
2. **Autograd Deneyi:**
   - `z = (x*w + b).sum()` ifadesi üzerinden backward() çağır.
   - `w.grad` ve `b.grad` değerlerini gözlemle.

### ✅ Çıktı Dosyası
📄 `02_pytorch_fundamentals.ipynb`

---

## **03 — Simple NN on Tabular Data**

### 🎯 Amaç
Küçük tabular veri üzerinde mini sinir ağı kurmak, eğitmek ve ML ile kıyaslamak.

### 📘 İçerik
- `nn.Sequential` vs `nn.Module` farkı
- Mini ağ yapısı: `Linear → ReLU → Linear → Sigmoid`
- Loss fonksiyonları (`BCELoss`, `MSELoss`)
- Optimizasyon: `SGD` ve `Adam`
- Eğitim döngüsü: `forward → loss → backward → step → zero_grad`
- Dataloader ve epoch kavramları

### 🧩 Görevler
1. **İlk Ağ:**
   - Sequential model oluştur, 50–100 epoch eğit.
   - Train/Validation ayrımı yap.
2. **Performans Takibi:**
   - Her epoch sonunda train/val loss değerlerini kaydet.
3. **Kıyaslama:**
   - Aynı problemi sklearn `LogisticRegression` ile çöz.
   - DL’in fazla/az geldiği durumları yorumla.

### ✅ Çıktı Dosyası
📄 `03_simple_nn_tabular.ipynb`

---

## 🔍 Ekstra Öneriler
- `03_simple_nn_tabular.ipynb` içerisine **loss/accuracy grafik çizimleri** ekle.
- **state_dict** ile en iyi modeli kaydetmeyi uygula.
- GPU kullanımını otomatik hale getir:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  ```

---

## 📁 Dosya Yapısı
```
VisionAI_Week1/
│
├── 01_theory_from_ml_to_dl.ipynb
├── 02_pytorch_fundamentals.ipynb
├── 03_simple_nn_tabular.ipynb
└── README.md
```

---

## 🧭 Kapanış Notu
Bu üç modül tamamlandığında, yalnızca DL’in neden ortaya çıktığını değil,  
aynı zamanda modern AI modellerinin **veriden öğrenme sürecini** kod seviyesinde anlamış olacaksın.

