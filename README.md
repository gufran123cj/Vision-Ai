# From ML to DL — Quick Guide

**Why DL?**
- Elle feature engineering yerine, **temsil öğrenimi** katmanlı olarak otomatikleşti.
- Büyük veri + GPU + yeni mimariler (CNN/RNN/Transformer) → **doğruluk ve ölçek**.
- Görüntü/ses/dil gibi karmaşık alanlarda **ML sınırlarını aştı** (ImageNet, konuşma, LLM’ler).

**Ne zaman DL?**
- Yüksek boyutlu ham veri (görüntü, ses, metin, zaman serileri) ve **çok veri** varsa.
- Özellikleri elle çıkarmanın **zor/masraflı** olduğu durumlarda.
- Doğruluğun kritik olduğu ve **nonlineer** ilişkilerin güçlü olduğu durumlarda.

**Ne zaman klasik ML?**
- **Az veri** + hızlı/yorumlanabilir çözümler gerektiğinde (lojistik regresyon, ağaçlar).
- Tablosal veride güçlü baseline’lar (Tree/GBM) sıklıkla yeterlidir.
- Eğitim/servis **maliyeti** sınırlıysa ve **açıklanabilirlik** önemliyse.

**Kısa kontrol listesi**
- Aktivasyon tek cümleleri, overfitting sinyali (loss↓/val_loss↑), learning rate uçları.
- LinearRegression (OLS) vs tek-parametreli “y = w x” kıyası.
