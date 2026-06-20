# 🕵️‍♀️ Akademik Metin AI Dedektörü (LLM Detector)

Bu proje, Türkçe akademik metinlerin (özetler, makaleler) gerçek bir insan tarafından mı yoksa Büyük Dil Modelleri (ChatGPT, Claude, Gemini vb.) tarafından mı yazıldığını tespit etmek amacıyla geliştirilmiş derin öğrenme tabanlı bir doğal dil işleme (NLP) sistemidir. 

Yazılım Mühendisliği bitirme projesi kapsamında, 110 milyon parametreli **BERTurk** modeli, izole edilmiş bir veri seti üzerinde özel bir hiperparametre stratejisiyle ince ayar (fine-tuning) işleminden geçirilerek eğitilmiştir.

## 🚀 Proje Özeti
* **Model Mimarisi:** dbmdz/bert-base-turkish-cased (BERTurk)
* **Kullanıcı Arayüzü:** Canlı metin analizi için Streamlit tabanlı interaktif web uygulaması.

---

## 📂 Proje Mimarisi ve Klasör Yapısı

Proje, veri bilimi standartlarına uygun olarak modüler bir yapıda tasarlanmıştır:

```text
📂 Bitirme_projesi/
 ├── 📁 Data/                
 │    ├── 📁 raw/            # İnternetten/API'lerden çekilmiş ham ve kirli veriler
 │    └── 📁 interim/        # Modele girmeye hazır, temizlenmiş ve etiketlenmiş CSV'ler
 ├── 📁 feature raporları/   # Metin özniteliklerinin istatistiksel analiz raporları
 ├── 📁 models/              # (Gitignored) Eğitilmiş modelin kaydedildiği dizin
 ├── 📁 sonuclar/            # Nihai test sonuçları, karmaşıklık matrisi (Confusion Matrix)
 ├── 📁 src/                 # Projenin motor kodları
 │    ├── analyze_features.py    # İnsan ve AI verisi arasındaki istatistiksel farkların analizi
 │    ├── clean_generated_ai.py  # Üretilen sentetik verilerin temizlenmesi
 │    ├── generate_*_ai.py       # LLM API'leri ile eğitim/test verilerinin üretilmesi
 │    ├── split_human_data.py    # İnsan verilerinin Train/Val/Test olarak izole edilmesi
 │    ├── train_local.py         # M1 MPS üzerinde BERTurk modelinin eğitilmesi
 │    └── test_local.py          # İzole test seti üzerinde modelin metrik değerlendirmesi
 ├── 📄 app.py               # Streamlit web arayüzü ana dosyası
 └── 📄 .gitignore           # Büyük model dosyalarını ve önbellekleri hariç tutma kuralları
