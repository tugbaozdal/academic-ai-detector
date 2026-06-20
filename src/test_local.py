import pandas as pd
import numpy as np
import os
import torch
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🚀 Nihai Test Süreci Başlıyor...")
    
    # Dosya yolları (Senin klasör yapına göre)
    model_yolu = "./models/berturk_final"
    human_test_yolu = "./Data/interim/human_test.csv"
    ai_test_yolu = "./Data/interim/ai_test_v2.csv"
    cikti_klasoru = "./sonuclar"
    
    os.makedirs(cikti_klasoru, exist_ok=True)

    # 1. TEST VERİLERİNİ OKU
    if not os.path.exists(human_test_yolu) or not os.path.exists(ai_test_yolu):
        print("❌ HATA: Test dosyaları Data/interim altında bulunamadı!")
        return

    print("📂 İzole edilmiş test verileri yükleniyor...")
    df_human = pd.read_csv(human_test_yolu)[['metin', 'label']].dropna()
    df_ai = pd.read_csv(ai_test_yolu)[['metin', 'label']].dropna()
    df_test = pd.concat([df_human, df_ai], ignore_index=True)
    
    metinler = df_test['metin'].tolist()
    y_true = df_test['label'].tolist()

    # 2. ŞAMPİYON MODELİ YÜKLE
    print("🧠 Eğitilen nihai model hafızaya alınıyor...")
    dedektor = pipeline(
        "text-classification", 
        model=model_yolu, 
        tokenizer=model_yolu,
        truncation=True, 
        max_length=256,
        device=-1
    )

    # 3. TAHMİNLERİ GERÇEKLEŞTİR
    print(f"🔍 {len(metinler)} adet hiç görülmemiş test metni üzerinde model tahmin yapıyor...")
    sonuclar = dedektor(metinler, batch_size=32)
    
    # Model çıktısındaki LABEL_0 ve LABEL_1 değerlerini 0 ve 1'e dönüştür
    y_pred = [int(s['label'].split('_')[-1]) for s in sonuclar]

    # 4. METRİKLERİ HESAPLA
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    rapor = classification_report(y_true, y_pred, digits=4, target_names=["İnsan (0)", "Yapay Zeka (1)"])

    # 5. EKRANA VE DOSYAYA YAZDIR
    metrik_metni = f"""==================================================
🏆 NİHAİ TEST SETİ PERFORMANS RAPORU
==================================================
Model Konumu: {model_yolu}
Test Edilen Toplam Metin: {len(metinler)} (İnsan: {len(df_human)} | AI: {len(df_ai)})

🧩 KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX):
{cm}

İnsan (0) -> Doğru Bilinen: {cm[0][0]} | Yanlışlıkla AI Denen: {cm[0][1]}
AI (1)    -> Yanlışlıkla İnsan Denen: {cm[1][0]} | Doğru Bilinen: {cm[1][1]}

🎯 ÖZET METRİKLER:
Genel Doğruluk (Accuracy): %{acc*100:.4f}
Kesinlik (Precision):      %{precision*100:.4f}
Duyarlılık (Recall):       %{recall*100:.4f}
F1-Skor (F1-Score):        %{f1*100:.4f}

📊 DETAYLI SINIFLANDIRMA RAPORU:
{rapor}
==================================================
"""
    
    print(metrik_metni)
    
    rapor_yolu = os.path.join(cikti_klasoru, "test_metrikleri.txt")
    with open(rapor_yolu, "w", encoding="utf-8") as f:
        f.write(metrik_metni)
        
    print(f"💾 Harika! Tüm bu metrikler tezin için '{rapor_yolu}' dosyasına kaydedildi.")

    # 6. MATPLOTLIB İLE GÖRSELLEŞTİRME VE KAYDETME
    print("🎨 Matplotlib ile görsel oluşturuluyor...")
    
    plt.figure(figsize=(8, 6)) 
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["İnsan (0)", "Yapay Zeka (1)"], 
                yticklabels=["İnsan (0)", "Yapay Zeka (1)"],
                annot_kws={"size": 14}) 
    
    plt.title('Nihai Model - Karmaşıklık Matrisi', fontsize=16, pad=15)
    plt.ylabel('Gerçek Değer', fontsize=12)
    plt.xlabel('Tahmin Edilen', fontsize=12)
    
    plt.tight_layout() 
    
    grafik_yolu = os.path.join(cikti_klasoru, "confusion_matrix.png")
    plt.savefig(grafik_yolu, dpi=300) 
    
    print(f"🖼️ Harika! Matris görselin yüksek çözünürlüklü olarak '{grafik_yolu}' dizinine kaydedildi.")

if __name__ == "__main__":
    main()