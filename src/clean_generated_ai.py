# src/clean_generated_ai.py
import pandas as pd
import re
import os
from pathlib import Path

def master_ai_cleaner(text):
    """
    Yapılan tüm üretimlerde metinlerin başına eklenen gereksiz etiketleri, 
    başlıkları ve kare (#) işaretlerini temizler. Metnin ortasına asla dokunmaz.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. ADIM: Kelimeli kalıpları metnin EN BAŞINDAN (^) siler.
    text = re.sub(
        r"^#+\s*(Araştırma Özeti|Tez Özeti|Akademik Metin|Özet|Abstract|Açıklama)[:\s]*", 
        "", 
        text, 
        flags=re.IGNORECASE
    )
    
    # 2. ADIM: Sadece '#' veya '##' bırakılan durumları metnin en başından temizler.
    text = re.sub(r"^#+\s*", "", text)
    
    # 3. ADIM: Fazla boşlukları teke indirir.
    return " ".join(text.split())

def main():
    ROOT_DIR = Path(__file__).parent.parent
    interim_dir = ROOT_DIR / "data" / "interim"
    
    # Temizlenecek hedef dosyaların listesi
    target_files = ["ai_val.csv", "ai_test.csv", "ai_train.csv"]
    
    print("🧹 [GÜVENLİ MASTER TEMİZLİK MOTORU BAŞLATILDI]")
    print("-" * 50)
    
    for file_name in target_files:
        file_path = interim_dir / file_name
        
        if not file_path.exists():
            print(f"⚠️ Atlandı: {file_name} dosyası bulunamadı.")
            continue
            
        print(f"📂 {file_name} yükleniyor ve hafızada temizleniyor...")
        df = pd.read_csv(file_path)
        
        if 'metin' not in df.columns:
            print(f"❌ Hata: {file_name} içinde 'metin' sütunu bulunamadı!")
            continue
            
        # Temizliği uygula
        df['metin'] = df['metin'].astype(str).apply(master_ai_cleaner)
        
        # --- KRİTİK DEĞİŞİKLİK: YENİ DOSYA ADI OLUŞTURMA ---
        # Örn: ai_train.csv -> ai_train_v2.csv
        new_file_name = file_name.replace(".csv", "_v2.csv")
        new_file_path = interim_dir / new_file_name
        
        # Yeni dosyaya kaydet (Orijinal dosya el değmeden arkada kalıyor)
        df.to_csv(new_file_path, index=False, encoding="utf-8")
        print(f"✨ Başarılı: Temizlenmiş veri '{new_file_name}' adıyla kaydedildi. (Satır: {len(df)})")
        print("-" * 50)
        
    print("\n🎯 [İŞLEM TAMAMLANDI] Orijinal dosyalarına dokunulmadı. '_v2' dosyaların eğitime hazır!")

if __name__ == "__main__":
    main()