import pandas as pd
import re
from pathlib import Path

# ===========================================================================
# 1. TEMİZLİK FONKSİYONU (Aynı Kalıyor)
# ===========================================================================
def profesyonel_akademik_temizleyici(metin):
    if not isinstance(metin, str) or len(metin.strip()) < 10:
        return ""
    
    # Kuyruğu (Anahtar Kelimeler) kes
    metin = re.split(r"\n?\s*(Anahtar Kelimeler|Anahtar Sözcükler|Keywords).*", metin, flags=re.IGNORECASE | re.DOTALL)[0]

    satirlar = metin.strip().split('\n')
    temiz_satirlar = []
    metadata_etiketleri = ["TEZ ÖZET", "Yazar :", "Üniversite :", "Anabilim Dalı :", "Tez Danışmanı :", "Sayfa Sayısı :"]
    baslangic_bulundu = False
    
    for satir in satirlar:
        s = satir.strip()
        if not s or any(etiket.lower() in s.lower() for etiket in metadata_etiketleri):
            continue
        if s.lower() in ["özet", "abstract", "özet:", "abstract:"]:
            continue
        if not baslangic_bulundu:
            if s.endswith(('.', '?', '!')) and len(s) > 20:
                baslangic_bulundu = True
            else:
                continue
        temiz_satirlar.append(s)

    final_metin = " ".join(temiz_satirlar)
    return re.sub(r'\s+', ' ', final_metin).strip()

# ===========================================================================
# 2. PROJE DİZİN YÖNETİMİ (SENİN YAPIYA ÖZEL)
# ===========================================================================

# Şu anki script: .../Bitirme_projesi/data_pipeline/prototip_temizlik.py
SCRIPT_DIR = Path(__file__).resolve().parent  # data_pipeline klasörü
ROOT_DIR = SCRIPT_DIR.parent                   # Bitirme_projesi (Ana Dizin)

# GİRDİ: data_pipeline/data/ klasörü içindeki dosyalar
insan_csv = SCRIPT_DIR / "data" / "insan_verisi_temiz.csv"
yz_csv = SCRIPT_DIR / "data" / "gemini_tezleri_v2.csv"

# ÇIKTI: Ana dizinde (Bitirme_projesi) prototip/data/ klasörü
output_dir = ROOT_DIR / "prototip" / "data"
output_dir.mkdir(parents=True, exist_ok=True)
cikti_dosyasi = output_dir / "prototip_dataset.csv"

# ===========================================================================
# 3. İŞLEM VE KAYIT
# ===========================================================================

print(f"🚀 İşlem Başlıyor...")
print(f"📥 Girdi Dosyaları: {SCRIPT_DIR}/data/")
print(f"📤 Çıktı Hedefi: {output_dir}")

# Verileri oku ve temizle
df_insan = pd.read_csv(insan_csv).head(500)
df_insan['metin'] = df_insan['metin'].apply(profesyonel_akademik_temizleyici)
df_insan['label'] = 0

df_yz = pd.read_csv(yz_csv)
df_yz['metin'] = df_yz['metin'].apply(profesyonel_akademik_temizleyici)
df_yz['label'] = 1

# Birleştir, karıştır ve filtrele
df_prototip = pd.concat([df_insan[['metin', 'label']], df_yz[['metin', 'label']]], ignore_index=True)
df_prototip = df_prototip.sample(frac=1, random_state=42).reset_index(drop=True)
df_prototip = df_prototip[df_prototip['metin'].str.len() > 100]

# Kaydet
df_prototip.to_csv(cikti_dosyasi, index=False, encoding="utf-8")

print(f"✅ Bitti! Prototip veri setin ana dizindeki '{output_dir.relative_to(ROOT_DIR)}' klasöründe.")