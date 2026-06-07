import pandas as pd
import re
from pathlib import Path

# ===========================================================================
# 1. ENTEGRE TEMİZLİK FONKSİYONU
# ===========================================================================

def master_akademik_temizleyici(metin):
    if not isinstance(metin, str) or len(metin.strip()) < 10:
        return ""

    # --- AŞAMA 1: KARAKTER ONARIMI (Senin Kuralların - Satır sonları hariç) ---
    # Mojibake ve Noktalama onarımlarını başta yapalım ki metadata daha rahat okunsun
    metin = metin.replace('Ã»', 'û')
    metin = re.sub(r'\.(?=[A-Za-zÇĞİÖŞÜçğıöşü])', '. ', metin)
    metin = re.sub(r',(?=[A-Za-zÇĞİÖŞÜçğıöşü])', ', ', metin)
    
    # --- AŞAMA 2: YAPISAL TEMİZLİK (Metadata & Başlık & Anahtar Kelimeler) ---
    # Önce alt bilgiyi (kuyruğu) keselim
    metin = re.split(r"\n?\s*(Anahtar Kelimeler|Anahtar Sözcükler|Keywords|ANAHTAR KELİMELER).*", metin, flags=re.IGNORECASE | re.DOTALL)[0]

    satirlar = metin.strip().split('\n')
    temiz_satirlar = []
    metadata_etiketleri = ["TEZ ÖZET", "Yazar :", "Üniversite :", "Anabilim Dalı :", "Tez Danışmanı :", "Sayfa Sayısı :", "Enstitü", "Jüri"]
    
    baslangic_bulundu = False
    for satir in satirlar:
        s = satir.strip()
        if not s or any(etiket.lower() in s.lower() for etiket in metadata_etiketleri):
            continue
        if s.lower() in ["özet", "abstract", "özet:", "abstract:"]:
            continue
        
        # Başlangıç tespiti
        if not baslangic_bulundu:
            if s.endswith(('.', '?', '!')) and len(s) > 20:
                baslangic_bulundu = True
            else:
                continue
        temiz_satirlar.append(s)

    # Satırları birleştiriyoruz (Artık \n karakterlerine veda edebiliriz)
    metin = " ".join(temiz_satirlar)

    # --- AŞAMA 3: SON RÖTUŞLAR (Senin Soru İşareti ve Boşluk Kuralların) ---
    metin = re.sub(r'(\d+)\s*\?\s*(\d+)', r'\1 - \2', metin)
    metin = re.sub(r'(?<=\s)\?(?=[A-Za-zÇĞİÖŞÜçğıöşü])', '', metin)
    metin = re.sub(r'\s+', ' ', metin).strip()
    
    return metin

# --- AŞAMA 4: ÇÖP VERİ (GARBAGE) KONTROLÜ (Senin Fonksiyonun) ---
def is_garbage(text):
    if not isinstance(text, str): return True
    kelimeler = text.split()
    if len(kelimeler) < 15: return True # Çok kısa metinleri doğrudan eliyoruz
    single_consonants = re.findall(r'\b[bcçdfgğhjklmnprsştvyz]\b', text.lower())
    if len(kelimeler) > 0 and (len(single_consonants) / len(kelimeler)) > 0.10:
        return True
    return False

# ===========================================================================
# 2. DOSYA OPERASYONU
# ===========================================================================

# Yollar (Senin proje yapına göre: Data_pipeline içindeyiz)
SCRIPT_DIR = Path(__file__).resolve().parent
input_file = SCRIPT_DIR / "data" / "insan_verisi_2500.csv"
output_file = SCRIPT_DIR / "data" / "insan_verisi_v3_temiz.csv" # Adını V3 yapalım, karışmasın

print(f"🚀 Master Temizlik Başlıyor... (Girdi: {input_file.name})")

df = pd.read_csv(input_file)
original_count = len(df)

# 1. Metin Temizliğini Uygula
df['metin'] = df['metin'].astype(str).apply(master_akademik_temizleyici)

# 2. Çöp Filtresini Uygula
df = df[~df['metin'].apply(is_garbage)]

# 3. 🎯 YENİ EKLEME: Kelime Sayısı Dağılım Filtresi (Outlier Temizliği)
# Geçici olarak kelime sayısını hesaplayıp 100 ile 350 kelime arasını filtreliyoruz
df["gecici_kelime_sayisi"] = df['metin'].apply(lambda x: len(str(x).split()))
df = df[(df["gecici_kelime_sayisi"] >= 100) & (df["gecici_kelime_sayisi"] <= 350)]

# İşimiz bitince geçici sütunu uçuralım
df = df.drop(columns=["gecici_kelime_sayisi"])

print("-" * 30)
print(f"✅ İşlem Tamamlandı!")
print(f"📊 Orijinal: {original_count} -> Outlier Öncesi: 2287 -> Tam Temiz (100-350 Kelime): {len(df)}")
print(f"💾 Kaydedildi: {output_file.name}")

df.to_csv(output_file, index=False, encoding="utf-8")