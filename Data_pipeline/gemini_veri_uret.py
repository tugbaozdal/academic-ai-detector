import pandas as pd
import re
import time
import os  
from pathlib import Path
from google import genai
from tqdm import tqdm
from dotenv import load_dotenv

# ===========================================================================
# 1. AYARLAR VE API YAPILANDIRMASI
# ===========================================================================

# .env dosyası ana dizinde olduğu için bir üst dizine bakıyoruz
current_folder = Path(__file__).parent
env_path = current_folder.parent / '.env'
load_dotenv(dotenv_path=env_path)

# API Key kontrolü
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("HATA: .env dosyası bulunamadı veya GOOGLE_API_KEY tanımlanmadı!")

client = genai.Client(api_key=GOOGLE_API_KEY)

# Dosya yolları (Data klasörü içinde)
data_folder = current_folder / "Data"
csv_path = data_folder / "insan_verisi_temiz.csv"
cikti_dosyasi = data_folder / "gemini_tezleri.csv" # Yeni isim

# ===========================================================================
# 2. TEMİZLİK FONKSİYONU
# ===========================================================================
def temizle_akilli_markdown(metin):
    metin = re.sub(r'(?m)^#+\s*', '', metin)
    metin = re.sub(r'\*\*+(.*?)\*\*+', r'\1', metin)
    metin = re.sub(r'\*+(.*?)\*+', r'\1', metin)
    metin = metin.replace("  ", " ").strip()
    return metin

# ===========================================================================
# 3. VERİ HAZIRLIĞI
# ===========================================================================
df_ana = pd.read_csv(csv_path)
asil_hedef_sayisi = 500 
df_hedef = df_ana.head(asil_hedef_sayisi).copy() 

# Kaldığı yeri kontrol etme (Eğer durdurup tekrar açarsan kaldığı yerden devam eder)
mevcut_tez_nolar = []
if cikti_dosyasi.exists():
    try:
        df_kayitli = pd.read_csv(cikti_dosyasi)
        mevcut_tez_nolar = df_kayitli['orijinal_tez_no'].tolist()
        print(f"🔄 Mevcut {len(mevcut_tez_nolar)} veri bulundu. Eksikler tamamlanacak...")
    except:
        print("🔄 Yeni dosya oluşturuluyor...")

# ===========================================================================
# 4. ÜRETİM DÖNGÜSÜ (KESİNTİSİZ MOD)
# ===========================================================================
print(f"🚀 Gemini 500 Maratonu (Firesiz) Başlıyor...")

for i, row in tqdm(df_hedef.iterrows(), total=asil_hedef_sayisi, desc="İlerleme"):
    o_tez_no = row['tez_no']
    baslik = row['baslik']
    
    # Kayma ve mükerrer kaydı önlemek için kontrol
    if o_tez_no in mevcut_tez_nolar:
        continue
    
    prompt = f"""Bir bitirme projesi üzerinde çalışıyorum. Amacım yapay zeka tarafından yazılan akademik metinleri tespit etmek. 
    Bu kapsamda senin bir yapay zeka olarak veri setime katkıda bulunmanı istiyorum.
    
    Aşağıdaki tez başlığı için 300-500 kelime uzunluğunda, akademik üsluba uygun, doğal bir Türkçe özet (abstract) yazar mısın? 
    Lütfen sadece özeti gönder, başka hiçbir açıklama ekleme.

    Tez Başlığı: {baslik}"""

    basarili = False
    while not basarili:
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite', 
                contents=prompt
            )
            
            ham_metin = response.text.strip()
            temiz_metin = temizle_akilli_markdown(ham_metin)
            
            # Yeni satır: source sütunu eklemek birleştirirken işini kolaylaştırır
            yeni_satir = pd.DataFrame([[i+1, o_tez_no, temiz_metin, 1, 'gemini']], 
                                      columns=['id', 'orijinal_tez_no', 'metin', 'label', 'source'])
            
            yeni_satir.to_csv(cikti_dosyasi, mode='a', index=False, header=not cikti_dosyasi.exists(), encoding="utf-8")
            
            basarili = True 
            time.sleep(1.2) # RPM limitini korumak için
            
        except Exception as e:
            # 503 veya bağlantı hatalarında 15 sn bekle ve aynı başlığı tekrar dene
            print(f"\n⚠️ Hata (Tez No {o_tez_no}): {e}. 15 saniye içinde tekrar deneniyor...")
            time.sleep(15)

print(f"\n🎯 İşlem tamamlandı! 'gemini_tezleri.csv' dosyan hazır.")