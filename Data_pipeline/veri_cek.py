import pandas as pd
from datasets import load_dataset

print("1. Hugging Face'ten 'Turkish Academic Theses' veri seti indiriliyor...")
print("(Bu işlem internet hızınıza bağlı olarak 1-2 dakika sürebilir, lütfen bekleyin...)")

# Veri setini yükle (Parquet formatında olduğu için oldukça hızlı inecektir)
ds = load_dataset("umutertugrul/turkish-academic-theses-dataset", split="train",token="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")
df = ds.to_pandas()

print(f"-> Başlangıçta indirilen toplam tez sayısı: {len(df)}")

# FİLTRELEME 1: Sadece 2019 ve öncesi (Temporal Shift'i önlemek için)
print("\n2. 2019 ve öncesi tezler filtreleniyor...")
df_clean = df[df["year"] <= 2019]

# FİLTRELEME 2: Özeti veya başlığı boş olan (NaN) hatalı satırları at
print("3. Eksik veya bozuk veriler temizleniyor...")
df_clean = df_clean.dropna(subset=["abstract_tr", "title_tr", "tez_no"])
df_clean = df_clean[df_clean["abstract_tr"].str.strip() != ""]

# ÖRNEKLEME: Çeşitlilik için rastgele 2500 tez seç
print("\n4. Modelin genel üslubu öğrenmesi için rastgele 2500 tez seçiliyor...")
# random_state=42 parametresi, kodu her çalıştırdığında aynı 2500 tezi seçmesini sağlar
df_insan = df_clean.sample(n=2500, random_state=42) 

# DÜZENLEME: Sadece ihtiyacımız olan sütunları al ve sınıf etiketini bas
df_final = df_insan[["tez_no", "title_tr", "abstract_tr"]].copy()
df_final["label"] = 0  # 0 = İnsan metni sınıfı

# Sütun isimlerini kendi veritabanı standartlarımıza göre Türkçeleştir
df_final.rename(columns={"title_tr": "baslik", "abstract_tr": "metin"}, inplace=True)

# KAYDETME: Temizlenmiş veriyi aynı klasöre CSV olarak dışa aktar
print("5. Veri seti CSV olarak kaydediliyor...")
df_final.to_csv("insan_verisi_2500.csv", index=False, encoding="utf-8")

print(f"\n🎉 BİTTİ! Tam {len(df_final)} satırlık 'insan_verisi_2500.csv' dosyası projenize başarıyla eklendi.")