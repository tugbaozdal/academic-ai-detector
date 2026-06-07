import pandas as pd

if __name__ == "__main__":
    # ⚠️ Orijinal insan verilerinin olduğu CSV dosyasının adını yaz
    girdi_dosyasi = "insan_verisi_full_temiz_V2.csv" 
    
    print(f"🔄 '{girdi_dosyasi}' dosyası yükleniyor...")
    df = pd.read_csv(girdi_dosyasi)
    
    # 1. Sadece İnsan Verilerini Filtrele (label = 0)
    # Not: Eğer dosmanda sadece insan verileri varsa ve label sütunu yoksa 
    # bu satırı 'insan_df = df' olarak değiştirebilirsin.
    insan_df = df
    
    # 2. Toplam Veri Sayısını Bul
    toplam_veri = len(insan_df)
    
    print("\n📊 --- İNSAN VERİLERİ ANALİZ SONUÇLARI ---")
    print("=" * 45)
    print(f"👤 Toplam İnsan Tez Özeti Sayısı: {toplam_veri}")
    print("=" * 45)
    
    if toplam_veri == 0:
        print("❌ Hata: Sınıfı '0' (İnsan) olan hiç veri bulunamadı!")
        print("Eğer dosyanızda 'label' sütunu yoksa kodun 11. satırını düzenleyin.")
    else:
        # 3. Her Metnin Kelime Sayısını Hesapla
        # 'text' sütununun adını kendi CSV'ne göre kontrol et ('text', 'metin', 'abstract' vb.)
        kelime_sayilari = insan_df['metin'].apply(lambda x: len(str(x).split()))
        
        # 4. İstatistiksel Dağılımı Yazdır
        print(kelime_sayilari.describe())
        print("=" * 45)
        
        # Ekstra Hızlı Bilgi
        print(f"💡 En çok karşılaşılan (Ortalama) uzunluk: {int(kelime_sayilari.mean())} kelime.")
        print(f"💡 Yapay zekaya verilecek en mantıklı aralık: {int(kelime_sayilari.quantile(0.25()))}-{int(kelime_sayilari.quantile(0.75()))} kelime arası.")