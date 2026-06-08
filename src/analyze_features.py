import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main():
    # Sütun gizleme sınırlarını kaldırıyoruz
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)

    ROOT_DIR = Path(__file__).parent.parent
    interim_dir = ROOT_DIR / "data" / "interim"
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True) # Raporlar için klasör yoksa oluşturur
    
    human_path = interim_dir / "human_train.csv"
    ai_path = interim_dir / "ai_train_v2.csv"
    
    if not human_path.exists() or not ai_path.exists():
        print("❌ Hata: Gerekli train dosyaları bulunamadı!")
        return

    print("📂 Dosyalar yükleniyor ve birleştiriliyor...")
    df_human = pd.read_csv(human_path)
    df_ai = pd.read_csv(ai_path)
    
    sutunlar = ['tez_no', 'baslik', 'metin', 'label']
    df = pd.concat([df_human[sutunlar], df_ai[sutunlar]], ignore_index=True)

    # MASTER ÖZELLİK ÇIKARIM FONKSİYONU
    def ozellikleri_cikar(metin):
        if pd.isna(metin):
            return pd.Series([0, 0, 0, 0, 0, 0])

        metin = str(metin).lower()
        kelimeler = re.findall(r'\b\w+\b', metin)
        toplam_kelime = len(kelimeler)

        if toplam_kelime == 0:
            return pd.Series([0, 0, 0, 0, 0, 0])

        tk = toplam_kelime
        cumleler = [c for c in re.split(r'(?<=[.!?])\s+', metin) if c.strip()]
        cu = toplam_kelime / len(cumleler) if len(cumleler) > 0 else 0
        kc = len(set(kelimeler)) / toplam_kelime
        vo = metin.count(',') / toplam_kelime
        
        baglaclar = ["ve", "ancak", "fakat", "bununla birlikte", "dolayısıyla", "ayrıca", "özellikle", "sonuç olarak", "öte yandan", "bu nedenle", "buna ek olarak"]
        bo = len(re.findall(r'\b(' + '|'.join(map(re.escape, baglaclar)) + r')\b', metin)) / toplam_kelime
        
        po = len(re.findall(r'\b\w+(miştir|mıştır|ilmiştir|ılmıştır|ulmuştur|ülmüştür|inmiştir)\b', metin)) / toplam_kelime

        return pd.Series([tk, cu, kc, vo, bo, po])

    print("🧠 Matematiksel ve dilbilimsel özellikler hesaplanıyor...")
    yeni_sutunlar = ['toplam_kelime', 'cumle_uzunlugu', 'kelime_cesitliligi', 'virgul_orani', 'baglac_orani', 'pasif_orani']
    df[yeni_sutunlar] = df['metin'].apply(ozellikleri_cikar)
    
    # -----------------------------------------------------
    # 1. RAPOR: ORTALAMALAR TABLOSU & CSV KAYDI
    # -----------------------------------------------------
    print("\n🎯 [RAPOR 1] TÜM ÖZELLİKLERİN SINIF BAZLI ORTALAMALARI")
    print("=" * 100)
    rapor_ortalamalar = df.groupby('label')[yeni_sutunlar].mean()
    print(rapor_ortalamalar)
    
    ortalamalar_csv_yolu = reports_dir / "feature_averages.csv"
    rapor_ortalamalar.to_csv(ortalamalar_csv_yolu)
    print(f"💾 Kaydedildi: {ortalamalar_csv_yolu}")
    print("-" * 100)

    # -----------------------------------------------------
    # 2. RAPOR: TEKLİ ÖZELLİK ANALİZLERİ (Katsayı & Başarı)
    # -----------------------------------------------------
    print("\n🔬 [RAPOR 2] TEKLİ ÖZELLİK LOGISTIC REGRESSION PERFORMANSLARI")
    print("-" * 100)
    
    tekli_sonuclar = []
    y = df['label']
    
    for feat in yeni_sutunlar:
        X_feat = df[[feat]]
        X_train, X_val, y_train, y_val = train_test_split(X_feat, y, test_size=0.2, random_state=42, stratify=y)
        
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        acc = lr.score(X_val, y_val)
        coef = lr.coef_[0][0]
        
        print(f"🔹 Özellik: {feat:<18} | Test Başarısı: %{acc*100:.2f} | Katsayı (Coefficient): {coef:.4f}")
        
        tekli_sonuclar.append({
            'ozellik': feat,
            'test_accuracy': acc,
            'coefficient': coef
        })
        
    df_tekli_rapor = pd.DataFrame(tekli_sonuclar)
    tekli_csv_yolu = reports_dir / "single_feature_performance.csv"
    df_tekli_rapor.to_csv(tekli_csv_yolu, index=False)
    print("-" * 100)
    print(f"💾 Kaydedildi: {tekli_csv_yolu}")

if __name__ == "__main__":
    main()