import pandas as pd
import re
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def main():
    # Sütun gizleme sınırlarını kaldırıyoruz
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    ROOT_DIR = Path(__file__).parent.parent
    interim_dir = ROOT_DIR / "data" / "interim"
    
    human_path = interim_dir / "human_train.csv"
    ai_path = interim_dir / "ai_train_v2.csv"
    
    if not human_path.exists() or not ai_path.exists():
        print("❌ Hata: Gerekli train dosyaları bulunamadı!")
        return

    df_human = pd.read_csv(human_path)
    df_ai = pd.read_csv(ai_path)
    
    sutunlar = ['tez_no', 'baslik', 'metin', 'label']
    df = pd.concat([df_human[sutunlar], df_ai[sutunlar]], ignore_index=True)
    
    # -----------------------------------------------------
    # ÖZELLİK ÇIKARIM MOTORU
    # -----------------------------------------------------
    def gelismis_ozellik_bul(metin):
        if pd.isna(metin): return pd.Series([0, 0, 0, 0, 0, 0])
        metin = str(metin).lower()
        kelimeler = re.findall(r'\b\w+\b', metin)
        toplam_kelime = len(kelimeler)
        if toplam_kelime == 0: return pd.Series([0, 0, 0, 0, 0, 0])

        # 1. Toplam Kelime
        tk = toplam_kelime
        # 2. Cümle Uzunluğu
        cumleler = [c for c in re.split(r'(?<=[.!?])\s+', metin) if c.strip()]
        cu = toplam_kelime / len(cumleler) if len(cumleler) > 0 else 0
        # 3. Kelime Çeşitliliği (TTR)
        kc = len(set(kelimeler)) / toplam_kelime
        # 4. Virgül Oranı
        vo = metin.count(',') / toplam_kelime
        # 5. Bağlaç Oranı
        baglaclar = ["ve", "ancak", "fakat", "bununla birlikte", "dolayısıyla", "ayrıca", "özellikle", "sonuç olarak", "öte yandan", "bu nedenle", "buna ek olarak"]
        bo = len(re.findall(r'\b(' + '|'.join(map(re.escape, baglaclar)) + r')\b', metin)) / toplam_kelime
        # 6. Pasif Oranı
        po = len(re.findall(r'\b\w+(miştir|mıştır|ilmiştir|ılmıştır|ulmuştur|ülmüştür|inmiştir)\b', metin)) / toplam_kelime

        return pd.Series([tk, cu, kc, vo, bo, po])

    print("🧠 Tüm özellikler ve kelime sayıları hesaplanıyor...")
    yeni_sutunlar = ['toplam_kelime', 'cumle_uzunlugu', 'kelime_cesitliligi', 'virgul_orani', 'baglac_orani', 'pasif_orani']
    df[yeni_sutunlar] = df['metin'].apply(gelismis_ozellik_bul)
    
    print("\n🎯 [ADIM 1] TÜM SÜTUNLARLA YENİLENMİŞ ORTALAMALAR")
    print("=" * 80)
    print(df.groupby('label')[yeni_sutunlar].mean())
    print("-" * 80)

    # Veriyi ayıralım
    y = df['label']
    X_train_all, X_val_all, y_train, y_val = train_test_split(df[yeni_sutunlar], y, test_size=0.2, random_state=42, stratify=y)

    # -----------------------------------------------------
    # TEST 1: SADECE TOPLAM KELİME İLE LOR
    # -----------------------------------------------------
    print("\n🚨 [TEST 1] SADECE 'toplam_kelime' İLE LOGISTIC REGRESSION")
    print("=" * 80)
    X_train_tk = X_train_all[['toplam_kelime']]
    X_val_tk = X_val_all[['toplam_kelime']]
    
    lr_tk = LogisticRegression()
    lr_tk.fit(X_train_tk, y_train)
    acc_tk = lr_tk.score(X_val_tk, y_val)
    print(f"📈 Sadece Kelime Sayısına Bakarak Başarı Oranı: %{acc_tk*100:.2f}")
    print(f"📉 Logistic Regression Katsayısı (Coefficient): {lr_tk.coef_[0][0]:.4f}")
    print("-" * 80)

    # -----------------------------------------------------
    # TEST 2: RANDOM FOREST FEATURE IMPORTANCE (TÜM ÖZELLİKLER)
    # -----------------------------------------------------
    print("\n🌲 [TEST 2] RANDOM FOREST FEATURE IMPORTANCE (AĞAÇLARIN GÖZÜNDEN EN ÖNEMLİ ÖZELLİK)")
    print("=" * 80)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_all, y_train)
    
    importances = rf.feature_importances_
    rf_acc = rf.score(X_val_all, y_val)
    
    importance_df = pd.DataFrame({
        'Özellik': yeni_sutunlar,
        'Önem Skoru (Importance)': importances
    }).sort_values(by='Önem Skoru (Importance)', ascending=False)
    
    print(f"📈 Tüm Özelliklerle Random Forest Başarısı: %{rf_acc*100:.2f}")
    print("\n📊 Özelliklerin Önem Sıralaması:")
    print(importance_df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    main()