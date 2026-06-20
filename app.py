import streamlit as st
from transformers import pipeline
import time

# 1. Sayfa Ayarları ve Başlık
st.set_page_config(page_title="AI Metin Dedektörü", page_icon="🕵️‍♀️", layout="centered")

st.title("🕵️‍♀️ Akademik Metin AI Dedektörü")
st.markdown("Bu araç, bitirme projesi kapsamında **Tuğba Sena Özdal** tarafından eğitilen **BERTurk** modeli ile metinlerin yapay zeka (LLM) tarafından yazılıp yazılmadığını analiz eder.")
st.divider()

# 2. Modeli Cache'e Alma (Arayüzde her işlem yapıldığında modeli baştan yüklemesin diye)
@st.cache_resource
def modeli_yukle():
    model_yolu = "./models/berturk_final" 
    # device=-1 diyerek arayüzün M1 işlemcinde (CPU modunda) sorunsuz çalışmasını garantiye alıyoruz
    return pipeline("text-classification", model=model_yolu, tokenizer=model_yolu, truncation=True, max_length=256, device=-1)

# Arayüz yüklenirken kullanıcıya bilgi ver
with st.spinner("🧠 Yapay Zeka Beyni Yükleniyor... (İlk açılışta birkaç saniye sürebilir)"):
    try:
        dedektor = modeli_yukle()
        model_hazir = True
    except Exception as e:
        st.error("⚠️ Model henüz bulunamadı! Lütfen 'models/berturk_final' klasörünün dolu olduğundan emin olun.")
        model_hazir = False

if model_hazir:
    # 3. Kullanıcı Girdisi
    kullanici_metni = st.text_area("Analiz edilecek metni buraya yapıştırın (Min 15-20 kelime tavsiye edilir):", height=200)

    # 4. Analiz Butonu ve Sonuç Gösterimi
    if st.button("🔍 Metni Analiz Et", use_container_width=True):
        if len(kullanici_metni.split()) < 5:
            st.warning("⚠️ Sağlıklı bir analiz için lütfen daha uzun bir metin giriniz.")
        else:
            with st.spinner("Metin derinlemesine analiz ediliyor..."):
                time.sleep(0.5) # Ekranda şık bir bekleme efekti için
                
                # Tahmini al
                sonuc = dedektor(kullanici_metni)[0]
                etiket = sonuc['label']
                oran = sonuc['score'] * 100
                
                st.divider()
                st.subheader("📊 Analiz Sonucu")
                
                # Sınıflandırma: LABEL_1 = Yapay Zeka (1), LABEL_0 = İnsan (0)
                if etiket == "LABEL_1":
                    st.error(f"🚨 **YÜKSEK İHTİMALLE YAPAY ZEKA!**")
                    st.progress(int(oran), text=f"Yapay Zeka Olma Olasılığı: %{oran:.2f}")
                    st.info("Bu metnin ChatGPT, Claude veya Gemini gibi bir dil modeli tarafından üretilmiş olma ihtimali çok yüksek.")
                else:
                    st.success(f"✅ **İNSAN TARAFINDAN YAZILMIŞ.**")
                    st.progress(int(oran), text=f"İnsan Yazısı Olma Olasılığı: %{oran:.2f}")
                    st.info("Bu metin doğal dil yapısına ve insan yazım tarzına uygun görünüyor.")