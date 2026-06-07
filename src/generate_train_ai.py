# src/generate_train_ai.py
import pandas as pd
import re
import time
import os
import random
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Resmi API İstemcileri
from google import genai
from google.genai import types  
from openai import OpenAI
from anthropic import Anthropic

# ===========================================================================
# 1. ORTAM DEĞİŞKENLERİ VE API BAĞLANTILARI
# ===========================================================================
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=ROOT_DIR / '.env')

try:
    gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    deepseek_client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"), 
        base_url="https://api.deepseek.com/v1"
    )
except Exception as e:
    print(f"⚠️ API istemcileri başlatılırken uyarı (İlgili anahtar eksik olabilir): {e}")

# ===========================================================================
# 2. PROMPT ŞABLONLARI
# ===========================================================================
PROMPTS = {
    "Prompt_1": "Aşağıdaki tez başlığı hakkında 180-250 kelimelik akademik bir metin yaz. İnsan yazımı gibi doğal olsun ama akademik üslup korunsun. Sadece doğrudan özet metnini gönder, başlık veya giriş/çıkış açıklaması ekleme.\n{baslik}",
    "Prompt_2": "Aşağıdaki konuyu tez özeti gibi, 200-300 kelime arasında açıkla. Çok uzun olmayan, doğrudan ve akademik bir dil kullan. Yalnızca hazırladığın özeti paylaş. Başına başlık, sonuna açıklama koyma.\n{baslik}",
    "Prompt_3": "Aşağıdaki konu hakkında 200-250 kelimelik açıklayıcı akademik metin yaz. Cümle yapıları doğal ve çeşitlendirilmiş olsun. Metne doğrudan başla. Başlık, selamlama veya açıklama cümleleri istemiyorum.\n{baslik}"
}

# ===========================================================================
# 3. GÜVENLİ VERİ TEMİZLEME MOTORU
# ===========================================================================
def clean_ai_output(text):
    if not text:
        return ""
    text = text.replace("**", "").replace("*", "")
    return " ".join(text.split())

# ===========================================================================
# 4. ORTAK API SÜRÜCÜSÜ
# ===========================================================================
def call_specific_llm(model_name, prompt):
    target_temp = round(random.uniform(0.6, 0.95), 2)
    
    if model_name == "gemini":
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-lite', 
            contents=prompt,
            config=types.GenerateContentConfig(temperature=target_temp)
        )
        return response.text
    
    elif model_name == "openai":
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=target_temp
        )
        return response.choices[0].message.content
        
    elif model_name == "deepseek":
        response = deepseek_client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=target_temp,
            extra_body={"thinking": {"type": "disabled"}}
        )
        return response.choices[0].message.content
        
    elif model_name == "claude":
        response = claude_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1000,
            temperature=target_temp,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    raise ValueError(f"Hatalı Model Tanımlaması: {model_name}")

# ===========================================================================
# 5. MATEMATİKSEL BÖLÜNTÜLÜ ANA DÖNGÜ (Main - TRAIN SÜRÜMÜ)
# ===========================================================================
def main():
    input_path = "data/interim/human_train.csv"
    output_path = "data/interim/ai_train.csv"
    
    if not os.path.exists(input_path):
        print(f"Hata: Girdi dosyası bulunamadı -> {input_path}")
        return

    df_human = pd.read_csv(input_path)
    total_rows = len(df_human)
    chunk_size = total_rows // 4 
    
    existing_ids = []
    if os.path.exists(output_path):
        try:
            df_existing = pd.read_csv(output_path)
            existing_ids = df_existing['orijinal_tez_no'].tolist()
            print(f"🔄 Önceden üretilmiş {len(existing_ids)} kayıt bulundu. Kaldığı yerden devam ediliyor...")
        except:
            print("🔄 Yeni çıktı dosyası hazırlanıyor...")

    # Loglar Train setine göre revize edildi
    print(f"🚀 TRAIN Seti Üretim Sistemi Başlatıldı. Model Başına Yük: {chunk_size} Satır.")

    # tqdm açıklaması 'Train Maratonu' olarak değiştirildi
    for index, row in tqdm(df_human.iterrows(), total=total_rows, desc="Train Maratonu"):
        o_tez_no = row['tez_no']
        baslik = row['baslik']
        
        if o_tez_no in existing_ids:
            continue
            
        if index < chunk_size:
            target_model = "gemini"    
        elif index < chunk_size * 2:
            target_model = "openai"    
        elif index < chunk_size * 3:
            target_model = "deepseek"  
        else:
            target_model = "claude"    

        relative_index = index % chunk_size
        prompt_index = (relative_index % 3) + 1  
        chosen_prompt_key = f"Prompt_{prompt_index}"
        
        final_prompt = PROMPTS[chosen_prompt_key].format(baslik=baslik)
        
        success = False
        while not success:
            try:
                raw_text = call_specific_llm(target_model, final_prompt)
                clean_text = clean_ai_output(raw_text)
                
                new_row = pd.DataFrame([[
                    f"AI_{o_tez_no}",
                    baslik,
                    clean_text,
                    1,                 
                    o_tez_no,          
                    target_model,      
                    chosen_prompt_key  
                ]], columns=['tez_no', 'baslik', 'metin', 'label', 'orijinal_tez_no', 'uretici_llm', 'kullanilan_prompt'])
                
                new_row.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path), encoding="utf-8")
                success = True
                time.sleep(1.0) 
                
            except Exception as e:
                print(f"\n⚠️ Bağlantı Kesintisi ({target_model} - Tez No {o_tez_no}): {e}")
                print("15 saniye boyunca bekleniyor ve ardından tekrar deneniyor...")
                time.sleep(15)

    print(f"\n🎯 [İŞLEM TAMAMLANDI] {output_path} dosyası simetrik matris kurallarıyla üretildi.")

if __name__ == "__main__":
    main()