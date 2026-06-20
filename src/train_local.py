import pandas as pd
import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed

def main():
    print("🚀 MacBook Üzerinde Lokal Eğitim Başlıyor...")
    set_seed(42)
    
    # Apple Silicon (M1/M2/M3) GPU Hızlandırması Kontrolü
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("⚡ Apple Metal (MPS) GPU Hızlandırması Aktif! Uçuşa hazırız.")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS bulunamadı, CPU üzerinden yavaş eğitim yapılacak.")

    # Çıktı klasörleri (Proje ana dizininden çalıştırılacağı varsayılarak)
    model_dir = "./models/berturk_final"
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Verileri Oku (Lokal 'Data/interim' klasöründen)
    print("📂 Eğitim ve doğrulama verileri okunuyor...")
    df_train = pd.concat([
        pd.read_csv("./Data/interim/human_train.csv")[['metin', 'label']],
        pd.read_csv("./Data/interim/ai_train_v2.csv")[['metin', 'label']]
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_val = pd.concat([
        pd.read_csv("./Data/interim/human_val.csv")[['metin', 'label']],
        pd.read_csv("./Data/interim/ai_val_v2.csv")[['metin', 'label']]
    ], ignore_index=True)

    # 2. PyTorch Dataset Sınıfı
    class FinalDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    # 3. Model ve Tokenizer
    print("🧠 BERTurk Modeli indiriliyor/yükleniyor...")
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    print("✂️ Metinler tokenize ediliyor...")
    train_encodings = tokenizer(df_train['metin'].tolist(), truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(df_val['metin'].tolist(), truncation=True, padding=True, max_length=256)

    # 🎯 Metrik Hesaplama Fonksiyonu
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=-1)
        # Sınıflarımız 0 (İnsan) ve 1 (AI) olduğu için average='binary' kullanıyoruz
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # 4. Hiperparametreler ve Eğitim (Güncellenmiş TrainingArguments)
    training_args = TrainingArguments(
        output_dir=model_dir, 
        num_train_epochs=3, 
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64, 
        weight_decay=0.01,
        warmup_ratio=0.1,  
        eval_strategy="epoch", 
        save_strategy="epoch", 
        load_best_model_at_end=True, 
        metric_for_best_model="accuracy"
        # use_mps_device parametresi sürüm güncellemeleri nedeniyle kaldırıldı.
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=FinalDataset(train_encodings, df_train['label'].tolist()),
        eval_dataset=FinalDataset(val_encodings, df_val['label'].tolist()),
        compute_metrics=compute_metrics # Fonksiyonu Trainer'a bağladık
    )
    
    print("🔥 Model eğitiliyor! (MacBook'un fanları biraz hızlanabilir, normaldir)...")
    trainer.train()
    
    # 5. Şampiyon Modeli Klasöre Kaydet
    print(f"\n✅ Eğitim Bitti! En başarılı model '{model_dir}' klasörüne kaydediliyor...")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("🎉 İşlem Tamam! Artık arayüzü kodlamaya geçebiliriz.")

if __name__ == "__main__":
    main()