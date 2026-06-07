# src/split_human_data.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    # Define file paths relative to the project root
    raw_data_path = "data/raw/insan_verisi_v3_temiz.csv"
    interim_dir = "data/interim"
    
    print(f"Reading raw human dataset from: {raw_data_path}")
    
    # Check if the raw data file exists
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found! Please place your source CSV file correctly.")
        return

    # Load raw human data
    df = pd.read_csv(raw_data_path)
    
    # First split: 80% Train and 20% Temporary (for Validation + Test)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Second split: Divide the 20% temporary data equally into Validation (10%) and Test (10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create the interim directory if it doesn't exist
    os.makedirs(interim_dir, exist_ok=True)
    
    # Save split human data into the interim folder
    train_df.to_csv(os.path.join(interim_dir, "human_train.csv"), index=False)
    val_df.to_csv(os.path.join(interim_dir, "human_val.csv"), index=False)
    test_df.to_csv(os.path.join(interim_dir, "human_test.csv"), index=False)
    
    # Console output report
    print("\n[SUCCESS] Human dataset split operation completed!")
    print(f"-> Total Raw Human Data: {len(df)} rows")
    print(f"-> human_train.csv      : {len(train_df)} rows (80%)")
    print(f"-> human_val.csv        : {len(val_df)} rows (10%)")
    print(f"-> human_test.csv       : {len(test_df)} rows (10%)")

if __name__ == "__main__":
    main()