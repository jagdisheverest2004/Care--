import pandas as pd
import os

def prepare_data():
    # Load the dataset
    df = pd.read_csv("data/mtsamples/mtsamples.csv")
    
    # Remove rows where transcription is missing
    df = df.dropna(subset=['transcription', 'description'])
    
    # Select only the relevant columns
    # Transcription = Full Doctor Note (Source)
    # Description = Summary (Target)
    clinical_data = df[['transcription', 'description']]
    
    # Add a prefix for T5 (T5 needs a task prefix like 'summarize: ')
    clinical_data['transcription'] = "summarize clinical note: " + clinical_data['transcription']
    
    # Save to a clean CSV for training
    os.makedirs("data/processed", exist_ok=True)
    clinical_data.to_csv("data/processed/clinical_summarization_train.csv", index=False)
    print(f"✅ Prepared {len(clinical_data)} medical records for training.")

if __name__ == "__main__":
    prepare_data()