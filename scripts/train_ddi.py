import argparse
import random
import os
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)

# --- 1. Data Preparation Logic ---
def _make_pairs(df: pd.DataFrame) -> Tuple[List[Tuple[str, str]], List[int]]:
    # Convert DataFrame to list of tuples
    positives = [(str(row["Drug 1"]), str(row["Drug 2"])) for _, row in df.iterrows()]
    positive_set = set(positives)
    
    labels = [1] * len(positives)
    
    # Get all unique drugs to generate negatives
    drugs = sorted(list(set(df["Drug 1"]).union(set(df["Drug 2"]))))
    negatives = set()
    
    print(f"Generating negatives from {len(drugs)} unique drugs...")
    
    # Generate equal number of negatives
    while len(negatives) < len(positives):
        a = random.choice(drugs)
        b = random.choice(drugs)
        if a == b: continue
        if (a, b) in positive_set or (b, a) in positive_set: continue
        negatives.add((a, b))
        
    pairs = positives + list(negatives)
    labels += [0] * len(negatives)
    
    return pairs, labels

# --- 2. Improved Dataset Class ---
class DdiDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], labels: List[int], tokenizer):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        drug_a, drug_b = self.pairs[idx]
        label = self.labels[idx]
        
        # CORRECT WAY: Pass both strings separately
        # This creates [CLS] DrugA [SEP] DrugB [SEP] AND sets token_type_ids correctly
        tokenized = self.tokenizer(
            drug_a, 
            drug_b, 
            truncation=True, 
            max_length=128, 
            padding="max_length", # Pad to max_length
            return_tensors="pt"
        )
        
        item = {key: val.squeeze(0) for key, val in tokenized.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# --- 3. Metric Calculation Function ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 4. Main Training Loop ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to csv")
    parser.add_argument("--output_dir", default="biobert_ddi")
    parser.add_argument("--epochs", type=int, default=3) # Increased to 3
    parser.add_argument("--batch_size", type=int, default=16) # Increased to 16
    args = parser.parse_args()

    print(f"Loading Data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    pairs, labels = _make_pairs(df)
    print(f"Total Pairs: {len(pairs)} (Balanced)")

    # Split Data
    train_pairs, val_pairs, train_labels, val_labels = train_test_split(
        pairs, labels, test_size=0.1, stratify=labels, random_state=42
    )

    print("Loading BioBERT...")
    model_name = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = DdiDataset(train_pairs, train_labels, tokenizer)
    val_dataset = DdiDataset(val_pairs, val_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",        # Check accuracy every epoch
        save_strategy="epoch",        # Save model every epoch
        learning_rate=2e-5,           # Slower, more careful learning
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,           # Keep only last 2 checkpoints
        load_best_model_at_end=True,  # Load the best accuracy model at end
        metric_for_best_model="accuracy",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics, # Add this!
    )

    print("Starting Training...")
    trainer.train()
    
    print("Saving Final Model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()