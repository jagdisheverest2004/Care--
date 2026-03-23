import pandas as pd
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# --- 1. Original Dataset Class ---
class DdiDataset(Dataset):
    def __init__(self, pairs, labels, tokenizer):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        drug_a, drug_b = self.pairs[idx]
        label = self.labels[idx]
        
        tokenized = self.tokenizer(
            drug_a, 
            drug_b, 
            truncation=True, 
            max_length=128, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        item = {key: val.squeeze(0) for key, val in tokenized.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# --- 2. Original Pair Generation Logic ---
def _make_pairs(df: pd.DataFrame):
    positives = [(str(row["Drug 1"]).strip(), str(row["Drug 2"]).strip()) for _, row in df.iterrows()]
    positive_set = set(positives)
    
    labels = [1] * len(positives)
    
    drugs = sorted(list(set(df["Drug 1"]).union(set(df["Drug 2"]))))
    negatives = set()
    
    print(f"Generating negatives from {len(drugs)} unique drugs...")
    
    random.seed(42) # Set seed to match training exactly
    while len(negatives) < len(positives):
        a = random.choice(drugs)
        b = random.choice(drugs)
        if a == b: continue
        if (a, b) in positive_set or (b, a) in positive_set: continue
        negatives.add((a, b))
        
    pairs = positives + list(negatives)
    labels += [0] * len(negatives)
    
    return pairs, labels

# --- 3. Evaluation and Plotting Logic ---
def evaluate_biobert(model_folder, csv_path):
    print(f"Loading Saved BioBERT from: {model_folder}")
    
    # 1. Load the SAVED model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    model = AutoModelForSequenceClassification.from_pretrained(model_folder, num_labels=2)

    # 2. Load and Prepare Data Exactly as in training
    print(f"Loading Data from {csv_path}...")
    df = pd.read_csv(csv_path)
    pairs, labels = _make_pairs(df)
    
    # 3. Extract the exact 10% Test/Validation set used during training (random_state=42)
    _, val_pairs, _, val_labels = train_test_split(
        pairs, labels, test_size=0.1, stratify=labels, random_state=42
    )
    
    total_test = len(val_pairs)
    hazards = sum(val_labels)
    safes = total_test - hazards
    print(f"Test Set Ready: {total_test} total pairs ({hazards} Hazardous, {safes} Safe)")
    
    test_dataset = DdiDataset(val_pairs, val_labels, tokenizer)

    # 4. Setup HuggingFace Trainer for Evaluation Only
    training_args = TrainingArguments(
        output_dir="./temp",
        per_device_eval_batch_size=16,
        report_to="none"
    )
    trainer = Trainer(model=model, args=training_args)

    # 5. Run Predictions
    print("Running AI predictions... This might take a minute.")
    predictions = trainer.predict(test_dataset)
    
    # 6. Convert logits to probabilities
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    y_pred = np.argmax(probs, axis=1)
    y_true = np.array(predictions.label_ids) if predictions.label_ids is not None else np.array([])
    y_probs_positive = probs[:, 1] # Probabilities for 'Hazardous' class

    # =========================================================
    #                    GENERATE METRICS & PLOTS
    # =========================================================
    
    # 1. Print Classification Report (Accuracy, Precision, Recall, F1)
    print("\n" + "="*50)
    print("  MODEL PERFORMANCE METRICS (BioBERT Drug Safety)")
    print("="*50)
    class_names = ['Safe (0)', 'Hazardous (1)']
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # 2. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.title('BioBERT Drug Safety - Confusion Matrix', fontsize=15, fontweight='bold')
    plt.ylabel('Actual Relationship', fontsize=12, fontweight='bold')
    plt.xlabel('AI Predicted Relationship', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('BioBERT_DDI_Confusion_Matrix.png', dpi=300)
    print("✅ Saved: BioBERT_DDI_Confusion_Matrix.png")
    plt.show()

    # 3. Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs_positive)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkred', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    plt.title('BioBERT Drug Safety - ROC Curve', fontsize=15, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('BioBERT_DDI_ROC_Curve.png', dpi=300)
    print("✅ Saved: BioBERT_DDI_ROC_Curve.png")
    plt.show()

if __name__ == "__main__":
    # Ensure these paths point to your actual folder/file locations
    evaluate_biobert(
        model_folder="models/biobert_ddi", # Path where your trained model is saved
        csv_path="data/drug-drug-interactions/db_drug_interactions.csv" # Path to your dataset
    )