import os
import torch
from typing import Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DrugSafetyEngine:
    def __init__(self, model_name: str = "models/biobert_ddi", rules_csv: Optional[str] = None) -> None:
        # Load the model and tokenizer from your trained folder
        print(f"Loading Safety Engine from: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.eval()
        
        # Load CSV for rule-based backup (optional)
        self.rules = self._load_rules(rules_csv)

    def _load_rules(self, rules_csv: Optional[str]) -> Dict[Tuple[str, str], str]:
        if rules_csv is None:
            # Note: I updated the folder name to 'data' based on our previous fix
            rules_csv = os.getenv("DDI_RULES_CSV", "data/drug-drug-interactions/db_drug_interactions.csv")
        
        if not os.path.exists(rules_csv):
            print(f"Warning: Rules CSV not found at {rules_csv}")
            return {}

        rules: Dict[Tuple[str, str], str] = {}
        try:
            with open(rules_csv, "r", encoding="utf-8") as handle:
                header = next(handle, None) # Skip header
                for line in handle:
                    parts = line.strip().split(",", 2)
                    if len(parts) != 3: continue
                    drug_a, drug_b, desc = parts
                    # Store both A-B and B-A directions
                    rules[(drug_a.strip().lower(), drug_b.strip().lower())] = desc.strip()
                    rules[(drug_b.strip().lower(), drug_a.strip().lower())] = desc.strip()
        except Exception as e:
            print(f"Error loading rules: {e}")
            
        return rules

    def check_interaction(self, drug_a: str, drug_b: str) -> Dict[str, object]:
        # 1. Rule-Based Lookup (Instant Check)
        key = (drug_a.strip().lower(), drug_b.strip().lower())
        if key in self.rules:
            return {
                "interaction_detected": True,
                "confidence": 1.0,
                "severity": "High",
                "description": self.rules[key],
                "source": "Database Rule"
            }

        # 2. AI Prediction (BioBERT)
        # CRITICAL FIX: Pass drug_a and drug_b as separate arguments!
        # This matches how we trained the model today.
        inputs = self.tokenizer(
            drug_a, 
            drug_b, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128, 
            padding="max_length"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        # Class 1 = Interaction, Class 0 = Safe
        confidence = float(probs[0, 1].item())
        
        # Set Threshold to 0.90 since your model is 96% accurate now
        THRESHOLD = 0.90 
        interaction_detected = confidence >= THRESHOLD

        result = {
            "interaction_detected": interaction_detected,
            "confidence": round(confidence, 4),
            "severity": "High" if interaction_detected else "Low",
            "status": "AI Warning" if interaction_detected else "Safe",
            "source": "BioBERT AI"
        }
        return result