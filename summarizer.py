from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments


class MedicalSummaryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: T5Tokenizer, max_input_length: int = 512, max_target_length: int = 150):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        source = f"summarize: {row['full_report']}"
        target = row["summary"]

        model_inputs = self.tokenizer(
            source,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        model_inputs["labels"] = labels.input_ids.squeeze(0)
        return {k: v.squeeze(0) for k, v in model_inputs.items()}


def train_summarizer(df: pd.DataFrame, output_dir: str = "t5_medical"):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    dataset = MedicalSummaryDataset(df, tokenizer)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        predict_with_generate=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


def _dedupe_sentences(text: str) -> str:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    seen = set()
    deduped = []
    for s in sentences:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return ". ".join(deduped) + ("." if deduped else "")


def generate_summary(text: str, model_dir: str = "t5_medical") -> str:
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)

    input_ids = tokenizer(
        f"summarize: {text}",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    outputs = model.generate(
        input_ids,
        max_length=150,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return _dedupe_sentences(summary)
