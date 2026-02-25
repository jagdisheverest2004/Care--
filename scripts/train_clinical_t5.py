import torch
from datasets import load_dataset
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "google/flan-t5-large"
DATA_PATH = "data/processed/clinical_summarization_train.csv"
OUTPUT_DIR = "models/clinical_t5_large"

def train():
    if not torch.cuda.is_available():
        print("❌ GPU NOT DETECTED!")
        return
    
    device = torch.device("cuda")
    print(f"✅ Training on: {torch.cuda.get_device_name(0)}")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # --- 1. ENABLE MEMORY SAVING FEATURES BEFORE LORA ---
    model.gradient_checkpointing_enable() # Reduce VRAM usage
    
    # ### CRITICAL FIX: This line connects the LoRA adapters to the Loss function ###
    model.enable_input_require_grads() 
    # ----------------------------------------------------

    # --- 2. CONFIGURE LORA ---
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q", "v"] 
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() 
    
    model.to(device)
    # -------------------------

    dataset = load_dataset("csv", data_files=DATA_PATH)
    # Safe filtering
    dataset = dataset.filter(lambda x: x["transcription"] is not None and x["description"] is not None)
    dataset = dataset["train"].train_test_split(test_size=0.1)

    def preprocess_function(examples):
        # Using 256 for more memory safety
        inputs = tokenizer(examples["transcription"], truncation=True, padding="max_length", max_length=256)
        targets = tokenizer(examples["description"], truncation=True, padding="max_length", max_length=128)
        
        # --- THE CRITICAL FIX FOR LOSS ---
        # Get the labels
        labels = targets["input_ids"]
        
        # Replace the tokenizer's pad_token_id (0) with -100
        # This tells PyTorch: "Do not calculate loss on empty padding spaces!"
        labels = [
            [-100 if token == tokenizer.pad_token_id else token for token in label]
            for label in labels
        ]
        
        inputs["labels"] = labels
        return inputs
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, 
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4, 
        warmup_steps=50,
        num_train_epochs=1,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=False, 
        bf16=True, # RTX 4060 loves bf16
        optim="adamw_torch", 
        gradient_checkpointing=True,
        predict_with_generate=True,
        push_to_hub=False,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer, # type: ignore
        data_collator=data_collator
    )

    # Disable cache to prevent warnings with gradient checkpointing
    model.config.use_cache = False # type: ignore
    
    print("--- Starting LoRA Training ---")
    trainer.train()
    
    # Save the adapter
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Specialized Clinical T5 (LoRA) saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()