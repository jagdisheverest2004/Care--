import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

BASE_MODEL = "google/flan-t5-large"
LORA_DIR = "models/clinical_t5_large"

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # 1. Load the Tokenizer
    print("⏳ Loading Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(LORA_DIR, legacy=False)

    # 2. Load the Base Model
    print("⏳ Loading Base FLAN-T5-Large...")
    base_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)

    # 3. Merge the Medical LoRA Adapter into the Base Model
    print("🧠 Injecting Medical Knowledge (LoRA)...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)
    model.eval() # Set to evaluation mode

    # 4. A Messy, Verbose Clinical Note (Input)
    clinical_note = """
    The patient is a 54-year-old male who presents to the emergency room today complaining of severe, 
    crushing chest pain that radiates down his left arm. He states the pain started about 2 hours ago 
    while he was mowing the lawn. He took an aspirin at home but it did not help. His blood pressure 
    is heavily elevated at 180/110 and his heart rate is 115 bpm. EKG shows ST elevation in leads V2-V4. 
    Patient has a past medical history of Type 2 Diabetes and is a heavy smoker. We will immediately 
    administer nitroglycerin and prep him for the cath lab for suspected anterior myocardial infarction.
    """

    print("\n" + "="*50)
    print("📝 ORIGINAL VERBOSE NOTE:")
    print(clinical_note.strip())
    print("="*50)

    # 5. Generate the Summary
    # Notice we add the same prefix we used during training!
    # CHANGE 1: Use a strict, structural prompt
    prompt = (
        "Extract the primary diagnosis, critical vitals, and treatment plan "
        "into a highly condensed medical summary: " + clinical_note
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=150,
            min_length=10,              # CHANGE 2: Remove the strict minimum quota
            num_beams=5,
            length_penalty=1.0,         # CHANGE 3: Stop forcing it to be artificially long
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n⚕️ AI GENERATED CLINICAL SUMMARY:")
    print(summary)
    print("="*50 + "\n")

if __name__ == "__main__":
    test_model()