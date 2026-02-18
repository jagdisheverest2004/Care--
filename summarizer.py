import torch
from transformers import pipeline

class MedicalSummarizer:
    def __init__(self, model_name="google/flan-t5-base"):
        print(f"--- Loading Advanced Generative AI: {model_name} ---")
        device = 0 if torch.cuda.is_available() else -1
        try:
            # Increased max_length to allow for longer reports
            self.generator = pipeline(
                "text2text-generation", 
                model=model_name, 
                device=device,
                max_length=256  # Allow longer output
            )
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Error loading Summarizer: {e}")
            self.model_loaded = False

    def generate_summary(self, result_dict):
        if not self.model_loaded:
            return "Error: Summarizer model not loaded."

        # 1. Extract Data
        modality = result_dict.get("modality", "Scan")
        part = result_dict.get("body_part", "Body Part")
        finding = result_dict.get("finding", "Unknown")
        conf = result_dict.get("confidence", 0)

        # 2. Advanced Prompt Engineering
        # We give the AI a "Persona" and specific instructions for tone.
        
        if finding == "Normal":
            input_text = (
                f"Act as a Radiologist. Write a reassuring clinical report for a {modality} of the {part}. "
                f"State that the findings are completely Normal (Confidence: {conf}%). "
                f"Mention that no fractures, lesions, or abnormalities are visible. "
                f"Conclude that the patient is healthy."
            )
        elif finding != "N/A":
            input_text = (
                f"Act as a Radiologist. Write a serious clinical diagnostic report for a {modality} of the {part}. "
                f"REPORT FINDINGS: Detected {finding} with high confidence ({conf}%). "
                f"Explain that this indicates a pathological abnormality requiring medical attention. "
                f"Suggest clinical correlation and further investigation."
            )
        else:
            input_text = (
                f"Write a standard medical note stating that a {modality} scan of the {part} was received, "
                f"but no specific diagnostic protocol exists for this region yet."
            )

        # 3. Generate
        try:
            # Increase length_penalty to encourage longer sentences
            output = self.generator(input_text, do_sample=True, temperature=0.7, max_length=150)
            text = output[0]['generated_text'] if output else '' # type: ignore
            return text
            
        except Exception as e:
            return f"Error generating summary: {e}"

# Test block
if __name__ == "__main__":
    summ = MedicalSummarizer()
    
    # Test 1: Pneumonia
    print("\n--- TEST 1: PNEUMONIA ---")
    print(summ.generate_summary({
        "modality": "X-Ray", "body_part": "Chest", "finding": "Pneumonia", "confidence": 99.2
    }))
    
    # Test 2: Normal Knee
    print("\n--- TEST 2: NORMAL KNEE ---")
    print(summ.generate_summary({
        "modality": "X-Ray", "body_part": "Knee", "finding": "Normal", "confidence": 94.5
    }))