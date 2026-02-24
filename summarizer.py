import torch
from transformers import pipeline

class MedicalSummarizer:
    def __init__(self, model_name="google/flan-t5-large"):
        print(f"--- Loading Professional Medical AI: {model_name} ---")
        # 0 is the ID for your first GPU (RTX 4060)
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            self.generator = pipeline(
                "text2text-generation", 
                model=model_name, 
                device=device,
                max_length=512  # Increased for detailed reports
            )
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_loaded = False

    def generate_summary(self, result_dict):
        if not self.model_loaded:
            return "Summarizer not available."

        # Extract Data
        modality = result_dict.get("modality", "X-Ray")
        part = result_dict.get("body_part", "Body Part")
        finding = result_dict.get("finding", "Unknown")
        conf = result_dict.get("confidence", 0)

        # THE "DOCTOR PERSONA" PROMPT
        # We give the model a specific role and clear instruction to be detailed.
        if finding == "Normal":
            input_text = (
                f"Identify the following medical scan. Modality: {modality}. Anatomy: {part}. "
                f"Finding: Normal. Confidence: {conf}%. "
                f"Task: Write a detailed, reassuring radiology report. Mention that the bone structures "
                f"and soft tissues appear healthy with no signs of acute pathology."
            )
        else:
            input_text = (
                f"Identify the following medical scan. Modality: {modality}. Anatomy: {part}. "
                f"Finding: {finding}. Confidence: {conf}%. "
                f"Task: Write a professional and urgent medical diagnostic report. "
                f"Explain the clinical significance of {finding} in the {part} and "
                f"recommend immediate clinical correlation and specialist consultation."
            )

        try:
            # We use a slight 'penalty' to force the model to write longer sentences
            output = self.generator(
                input_text, 
                max_length=200, 
                min_length=50, 
                repetition_penalty=2.5,
                do_sample=False # Keep it professional and factual
            )
            
            report = output[0].get('generated_text', '') # type: ignore
            
            # Clean up and add a professional header
            final_output = (
                f"{report}\n"
                f"Technical Note: Analysis performed with {conf}% confidence by Care++ Vision Engine."
            )
            return final_output
            
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    summ = MedicalSummarizer()
    # Test with Pneumonia
    test_case = {"modality": "X-Ray", "body_part": "Chest", "finding": "Pneumonia", "confidence": 99.2}
    print(summ.generate_summary(test_case))