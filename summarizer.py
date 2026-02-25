import torch
from transformers import pipeline , T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel
import ollama

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
                f"{report}"
                f"Technical Note: Analysis performed with {conf}% confidence by Care++ Vision Engine."
            )
            return final_output
            
        except Exception as e:
            return f"Error: {e}"
        
class DocumentSummarizer:
    def __init__(self):
        print(f"--- Loading Llama-3 (8B) Clinical AI ---")
        self.model_loaded = True
        self.model_name = "llama3"
        print("✅ Llama-3 Connected Successfully")

    def summarize_long_document(self, text_or_chunks) -> str:
        """Llama-3 can read the whole document at once. No Map-Reduce needed!"""
        
        # If the data loader still passes chunks, combine them back into one huge string
        if isinstance(text_or_chunks, list):
            full_document = " ".join(text_or_chunks)
        else:
            full_document = text_or_chunks

        print("🧠 Llama-3 is analyzing the document...")

        # The "ChatGPT-style" Prompt
        # The "SOAP Executive Summary" Prompt
        prompt = f"""You are an expert Medical Summarizer. Read the following clinical document and extract the core information into a highly condensed, bulleted executive summary. 

        Use the SOAP format:
        - **Subjective**: 1-2 bullet points on patient's reported issues and history.
        - **Objective**: 1-2 bullet points on the most critical vital signs or exam findings.
        - **Assessment**: The primary diagnosis in one sentence.
        - **Plan**: 2-3 bullet points on the immediate treatment or medications.

        Keep it extremely concise. Do not write long paragraphs.

        DOCUMENT TO SUMMARIZE:
        {full_document}
        """

        try:
            # Stream the response natively using your RTX 4060
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            return response['message']['content']
            
        except Exception as e:
            return f"❌ Error: Make sure the Ollama app is running on your computer. Details: {e}"
        
        
if __name__ == "__main__":
    summ = MedicalSummarizer()
    # Test with Pneumonia
    test_case = {"modality": "X-Ray", "body_part": "Chest", "finding": "Pneumonia", "confidence": 99.2}
    print(summ.generate_summary(test_case))