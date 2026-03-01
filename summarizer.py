import ollama

class MedicalSummarizer:
    """Handles short, urgent radiology/image reports."""
    def __init__(self):
        print("--- Loading Llama-3 (8B) Vision/Radiology AI ---")
        self.model_name = "llama3"
        self.model_loaded = True

    def generate_summary(self, result_dict):
        modality = result_dict.get("modality", "X-Ray")
        part = result_dict.get("body_part", "Body Part")
        finding = result_dict.get("finding", "Unknown")
        conf = result_dict.get("confidence", 0)

        # AGGRESSIVE PROMPT: We force the AI to drop the "Chatbot" act.
        if finding == "Normal":
            prompt = (
                f"You are an expert Radiologist. The AI analyzed a {modality} of the {part} and found it 'Normal' with {conf}% confidence. "
                f"Write a brief, professional 2-sentence radiology report stating bone and soft tissues are healthy. "
                f"CRITICAL RULES: Output ONLY the raw medical text. Do NOT say 'Here is your report'. Do NOT add any disclaimers. Do NOT use bullet points or titles."
            )
        else:
            prompt = (
                f"You are an expert Radiologist. The AI analyzed a {modality} of the {part} and detected '{finding}' with {conf}% confidence. "
                f"Write a highly professional, urgent 3-sentence medical diagnostic report explaining the clinical significance and recommending specialist consultation. "
                f"CRITICAL RULES: Output ONLY the raw medical text. Do NOT say 'Here is your report'. Do NOT add any disclaimers. Do NOT use bullet points, greetings, or titles."
            )

        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt}
            ])
            raw_report = response['message']['content']
            
            # STRING CLEANING: Remove the annoying \n line breaks and extra spaces
            clean_report = raw_report.replace('\n', ' ').replace('**', '').strip()
            
            # Ensure there is no accidental double-spacing
            clean_report = " ".join(clean_report.split())
            
            final_output = f"{clean_report}Technical Note: Analysis performed with {conf}% confidence by Care++ Vision Engine."
            return final_output
            
        except Exception as e:
            return f"Error connecting to Ollama: {e}"
        

class DocumentSummarizer:
    """Handles long, multi-page clinical PDFs."""
    def __init__(self):
        print(f"--- Loading Llama-3 (8B) Clinical Document AI ---")
        self.model_name = "llama3"
        self.model_loaded = True

    def summarize_long_document(self, text_or_chunks) -> str:
        if isinstance(text_or_chunks, list):
            full_document = " ".join(text_or_chunks)
        else:
            full_document = text_or_chunks

        print("🧠 Llama-3 is analyzing the document...")

        # Restored your preferred Narrative Prompt!
        prompt = f"""You are an expert Chief Medical Officer. Read the following clinical document and write a highly professional, articulate, and comprehensive medical narrative summary. 
        
        Your summary MUST include:
        1. Chief Complaint & History of Present Illness
        2. Critical Objective Findings (Vitals, Physical Exam)
        3. The Assessment and Detailed Treatment Plan

        Do NOT just copy and paste sentences. Synthesize the information into your own flowing, professional medical prose.

        DOCUMENT TO SUMMARIZE:
        {full_document}
        """

        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            raw_summary = response['message']['content']
            clean_summary = raw_summary.replace('**', '').replace('\n', ' ').strip()
            return " ".join(clean_summary.split())
            
        except Exception as e:
            return f"Error connecting to Ollama: {e}"
        
        
class SafetyExplainer:
    """Explains the safety and limitations of the AI's medical summaries."""
    def __init__(self):
        print("--- Loading Llama-3 (8B) Safety Explainer ---")
        self.model_name = "llama3"
        self.model_loaded = True

    def explain_interaction(self, drug_a, drug_b, raw_description):
        """Uses Llama-3 to turn a dry DDI rule into a helpful clinical explanation."""
        print(f"🧠 Llama-3 is explaining interaction: {drug_a} + {drug_b}")
        
        prompt = f"""You are a Clinical Pharmacologist. 
        The system detected a potential interaction between {drug_a} and {drug_b}.
        Raw Database Note: {raw_description}
        
        Task: Write a concise, professional 2-sentence explanation of why this interaction is concerning and what the patient should do (e.g., consult a doctor, monitor for specific symptoms).
        
        CRITICAL RULES:
        - Output ONLY the explanation.
        - Do NOT say "Here is the explanation".
        - Do NOT use bolding or bullet points.
        - Be direct and professional.
        """
        
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt}
            ])
            # Clean the string like we do for other reports
            return response['message']['content'].replace('\n', ' ').strip()
        except Exception as e:
            return f"Warning: {raw_description} (Llama-3 explanation unavailable)"