from typing import List, Dict, Any

from data_loader import DocumentLoader
from summarizer import generate_summary
from safety_engine import DrugSafetyEngine


loader = DocumentLoader()
engine = DrugSafetyEngine()


def summarize_pdf(pdf_path: str) -> Dict[str, Any]:
    text = loader.extract_text_from_pdf(pdf_path)
    summary = generate_summary(text)
    return {"original_text_len": len(text), "summary": summary}


def check_prescription(patient_history: List[str], new_drug: str) -> Dict[str, Any]:
    warnings = []
    for history_drug in patient_history:
        result = engine.check_interaction(history_drug, new_drug)
        if result.get("interaction_detected"):
            warnings.append(result)
    return {"is_safe": len(warnings) == 0, "warnings": warnings}
