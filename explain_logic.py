from typing import List, Dict

from safety_engine import DrugSafetyEngine


def generate_explanation(interaction_result: Dict[str, object], drug_a: str, drug_b: str) -> str:
    if not interaction_result.get("interaction_detected", False):
        return "Safe to administer. No known adverse interactions found in database."

    confidence = interaction_result.get("confidence", 0.0)
    interaction_type = "potential interaction"
    return (
        f"Warning: Co-administration of {drug_a} and {drug_b} may lead to "
        f"{interaction_type}. Risk level is {confidence:.2f}."
    )


def process_prescription(patient_history_list: List[str], new_drug: str) -> Dict[str, object]:
    engine = DrugSafetyEngine()
    warnings = []

    for history_drug in patient_history_list:
        result = engine.check_interaction(history_drug, new_drug)
        if result.get("interaction_detected", False):
            warnings.append(generate_explanation(result, history_drug, new_drug))

    return {
        "is_safe": len(warnings) == 0,
        "warnings": warnings,
    }
