import os
from typing import Dict, Any, List

from flask import Flask, request, jsonify
from flask_cors import CORS

from data_loader import DocumentLoader


app = Flask(__name__)
CORS(app)


class MockResNet50:
    def predict(self, image_path: str) -> Dict[str, str]:
        return {"body_part": "Chest", "finding": "Pneumonia"}


class MockT5Summarizer:
    def generate_summary(self, text: str) -> str:
        return text[:200] + ("..." if len(text) > 200 else "")


class MockBioBERT:
    def check_interaction(self, drug_a: str, drug_b: str) -> Dict[str, Any]:
        return {"interaction_detected": False, "confidence": 0.1, "severity": "Low"}


loader = DocumentLoader()

# Force usage of real models
USE_MOCKS = False 
USE_REAL_VISION = False       # Set True if you have the vision model ready
USE_REAL_SAFETY = True        # Set True for BioBERT DDI
USE_REAL_SUMMARIZER = False   # Set True if you have the T5 model ready

vision_model = MockResNet50()
summarizer = MockT5Summarizer()
safety_engine = MockBioBERT()

if USE_REAL_VISION:
    try:
        from vision_model import load_trained_model, predict_image

        _vision_model, _class_names = load_trained_model()

        class RealVision:
            def predict(self, image_path: str) -> Dict[str, Any]:
                label, confidence = predict_image(_vision_model, image_path)
                return {"body_part": "Unknown", "finding": label, "confidence": confidence}

        vision_model = RealVision()
    except FileNotFoundError:
        pass

if USE_REAL_SUMMARIZER:
    from summarizer import generate_summary

    class RealSummarizer:
        def generate_summary(self, text: str) -> str:
            return generate_summary(text)

    summarizer = RealSummarizer()

if USE_REAL_SAFETY:
    from safety_engine import DrugSafetyEngine

    ddi_model_dir = os.getenv("DDI_MODEL_DIR", "biobert_ddi")
    if os.path.isdir(ddi_model_dir):
        safety_engine = DrugSafetyEngine(model_name=ddi_model_dir)
    else:
        safety_engine = DrugSafetyEngine()


def _get_current_meds_from_db(patient_id: str) -> List[str]:
    # TODO: Replace with real DB lookup (MySQL/PostgreSQL).
    # Stub returns empty list for now.
    return []


@app.route("/analyze_xray", methods=["POST"])
def analyze_xray():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return jsonify({"error": "Unsupported file type"}), 400

    temp_path = os.path.join("temp_xray" + os.path.splitext(file.filename)[1])
    file.save(temp_path)

    try:
        loader.preprocess_xray(temp_path)
        result = vision_model.predict(temp_path)
        return jsonify(result)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/summarize_report", methods=["POST"])
def summarize_report():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Unsupported file type"}), 400

    temp_path = os.path.join("temp_report.pdf")
    file.save(temp_path)

    try:
        text = loader.extract_text_from_pdf(temp_path)
        summary = summarizer.generate_summary(text)
        return jsonify({"original_text_len": len(text), "summary": summary})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/check_safety", methods=["POST"])
def check_safety():
    payload = request.get_json(silent=True) or {}
    current_meds = payload.get("current_meds", [])
    new_drug = payload.get("new_drug", "")

    if not isinstance(current_meds, list) or not new_drug:
        return jsonify({"error": "Invalid payload"}), 400

    warnings = []
    for med in current_meds:
        result = safety_engine.check_interaction(med, new_drug)
        if result.get("interaction_detected"):
            warnings.append(result)

    return jsonify({"is_safe": len(warnings) == 0, "warnings": warnings})


@app.route("/check_safety_batch", methods=["POST"])
def check_safety_batch():
    payload = request.get_json(silent=True) or {}
    patient_id = payload.get("patient_id", "")
    new_drugs = payload.get("new_drugs", [])
    current_meds = payload.get("current_meds")

    if not isinstance(new_drugs, list) or not new_drugs:
        return jsonify({"error": "Invalid payload"}), 400

    if current_meds is None:
        if not patient_id:
            return jsonify({"error": "patient_id required when current_meds not provided"}), 400
        current_meds = _get_current_meds_from_db(patient_id)

    if not isinstance(current_meds, list):
        return jsonify({"error": "current_meds must be a list"}), 400

    results = []
    overall_safe = True
    for new_drug in new_drugs:
        warnings = []
        for med in current_meds:
            result = safety_engine.check_interaction(med, new_drug)
            if result.get("interaction_detected"):
                warnings.append({"current_med": med, "result": result})
        is_safe = len(warnings) == 0
        if not is_safe:
            overall_safe = False
        results.append({"new_drug": new_drug, "is_safe": is_safe, "warnings": warnings})

    return jsonify({"patient_id": patient_id, "overall_safe": overall_safe, "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
