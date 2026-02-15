import os
from typing import Dict, Any, List

from flask import Flask, request, jsonify
from flask_cors import CORS

from data_loader import DocumentLoader


app = Flask(__name__)
CORS(app)


# --- MOCK CLASSES (For Testing when models aren't trained yet) ---
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

# --- CONFIGURATION SWITCHES ---
# Set these to TRUE when you are ready to use the real AI models
USE_MOCKS = False 
USE_REAL_VISION = True        # Set True for ResNet50
USE_REAL_SAFETY = True        # Set True for BioBERT DDI
USE_REAL_SUMMARIZER = False   # Set True for T5

vision_model = MockResNet50()
summarizer = MockT5Summarizer()
safety_engine = MockBioBERT()

# --- 1. SETUP VISION MODEL ---
if USE_REAL_VISION:
    try:
        from vision_model import load_trained_model, predict_image
        
        # DEFINITION: Which model do you want to load? 
        # For Phase 3, you might want "models/chest_specialist.pth"
        MODEL_PATH = "models/chest_specialist.pth" 

        if os.path.exists(MODEL_PATH):
            # FIX 1: Pass the required MODEL_PATH argument
            _vision_model, _class_names = load_trained_model(MODEL_PATH)

            class RealVision:
                def predict(self, image_path: str) -> Dict[str, Any]:
                    # FIX 2: Pass _class_names to the prediction function
                    label, confidence = predict_image(_vision_model, _class_names, image_path)
                    return {"body_part": "Unknown", "finding": label, "confidence": confidence}

            vision_model = RealVision()
            print(f"✅ Vision Model Loaded from {MODEL_PATH}")
        else:
            print(f"⚠️ Warning: Vision model file not found at {MODEL_PATH}. Using Mock.")

    except ImportError:
        print("⚠️ Warning: vision_model.py not found.")
    except Exception as e:
        print(f"❌ Error loading Vision Model: {e}")

# --- 2. SETUP SUMMARIZER ---
if USE_REAL_SUMMARIZER:
    try:
        from summarizer import generate_summary

        class RealSummarizer:
            def generate_summary(self, text: str) -> str:
                return generate_summary(text)

        summarizer = RealSummarizer()
        print("✅ Summarizer Model Loaded")
    except ImportError:
        print("⚠️ Warning: summarizer.py not found.")

# --- 3. SETUP SAFETY ENGINE ---
if USE_REAL_SAFETY:
    try:
        from safety_engine import DrugSafetyEngine

        ddi_model_dir = os.getenv("DDI_MODEL_DIR", "biobert_ddi")
        if os.path.isdir(ddi_model_dir):
            safety_engine = DrugSafetyEngine(model_name=ddi_model_dir)
            print(f"✅ Safety Engine Loaded from {ddi_model_dir}")
        else:
            # Fallback to standard initialization if folder doesn't exist
            safety_engine = DrugSafetyEngine()
            print("✅ Safety Engine Loaded (Default/Rules Only)")
    except ImportError:
        print("⚠️ Warning: safety_engine.py not found.")


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

    # Create temp directory if it doesn't exist
    os.makedirs("temp_xray", exist_ok=True)
    temp_path = os.path.join("temp_xray", file.filename)
    file.save(temp_path)

    try:
        # loader.preprocess_xray(temp_path) # Optional depending on your loader logic
        result = vision_model.predict(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/check_safety", methods=["POST"])
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
        interactions_analysis = [] 
        drug_is_safe = True
        
        for med in current_meds:
            result = safety_engine.check_interaction(med, new_drug)
            
            # Update safety flags ONLY if interaction is found
            if result.get("interaction_detected"):
                drug_is_safe = False
                overall_safe = False
            
            # ALWAYS add the result, whether Safe or Unsafe
            interactions_analysis.append({
                "current_med": med, 
                "new_drug": new_drug,
                "result": result
            })

        results.append({
            "new_drug": new_drug, 
            "is_safe": drug_is_safe, 
            "analysis_results": interactions_analysis 
        })

    return jsonify({"patient_id": patient_id, "overall_safe": overall_safe, "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)