import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS

from pipeline import MedicalPipeline
from safety_engine import DrugSafetyEngine 
from data_loader import DocumentLoader        
from summarizer import DocumentSummarizer, SafetyExplainer

app = Flask(__name__)
CORS(app)

print("--- Starting Care++ API Server ---")
medical_ai = MedicalPipeline()
safety_engine = DrugSafetyEngine() 

doc_loader = DocumentLoader()
doc_summarizer = DocumentSummarizer()

safety_explainer = SafetyExplainer() 

TEMP_DIR = "api_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route("/summarize_reports", methods=["POST"])
def summarize_reports():
    """
    Endpoint for uploading single or multiple clinical PDFs.
    Returns a list of JSON objects containing the AI-generated medical narrative for each.
    """
    
    if "report" not in request.files:
        return jsonify({"error": "No files provided under the key 'report'"}), 400

    files = request.files.getlist("report")
    
    if not files or (len(files) == 1 and files[0].filename == ''):
        return jsonify({"error": "No selected files"}), 400

    all_results = []
    
    for file in files:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join(TEMP_DIR, unique_filename)
        
        try:
            file.save(temp_path)
            chunks = doc_loader.extract_and_chunk_pdf(temp_path)
            master_summary = doc_summarizer.summarize_long_document(chunks)
            all_results.append({
                "file": file.filename,
                "summary": master_summary,
                "status": "success"
            })

        except Exception as e:
            print(f"❌ Error summarizing {file.filename}: {e}")
            all_results.append({
                "file": file.filename,
                "error": "Summarization failed",
                "details": str(e),
                "status": "failed"
            })
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return jsonify(all_results)

@app.route("/analyze_xray", methods=["POST"])
def analyze_xray():
    """
    Endpoint for uploading multiple Radiology images.
    Returns a list of JSON objects for each image analyzed.
    """
    if "file" not in request.files:
        return jsonify({"error": "No files provided under the key 'file'"}), 400

    files = request.files.getlist("file")
    
    if not files or (len(files) == 1 and files[0].filename == ''):
        return jsonify({"error": "No selected files"}), 400

    all_results = []
    for file in files:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join(TEMP_DIR, unique_filename)
        
        try:
            file.save(temp_path)
            result = medical_ai.analyze_image(temp_path)
            all_results.append(result)

        except Exception as e:
            print(f"❌ Error analyzing {file.filename}: {e}")
            all_results.append({
                "file": file.filename,
                "error": "Analysis failed",
                "details": str(e)
            })
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return jsonify(all_results)
@app.route("/check_safety", methods=["POST"])
def check_safety_batch():
    payload = request.get_json(silent=True) or {}
    new_drugs = payload.get("new_drugs", [])
    current_meds = payload.get("current_meds", [])

    if not isinstance(new_drugs, list) or not new_drugs:
        return jsonify({"error": "new_drugs must be a list"}), 400

    results = []
    overall_safe = True

    for new_drug in new_drugs:
        interactions = []
        drug_is_safe = True
        for med in current_meds:
            res = safety_engine.check_interaction(med, new_drug)
            if res.get("interaction_detected"):
                drug_is_safe = False
                overall_safe = False
                
            enhanced_desc = safety_explainer.explain_interaction(
                    med, 
                    new_drug, 
                    res.get("description")
                )
            res["description"] = enhanced_desc
        
            interactions.append({"current_med": med, "new_drug": new_drug, "result": res})
        
        results.append({"new_drug": new_drug, "is_safe": drug_is_safe, "analysis_results": interactions})

    return jsonify({"overall_safe": overall_safe, "results": results})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "models_loaded": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)