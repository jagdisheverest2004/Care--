import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your custom modules
from pipeline import MedicalPipeline
from safety_engine import DrugSafetyEngine  # Assuming you still want the safety feature
from data_loader import DocumentLoader        
from summarizer import DocumentSummarizer

app = Flask(__name__)
CORS(app)

# --- 1. INITIALIZE THE BRAIN ---
print("--- Starting Care++ API Server ---")
medical_ai = MedicalPipeline()
safety_engine = DrugSafetyEngine() 

# <-- ADD THESE TWO LINES -->
doc_loader = DocumentLoader()
doc_summarizer = DocumentSummarizer()

# Ensure temp directory exists
TEMP_DIR = "api_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


# --- NEW: THE CLINICAL REPORT SUMMARIZER ENDPOINT ---
@app.route("/summarize_reports", methods=["POST"])
def summarize_reports():
    """
    Endpoint for uploading single or multiple clinical PDFs.
    Returns a list of JSON objects containing the AI-generated medical narrative for each.
    """
    # 1. Check if the 'report' key exists in the request
    if "report" not in request.files:
        return jsonify({"error": "No files provided under the key 'report'"}), 400

    # 2. Get the list of all uploaded files (handles 1 or many)
    files = request.files.getlist("report")
    
    if not files or (len(files) == 1 and files[0].filename == ''):
        return jsonify({"error": "No selected files"}), 400

    all_results = []

    # 3. Process each PDF in a loop
    for file in files:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join(TEMP_DIR, unique_filename)
        
        try:
            # Save PDF temporarily
            file.save(temp_path)

            # 4. Extract Text and Chunk it using PyMuPDF
            chunks = doc_loader.extract_and_chunk_pdf(temp_path)

            # 5. Generate the Narrative Summary using Llama-3
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
            # 6. Cleanup: Remove the temporary PDF
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 7. Return the results
    return jsonify(all_results)

# --- 2. THE X-RAY ANALYSIS ENDPOINT ---
@app.route("/analyze_xray", methods=["POST"])
def analyze_xray():
    """
    Endpoint for uploading multiple Radiology images.
    Returns a list of JSON objects for each image analyzed.
    """
    # 1. Check if the 'file' key exists in the request
    if "file" not in request.files:
        return jsonify({"error": "No files provided under the key 'file'"}), 400

    # 2. Get the list of all uploaded files using getlist
    files = request.files.getlist("file")
    
    if not files or (len(files) == 1 and files[0].filename == ''):
        return jsonify({"error": "No selected files"}), 400

    all_results = []

    # 3. Process each file in a loop
    for file in files:
        # Generate a unique filename to prevent collisions
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join(TEMP_DIR, unique_filename)
        
        try:
            file.save(temp_path)

            # 4. Run the Pipeline for this specific image
            # This calls: Modality -> Anatomy -> Specialist -> T5 Summarizer
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
            # 5. Cleanup: Remove the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 6. Return the FULL LIST of results as a JSON array
    return jsonify(all_results)
# --- 3. DRUG SAFETY ENDPOINT (Kept from your previous code) ---
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
            interactions.append({"current_med": med, "new_drug": new_drug, "result": res})
        
        results.append({"new_drug": new_drug, "is_safe": drug_is_safe, "analysis_results": interactions})

    return jsonify({"overall_safe": overall_safe, "results": results})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "models_loaded": True})

if __name__ == "__main__":
    # We use threaded=False if using CUDA sometimes, but True is usually fine for Flask
    app.run(host="0.0.0.0", port=5000, debug=False)