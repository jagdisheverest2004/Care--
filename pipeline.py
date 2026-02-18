import os
import torch
from vision_model import MedicalModel
from summarizer import MedicalSummarizer  # Ensure summarizer.py exists

# --- CONFIGURATION ---
MODEL_DIR = "models"

MODALITY_CLASSES = {0: "Other", 1: "X-Ray"}

# FINAL VERIFIED MAPPING (Derived from inspect_model.py output)
# Index -> Folder Name -> Body Part
ANATOMY_MAP = {
    0: "Abdomen",        # Folder 0
    1: "Ankle",          # Folder 1
    2: "Hip",            # Folder 10
    3: "Knee",           # Folder 11
    4: "Lower Leg",      # Folder 12
    5: "Lumbar Spine",   # Folder 13
    6: "Others",         # Folder 14
    7: "Pelvis",         # Folder 15
    8: "Shoulder",       # Folder 16
    9: "Sinus",          # Folder 17
    10: "Skull",         # Folder 18
    11: "Thigh",         # Folder 19
    12: "Cervical Spine",# Folder 2
    13: "Thoracic Spine",# Folder 20
    14: "Wrist",         # Folder 21
    15: "Chest",         # Folder 3
    16: "Clavicles",     # Folder 4
    17: "Elbow",         # Folder 5
    18: "Feet",          # Folder 6
    19: "Finger",        # Folder 7
    20: "Forearm",       # Folder 8
    21: "Hand"           # Folder 9
}

class MedicalPipeline:
    def __init__(self):
        print("--- Initializing Medical AI Pipeline ---")
        
        # Load Models
        self.modality_model = MedicalModel(os.path.join(MODEL_DIR, "modality_router.pth"), num_classes=2)
        self.anatomy_model = MedicalModel(os.path.join(MODEL_DIR, "anatomy_router.pth"), num_classes=22)
        self.chest_model = MedicalModel(os.path.join(MODEL_DIR, "chest_specialist.pth"), num_classes=2)
        self.bone_model = MedicalModel(os.path.join(MODEL_DIR, "bone_specialist.pth"), num_classes=2)
        self.knee_model = MedicalModel(os.path.join(MODEL_DIR, "knee_specialist.pth"), num_classes=2)
        
        # Load Summarizer
        self.summarizer = MedicalSummarizer(model_name="google/flan-t5-base")

    def analyze_image(self, image_path):
        result = {
            "file": os.path.basename(image_path),
            "modality": "Unknown",
            "body_part": "Unknown",
            "finding": "N/A",
            "confidence": 0.0,
            "generated_report": ""
        }

        # --- STEP 1: Modality ---
        idx, conf = self.modality_model.predict(image_path)
        modality = MODALITY_CLASSES.get(int(idx) if idx is not None else 0, "Unknown")
        result["modality"] = modality
        
        if modality != "X-Ray":
            result["generated_report"] = f"System detected {modality}. Analysis halted as this is not an X-Ray."
            return result

        # --- STEP 2: Anatomy ---
        idx, conf = self.anatomy_model.predict(image_path)
        body_part = ANATOMY_MAP.get(int(idx) if idx is not None else 0, "Unknown")
        result["body_part"] = body_part

        # --- STEP 3: Specialist ---
        s_idx, s_conf = 0, 0.0
        finding = "N/A"
        
        # Routing Logic
        if body_part == "Chest":
            s_idx, s_conf = self.chest_model.predict(image_path)
            finding = "Pneumonia" if s_idx == 1 else "Normal"
            
        elif body_part in ['Elbow', 'Finger', 'Forearm', 'Hand', 'Humerus', 'Shoulder', 'Wrist']:
            s_idx, s_conf = self.bone_model.predict(image_path)
            finding = "Fracture/Abnormality" if s_idx == 1 else "Normal"
            
        elif body_part == "Knee":
            s_idx, s_conf = self.knee_model.predict(image_path)
            finding = "Arthritis/Abnormality" if s_idx == 1 else "Normal"
            
        else:
            result["generated_report"] = f"Identified {body_part} X-Ray. No specialist model available for this body part."
            return result

        # --- STEP 4: Final Data ---
        result["finding"] = finding
        result["confidence"] = round(s_conf * 100, 2)
        
        # --- STEP 5: Generate Report ---
        result["generated_report"] = self.summarizer.generate_summary(result)
        
        return result