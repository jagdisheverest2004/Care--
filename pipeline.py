import os
import torch
from vision_model import MedicalModel

# --- CONFIGURATION ---
MODEL_DIR = "models"

# Class Mappings (Must match your training folders exactly!)
MODALITY_CLASSES = {0: "Other", 1: "X-Ray"} # Based on ROCO training
# Note: For Anatomy, you need to know the EXACT index-to-name mapping from your training.
# Assuming standard alphabetical sort of folders.
ANATOMY_CLASSES_UNIFESP = [
    'Abdomen', 'Ankle', 'Cervical Spine', 'Chest', 'Clavicles', 'Elbow', 
    'Feet', 'Finger', 'Forearm', 'Hand', 'Hip', 'Knee', 'Lower Leg', 
    'Lumbar Spine', 'Others', 'Pelvis', 'Shoulder', 'Sinus', 'Skull', 
    'Thigh', 'Thoracic Spine', 'Wrist'
]
# If you used 22 classes, check your data/unifesp_sorted/train folder names order!

BINARY_CLASSES = {0: "Normal", 1: "Abnormal"}

class MedicalPipeline:
    def __init__(self):
        print("--- Initializing Medical AI Pipeline ---")
        
        # 1. Load Modality Router
        self.modality_model = MedicalModel(
            os.path.join(MODEL_DIR, "modality_router.pth"), num_classes=2
        )
        
        # 2. Load Anatomy Router
        self.anatomy_model = MedicalModel(
            os.path.join(MODEL_DIR, "anatomy_router.pth"), num_classes=22
        )
        
        # 3. Load Specialists
        self.chest_model = MedicalModel(
            os.path.join(MODEL_DIR, "chest_specialist.pth"), num_classes=2
        )
        self.bone_model = MedicalModel(
            os.path.join(MODEL_DIR, "bone_specialist.pth"), num_classes=2
        )
        self.knee_model = MedicalModel(
            os.path.join(MODEL_DIR, "knee_specialist.pth"), num_classes=2
        )

    def analyze_image(self, image_path):
        result = {
            "file": os.path.basename(image_path),
            "modality": "Unknown",
            "body_part": "Unknown",
            "finding": "N/A",
            "confidence": 0.0,
            "description": ""
        }

        # --- STEP 1: Modality Check ---
        idx, conf = self.modality_model.predict(image_path)
        modality = MODALITY_CLASSES.get(idx, "Unknown") # type: ignore
        result["modality"] = modality
        
        if modality != "X-Ray":
            result["description"] = f"Detected {modality}. This system currently only analyzes X-Rays."
            return result

        # --- STEP 2: Anatomy Detection ---
        idx, conf = self.anatomy_model.predict(image_path)
        # Map index to class name safely
        if 0 <= idx < len(ANATOMY_CLASSES_UNIFESP): # type: ignore
            body_part = ANATOMY_CLASSES_UNIFESP[idx] # type: ignore
        else:
            body_part = "Unknown"
        
        result["body_part"] = body_part

        # --- STEP 3: Specialist Routing ---
        specialist_result = None
        specialist_name = ""

        # Router Logic
        if body_part == "Chest":
            specialist_name = "Chest Specialist (Pneumonia)"
            s_idx, s_conf = self.chest_model.predict(image_path)
            # Chest Data was: 0=Normal, 1=Pneumonia (Check your folder alphabetic order!)
            # usually N comes before P.
            finding = "Pneumonia" if s_idx == 1 else "Normal"
            specialist_result = (finding, s_conf)

        elif body_part in ['Elbow', 'Finger', 'Forearm', 'Hand', 'Humerus', 'Shoulder', 'Wrist']:
            specialist_name = "Upper Limb Bone Specialist"
            s_idx, s_conf = self.bone_model.predict(image_path)
            # MURA: 0_Normal, 1_Abnormal
            finding = "Fracture/Abnormality" if s_idx == 1 else "Normal"
            specialist_result = (finding, s_conf)

        elif body_part == "Knee":
            specialist_name = "Knee Specialist (Arthritis)"
            s_idx, s_conf = self.knee_model.predict(image_path)
            # Knee: 0_Normal, 1_Abnormal
            finding = "Arthritis/Abnormality" if s_idx == 1 else "Normal"
            specialist_result = (finding, s_conf)

        else:
            result["description"] = f"Identified {body_part} X-Ray. No specific specialist model trained for this part yet."
            return result

        # --- STEP 4: Final Aggregation ---
        if specialist_result:
            finding, conf = specialist_result
            result["finding"] = finding
            result["confidence"] = round(conf * 100, 2)
            result["description"] = f"AI Analysis ({specialist_name}): Detected {finding} with {result['confidence']}% confidence."

        return result

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Create a dummy image for testing if none exists
    # Or point to a real image path here
    test_image = "data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg" # Example path
    
    if os.path.exists(test_image):
        pipeline = MedicalPipeline()
        report = pipeline.analyze_image(test_image)
        print("\n--- FINAL REPORT ---")
        print(report)
    else:
        print("Please set a valid 'test_image' path in the code to test.")