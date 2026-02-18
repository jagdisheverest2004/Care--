from pipeline import MedicalPipeline
import os

# Define paths to test images from your datasets
test_images = [
    # 1a. A Chest X-ray (Pneumonia)
    "data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg",
    
    # 1b. A Chest X-ray (Normal)
    "data/chest_xray/test/NORMAL/IM-0003-0001.jpeg",
    
    
    # 2a. A Knee X-ray (From your sorted dataset)
    "data/knee_sorted/test/1_Abnormal/Grade3_9429101R.png",
    
    # 2b. A Knee X-ray (Normal)
    "data/knee_sorted/test/0_Normal/Grade0_9003175L.png",
    
    # 3a. An Upper Limb X-ray (From MURA, e.g., Elbow)
    "data/mura_sorted/val/0_Normal/valid_XR_ELBOW_patient11204_study1_negative__image1.png",
    
    # 3b. An Upper Limb X-ray (Abnormal)
    "data/mura_sorted/val/1_Abnormal/valid_XR_ELBOW_patient11186_study1_positive__image1.png"
    
]

def main():
    # Initialize Pipeline
    pipeline = MedicalPipeline()
    
    print("\n" + "="*50)
    print("STARTING PIPELINE TEST")
    print("="*50 + "\n")

    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"📸 Analyzing: {img_path} ...")
            report = pipeline.analyze_image(img_path)
            
            # Print the JSON output clearly
            import json
            print(json.dumps(report, indent=4))
            print("-" * 30)
        else:
            print(f"⚠️ Image not found: {img_path}")

if __name__ == "__main__":
    main()