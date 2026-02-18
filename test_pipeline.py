from pipeline import MedicalPipeline
import os

# Define paths to test images from your datasets
test_images = [
    # 1. Chest X-ray 
    "data/unifesp_sorted/train/3/10053755320637729867508668285241208441.jpg",
    
    # 2. A Knee X-ray
    "data/unifesp_sorted/train/11/10242799675195671634897807131985000448.jpg",
    
    # 3a. An Elbow X-ray
    "data/unifesp_sorted/train/5/11504724594984965602352794642021327380.jpg",

    
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