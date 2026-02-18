from pipeline import MedicalPipeline
import os

# Define paths to test images from your datasets
test_images = [
    # 1. A Chest X-ray (Pneumonia)
    "data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg",
    
    # 2. A Knee X-ray (From your sorted dataset)
    # Pick a random file from your knee folder
    "data/knee_sorted/test/1_Abnormal/Grade4_9001695L.png" 
    # (Adjust filename to one that actually exists in your folder!)
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