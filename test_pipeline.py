from pipeline import MedicalPipeline
import os

# Define paths to test images from your datasets
test_images = [
    "data/knee_sorted/test/1_Abnormal/Grade3_9429101R.png"
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