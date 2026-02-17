import os
import shutil
import pandas as pd

# --- CONFIGURATION ---
SOURCE_ROOT = "data/mura-v11"
OUTPUT_ROOT = "data/mura_sorted"

def process_phase(phase_name, csv_path):
    print(f"--- Processing {phase_name} ---")
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV not found at {csv_path}")
        return

    # Create Output Dirs (0_Normal, 1_Abnormal)
    dest_neg = os.path.join(OUTPUT_ROOT, phase_name, "0_Normal")
    dest_pos = os.path.join(OUTPUT_ROOT, phase_name, "1_Abnormal")
    os.makedirs(dest_neg, exist_ok=True)
    os.makedirs(dest_pos, exist_ok=True)

    # Read CSV (No header in MURA CSVs usually)
    # Col 0: Path, Col 1: Label (0 or 1)
    df = pd.read_csv(csv_path, header=None, names=["path", "label"])
    
    success_count = 0
    missing_count = 0

    for i, (index, row) in enumerate(df.iterrows()):
        # Clean up path (MURA paths in CSV start with "MURA-v1.1/")
        # We need to map this to "data/mura-v11/..."
        rel_path = row['path']
        label = row['label']
        
        # Fix path prefix if needed
        if rel_path.startswith("MURA-v1.1/"):
            rel_path = rel_path.replace("MURA-v1.1/", "")
            
        full_src_path = os.path.join(SOURCE_ROOT, rel_path)
        
        # The CSV points to a FOLDER (study), not an image.
        # We need to find all images inside that study folder.
        if os.path.isdir(full_src_path):
            for img_name in os.listdir(full_src_path):
                if img_name.endswith(".png") or img_name.endswith(".jpg"):
                    src_img = os.path.join(full_src_path, img_name)
                    
                    # Generate unique filename to avoid collisions
                    # e.g., XR_ELBOW_patient001_study1_image1.png
                    clean_name = rel_path.replace("/", "_") + "_" + img_name
                    
                    if label == 1:
                        dst_img = os.path.join(dest_pos, clean_name)
                    else:
                        dst_img = os.path.join(dest_neg, clean_name)
                    
                    # Copy
                    if not os.path.exists(dst_img):
                        shutil.copy2(src_img, dst_img)
                    
                    success_count += 1
        else:
            missing_count += 1
            
        if i % 100 == 0:
            print(f"Processed {i} studies... (Images: {success_count})", end="\r")

    print(f"\n✅ {phase_name} Done! Extracted {success_count} images.")

def main():
    # 1. Process Train
    process_phase("train", "data/mura-v11/train_labeled_studies.csv")
    
    # 2. Process Valid
    process_phase("val", "data/mura-v11/valid_labeled_studies.csv")
    
    print(f"\n🎉 MURA Organization Complete. Data ready at: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()