import os
import shutil
import pandas as pd
import random

# --- CONFIGURATION ---
# Path to the downloaded JPEG dataset
SOURCE_IMAGES_DIR = "data/unifesp_jpeg/train"
CSV_PATH = "data/unifesp_jpeg/train.csv"

# Path where we want the final sorted data
OUTPUT_DIR = "data/unifesp_sorted"
VAL_SPLIT = 0.2  # 20% for testing

def main():
    print("--- Organizing UNIFESP JPEG Dataset ---")
    
    # 1. Setup Directories
    train_dir = os.path.join(OUTPUT_DIR, "train")
    val_dir = os.path.join(OUTPUT_DIR, "val")
    
    for d in [train_dir, val_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # 2. Read CSV
    print(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # 3. Process
    success_count = 0
    missing_count = 0
    
    print(f"Processing {len(df)} entries...")
    
    for index, row in df.iterrows():
        # Get filename and label from CSV
        # Note: In your CSV snippet, the first column 'file_name' is the ID
        file_id = str(row.iloc[0]).strip() 
        target_str = str(row['Target']).strip()
        
        # Handle multi-labels "0 12" -> take "0"
        target_class = target_str.split()[0]
        
        # Construct current file path (The download has .jpg extension)
        src_file = os.path.join(SOURCE_IMAGES_DIR, file_id + ".jpg")
        
        # Check if file exists
        if not os.path.exists(src_file):
            # Try .png just in case
            src_file = os.path.join(SOURCE_IMAGES_DIR, file_id + ".png")
        
        if os.path.exists(src_file):
            # Decide: Train or Val?
            dest_root = val_dir if random.random() < VAL_SPLIT else train_dir
            
            # Create Class Folder
            class_folder = os.path.join(dest_root, target_class)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            
            # Copy File
            dest_file = os.path.join(class_folder, os.path.basename(src_file))
            if not os.path.exists(dest_file):
                shutil.copy2(src_file, dest_file)
                
            success_count += 1
            if success_count % 100 == 0:
                print(f"Organized {success_count} images...", end="\r")
        else:
            missing_count += 1

    print(f"\n✅ DONE! Organized: {success_count} | Missing: {missing_count}")
    print(f"Ready for training at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()