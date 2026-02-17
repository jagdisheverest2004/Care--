import os
import shutil

# This must match your folder structure exactly
BASE_DIR = "data/roco-dataset/all_data"

def fix_folder(phase, category):
    # Path to the class folder (e.g., train/radiology)
    target_dir = os.path.join(BASE_DIR, phase, category)
    # Path to the nested images folder (e.g., train/radiology/images)
    source_dir = os.path.join(target_dir, "images")

    if not os.path.exists(source_dir):
        print(f"Skipping {source_dir} (Folder 'images' not found - maybe already fixed?)")
        return

    print(f"Processing {source_dir}...")
    files = os.listdir(source_dir)
    
    count = 0
    for filename in files:
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, filename)
        # Move file only if it doesn't exist at destination
        if not os.path.exists(dst_path):
            shutil.move(src_path, dst_path)
            count += 1
    
    # Remove the empty 'images' folder
    try:
        os.rmdir(source_dir)
        print(f"✅ Moved {count} images and removed empty 'images' folder.")
    except OSError:
        print(f"⚠️ Moved {count} images, but 'images' folder is not empty (files might be remaining).")

def main():
    if not os.path.exists(BASE_DIR):
        print(f"❌ Error: Could not find {BASE_DIR}. Check your path!")
        return

    print("--- Fixing Folder Structure ---")
    # Fix Train Folders
    # fix_folder("train", "radiology")
    # fix_folder("train", "non-radiology")
    
    # # Fix Validation Folders
    # fix_folder("validation", "radiology")
    # fix_folder("validation", "non-radiology")
    
    # Fix Test Folders
    fix_folder("test", "radiology")
    fix_folder("test", "non-radiology")
    
    
    print("\n🎉 DONE! You can now run the training command.")

if __name__ == "__main__":
    main()