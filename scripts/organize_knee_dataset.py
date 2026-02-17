import os
import shutil

# --- CONFIGURATION ---
SOURCE_ROOT = "data/knee"
OUTPUT_ROOT = "data/knee_sorted"

def main():
    print("--- Organizing Knee Dataset (Merging Classes 1-4) ---")

    # We process train, val, and test folders
    for phase in ["train", "val", "test"]:
        src_phase_path = os.path.join(SOURCE_ROOT, phase)
        
        if not os.path.exists(src_phase_path):
            print(f"Skipping {phase} (not found)")
            continue

        print(f"Processing {phase}...")

        # Create Destination Folders
        dest_neg = os.path.join(OUTPUT_ROOT, phase, "0_Normal")
        dest_pos = os.path.join(OUTPUT_ROOT, phase, "1_Abnormal")
        os.makedirs(dest_neg, exist_ok=True)
        os.makedirs(dest_pos, exist_ok=True)

        # Iterate through original classes 0, 1, 2, 3, 4
        for class_idx in range(5):
            class_str = str(class_idx)
            src_class_path = os.path.join(src_phase_path, class_str)
            
            if not os.path.exists(src_class_path):
                continue

            # Decide: 0 goes to Normal, 1-4 goes to Abnormal
            if class_idx == 0:
                target_dir = dest_neg
            else:
                target_dir = dest_pos

            # Copy files
            files = os.listdir(src_class_path)
            count = 0
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_file = os.path.join(src_class_path, fname)
                    
                    # Rename to avoid collisions (e.g., "Grade2_image.png")
                    new_name = f"Grade{class_idx}_{fname}"
                    dst_file = os.path.join(target_dir, new_name)
                    
                    shutil.copy2(src_file, dst_file)
                    count += 1
            
            print(f"  - Copied {count} images from Class {class_idx} to {'Normal' if class_idx==0 else 'Abnormal'}")

    print(f"\n✅ Knee Organization Complete. Data ready at: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()