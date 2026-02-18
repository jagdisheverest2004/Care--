import torch
import os

MODEL_PATH = "models/anatomy_router.pth"

def inspect():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    print(f"🔍 Loading {MODEL_PATH}...")
    try:
        # Load the checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        
        # Check if 'class_names' key exists
        if "class_names" in checkpoint:
            classes = checkpoint["class_names"]
            print("\n✅ FOUND EXACT CLASS MAPPING:")
            print("="*30)
            for idx, name in enumerate(classes):
                print(f"Index {idx}: {name}")
            print("="*30)
            
            # Print the list formatted for Python code
            print("\nCOPY THIS LIST INTO YOUR PIPELINE.PY:")
            print(f"ANATOMY_CLASSES_UNIFESP = {classes}")
            
        else:
            print("⚠️ 'class_names' key NOT found in .pth file.")
            print("Keys found:", checkpoint.keys())
            
    except Exception as e:
        print(f"❌ Error reading model: {e}")

if __name__ == "__main__":
    inspect()