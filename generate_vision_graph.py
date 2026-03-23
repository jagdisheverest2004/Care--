import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Import your exact build_model function from your vision_model.py
from vision_model import build_model 

def evaluate_saved_vision_model(model_path: str, test_data_dir: str, model_title: str):
    print(f"--- Evaluating {model_title} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return
    if not os.path.exists(test_data_dir):
        print(f"❌ Error: Test data directory not found at {test_data_dir}")
        return

    # 1. Load the Saved Checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract class names and number of classes directly from your saved .pth file
    class_names = checkpoint.get("class_names", [])
    num_classes = len(class_names)
    if num_classes == 0:
        print("❌ Error: 'class_names' not found in the checkpoint. Check your training save logic.")
        return
        
    print(f"Loaded Classes: {class_names}")

    # 2. Build Model and Load Weights
    model = build_model(num_classes).to(device)
    
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 3. Prepare Testing Data using the exact transform from your vision_model.py
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # We use ImageFolder because your test data should be organized in folders by class
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    y_true = []
    y_pred = []
    y_probs = []

    # 4. Run Inference
    print("Running predictions... Please wait.")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
            # Store probabilities for the positive class (useful for ROC if binary)
            if num_classes == 2:
                y_probs.extend(probs[:, 1].cpu().numpy())

    # =========================================================
    #                    GENERATE METRICS & PLOTS
    # =========================================================

    # 1. Model Performance Metrics (Accuracy, Precision, Recall, F1)
    print("\n" + "="*50)
    print(f"  MODEL PERFORMANCE METRICS ({model_title})")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.title(f'{model_title} - Confusion Matrix', fontsize=15, fontweight='bold')
    plt.ylabel('Actual Diagnosis', fontsize=12, fontweight='bold')
    plt.xlabel('AI Predicted Diagnosis', fontsize=12, fontweight='bold')
    plt.tight_layout()
    cm_filename = f'{model_title.replace(" ", "_")}_Confusion_Matrix.png'
    plt.savefig(cm_filename, dpi=300)
    print(f"✅ Saved: {cm_filename}")
    plt.show()

    # 3. ROC Curve (Only generated for Binary Classification, e.g., Pneumonia vs Normal)
    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
        plt.title(f'{model_title} - ROC Curve', fontsize=15, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        roc_filename = f'{model_title.replace(" ", "_")}_ROC_Curve.png'
        plt.savefig(roc_filename, dpi=300)
        print(f"✅ Saved: {roc_filename}")
        plt.show()

if __name__ == "__main__":
    # --- Instructions ---
    # Change the model_path to point to your saved .pth file.
    # Change the test_data_dir to point to the 'test' or 'val' folder for that specific model.
    
    # evaluate_saved_vision_model(
    #     model_path="models/chest_specialist.pth", 
    #     test_data_dir="data/chest_xray/test", # Example path, change to yours
    #     model_title="Chest Pathology Model"
    # )
    
    # You can call it again for other models:
    evaluate_saved_vision_model(
        model_path="models/anatomy_router.pth", 
        test_data_dir="data/unifesp_sorted/val", # Example path, change to yours
        model_title="Anatomy Classification Model"
    )