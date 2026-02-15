import os
from typing import Tuple, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# --- 1. Dataset Class ---
class MedicalXrayDataset(Dataset):
    def __init__(self, root_dir: str, transform=None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_names: List[str] = []

        # Sort folders to ensure consistent class indices
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_names.append(class_name)
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_path, file_name), idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# --- 2. Model Builder (Optimized for Medical Imaging) ---
from torchvision.models import ResNet

def build_model(num_classes: int) -> nn.Module:
    model: ResNet = models.resnet50(pretrained=True)
    
    # 1. Freeze early layers (Edges, Shapes, Colors - ImageNet knows this well)
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Unfreeze the LAST Block (Layer 4)
    # This allows the model to learn "Medical Textures" (Bone opacity, lesions)
    for param in model.layer4.parameters(): # type: ignore
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# --- 3. Training Function ---
def train_model(train_dir: str, val_dir: str, num_classes: int, output_path: str, epochs: int = 5, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} for {num_classes} classes...")
    print(f"Model will be saved to: {output_path}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Augmentation: Mirroring
        transforms.RandomRotation(10),     # Augmentation: Rotation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MedicalXrayDataset(train_dir, transform=train_transform)
    val_dataset = MedicalXrayDataset(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Use a lower learning rate because we are Fine-Tuning
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # --- Validation & Metrics Phase ---
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                # Store predictions and labels for sklearn metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate Metrics
        acc = accuracy_score(all_labels, all_preds)
        
        # 'weighted' average handles class imbalance better than 'binary' for multi-class tasks
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")
        print(f"   >> Val Acc: {acc:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        # Save Best Model based on F1 Score (Standard for Medical AI)
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()

    if best_state is not None:
        torch.save({"model_state": best_state, "class_names": train_dataset.class_names}, output_path)
        print(f"Training Complete. Best Model saved to {output_path} (Best F1: {best_f1:.4f})")

    return model


# --- 4. Loading & Prediction (Cleaned Up - No Defaults) ---

def load_trained_model(weights_path: str) -> Tuple[nn.Module, List[str]]:
    """
    Loads a model from a specific .pth file.
    Args:
        weights_path (str): REQUIRED. Path to the .pth file (e.g., 'models/chest_specialist.pth')
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    class_names = checkpoint.get("class_names", [])
    num_classes = max(1, len(class_names))
    
    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    return model, class_names


def predict_image(model: nn.Module, class_names: List[str], image_path: str) -> Tuple[str, float]:
    """
    Predicts using an already loaded model. 
    We pass class_names explicitly to avoid reloading them.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor: torch.Tensor = transform(image)  # type: ignore
    if isinstance(tensor, torch.Tensor) and tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    # Move tensor to same device as model
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = class_names[int(predicted.item())] if class_names else str(predicted.item())
    return label, float(confidence.item())