import os
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


class MedicalXrayDataset(Dataset):
    def __init__(self, root_dir: str, transform=None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_names: List[str] = []

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


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(train_dir: str, val_dir: str, num_classes: int, epochs: int = 5, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()

    if best_state is not None:
        torch.save({"model_state": best_state, "class_names": train_dataset.class_names}, "resnet50_chest_xray.pth")

    return model


def load_trained_model(weights_path: str = "resnet50_chest_xray.pth") -> Tuple[nn.Module, List[str]]:
    checkpoint = torch.load(weights_path, map_location="cpu")
    class_names = checkpoint.get("class_names", [])
    num_classes = max(1, len(class_names))
    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names


def predict_image(model: nn.Module, image_path: str, weights_path: str = "resnet50_chest_xray.pth") -> Tuple[str, float]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        class_names = checkpoint.get("class_names", [])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
    else:
        class_names = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)
    if isinstance(tensor, torch.Tensor) and tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = class_names[int(predicted.item())] if class_names else str(predicted.item())
    return label, float(confidence.item())
