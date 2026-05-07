import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from models import EfficientNetTransfer
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder("data/processed/train", transform=train_transform)
    val_dataset = datasets.ImageFolder("data/processed/val", transform=val_transform)

    print("Classes:", train_dataset.classes)
    print("Number of classes:", len(train_dataset.classes))
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    class_counts = [train_dataset.targets.count(i) for i in range(len(train_dataset.classes))]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for label in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = EfficientNetTransfer(num_classes=len(train_dataset.classes)).to(device)

    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.classifier.parameters(), lr=0.001)

    epochs = 15
    print("Phase 1: Training classifier head only...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
        acc = correct / len(val_dataset)
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, val_acc={acc:.4f}")

    print("Phase 2: Fine-tuning entire model...")
    for param in model.model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    epochs2 = 10
    for epoch in range(epochs2):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
        acc = correct / len(val_dataset)
        print(f"Fine-tune Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, val_acc={acc:.4f}")

    torch.save(model.state_dict(), "efficientnet_7classes_fixed.pth")
    print("Model saved as efficientnet_7classes_fixed.pth")

if __name__ == '__main__':
    main()