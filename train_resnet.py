import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import ResNetTransfer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"جهاز التدريب: {device}")

# استخدم حجم 224 لأن ResNet50 يتوقع 224x224
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder("data/processed/train", transform=transform_train)
val_dataset = datasets.ImageFolder("data/processed/val", transform=transform_val)

print(f"عدد الفئات: {len(train_dataset.classes)}")  # يجب أن يكون 7
print(f"الفئات: {train_dataset.classes}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# إنشاء النموذج بـ 7 فئات
model = ResNetTransfer(num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 8  # يمكنك زيادتها لنتائج أفضل
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
    
    # التقييم على validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Val Acc={acc:.4f}")

torch.save(model.state_dict(), "resnet50.pth")
print("تم حفظ النموذج resnet50.pth")