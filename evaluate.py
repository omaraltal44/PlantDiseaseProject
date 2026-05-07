import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from models import SimpleCNN, DeeperCNN, ResNetTransfer, EfficientNetTransfer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluation device: {device}")

test_dataset_raw = datasets.ImageFolder("data/processed/test")
class_names = test_dataset_raw.classes
num_classes = len(class_names)
print("Classes:", class_names)
print(f"Number of test images: {len(test_dataset_raw)}")

model_configs = {
    "SimpleCNN": {
        "class": SimpleCNN,
        "weight": "simplecnn_7classes.pth",
        "img_size": 128
    },
    "DeeperCNN": {
        "class": DeeperCNN,
        "weight": "deepercnn_7classes.pth",
        "img_size": 128
    },
    "ResNet50": {
        "class": ResNetTransfer,
        "weight": "resnet50_7classes.pth",
        "img_size": 224
    },
    "EfficientNet": {
        "class": EfficientNetTransfer,
        "weight": "efficientnet_7classes_fixed.pth",
        "img_size": 224
    }
}

def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

results = {}
for name, config in model_configs.items():
    weight_path = config["weight"]
    if not os.path.exists(weight_path):
        print(f"⚠️ {name} not found (missing {weight_path}) – skipping")
        continue
    
    print(f"\n🔍 Evaluating {name}...")

    transform = get_transform(config["img_size"])
    test_dataset = datasets.ImageFolder("data/processed/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = config["class"](num_classes=7)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    results[name] = {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1': f1}
    print(f"{name} -> Acc: {acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'confusion_{name}.png')
    plt.close()

if results:
    df = pd.DataFrame(results).T
    print("\n===== Final Comparison Table =====")
    print(df)
    df.to_csv("comparison.csv")
    print("Comparison table and all confusion matrices saved.")
else:
    print("No model evaluated. Make sure weight files exist.")