from torchvision import datasets
import os

train_dir = "data/processed/train"
val_dir = "data/processed/val"
test_dir = "data/processed/test"

train_dataset = datasets.ImageFolder(train_dir)
val_dataset = datasets.ImageFolder(val_dir)
test_dataset = datasets.ImageFolder(test_dir)

print(f"Number of training classes: {len(train_dataset.classes)}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}")