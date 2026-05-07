# -*- coding: utf-8 -*-
from torchvision import datasets
import os

train_dir = "data/processed/train"
val_dir = "data/processed/val"
test_dir = "data/processed/test"

train_dataset = datasets.ImageFolder(train_dir)
val_dataset = datasets.ImageFolder(val_dir)
test_dataset = datasets.ImageFolder(test_dir)

print(f"عدد فئات التدريب: {len(train_dataset.classes)}")
print(f"صور التدريب: {len(train_dataset)}")
print(f"صور التحقق: {len(val_dataset)}")
print(f"صور الاختبار: {len(test_dataset)}")