import os
import shutil
import random
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

# المسارات
source_dir = "data/raw/Tomato_Leaf_Dataset"
train_dir = "data/processed/train"
val_dir = "data/processed/val"
test_dir = "data/processed/test"

# حذف المجلدات القديمة لو موجودة
for d in [train_dir, val_dir, test_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

# الحصول على قائمة الفئات (الأمراض)
classes = [c for c in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, c))]
print("الفئات:", classes)

# تقسيم كل فئة على حدة
for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # تقسيم 70-15-15
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    # نسخ الصور إلى المجلدات المناسبة
    for img in train_imgs:
        dest = os.path.join(train_dir, cls)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(os.path.join(cls_path, img), os.path.join(dest, img))
    
    for img in val_imgs:
        dest = os.path.join(val_dir, cls)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(os.path.join(cls_path, img), os.path.join(dest, img))
    
    for img in test_imgs:
        dest = os.path.join(test_dir, cls)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(os.path.join(cls_path, img), os.path.join(dest, img))
    
    print(f"{cls}: Train {len(train_imgs)}, Val {len(val_imgs)}, Test {len(test_imgs)}")

print("✅ تم تقسيم البيانات إلى train/val/test")

# تعريف تحويلات الصور (Augmentation + Normalization)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("✅ تم تجهيز التحويلات (transforms)")