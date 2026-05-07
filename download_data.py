import kagglehub
import os
import shutil

print("جاري تحميل dataset الطماطم...")
path = kagglehub.dataset_download("syedhashirali260/tomato-leaf-disease-dataset-6-classes")
print(f"تم التحميل إلى: {path}")

# نقل الملفات إلى مجلد data/raw
os.makedirs("data/raw", exist_ok=True)

for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join("data/raw", item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("✅ البيانات اتنقلت إلى مجلد data/raw")