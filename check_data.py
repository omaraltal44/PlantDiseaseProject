import os

data_path = "data/raw/Tomato_Leaf_Dataset"

if os.path.exists(data_path):
    classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print("✅ الفئات الموجودة:")
    for cls in classes:
        num = len(os.listdir(os.path.join(data_path, cls)))
        print(f"  - {cls}: {num} صورة")
else:
    print("المسار لا يزال غير صحيح، تحقق من وجود المجلد يدوياً.")