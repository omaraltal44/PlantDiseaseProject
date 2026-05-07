import os
import shutil
import random
from sklearn.model_selection import train_test_split

# المصدر: المجلد الرئيسي الذي يحتوي على مجلدات (bird, flower, ...)
source_dir = "data/random_images/Not_Tomato"

# الوجهة: مجلدات المشروع
target_base = "data/processed"

# قائمة لجمع كل مسارات الصور
all_image_paths = []

# اجتياز كل المجلدات والأدلة الفرعية للوصول إلى الصور
print("جاري البحث عن الصور...")
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(root, file)
            all_image_paths.append(full_path)

print(f"✅ إجمالي الصور المجمعة: {len(all_image_paths)}")

if len(all_image_paths) == 0:
    print("❌ لم يتم العثور على أي صور. تأكد من وجود صور بصيغة jpg/png داخل مجلد Not_Tomato.")
    exit()

# تقسيم الصور إلى train/val/test (70% / 15% / 15%)
train_paths, temp_paths = train_test_split(all_image_paths, test_size=0.3, random_state=42)
val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)

# نسخ الصور إلى المجلدات المناسبة تحت اسم "Not_Tomato"
for split, path_list in zip(['train', 'val', 'test'], [train_paths, val_paths, test_paths]):
    target_dir = os.path.join(target_base, split, "Not_Tomato")
    os.makedirs(target_dir, exist_ok=True)
    
    for src_path in path_list:
        img_name = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, img_name)
        shutil.copy2(src_path, dst_path)
    
    print(f"✅ {split}: تم نسخ {len(path_list)} صورة إلى {target_dir}")

print("🎉 تم تجهيز فئة Not_Tomato بنجاح!")