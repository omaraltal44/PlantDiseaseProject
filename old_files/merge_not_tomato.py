import os
import shutil
import random
from sklearn.model_selection import train_test_split

source_dir = "data/random_images/Not_Tomato"
target_base = "data/processed"
all_image_paths = []

print("Searching for images...")
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(root, file)
            all_image_paths.append(full_path)

print(f"Total images collected: {len(all_image_paths)}")

if len(all_image_paths) == 0:
    print("No images found. Make sure there are jpg/png files inside Not_Tomato folder.")
    exit()

train_paths, temp_paths = train_test_split(all_image_paths, test_size=0.3, random_state=42)
val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)

for split, path_list in zip(['train', 'val', 'test'], [train_paths, val_paths, test_paths]):
    target_dir = os.path.join(target_base, split, "Not_Tomato")
    os.makedirs(target_dir, exist_ok=True)
    
    for src_path in path_list:
        img_name = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, img_name)
        shutil.copy2(src_path, dst_path)
    
    print(f"{split}: Copied {len(path_list)} images to {target_dir}")

print("Not_Tomato class prepared successfully!")