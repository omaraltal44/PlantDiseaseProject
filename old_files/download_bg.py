from openimages.download import download_dataset
import os

classes = ["Tree", "Flower", "Fruit", "Vegetable", "Bird"]

save_path = "data/random_images/Not_Tomato"

for cls in classes:
    print(f"--- Starting: {cls} ---")
    try:
        download_dataset(
            dest_dir=save_path,
            class_labels=[cls],
            limit=100
        )
    except Exception as e:
        print(f"Error with {cls}: {e}")