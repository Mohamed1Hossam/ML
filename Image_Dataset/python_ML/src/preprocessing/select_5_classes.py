import os
import shutil


RAW_DIR = r"C:\Users\Mina_\Desktop\python_ML\data\raw\Images"
SELECTED_DIR = r"C:\Users\Mina_\Desktop\python_ML\data\selected_5_classes"

selected_classes = {
    "golden_retriever": "n02099601-Golden_retriever",
    "pug": "n02110958-Pug",
    "beagle": "n02088364-Beagle",
    "german_shepherd": "n02106662-German_shepherd",
    "chihuahua": "n02085620-Chihuahua"
}

os.makedirs(SELECTED_DIR, exist_ok=True)

for breed, folder_name in selected_classes.items():
    src_folder = os.path.join(RAW_DIR, folder_name)
    dst_folder = os.path.join(SELECTED_DIR, breed)
    os.makedirs(dst_folder, exist_ok=True)

    # Copy images
    for file_name in os.listdir(src_folder):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            shutil.copy2(os.path.join(src_folder, file_name), os.path.join(dst_folder, file_name))

    print(f"Copied {len(os.listdir(dst_folder))} images for class '{breed}'")

print("Selected 5 classes dataset created successfully!")
