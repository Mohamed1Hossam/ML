
import os
from PIL import Image

def load_images_from_folder(root_dir, selected_classes=None):

    images = []
    labels = []


    class_names = sorted(os.listdir(root_dir))
    if selected_classes:
        class_names = [cls for cls in class_names if cls in selected_classes]

    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    for cls in class_names:
        cls_folder = os.path.join(root_dir, cls)
        for file in os.listdir(cls_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cls_folder, file)
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    labels.append(class_to_idx[cls])
                except Exception as e:
                    print(f"Failed to load {img_path}: {e}")

    return images, labels, class_to_idx

if __name__ == "__main__":
    root = "../../data/raw/Images"
    selected_classes = ["golden_retriever", "pug", "beagle", "german_shepherd", "chihuahua"]
    images, labels, class_to_idx = load_images_from_folder(root, selected_classes)
    print(f"Loaded {len(images)} images from {len(class_to_idx)} classes")
