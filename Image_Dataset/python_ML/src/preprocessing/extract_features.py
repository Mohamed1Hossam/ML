# src/preprocessing/extract_features.py

import torch
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from resize_crop import preprocess_image

class DogDataset(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = preprocess_image(self.images[idx])
        label = self.labels[idx]
        return img, label

def extract_features(model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            feats = model(imgs)
            feats = feats.view(feats.size(0), -1)  # flatten
            features_list.append(feats.cpu().numpy())
            labels_list.extend(labels.numpy())

    X = np.concatenate(features_list, axis=0)
    y = np.array(labels_list)
    return X, y

if __name__ == "__main__":
    from image_loader import load_images_from_folder


    selected_classes = ["golden_retriever", "pug", "beagle", "german_shepherd", "chihuahua"]
    images, labels, class_to_idx = load_images_from_folder("../../data/raw/Images", selected_classes)


    dataset = DogDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)


    model = resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove FC layer

    X, y = extract_features(model, dataloader)

    np.save("../../data/features/X_features.npy", X)
    np.save("../../data/features/y_labels.npy", y)
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

    import json
    with open("../../data/features/class_names.json", "w") as f:
        json.dump(class_to_idx, f)
