
from torchvision import transforms
from PIL import Image

# Define preprocessing transform
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

def preprocess_image(img):
    """
    Preprocess a single PIL image: resize + normalize.

    Parameters:
        img (PIL.Image): Input image.

    Returns:
        tensor: Torch tensor of shape [3, 224, 224]
    """
    return preprocess_transform(img)

if __name__ == "__main__":
    from PIL import Image
    img = Image.open("../../data/raw/Images/golden_retriever/sample.jpg").convert("RGB")
    tensor_img = preprocess_image(img)
    print(tensor_img.shape)
