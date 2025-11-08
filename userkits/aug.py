import os
import random
from PIL import Image, ImageEnhance
from torchvision import transforms

# ===============================
# Config
# ===============================
dataset_dir = "train_data"
target_class = "tall_birch_forest"
target_augment_times = 5
other_augment_times = 1  # minimal augmentation

# Random seed
random.seed(42)

# ===============================
# Define augmentation
# ===============================
def augment_image(img):
    """Apply small rotation, scale, or noise"""
    # Convert to RGB (just in case)
    img = img.convert("RGB")

    # Random small rotation
    angle = random.uniform(-15, 15)
    img = img.rotate(angle)

    # Random slight scale (resize + crop)
    scale_factor = random.uniform(0.9, 1.1)
    w, h = img.size
    new_w, new_h = int(w*scale_factor), int(h*scale_factor)
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    # Center crop/pad back to original size
    img = transforms.CenterCrop((h, w))(img)

    # Optional: add slight Gaussian noise
    # Convert to tensor, add noise, convert back
    tensor = transforms.ToTensor()(img)
    noise = torch.randn_like(tensor) * 0.02  # small noise
    tensor = torch.clamp(tensor + noise, 0.0, 1.0)
    img = transforms.ToPILImage()(tensor)

    return img

# ===============================
# Apply augmentation
# ===============================
import torch

for cls in os.listdir(dataset_dir):
    cls_path = os.path.join(dataset_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    # List of image files
    images = [f for f in os.listdir(cls_path)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Determine how many times to augment
    if cls == target_class:
        times = target_augment_times
    else:
        times = other_augment_times

    for img_name in images:
        img_path = os.path.join(cls_path, img_name)
        img = Image.open(img_path)

        for i in range(times):
            aug_img = augment_image(img)
            base, ext = os.path.splitext(img_name)
            new_name = f"{base}_aug{i}{ext}"
            aug_img.save(os.path.join(cls_path, new_name))

print("Data augmentation completed!")