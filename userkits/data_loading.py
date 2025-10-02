import os
import cv2
import torch
import numpy as np
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def load_data(method='numpy',
              data_dir='../data', # Adjust path as needed
              test_size=0.2,
              random_state=42,
              batch_size=32):
    """
    Load image data for either numpy or PyTorch pipelines.

    Parameters:
        method (str): 'numpy' or 'pytorch'
        data_dir (str): Path to the dataset directory.
        test_size (float): Fraction of data to use for testing.
        random_state (int): Random seed for reproducibility.
        batch_size (int): Batch size for PyTorch DataLoader.

    Returns:
        For method='numpy':
            A dict with keys 'X_train', 'y_train', 'X_test', 'y_test'
            where X_train/X_test are lists of numpy arrays (images)
            and y_train/y_test are their corresponding labels.
        For method='pytorch':
            A dict with keys 'train_loader', 'test_loader',
            'train_dataset', and 'test_dataset'
            for use in PyTorch training.
    """
    if method == 'numpy':
        # Use cv2 to load images and gather features
        X = []
        y = []
        for biome_name in os.listdir(data_dir):
            biome_path = os.path.join(data_dir, biome_name)
            if not os.path.isdir(biome_path):
                continue
            for img_name in os.listdir(biome_path):
                img_path = os.path.join(biome_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                # You’ll add feature extraction here if needed.
                X.append(img)
                y.append(biome_name)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
        }
    
    elif method == 'pytorch':

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        total_size = len(full_dataset)
        train_size = int((1 - test_size) * total_size)
        test_size_local = total_size - train_size
        
        train_dataset, test_dataset = random_split(
            full_dataset, [train_size, test_size_local],
            generator=torch.Generator().manual_seed(random_state)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
        }
    
    else:
        raise ValueError("Method must be either 'numpy' or 'pytorch'")
