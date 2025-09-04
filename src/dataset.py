# src/dataset.py
import os
import logging
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNetSingleViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for instance in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance)
                if not os.path.isdir(instance_path):
                    continue

                # Random single-view image
                images = [
                    f for f in os.listdir(instance_path) if f.lower().endswith(".png")
                ]
                if not images:
                    continue
                random_image = random.choice(images)
                image_path = os.path.join(instance_path, random_image)

                voxel_path = os.path.join(instance_path, "voxel.npy")
                if os.path.exists(voxel_path):
                    self.data.append((image_path, voxel_path))

        logging.info(f"Found {len(self.data)} samples in {root_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, voxel_path = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        voxel = np.load(voxel_path)
        voxel = voxel.squeeze()
        voxel = torch.tensor(voxel, dtype=torch.float32)

        return image, voxel
