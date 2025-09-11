# src/dataset.py
import os
import logging
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelNetSingleViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_pairs = []

        # --- Scan the directory and store paths ---
        # Checks for top-level directories (e.g., 'chair', 'table')
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            # Checks for instance-level directories (e.g., 'chair_0001')
            for instance_name in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance_name)
                if not os.path.isdir(instance_path):
                    continue

                # Find all .png images and the .npy voxel file
                all_images = [
                    f for f in os.listdir(instance_path) if f.lower().endswith(".png")
                ]
                voxel_files = [
                    f for f in os.listdir(instance_path) if f.lower().endswith(".npy")
                ]

                # Only add to dataset if both images and a voxel file exist
                if all_images and voxel_files:
                    # Store the directory of images and the path to the voxel file
                    image_dir = instance_path
                    voxel_path = os.path.join(
                        instance_path, voxel_files[0]
                    )  # Use the first .npy file found
                    self.data_pairs.append((image_dir, voxel_path))
                else:
                    logging.warning(
                        f"Skipping {instance_path}: missing images or voxel file."
                    )

        if not self.data_pairs:
            logging.error(f"No data found in {root_dir}. Check your dataset structure.")
        else:
            logging.info(f"Found {len(self.data_pairs)} instances in {root_dir}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_dir, voxel_path = self.data_pairs[idx]

        try:
            # --- Suggestion 1: Select a random view dynamically ---
            image_name = random.choice(os.listdir(image_dir))
            img_path = os.path.join(image_dir, image_name)

            # --- Load and process image ---
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            # --- Load and process voxel data ---
            voxel = np.load(voxel_path)  # Load the 3D voxel grid

            # --- Suggestion 2: Ensure correct voxel shape and type ---
            # Ensure it's a binary occupancy grid
            voxel = (voxel > 0).astype(np.float32)

            # Add a channel dimension: (D, H, W) -> (1, D, H, W)
            voxel = torch.from_numpy(voxel).unsqueeze(0)

            return image, voxel

        except Exception as e:
            # --- Suggestion 3: Robust error handling ---
            logging.error(f"Error loading data at index {idx} (path: {image_dir}): {e}")
            # Return None or a dummy sample; the DataLoader's collate_fn should handle this
            return None, None


# A custom collate function to filter out failed samples
def collate_fn(batch):
    # Filter out samples where data loading failed (returned None)
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return (
            torch.Tensor(),
            torch.Tensor(),
        )  # Return empty tensors if a whole batch fails
    # Stack the valid samples into a batch
    return torch.utils.data.dataloader.default_collate(batch)
