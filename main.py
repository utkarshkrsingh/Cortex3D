# main.py (Revised)
import torch
import time
from src.dataset import ModelNetSingleViewDataset

# --- Import your model, optimizer, and loss ---
from src.model import (
    SingleViewReconstructionNet,
)  # Assuming your model class is named this
from src.train import train_model
from src.config import *
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def main():
    # --- Step 1: Transformations ---
    transform = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    )

    # --- Step 2: Load dataset ---
    print("[INFO] Loading dataset...")
    train_dataset = ModelNetSingleViewDataset(
        root_dir=DATA_PATH + "train", transform=transform
    )
    if len(train_dataset) == 0:
        print("[ERROR] No training data found! Check DATA_PATH.")
        return
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"[INFO] Dataset loaded. Total samples: {len(train_dataset)}")

    # --- Step 3: Setup Model, Optimizer, and Loss ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training will run on: {device}")

    # Instantiate the model and move it to the correct device
    model = ReconstructionNet().to(device)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Setup loss function
    criterion = nn.BCELoss()  # A common choice for voxel occupancy

    # --- Step 4: Train model ---
    print("[INFO] Starting training...")
    start_time = time.time()

    # Pass the necessary components to the training function
    train_model(model, train_loader, optimizer, criterion, device)

    end_time = time.time()
    print(f"[INFO] Training complete in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
