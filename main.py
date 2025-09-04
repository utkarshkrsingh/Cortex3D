# main.py
import torch
import time
from src.dataset import ModelNetSingleViewDataset
from src.train import train_model
from src.config import *
from torchvision import transforms
from torch.utils.data import DataLoader


def main():
    # --- Step 1: Transformations for image data ---
    transform = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    )

    # --- Step 2: Load dataset ---
    print("[INFO] Loading dataset...")
    train_dataset = ModelNetSingleViewDataset(
        root_dir=DATA_PATH + "train", transform=transform
    )

    if len(train_dataset) == 0:
        print("[ERROR] No training data found! Check DATA_PATH or dataset structure.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"[INFO] Dataset loaded. Total samples: {len(train_dataset)}")

    # --- Step 3: Check GPU/CPU ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training will run on: {device}")

    # --- Step 4: Train model ---
    print("[INFO] Starting training...")
    start_time = time.time()

    train_model(train_loader)

    end_time = time.time()
    print(f"[INFO] Training complete in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
