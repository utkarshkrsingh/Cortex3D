# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Pix2VoxMini
from src.config import *
from src.utils import plot_loss_curve


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Runs one epoch of training
    """
    model.train()
    epoch_loss = 0

    for batch_idx, (images, voxels) in enumerate(loader):
        # Move data to GPU/CPU
        images, voxels = images.to(device), voxels.to(device)

        # Debugging shapes (only on first batch of first epoch)
        if batch_idx == 0:
            print(f"[DEBUG] Images shape: {images.shape}")  # Expected: [B, 3, 128, 128]
            print(f"[DEBUG] Voxels shape: {voxels.shape}")  # Expected: [B, 32, 32, 32]

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)  # [B, 1, 32, 32, 32]

        # Voxels need a channel dimension for BCELoss
        loss = criterion(outputs, voxels.unsqueeze(1))

        # Backpropagation
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def train_model(train_loader):
    """
    Main training function that:
    - Initializes model, optimizer, loss function
    - Runs for specified epochs
    - Saves trained model
    - Returns trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}")

    # Initialize model
    model = Pix2VoxMini().to(device)

    # Binary Cross-Entropy Loss for voxel occupancy
    criterion = nn.BCELoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses = []

    # Training loop
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(avg_loss)
        print(f"[INFO] Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    # Plot loss curve after training
    plot_loss_curve(train_losses)

    # Save trained model
    torch.save(model.state_dict(), "pix2vox_mini.pth")
    print("[INFO] Model saved as pix2vox_mini.pth")

    return model
