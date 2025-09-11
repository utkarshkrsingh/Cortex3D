# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm  # A library to create smart progress bars

# --- Make sure you have a function to calculate IoU in utils.py ---
from .utils import plot_loss_curve, calculate_iou
from .config import *  # Import all config variables

# ==========================================================================================
# Epoch-Level Functions
# ==========================================================================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs one epoch of training."""
    model.train()
    epoch_loss = 0.0

    # Use tqdm for a progress bar
    for i, (images, voxels) in enumerate(tqdm(loader, desc="Training")):
        images, voxels = images.to(device), voxels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Ensure target voxels have the same shape as outputs [B, 1, D, H, W]
        # NOTE: If your dataloader already returns [B, 1, D, H, W], remove .unsqueeze(1)
        if voxels.dim() == 4:
            voxels = voxels.unsqueeze(1)

        # Calculate loss
        loss = criterion(outputs, voxels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # --- Suggestion 3: Implement DEBUG mode ---
        if DEBUG and i >= DEBUG_BATCHES - 1:
            print("[DEBUG] Stopping epoch early due to DEBUG mode.")
            break

    return epoch_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    """Runs one epoch of validation."""
    model.eval()
    epoch_loss = 0.0
    all_iou_scores = []

    with torch.no_grad():  # Disable gradient calculation
        for i, (images, voxels) in enumerate(tqdm(loader, desc="Validation")):
            images, voxels = images.to(device), voxels.to(device)

            # Forward pass
            outputs = model(images)

            # Ensure target voxels have the same shape as outputs
            if voxels.dim() == 4:
                voxels = voxels.unsqueeze(1)

            # Calculate loss
            loss = criterion(outputs, voxels)
            epoch_loss += loss.item()

            # --- Suggestion 1: Calculate IoU metric ---
            iou = calculate_iou(outputs, voxels)
            all_iou_scores.append(iou)

            if DEBUG and i >= DEBUG_BATCHES - 1:
                print("[DEBUG] Stopping validation early due to DEBUG mode.")
                break

    avg_loss = epoch_loss / len(loader)
    avg_iou = torch.mean(torch.tensor(all_iou_scores))
    return avg_loss, avg_iou


# ==========================================================================================
# Main Training Orchestrator
# ==========================================================================================


# --- Suggestion 2: Refactor for better flexibility ---
def train_model(model, train_loader, val_loader, optimizer, criterion, device):
    """Main training function."""

    train_losses, val_losses, val_ious = [], [], []
    best_val_iou = 0.0

    print("[INFO] Starting training...")
    for epoch in range(EPOCHS):
        # Training step
        avg_train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(avg_train_loss)

        # Validation step
        avg_val_loss, avg_val_iou = validate_one_epoch(
            model, val_loader, criterion, device
        )
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)

        print(
            f"[INFO] Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val IoU: {avg_val_iou:.4f}"
        )

        # Save the model if it has the best validation IoU so far
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            save_path = os.path.join(SAVE_PATH, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Model saved to {save_path} (IoU: {best_val_iou:.4f})")

    # Plotting after training is complete
    plot_loss_curve(train_losses, val_losses)
    print(f"[INFO] Training complete. Best validation IoU: {best_val_iou:.4f}")

    return model
