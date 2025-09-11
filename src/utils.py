# src/utils.py
import matplotlib.pyplot as plt
import torch
import numpy as np

# ==========================================================================================
# Visualization Functions
# ==========================================================================================


def show_image(image_tensor):
    """
    Displays a single image tensor (C, H, W).
    """
    if image_tensor.is_cpu:
        image = image_tensor.permute(1, 2, 0).numpy()  # Convert to HWC
    else:
        image = image_tensor.cpu().permute(1, 2, 0).numpy()

    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")
    plt.show()


def show_voxel_grid(voxel_tensor, threshold=0.5):
    """
    Displays a 3D voxel grid as a 3D scatter plot.
    """
    if voxel_tensor.dim() > 3:  # Handle batches or channel dims
        voxel_tensor = voxel_tensor.squeeze()

    # Ensure tensor is on CPU and converted to numpy
    if isinstance(voxel_tensor, torch.Tensor):
        voxel_tensor = voxel_tensor.cpu().numpy()

    # Apply threshold to get a binary grid
    binary_grid = voxel_tensor > threshold

    # Get coordinates of occupied voxels
    x, y, z = np.where(binary_grid)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="blue", marker="s")  # 's' for square markers

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Voxel Grid")
    plt.show()


def plot_loss_curve(train_losses, val_losses):
    """
    Plots the training and validation loss curves.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==========================================================================================
# Metric Calculation Functions
# ==========================================================================================


def calculate_iou(outputs, targets, threshold=0.5):
    """
    Calculates the Intersection over Union (IoU) metric for a batch.
    """
    # Apply threshold to model outputs to get binary predictions
    preds = (outputs > threshold).float()

    # Ensure targets are also binary (0 or 1)
    targets = (targets > 0).float()

    # Intersection is where both preds and targets are 1
    # The multiplication operation (preds * targets) results in 1 only where both are 1
    intersection = (preds * targets).sum(
        dim=(1, 2, 3, 4)
    )  # Sum over all dimensions except batch

    # Union is where either preds or targets (or both) are 1
    # Sum of both minus the intersection (to avoid double-counting)
    union = preds.sum(dim=(1, 2, 3, 4)) + targets.sum(dim=(1, 2, 3, 4)) - intersection

    # IoU is intersection / union. Add a small epsilon to avoid division by zero.
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Return the mean IoU for the batch
    return iou.mean().item()
