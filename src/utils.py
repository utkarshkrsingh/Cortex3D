# src/utils.py
import matplotlib.pyplot as plt
import torch


def show_image(image_tensor):
    """
    Display a single image tensor (C, H, W)
    """
    image = image_tensor.permute(1, 2, 0)  # Convert to HWC
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")
    plt.show()


def show_voxel_slice(voxel_tensor, slice_index=16):
    """
    Display a single slice of a 3D voxel grid
    """
    if isinstance(voxel_tensor, torch.Tensor):
        voxel_tensor = voxel_tensor.numpy()

    plt.imshow(voxel_tensor[slice_index], cmap="gray")
    plt.title(f"Voxel Slice {slice_index}")
    plt.axis("off")
    plt.show()


def plot_loss_curve(train_losses):
    """
    Plot the training loss curve
    """
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()
