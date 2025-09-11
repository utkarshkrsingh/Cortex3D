# tests/test_visualization.py
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Import all necessary components from your src folder
from src.dataset import ModelNetSingleViewDataset, collate_fn
from src.utils import show_image, show_voxel_grid  # Updated to use the 3D visualizer


def test_visualization():
    """
    Loads one sample from the dataset and visualizes the input image
    and the corresponding 3D voxel grid.
    """
    print("[INFO] Setting up dataset and dataloader for visualization...")

    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    # Use a try-except block to handle cases where the dataset path is wrong
    try:
        dataset = ModelNetSingleViewDataset(root_dir="data/train", transform=transform)
        # Suggestion 2: Use the robust collate_fn
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        print(
            "Please ensure the 'data/train' directory exists and is structured correctly."
        )
        return

    # Suggestion 2: Add a check for an empty loader
    if len(loader) == 0:
        print("[ERROR] Dataloader is empty. No data found.")
        return

    # Get one image and its corresponding voxel grid
    print("[INFO] Fetching one sample...")
    images, voxels = next(iter(loader))

    # Check if the fetched batch is valid
    if images.nelement() == 0:
        print(
            "[WARN] Fetched an empty batch, possibly due to a data loading error. Trying again..."
        )
        images, voxels = next(iter(loader))  # Try one more time
        if images.nelement() == 0:
            print("[ERROR] Could not fetch a valid sample from the dataloader.")
            return

    print("[INFO] Displaying sample...")

    # Show the 2D input image (first item in the batch)
    show_image(images[0])

    # Suggestion 1: Show the full 3D voxel grid (first item in the batch)
    show_voxel_grid(voxels[0])


if __name__ == "__main__":
    test_visualization()
