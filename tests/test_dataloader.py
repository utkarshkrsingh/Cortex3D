# tests/test_dataloader.py
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
import pytest

# Import the dataset class to be tested
from src.dataset import ModelNetSingleViewDataset, collate_fn


# --- Suggestion 2: Create a fixture for a temporary, fake dataset ---
@pytest.fixture(scope="module")
def fake_dataset_path(tmpdir_factory):
    """
    Creates a temporary directory with a fake dataset structure for testing.
    This runs only once per test module.
    """
    # Create a temporary root directory
    path = tmpdir_factory.mktemp("data")

    # Create a structure mimicking your real data: root/class/instance
    instance_path = path.joinpath("chair", "chair_001")
    os.makedirs(instance_path)

    # Create a dummy PNG image (10x10 pixels)
    dummy_image = Image.new("RGB", (10, 10), color="red")
    dummy_image.save(instance_path / "view_01.png")
    dummy_image.save(instance_path / "view_02.png")

    # Create a dummy voxel .npy file (32x32x32)
    dummy_voxel = np.random.rand(32, 32, 32).astype(np.float32)
    np.save(instance_path / "model.npy", dummy_voxel)

    # The fixture returns the path to the root of this fake dataset
    return str(path)


# --- Suggestion 1 & 3: Use pytest structure and assertions ---
def test_dataloader_output_shapes(fake_dataset_path):
    """
    Tests that the DataLoader produces batches with the correct tensor shapes.
    """
    # Constants for the test
    BATCH_SIZE = 4
    IMG_SIZE = 128
    VOXEL_SIZE = 32

    # Standard transformations
    transform = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    )

    # Initialize dataset with the path to our fake data
    dataset = ModelNetSingleViewDataset(root_dir=fake_dataset_path, transform=transform)

    # Since our fake dataset only has one item, we need to handle batching
    # We'll create a list of this single item to simulate a larger dataset
    dataset.data_pairs = dataset.data_pairs * BATCH_SIZE

    # Use a dataloader
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Fetch one batch
    images, voxels = next(iter(loader))

    # --- Use Assertions instead of print statements ---
    # The test will FAIL automatically if any of these are false

    # 1. Assert that the batch is not empty
    assert images is not None and voxels is not None, (
        "Dataloader returned an empty batch."
    )

    # 2. Assert the shape of the images tensor
    expected_img_shape = (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    assert images.shape == expected_img_shape, (
        f"Image shape is {images.shape}, but expected {expected_img_shape}"
    )

    # 3. Assert the shape of the voxels tensor
    # Note: Assumes the dataloader returns voxels with a channel dim [B, 1, D, H, W]
    expected_voxel_shape = (BATCH_SIZE, 1, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
    assert voxels.shape == expected_voxel_shape, (
        f"Voxel shape is {voxels.shape}, but expected {expected_voxel_shape}"
    )

    # 4. Assert the data types
    assert images.dtype == torch.float32
    assert voxels.dtype == torch.float32

    print("\n[SUCCESS] Dataloader shape and type tests passed!")
