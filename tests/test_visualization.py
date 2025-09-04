# tests/test_visualization.py
from src.dataset import ModelNetSingleViewDataset
from src.utils import show_image, show_voxel_slice
from torchvision import transforms
from torch.utils.data import DataLoader


def test_visualization():
    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    dataset = ModelNetSingleViewDataset(root_dir="data/train", transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Get one image and voxel
    images, voxels = next(iter(loader))

    # Show first image
    show_image(images[0])

    # Show voxel grid middle slice
    show_voxel_slice(voxels[0], slice_index=16)


if __name__ == "__main__":
    test_visualization()
