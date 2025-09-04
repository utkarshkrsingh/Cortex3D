# tests/test_dataloader.py

from src.dataset import ModelNetSingleViewDataset
from torchvision import transforms
from torch.utils.data import DataLoader


def test_dataloader():
    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    dataset = ModelNetSingleViewDataset(root_dir="data/train", transform=transform)
    if len(dataset) == 0:
        print("[ERROR] No samples found. Please check your dataset path and structure.")
        return
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Fetch one batch
    images, voxels = next(iter(loader))

    print("[INFO] Images batch shape:", images.shape)
    print("[INFO] Voxels batch shape:", voxels.shape)


if __name__ == "__main__":
    test_dataloader()
