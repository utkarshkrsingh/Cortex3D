# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """
    Encoder module using a pre-trained ResNet-18 to extract features from an image.
    """

    def __init__(self, freeze_params=True):
        super(Encoder, self).__init__()
        # Load pre-trained ResNet-18
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove the final fully connected layer to get feature maps
        layers = list(base_model.children())[:-1]
        self.features = nn.Sequential(*layers)

        # Freeze the parameters of the pre-trained layers if specified
        if freeze_params:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Input x shape: [Batch, Channels, Height, Width] e.g., [B, 3, 128, 128]
        x = self.features(x)  # Output shape: [B, 512, 1, 1]

        # Flatten the features to a vector
        return x.view(x.size(0), -1)  # Output shape: [B, 512]


class Decoder(nn.Module):
    """
    Decoder module that takes a feature vector and upsamples it to a 3D voxel grid.
    """

    def __init__(self, feature_dim=512, voxel_size=32):
        super(Decoder, self).__init__()

        # Ensure voxel_size is divisible by 8 for the 3 upsampling steps (4 -> 8 -> 16 -> 32)
        if voxel_size != 32:
            raise ValueError("This decoder is hardcoded for 32x32x32 voxel output.")

        # Fully connected layer to project the feature vector into a starting 3D volume
        self.fc = nn.Linear(feature_dim, 64 * 4 * 4 * 4)

        # 3D Transposed Convolutional layers for upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),  # Added for stability
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),  # Added for stability
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Sigmoid to output occupancy probabilities (0 to 1)
        )

    def forward(self, x):
        # Input x shape: [B, 512]
        x = self.fc(x)

        # Reshape to a 4x4x4 volume with 64 channels
        x = x.view(-1, 64, 4, 4, 4)  # Shape: [B, 64, 4, 4, 4]

        # Upsample to the final voxel grid
        x = self.decoder(x)  # Output shape: [B, 1, 32, 32, 32]
        return x


class SingleViewReconstructionNet(nn.Module):
    """
    A simple Encoder-Decoder network for single-view 3D reconstruction.
    This serves as the generator for "coarse 3D volumes" in the main project.
    """

    def __init__(self):
        super(SingleViewReconstructionNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        features = self.encoder(x)
        voxels = self.decoder(features)
        return voxels


# --- Placeholders for Your Project's Core Modules ---


class ContextAwareFusion(nn.Module):
    """
    Placeholder for the fusion module.
    This module will take multiple coarse voxel grids and fuse them.
    """

    def __init__(self):
        super(ContextAwareFusion, self).__init__()
        # TODO: Implement the fusion logic.
        # This could be a simple max-pooling, an average, or a more complex
        # 3D convolutional network that learns to weigh different views.
        pass

    def forward(self, coarse_volumes):
        # Input coarse_volumes shape: [Num_Views, Batch, 1, D, H, W]
        # For simplicity, you might process it as [Batch, Num_Views, D, H, W]

        # Example: Simple max-pooling fusion
        fused_volume, _ = torch.max(coarse_volumes, dim=0)
        return fused_volume


class Refiner(nn.Module):
    """
    Placeholder for the 3D Refiner CNN.
    This module takes the fused volume and improves its quality.
    """

    def __init__(self):
        super(Refiner, self).__init__()
        # TODO: Implement the refiner network.
        # This will likely be a series of 3D convolutional layers (e.g., a 3D U-Net)
        # that corrects defects and smoothes surfaces.
        self.refiner_net = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        pass

    def forward(self, fused_volume):
        # Input fused_volume shape: [Batch, 1, D, H, W]
        refined_volume = self.refiner_net(fused_volume)
        return refined_volume
