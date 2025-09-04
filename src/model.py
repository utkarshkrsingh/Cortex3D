# src/model.py
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        base_model = models.resnet18(pretrained=True)
        layers = list(base_model.children())[:-1]  # Remove last FC layer
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)  # [B, 512, 1, 1]
        return x.view(x.size(0), -1)  # Flatten to [B, 512]


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),  # Output between 0 and 1
        )

        # Map encoder features (512) -> 3D volume input (64x4x4x4)
        self.fc = nn.Linear(512, 64 * 4 * 4 * 4)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 4, 4, 4)
        return self.decoder(x)


class Pix2VoxMini(nn.Module):
    def __init__(self):
        super(Pix2VoxMini, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        features = self.encoder(x)
        voxels = self.decoder(features)
        return voxels
