import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class MultiTaskUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet"):
        super().__init__()

        self.backbone = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=32, activation=None)

        self.sky_head = nn.Conv2d(32, 2, kernel_size=1)
        self.depth_head = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        features = self.backbone(x)
        sky_logits = self.sky_head(features)
        depth = self.depth_head(features)
        return {"sky_logits": sky_logits, "depth": depth}


def build_model():
    return MultiTaskUNet()