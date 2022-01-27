import torch
import torch.nn as nn


class BYOL(nn.Module):
    """Implementation of the BYOL architecture.
    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection mlp).
    """

    def __init__(self,
                 backbone: nn.Module,
                 num_ftrs: int = 2048):
        super(BYOL, self).__init__()
        last_conv_channels = list(backbone.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self,
                x0: torch.Tensor):
        f0 = self.backbone(x0)
        return f0