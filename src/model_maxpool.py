import torch
import torch.nn as nn
import torchvision.models as models

class MaxPoolMIL(nn.Module):
    def __init__(self, pretrained=True):
        super(MaxPoolMIL, self).__init__()

        backbone = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # Remove FC layer

        self.embedding_dim = 512
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        features = self.feature_extractor(x)
        features = features.view(B, N, self.embedding_dim)

        pooled, _ = torch.max(features, dim=1)

        output = self.classifier(pooled)
        return output.squeeze(1)
