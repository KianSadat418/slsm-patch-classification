import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    def __init__(self, pretrained=True, dropout=0.5, patch_dropout=0.0):
        super(AttentionMIL, self).__init__()

        backbone = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.embedding_dim = 512
        self.attention_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, 1),
        )

        self.patch_dropout = patch_dropout

    def forward(self, x, return_features=False):
        B, N, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        features = self.feature_extractor(x)
        features = features.view(B, N, self.embedding_dim)

        if self.training and self.patch_dropout > 0:
            mask = torch.rand(B, N, 1, device=features.device) > self.patch_dropout
            features = features * mask

        A = self.attention_layer(features)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1)

        M = torch.bmm(A, features).squeeze(1)

        output = self.classifier(M)
        if return_features:
            return output.squeeze(1), A.squeeze(1), features
        return output.squeeze(1), A.squeeze(1)
