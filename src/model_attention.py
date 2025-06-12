import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    def __init__(self, pretrained=True, dropout=0.5):
        super(AttentionMIL, self).__init__()

        backbone = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.embedding_dim = 512
        self.attention_layer = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(self.embedding_dim, 128),
            nn.Tanh(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, 2),
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        features = self.feature_extractor(x)
        features = features.view(B, N, self.embedding_dim)

        A = self.attention_layer(features)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1)

        M = torch.bmm(A, features).squeeze(1)

        logits = self.classifier(M)
        return logits, A.squeeze(1)
