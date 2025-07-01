import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=512, D=256, dropout=True):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(0.25) if dropout else nn.Identity()
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(0.25) if dropout else nn.Identity()
        )
        self.attention_c = nn.Linear(D, 1)

    def forward(self, x):  # x: (B, N, L)
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)  # (B, N, 1)
        return A

class AttentionMIL(nn.Module):
    def __init__(self, dropout=0.5):
        super(AttentionMIL, self).__init__()

        self.feature_embed = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout)
        )
        self.embedding_dim = 512
        self.attention_module = Attn_Net_Gated(L=self.embedding_dim, D=256, dropout=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )

        self.k_sample = 8
        self.n_classes = 2
        self.instance_loss_fn = nn.CrossEntropyLoss()
        self.instance_classifier = nn.Linear(self.embedding_dim, 2)

    def forward(self, x, label=None, instance_eval=False):
        x = x.squeeze(0)
        x = self.feature_embed(x)  # Project to 512
        A = self.attention_module(x)  # (N, 1)
        A = torch.transpose(A, 1, 0)  # (1, N)
        A = F.softmax(A, dim=1)

        M = torch.mm(A, x)  # (1, 512)
        logits = self.classifier(M)  # (1, 2)

        inst_loss = torch.tensor(0.0, device=x.device)
        if instance_eval and label is not None:
            A_flat = A.view(-1)
            num_instances = A_flat.size(0)
            k = min(self.k_sample, num_instances)  # clip k to available instances
            top_ids = torch.topk(A_flat, k=k).indices
            inst_feats = x[top_ids]  # (k, 512)
            inst_targets = torch.full((k,), label.item(), dtype=torch.long, device=x.device)
            inst_logits = self.instance_classifier(inst_feats)
            inst_loss = self.instance_loss_fn(inst_logits, inst_targets)

        return logits, A, inst_loss