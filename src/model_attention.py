import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=512, D=128, dropout=True):
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

        self.embedding_dim = 2048
        self.attention_module = Attn_Net_Gated(L=self.embedding_dim, D=128, dropout=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, 2),
        )

        self.k_sample = 8
        self.n_classes = 2
        self.instance_loss_fn = nn.CrossEntropyLoss()
        self.instance_classifiers = nn.ModuleList([
            nn.Linear(self.embedding_dim, 2) for _ in range(self.n_classes)
        ])
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()
    
    def inst_eval(self, A, h, classifier):
        A = A.view(-1)  # shape: (N,)
        top_p_ids = torch.topk(A, self.k_sample)[1]
        top_n_ids = torch.topk(-A, self.k_sample)[1]

        top_p = h[top_p_ids]
        top_n = h[top_n_ids]

        all_instances = torch.cat([top_p, top_n], dim=0)
        all_targets = torch.cat([
            self.create_positive_targets(self.k_sample, h.device),
            self.create_negative_targets(self.k_sample, h.device)
        ])

        logits = classifier(all_instances)
        instance_loss = self.instance_loss_fn(logits, all_targets)

        return instance_loss

    def forward(self, x, label=None, instance_eval=False):
        features = x

        A = self.attention_module(features)  # (B, N, 1)
        A = torch.transpose(A, 1, 2)         # (B, 1, N)
        A = F.softmax(A, dim=-1)             # attention weights

        M = torch.bmm(A, features).squeeze(1)  # (B, 512)

        logits = self.classifier(M)            # (B, 2)

        if instance_eval and label is not None:
            classifier = self.instance_classifiers[label.item()]
            inst_loss = self.inst_eval(A[0], features[0], classifier)
            return logits, A.squeeze(1), inst_loss

        return logits, A.squeeze(1)