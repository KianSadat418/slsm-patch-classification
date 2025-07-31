import torch
import torch.nn as nn
import torch.nn.functional as F
from topk.svm import SmoothTop1SVM

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

        # Feature embedding network
        self.feature_embed = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.25)
        )
        
        self.embedding_dim = 512
        
        # Attention network
        self.attention_module = Attn_Net_Gated(L=self.embedding_dim, D=256, dropout=True)
        
        # Bag classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )

        # CLAM specific parameters
        self.k_sample = 10  # Number of top/bottom instances to sample
        self.n_classes = 2
        
        # Instance classifier
        self.instance_classifier = nn.Linear(self.embedding_dim, 2)
        
        # Loss functions
        self.instance_loss_fn = SmoothTop1SVM(n_classes=self.n_classes)
        self.bag_loss_fn = nn.CrossEntropyLoss()

    def _instance_eval(self, x, label):
        """
        CLAM's instance-level evaluation
        Returns instance loss and attention scores
        """
        # Get attention scores
        A = self.attention_module(x)  # (N, 1)
        A = torch.transpose(A, 1, 0)  # (1, N)
        A = F.softmax(A, dim=1)
        
        # Get top-k and bottom-k instances
        A_flat = A.view(-1)
        num_instances = A_flat.size(0)
        k = min(self.k_sample, num_instances // 2)  # Ensure we don't take more than half
        
        # Get top-k and bottom-k indices
        topk_vals, topk_idxs = torch.topk(A_flat, k)
        botk_vals, botk_idxs = torch.topk(-A_flat, k)
        botk_idxs = botk_idxs[botk_vals < 0]  # Only take actual bottom-k
        
        # Get instance features
        topk_feats = x[topk_idxs]  # (k, 512)
        botk_feats = x[botk_idxs]  # (k, 512)
        
        # Create instance targets (1 for positive class, 0 for negative)
        inst_targets = torch.full((k,), label.item(), dtype=torch.long, device=x.device)
        
        # Get predictions from both instance classifiers
        topk_preds = self.instance_classifier(topk_feats)  # Positive evidence
        botk_preds = self.instance_classifier(botk_feats)  # Negative evidence
        
        # Calculate instance losses
        inst_loss = self.instance_loss_fn(topk_preds, inst_targets)  # Positive instances should be positive
        inst_loss += self.instance_loss_fn(botk_preds, 1 - inst_targets)  # Negative instances should be negative
        
        # Normalize by number of instance classifiers (2)
        inst_loss /= 2.0
        
        return inst_loss, A

    def forward(self, x, label=None, instance_eval=False):
        # x shape: (1, N, 2048) where N is number of instances
        x = x.squeeze(0)  # (N, 2048)
        
        # Project to lower dimension
        x = self.feature_embed(x)  # (N, 512)
        
        # Get attention scores and instance loss if needed
        inst_loss = torch.tensor(0.0, device=x.device)
        if instance_eval and label is not None:
            inst_loss, A = self._instance_eval(x, label)
        else:
            # Just get attention scores without computing instance loss
            A = self.attention_module(x)
            A = torch.transpose(A, 1, 0)  # (1, N)
            A = F.softmax(A, dim=1)
        
        # Compute bag-level representation
        M = torch.mm(A, x)  # (1, 512)
        
        # Get bag-level predictions
        logits = self.classifier(M)  # (1, 2)
        
        return logits, A, inst_loss