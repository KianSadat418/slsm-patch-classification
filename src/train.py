import argparse
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import MILDataset, mil_transform, mil_collate
from model_attention import AttentionMIL
from model_maxpool import MaxPoolMIL


def get_args():
    parser = argparse.ArgumentParser(description="Train MIL model")
    parser.add_argument("--bags", type=Path, required=True, help="Path to bag_to_patches.json")
    parser.add_argument("--labels", type=Path, required=True, help="Path to bag_labels.json")
    parser.add_argument("--folds", type=Path, required=True, help="Path to bag_folds.json")
    parser.add_argument("--fold", type=int, default=0, help="Fold id to use for validation")
    parser.add_argument("--model", choices=["attention", "maxpool"], default="attention")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--plot-loss", type=Path, help="Optional path to save loss curve plot")
    parser.add_argument("--plot-auc", type=Path, help="Optional path to save AUC curve plot")
    parser.add_argument("--output", type=Path, default=Path("model.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)

    # Determine training folds (all except the specified fold)
    with open(args.folds, "r") as f:
        fold_map = json.load(f)
    all_folds = set(fold_map.values())
    train_folds = [f for f in all_folds if f != args.fold]

    train_ds = MILDataset(
        args.bags,
        args.labels,
        args.folds,
        train_folds,
        transform=mil_transform,
    )
    val_ds = MILDataset(
        args.bags,
        args.labels,
        args.folds,
        [args.fold],
        transform=mil_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=mil_collate,
    )
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=mil_collate)

    if args.model == "attention":
        model = AttentionMIL(pretrained=True, dropout=args.dropout)
        # Freeze first convolutional layer for the first few epochs
        modules = list(model.feature_extractor.children())
        for m in modules[:2]:
            for param in m.parameters():
                param.requires_grad = False
    else:
        model = MaxPoolMIL(pretrained=True, dropout=args.dropout)

    model.to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt

    print(f"Training on {device} for {args.epochs} epochs")
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []

    for epoch in range(args.epochs):
        if args.model == "attention":
            if epoch == 2:
                for p in modules[4].parameters():
                    p.requires_grad = True
            if epoch == 4:
                for p in modules[5].parameters():
                    p.requires_grad = True
        model.train()
        epoch_losses = []
        preds = []
        labels_list = []
        for bags, labels, _ in train_loader:
            for bag, label in zip(bags, labels):
                N = bag.size(0)
                k = int(0.9 * N)
                idx = torch.randperm(N)[:k]
                bag_sub = bag[idx]
                bag = bag_sub.unsqueeze(0).to(device)
                label = label.unsqueeze(0).long().to(device)
                logits, _ = model(bag)
                loss = criterion(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                probs = F.softmax(logits, dim=1)[:, 1]
                preds.extend(probs.detach().cpu().tolist())
                labels_list.extend(label.cpu().tolist())
        train_loss = sum(epoch_losses) / len(epoch_losses)
        try:
            train_auc = roc_auc_score(labels_list, preds)
        except Exception:
            train_auc = 0.0

        model.eval()
        val_epoch_losses = []
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for bags, labels, _ in val_loader:
                for bag, label in zip(bags, labels):
                    bag = bag.unsqueeze(0).to(device)
                    label = label.unsqueeze(0).long().to(device)
                    logits, _ = model(bag)
                    loss = criterion(logits, label)
                    val_epoch_losses.append(loss.item())
                    val_probs = F.softmax(logits, dim=1)[:, 1]
                    val_preds.extend(val_probs.cpu().tolist())
                    val_labels.extend(label.cpu().tolist())
        val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except Exception:
            val_auc = 0.0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_auc={train_auc:.3f} "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.3f}"
        )

    if args.plot_loss:
        plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot_loss)
        print(f"Saved loss plot to {args.plot_loss}")

    if args.plot_auc:
        plt.figure()
        plt.plot(train_aucs, label="train")
        plt.plot(val_aucs, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot_auc)
        print(f"Saved AUC plot to {args.plot_auc}")

    torch.save(model.state_dict(), args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
