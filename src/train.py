import argparse
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import CrossFoldDataset, mil_collate
from model_attention import AttentionMIL
from model_attention import FocalLoss
from model_maxpool import MaxPoolMIL


def get_args():
    parser = argparse.ArgumentParser(description="Train MIL model")
    parser.add_argument("--annotations", type=Path, required=True, help="Path to annotations_new.csv")
    parser.add_argument("--fold-assignments", type=Path, required=True, help="Path to fold_assignments_new.json")
    parser.add_argument("--patch-dir", type=Path, required=True, help="Directory with extracted feature tensors")
    parser.add_argument("--fold", type=int, default=1, help="Fold id to use (1-5)")
    parser.add_argument("--crossval", action="store_true", help="Run 5-fold cross validation")
    parser.add_argument("--model", choices=["attention", "maxpool"], default="attention")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--plot-loss", type=Path, help="Optional path to save loss curve plot")
    parser.add_argument("--plot-auc", type=Path, help="Optional path to save AUC curve plot")
    parser.add_argument("--plot-acc", type=Path, help="Optional path to save accuracy curve plot")
    parser.add_argument("--plot-roc", type=Path, help="Optional path to save ROC curve plot")
    parser.add_argument("--output", type=Path, default=Path("model.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)

    with open(args.fold_assignments, "r") as f:
        fold_assignments = json.load(f)

    def run_training(fold_key: str):
        train_ids = fold_assignments[fold_key]["train"]
        val_ids = fold_assignments[fold_key]["val"]

        train_ds = CrossFoldDataset(args.annotations, args.patch_dir, train_ids)
        val_ds = CrossFoldDataset(args.annotations, args.patch_dir, val_ids)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=mil_collate,
        )
        val_loader = DataLoader(val_ds, batch_size=1, collate_fn=mil_collate)

        if args.model == "attention":
            model = AttentionMIL(dropout=args.dropout)
        else:
            model = MaxPoolMIL(dropout=args.dropout)

        model.to(device)
        model.train()

        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        from sklearn.metrics import roc_auc_score, roc_curve
        import matplotlib.pyplot as plt

        print(f"Training {fold_key} on {device} for {args.epochs} epochs")

        train_losses, val_losses = [], []
        train_aucs, val_aucs = [], []
        train_accs, val_accs = [], []

        for epoch in range(args.epochs):
            model.train()
            epoch_losses = []
            preds = []
            labels_list = []
            train_correct = 0
            train_total = 0
            for bags, labels, _ in train_loader:
                for bag, label in zip(bags, labels):
                    bag = bag.unsqueeze(0).to(device)
                    label = label.unsqueeze(0).long().to(device)
                    logits, _, inst_loss = model(bag, label=label, instance_eval=True)
                    bag_loss = criterion(logits, label)
                    loss = bag_loss + 0.5 * inst_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())
                    probs = F.softmax(logits, dim=1)[:, 1]
                    preds.extend(probs.detach().cpu().tolist())
                    labels_list.extend(label.cpu().tolist())
                    pred_class = torch.argmax(logits, dim=1)
                    train_correct += (pred_class == label).sum().item()
                    train_total += label.size(0)
            train_loss = sum(epoch_losses) / len(epoch_losses)
            try:
                train_auc = roc_auc_score(labels_list, preds)
            except Exception:
                train_auc = 0.0
            train_acc = train_correct / train_total if train_total else 0.0

            model.eval()
            val_epoch_losses = []
            val_preds = []
            val_labels = []
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for bags, labels, _ in val_loader:
                    for bag, label in zip(bags, labels):
                        bag = bag.unsqueeze(0).to(device)
                        label = label.unsqueeze(0).long().to(device)
                        logits, _, inst_loss = model(bag, label=label, instance_eval=True)
                        bag_loss = criterion(logits, label)
                        loss = bag_loss + 0.5 * inst_loss
                        val_epoch_losses.append(loss.item())
                        val_probs = F.softmax(logits, dim=1)[:, 1]
                        val_preds.extend(val_probs.cpu().tolist())
                        val_labels.extend(label.cpu().tolist())
                        val_pred_class = torch.argmax(logits, dim=1)
                        val_correct += (val_pred_class == label).sum().item()
                        val_total += label.size(0)
            val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
            try:
                val_auc = roc_auc_score(val_labels, val_preds)
            except Exception:
                val_auc = 0.0
            val_acc = val_correct / val_total if val_total else 0.0

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_aucs.append(train_auc)
            val_aucs.append(val_auc)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_auc={train_auc:.3f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_auc={val_auc:.3f} val_acc={val_acc:.3f}"
            )

        if args.plot_loss:
            plt.figure()
            plt.plot(train_losses, label="train")
            plt.plot(val_losses, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            loss_path = args.plot_loss.with_name(args.plot_loss.stem + f"_{fold_key}" + args.plot_loss.suffix)
            plt.savefig(loss_path)
            print(f"Saved loss plot to {loss_path}")

        if args.plot_auc:
            plt.figure()
            plt.plot(train_aucs, label="train")
            plt.plot(val_aucs, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("AUC")
            plt.legend()
            plt.tight_layout()
            auc_path = args.plot_auc.with_name(args.plot_auc.stem + f"_{fold_key}" + args.plot_auc.suffix)
            plt.savefig(auc_path)
            print(f"Saved AUC plot to {auc_path}")

        if args.plot_acc:
            plt.figure()
            plt.plot(train_accs, label="train")
            plt.plot(val_accs, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            acc_path = args.plot_acc.with_name(args.plot_acc.stem + f"_{fold_key}" + args.plot_acc.suffix)
            plt.savefig(acc_path)
            print(f"Saved accuracy plot to {acc_path}")

        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(val_labels, val_preds)

        if args.plot_roc:
            plt.figure()
            plt.plot(fpr, tpr, label="ROC")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve {fold_key}")
            plt.text(0.95, 0.05, f"AUC={val_auc:.3f}", ha="right", va="bottom", transform=plt.gca().transAxes)
            plt.tight_layout()
            roc_path = args.plot_roc.with_name(args.plot_roc.stem + f"_{fold_key}" + args.plot_roc.suffix)
            plt.savefig(roc_path)
            plt.close()
            print(f"Saved ROC curve to {roc_path}")

        model_path = args.output.with_name(args.output.stem + f"_{fold_key}" + args.output.suffix)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")
        return train_accs[-1], val_accs[-1], fpr, tpr, val_auc

    if args.crossval:
        roc_data = []
        for key in sorted(fold_assignments.keys()):
            _, _, fpr, tpr, auc_val = run_training(key)
            roc_data.append((key, fpr, tpr, auc_val))

        if args.plot_roc and roc_data:
            import numpy as np
            import matplotlib.pyplot as plt

            plt.figure()

            mean_fpr = np.linspace(0.0, 1.0, 100)
            tprs = []
            aucs = []
            for fold_key, fpr, tpr, auc_val in roc_data:
                plt.plot(fpr, tpr, label=f"{fold_key} (AUC={auc_val:.3f})")
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(auc_val)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            plt.plot(mean_fpr, mean_tpr, color="black", linestyle="--", label=f"Mean (AUC={mean_auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.text(0.95, 0.05, f"Mean AUC={mean_auc:.3f}", ha="right", va="bottom", transform=plt.gca().transAxes)
            plt.tight_layout()
            all_path = args.plot_roc.with_name(args.plot_roc.stem + "_all" + args.plot_roc.suffix)
            plt.savefig(all_path)
            plt.close()
            print(f"Saved cross-validation ROC curve to {all_path}")
    else:
        run_training(f"fold{args.fold}")


if __name__ == "__main__":
    main()
