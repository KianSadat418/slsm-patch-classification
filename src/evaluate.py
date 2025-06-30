import argparse
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import CrossFoldDataset, mil_collate
from model_attention import AttentionMIL
from model_maxpool import MaxPoolMIL


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate MIL model")
    parser.add_argument("--annotations", type=Path, required=True, help="Path to annotations_new.csv")
    parser.add_argument("--fold-assignments", type=Path, required=True, help="Path to fold_assignments_new.json")
    parser.add_argument("--patch-dir", type=Path, required=True, help="Directory with extracted features")
    parser.add_argument("--fold", type=int, default=1, help="Fold id to evaluate (1-5)")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Split to evaluate")
    parser.add_argument("--model", choices=["attention", "maxpool"], default="attention")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained weights")
    parser.add_argument("--save-scores", type=Path, help="Optional path to save attention scores as JSON")
    parser.add_argument("--plot-roc", type=Path, help="Optional path to save ROC curve plot")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for evaluation")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)

    with open(args.fold_assignments, "r") as f:
        folds = json.load(f)
    fold_key = f"fold{args.fold}"
    bag_ids = folds[fold_key][args.split]

    dataset = CrossFoldDataset(
        args.annotations,
        args.patch_dir,
        bag_ids,
        transform=None,
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=mil_collate)

    if args.model == "attention":
        model = AttentionMIL(dropout=0.5)  # pretrained=False by default now
    else:
        model = MaxPoolMIL(dropout=0.5)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    patch_dict = {}
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for bags, labels, bag_ids in loader:
            for bag, label, bag_id in zip(bags, labels, bag_ids):
                bag = bag.unsqueeze(0).to(device)
                label = label.unsqueeze(0).to(device)
                logits, attention_scores = model(bag)

                probs = F.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1)
                correct += (pred_class == label).sum().item()
                total += label.size(0)

                patch_dict[bag_id] = attention_scores.squeeze(0).cpu().tolist()
                all_labels.extend(label.cpu().tolist())
                all_probs.extend(probs[:, 1].cpu().tolist())  # class 1 probability

    acc = correct / total if total else 0
    print(f"Accuracy: {acc * 100:.2f}%")

    from sklearn.metrics import roc_auc_score, RocCurveDisplay
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    print(f"AUC: {auc:.3f}")

    if args.plot_roc:
        RocCurveDisplay.from_predictions(all_labels, all_probs)
        plt.tight_layout()
        plt.savefig(args.plot_roc)
        print(f"Saved ROC curve to {args.plot_roc}")

    if args.save_scores:
        with open(args.save_scores, "w") as f:
            json.dump(patch_dict, f)
        print(f"Saved patch scores to {args.save_scores}")


if __name__ == "__main__":
    main()
