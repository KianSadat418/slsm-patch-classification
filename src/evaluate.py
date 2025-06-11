import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import MILDataset, mil_transform
from model_attention import AttentionMIL
from model_maxpool import MaxPoolMIL


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate MIL model")
    parser.add_argument("--bags", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1, help="Fold id to evaluate")
    parser.add_argument("--model", choices=["attention", "maxpool"], default="attention")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained weights")
    parser.add_argument("--save-scores", type=Path, help="Optional path to save patch-level scores as JSON")
    parser.add_argument("--auc", action="store_true", help="Compute ROC AUC in addition to accuracy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for evaluation")
    return parser.parse_args()


def main():
    args = get_args()

    device = torch.device(args.device)

    dataset = MILDataset(
        args.bags,
        args.labels,
        args.folds,
        [args.fold],
        transform=mil_transform,
    )
    loader = DataLoader(dataset, batch_size=1)

    if args.model == "attention":
        model = AttentionMIL(pretrained=False)
    else:
        model = MaxPoolMIL(pretrained=False)
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
            bags = bags.to(device)
            labels = labels.to(device)
            outputs, patch_scores = model(bags)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
            patch_dict[bag_ids[0]] = patch_scores.squeeze(0).cpu().tolist()
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(outputs.cpu().tolist())
    acc = correct / total if total else 0
    print(f"Accuracy: {acc*100:.2f}%")
    if args.auc:
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.0
        print(f"AUC: {auc:.3f}")

    if args.save_scores:
        import json
        with open(args.save_scores, "w") as f:
            json.dump(patch_dict, f)
        print(f"Saved patch scores to {args.save_scores}")


if __name__ == "__main__":
    main()
