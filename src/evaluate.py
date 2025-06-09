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
    return parser.parse_args()


def main():
    args = get_args()

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
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for bags, labels, _ in loader:
            outputs, *_ = model(bags)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    acc = correct / total if total else 0
    print(f"Accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
