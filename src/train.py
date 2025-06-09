import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import MILDataset, mil_transform
from model_attention import AttentionMIL
from model_maxpool import MaxPoolMIL


def get_args():
    parser = argparse.ArgumentParser(description="Train MIL model")
    parser.add_argument("--bags", type=Path, required=True, help="Path to bag_to_patches.json")
    parser.add_argument("--labels", type=Path, required=True, help="Path to bag_labels.json")
    parser.add_argument("--folds", type=Path, required=True, help="Path to bag_folds.json")
    parser.add_argument("--fold", type=int, default=0, help="Fold id to use for training")
    parser.add_argument("--model", choices=["attention", "maxpool"], default="attention")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("model.pt"))
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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.model == "attention":
        model = AttentionMIL(pretrained=False)
    else:
        model = MaxPoolMIL(pretrained=False)
    model.train()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for bags, labels, _ in loader:
            outputs, *_ = model(bags)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
