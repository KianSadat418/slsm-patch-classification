import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json


def get_args():
    parser = argparse.ArgumentParser(description="Basic data exploration for MIL patches")
    parser.add_argument("--labels-csv", type=Path, required=True, help="Path to labels.csv")
    parser.add_argument("--patch-dir", type=Path, required=True, help="Root directory containing patch images organized as study_id/biopsy_id/*.png")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis"), help="Directory to save figures and summary")
    return parser.parse_args()


def main():
    args = get_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels_csv)

    label_counts = df["label"].value_counts().sort_index()
    label_counts.plot(kind="bar")
    plt.xlabel("Label")
    plt.ylabel("Number of bags")
    plt.title("Bag label distribution")
    plt.tight_layout()
    label_plot = args.out_dir / "label_distribution.png"
    plt.savefig(label_plot)
    plt.close()

    patch_counts = []
    for _, row in df.iterrows():
        patch_folder = args.patch_dir / str(row["study_id"]) / str(row["Biopsy_image_id"]) 
        patch_counts.append(len(list(patch_folder.glob("*.png"))))
    df["patch_count"] = patch_counts

    df["patch_count"].plot(kind="hist", bins=20)
    plt.xlabel("Patches per bag")
    plt.ylabel("Number of bags")
    plt.title("Patch count distribution")
    plt.tight_layout()
    patch_plot = args.out_dir / "patch_count_distribution.png"
    plt.savefig(patch_plot)
    plt.close()

    summary = {
        "num_bags": int(len(df)),
        "mean_patches_per_bag": float(df["patch_count"].mean()),
        "std_patches_per_bag": float(df["patch_count"].std()),
        "label_counts": label_counts.to_dict(),
    }
    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary to {summary_path}")
    print("Key observations:")
    print(f"- Bags: {summary['num_bags']}")
    print(f"- Avg patches per bag: {summary['mean_patches_per_bag']:.2f} ± {summary['std_patches_per_bag']:.2f}")
    print(f"- Label distribution: {summary['label_counts']}")
    print("Check for class imbalance and extremely small or large bags before training.")


if __name__ == "__main__":
    main()
