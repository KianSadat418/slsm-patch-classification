import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
from PIL import Image
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Basic data exploration for MIL patches")
    parser.add_argument("--labels-csv", type=Path, required=True, help="Path to labels.csv")
    parser.add_argument("--patch-dir", type=Path, required=True, help="Root directory containing patch images organized as study_id/biopsy_id/*.png")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis"), help="Directory to save figures and summary")
    return parser.parse_args()


def calc_stats(img_path: Path):
    """Return mean intensity and Shannon entropy for an image."""
    img = Image.open(img_path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean_val = float(arr.mean())
    hist, _ = np.histogram(arr, bins=256, range=(0, 1), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum())
    return mean_val, entropy


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
    mean_intensities = []
    entropies = []
    bag_detail = []
    for _, row in df.iterrows():
        patch_folder = args.patch_dir / str(row["study_id"]) / str(row["Biopsy_image_id"])
        patches = list(patch_folder.glob("*.png"))
        patch_counts.append(len(patches))
        patch_ints = []
        patch_ents = []
        for p in patches:
            m, e = calc_stats(p)
            patch_ints.append(m)
            patch_ents.append(e)
            mean_intensities.append(m)
            entropies.append(e)
        if patch_ents:
            bag_detail.append({
                "biopsy_id": row["Biopsy_image_id"],
                "mean_entropy": float(np.mean(patch_ents)),
                "low_detail_frac": float(np.mean(np.array(patch_ents) < 2.0)),
            })
        else:
            bag_detail.append({
                "biopsy_id": row["Biopsy_image_id"],
                "mean_entropy": 0.0,
                "low_detail_frac": 1.0,
            })
    df["patch_count"] = patch_counts

    df["patch_count"].plot(kind="hist", bins=20)
    plt.xlabel("Patches per bag")
    plt.ylabel("Number of bags")
    plt.title("Patch count distribution")
    plt.tight_layout()
    patch_plot = args.out_dir / "patch_count_distribution.png"
    plt.savefig(patch_plot)
    plt.close()

    # Patch-level intensity distribution
    plt.figure()
    plt.hist(mean_intensities, bins=20, color="tab:blue")
    plt.xlabel("Mean intensity")
    plt.ylabel("Number of patches")
    plt.title("Patch intensity distribution")
    plt.tight_layout()
    intensity_plot = args.out_dir / "patch_intensity_distribution.png"
    plt.savefig(intensity_plot)
    plt.close()

    # Patch-level entropy distribution
    plt.figure()
    plt.hist(entropies, bins=20, color="tab:green")
    plt.xlabel("Shannon entropy")
    plt.ylabel("Number of patches")
    plt.title("Patch entropy distribution")
    plt.tight_layout()
    entropy_plot = args.out_dir / "patch_entropy_distribution.png"
    plt.savefig(entropy_plot)
    plt.close()

    # Bag level detail
    detail_df = pd.DataFrame(bag_detail)
    top_detail = detail_df.sort_values("mean_entropy", ascending=False).head(10)
    low_detail = detail_df.sort_values("mean_entropy", ascending=True).head(10)
    plt.figure(figsize=(10,4))
    plt.bar(top_detail["biopsy_id"].astype(str), top_detail["mean_entropy"], color="tab:orange")
    plt.xticks(rotation=90)
    plt.ylabel("Mean entropy")
    plt.title("Top 10 biopsies with highest detail")
    plt.tight_layout()
    high_detail_plot = args.out_dir / "high_detail_biopsies.png"
    plt.savefig(high_detail_plot)
    plt.close()

    plt.figure(figsize=(10,4))
    plt.bar(low_detail["biopsy_id"].astype(str), low_detail["mean_entropy"], color="tab:red")
    plt.xticks(rotation=90)
    plt.ylabel("Mean entropy")
    plt.title("Bottom 10 biopsies with lowest detail")
    plt.tight_layout()
    low_detail_plot = args.out_dir / "low_detail_biopsies.png"
    plt.savefig(low_detail_plot)
    plt.close()

    summary = {
        "num_bags": int(len(df)),
        "mean_patches_per_bag": float(df["patch_count"].mean()),
        "std_patches_per_bag": float(df["patch_count"].std()),
        "label_counts": label_counts.to_dict(),
        "mean_patch_intensity": float(np.mean(mean_intensities)) if mean_intensities else 0.0,
        "mean_patch_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }
    summary_path = args.out_dir / "summary.json"
    detail_path = args.out_dir / "bag_detail.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(detail_path, "w") as f:
        json.dump(bag_detail, f, indent=2)

    print(f"Wrote summary to {summary_path}")
    print(f"Wrote bag details to {detail_path}")
    print("Key observations:")
    print(f"- Bags: {summary['num_bags']}")
    print(f"- Avg patches per bag: {summary['mean_patches_per_bag']:.2f} ± {summary['std_patches_per_bag']:.2f}")
    print(f"- Mean patch intensity: {summary['mean_patch_intensity']:.3f}")
    print(f"- Mean patch entropy: {summary['mean_patch_entropy']:.3f}")
    print("Check for class imbalance and extremely small or large bags before training.")


if __name__ == "__main__":
    main()
