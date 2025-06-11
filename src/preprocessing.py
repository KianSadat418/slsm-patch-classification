import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json


def get_args():
    parser = argparse.ArgumentParser(description="Generate MIL dataset json files")
    parser.add_argument("--patch-dir", type=Path, required=True,
                        help="Directory containing patch images organized as study/biopsy")
    parser.add_argument("--labels-csv", type=Path, required=True,
                        help="CSV with columns study_id,Biopsy_image_id,label,fold")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory to write json files")
    return parser.parse_args()


def main():
    args = get_args()
    df = pd.read_csv(args.labels_csv)

    bag_to_patches = defaultdict(list)
    bag_labels = {}
    bag_folds = {}

    for _, row in df.iterrows():
        study = row["study_id"]
        biopsy_img_id = row["Biopsy_image_id"]
        label = row["label"]
        fold = row["fold"]

        patch_folder = args.patch_dir / str(study) / str(biopsy_img_id)
        patches = sorted(patch_folder.glob("*.png"))

        if patches:
            bag_to_patches[biopsy_img_id] = [str(p) for p in patches]
            bag_labels[biopsy_img_id] = int(label)
            bag_folds[biopsy_img_id] = int(fold)
        else:
            print(f"Warning: no patches found in {patch_folder}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "bag_to_patches.json").write_text(json.dumps(bag_to_patches))
    (args.out_dir / "bag_labels.json").write_text(json.dumps(bag_labels))
    (args.out_dir / "bag_folds.json").write_text(json.dumps(bag_folds))
    print(f"Wrote dataset files to {args.out_dir}")


if __name__ == "__main__":
    main()
