import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

PATCH_DIR = Path("Kian Code/lsm_mil_project/data/patch_images")
df = pd.read_csv("Kian Code/lsm_mil_project/data/labels.csv")

print(df.head())

bag_to_patches = defaultdict(list)
bag_labels = {}
bag_folds = {}

for _, row in df.iterrows():
    study = row["study_id"]
    biopsy_img_id = row["Biopsy_image_id"]
    label = row["label"]
    fold = row["fold"]

    patch_folder = PATCH_DIR / study / biopsy_img_id

    patches = list(patch_folder.glob("*.png"))

    if patches:
        bag_to_patches[biopsy_img_id] = [str(p) for p in sorted(patches)]
        bag_labels[biopsy_img_id] = label
        bag_folds[biopsy_img_id] = fold
    else:
        print(f"Error: no patches found in {patch_folder}")

print(f"Total bags: {len(bag_to_patches)}")
example_bag = next(iter(bag_to_patches))
print("Example bag:", example_bag)
print("Patch paths:", bag_to_patches[example_bag])
print("Label:", bag_labels[example_bag])

with open("Kian Code/lsm_mil_project/data/bag_to_patches.json", "w") as f:
    json.dump(bag_to_patches, f)
with open("Kian Code/lsm_mil_project/data/bag_labels.json", "w") as f:
    json.dump(bag_labels, f)
with open("Kian Code/lsm_mil_project/data/bag_folds.json", "w") as f:
    json.dump(bag_folds, f)

