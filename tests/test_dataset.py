from pathlib import Path
import json
import tempfile

from PIL import Image
import torch

from src.dataset import MILDataset, mil_transform


def create_dummy_data(tmpdir: Path):
    patch_dir = tmpdir / "patches"
    patch_dir.mkdir()
    bag_to_patches = {"bag1": []}
    bag_labels = {"bag1": 1}
    bag_folds = {"bag1": 0}
    for i in range(2):
        img_path = patch_dir / f"img_{i}.png"
        Image.new("RGB", (224, 224)).save(img_path)
        bag_to_patches["bag1"].append(str(img_path))
    bags = tmpdir / "bags.json"
    labels = tmpdir / "labels.json"
    folds = tmpdir / "folds.json"
    bags.write_text(json.dumps(bag_to_patches))
    labels.write_text(json.dumps(bag_labels))
    folds.write_text(json.dumps(bag_folds))
    return bags, labels, folds


def test_dataset_load():
    with tempfile.TemporaryDirectory() as tmp:
        bags, labels, folds = create_dummy_data(Path(tmp))
        ds = MILDataset(bags, labels, folds, [0], transform=mil_transform)
        assert len(ds) == 1
        bag, label, bag_id = ds[0]
        assert bag.shape[0] == 2  # two patches
        assert label.item() == 1
        assert bag_id == "bag1"
