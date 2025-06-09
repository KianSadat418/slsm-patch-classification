from pathlib import Path
import json
import tempfile

from PIL import Image
import torch

from src.dataset import MILDataset, mil_transform
from src.model_attention import AttentionMIL
from src.model_maxpool import MaxPoolMIL


def create_dummy_dataset(tmpdir: Path):
    patch_dir = tmpdir / "patches"
    patch_dir.mkdir()
    bag_to_patches = {"bag1": []}
    bag_labels = {"bag1": 0}
    bag_folds = {"bag1": 0}
    for i in range(3):
        img_path = patch_dir / f"img_{i}.png"
        Image.new("RGB", (224, 224)).save(img_path)
        bag_to_patches["bag1"].append(str(img_path))
    bags = tmpdir / "bags.json"
    labels = tmpdir / "labels.json"
    folds = tmpdir / "folds.json"
    bags.write_text(json.dumps(bag_to_patches))
    labels.write_text(json.dumps(bag_labels))
    folds.write_text(json.dumps(bag_folds))
    ds = MILDataset(bags, labels, folds, [0], transform=mil_transform)
    bag, label, _ = ds[0]
    return bag.unsqueeze(0), label.unsqueeze(0)


def test_model_forward_attention():
    with tempfile.TemporaryDirectory() as tmp:
        bag, label = create_dummy_dataset(Path(tmp))
        model = AttentionMIL(pretrained=False)
        out, attn = model(bag)
        assert out.shape == label.shape
        assert attn.shape[0] == 1


def test_model_forward_maxpool():
    with tempfile.TemporaryDirectory() as tmp:
        bag, label = create_dummy_dataset(Path(tmp))
        model = MaxPoolMIL(pretrained=False)
        out = model(bag)
        assert out.shape == label.shape
