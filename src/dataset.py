import torch
from torch.utils.data import Dataset
import json

class MILDataset(Dataset):
    def __init__(self, bag_to_patches_path, bag_labels_path, bag_folds_path, folds_to_include, transform=None):
        with open(bag_to_patches_path, "r") as f:
            all_bags = json.load(f)
        with open(bag_labels_path, "r") as f:
            all_labels = json.load(f)
        with open(bag_folds_path, "r") as f:
            all_folds = json.load(f)

        self.bag_ids = [k for k in all_bags.keys() if all_folds[k] in folds_to_include]
        self.bag_to_patches = all_bags
        self.bag_labels = all_labels
        self.transform = transform

    def __len__(self):
        return len(self.bag_ids)

    def __getitem__(self, idx):
        bag_id = self.bag_ids[idx]
        feat_path = self.bag_to_patches[bag_id]
        label = self.bag_labels[bag_id]

        bag_tensor = torch.load(feat_path)
        return bag_tensor, torch.tensor(label, dtype=torch.long), bag_id

def mil_collate(batch):
    """Custom collate_fn to handle bags with different numbers of patches."""
    bags, labels, bag_ids = zip(*batch)
    labels = torch.stack(labels)
    return list(bags), labels, list(bag_ids)
