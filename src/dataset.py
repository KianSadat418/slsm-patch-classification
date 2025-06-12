import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
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
        patch_paths = self.bag_to_patches[bag_id]
        label = self.bag_labels[bag_id]

        patch_tensors = []
        for patch_path in patch_paths:
            img = Image.open(patch_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            patch_tensors.append(img)

        bag_tensor = torch.stack(patch_tensors)

        return bag_tensor, torch.tensor(label, dtype=torch.float32), bag_id

mil_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
])


def mil_collate(batch):
    """Custom collate_fn to handle bags with different numbers of patches."""
    bags, labels, bag_ids = zip(*batch)
    labels = torch.stack(labels)
    return list(bags), labels, list(bag_ids)
