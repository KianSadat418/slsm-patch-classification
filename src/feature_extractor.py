import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description="Extract CNN features from patch folders")
    parser.add_argument("--patch-dir", type=Path, required=True,
                        help="Directory with patch images in study_id/biopsy_id/*.png structure")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory to save .pt feature files")
    return parser.parse_args()

def extract_features_from_folder(patch_folder, model, transform):
    patch_paths = sorted(list(patch_folder.glob("*.png")))
    if not patch_paths:
        return None

    feats = []
    for p in patch_paths:
        img = Image.open(p).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor).squeeze(0).cpu()  # [2048]
        feats.append(feat)

    return torch.stack(feats)  # [N_patches, 2048]

def main():
    args = get_args()

    # Load ResNet50 and remove classification head
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.to(device).eval()


    # Transform images to match ResNet50 input requirements
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for study_dir in tqdm(sorted(args.patch_dir.iterdir()), desc="Studies"):
        if not study_dir.is_dir():
            continue
        for biopsy_dir in sorted(study_dir.iterdir()):
            if not biopsy_dir.is_dir():
                continue
            biopsy_id = biopsy_dir.name
            save_path = args.out_dir / f"{biopsy_id}.pt"
            if save_path.exists():
                continue
            features = extract_features_from_folder(biopsy_dir, model, transform)
            if features is not None:
                torch.save(features, save_path)
            else:
                print(f"[Warning] No patches found in {biopsy_dir}")

    print(f"Feature extraction completed. Saved to {args.out_dir}")

if __name__ == "__main__":
    main()