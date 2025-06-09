# SLSM Patch Classification

This project explores Multiple Instance Learning (MIL) approaches for classifying slide-level samples from patch images.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate dataset JSON files by running the preprocessing script (edit paths inside `src/preprocessing.py` to point to your data):
   ```bash
   python src/preprocessing.py
   ```

## Training

Train a model using the generated JSON files:

```bash
python src/train.py --bags path/to/bag_to_patches.json \
                    --labels path/to/bag_labels.json \
                    --folds path/to/bag_folds.json \
                    --fold 0 --model attention --epochs 10
```

The trained weights are saved to `model.pt` by default.

## Evaluation

Evaluate a trained model on another fold:

```bash
python src/evaluate.py --bags path/to/bag_to_patches.json \
                      --labels path/to/bag_labels.json \
                      --folds path/to/bag_folds.json \
                      --fold 1 --model attention --weights model.pt
```

## Testing

Unit tests cover dataset loading and model forward passes. Run them with:

```bash
pytest
```
