# SLSM Patch Classification

This project explores Multiple Instance Learning (MIL) approaches for classifying slide-level samples from patch images.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate dataset JSON files by running the preprocessing script:
   ```bash
   python src/preprocessing.py --patch-dir /path/to/patches \
                              --labels-csv /path/to/labels.csv \
                              --out-dir dataset_json
   ```

## Training

Train a model using the generated JSON files. The training script automatically
uses all folds except the one specified by `--fold` for validation.
Loss curves can optionally be saved with `--plot-loss` and AUC curves with
`--plot-auc`.

```bash
python src/train.py --bags path/to/bag_to_patches.json \
                    --labels path/to/bag_labels.json \
                    --folds path/to/bag_folds.json \
                    --fold 0 --model attention --epochs 10 \
                    --plot-loss loss.png --plot-auc auc.png --device cuda
```

The trained weights are saved to `model.pt` by default.

## Evaluation

Evaluate a trained model on another fold. Use `--auc` to report ROC AUC in addition
to accuracy and `--save-scores` to store patch-level predictions:

```bash
python src/evaluate.py --bags path/to/bag_to_patches.json \
                      --labels path/to/bag_labels.json \
                      --folds path/to/bag_folds.json \
                      --fold 1 --model attention --weights model.pt \
                      --save-scores patch_scores.json --auc --device cuda
```

## Testing

Unit tests cover dataset loading and model forward passes. Run them with:

```bash
pytest
```
