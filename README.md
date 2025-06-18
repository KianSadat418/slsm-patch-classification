# SLSM Patch Classification

This project explores Multiple Instance Learning (MIL) approaches for classifying slide-level samples from patch images.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate dataset JSON files by running the preprocessing script:
   ```bash
   python src/preprocessing.py (associated arguments)
   ```

## Training

Train a model using the generated JSON files. The training script automatically
uses all folds except the one specified by `--fold` for validation.
Loss curves can optionally be saved with `--plot-loss` and AUC curves with
`--plot-auc`.

```bash
python src/train.py (associated arguments)
```

The trained weights are saved to `model.pt` by default.

## Evaluation

Evaluate a trained model on another fold. Use `--auc` to report ROC AUC in addition
to accuracy and `--save-scores` to store patch-level predictions:

```bash
python src/evaluate.py (associated arguments)
```

## Testing

Unit tests cover dataset loading and model forward passes. Run them with:

```bash
pytest
```
