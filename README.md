# SLSM Patch Classification

This project explores Multiple Instance Learning (MIL) approaches for classifying slide-level samples from patch images.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Extract patch-level features using `src/feature_extractor.py` and store them under `<patch_dir>/features/*.pt`.
3. The training and evaluation scripts consume `annotations_new.csv` and `fold_assignments_new.json` directly, so no preprocessing step is required.

## Training

Train a model using the provided annotation and fold assignment files. The
training script trains on the fold specified by `--fold`. Use `--crossval` to
run all five folds sequentially.
Loss curves can optionally be saved with `--plot-loss`, AUC curves with
`--plot-auc`, and ROC curves with `--plot-roc`. ROC plots include a dashed
diagonal reference line and display the AUC value on the plot. When running
five-fold cross validation with `--crossval` and `--plot-roc`, an additional
plot will show all fold ROC curves along with the mean curve and AUC.

```bash
python src/train.py (associated arguments)
```

The trained weights are saved to `model.pt` by default.

## Evaluation

Evaluate a trained model on the validation or test split of a fold. Use
`--save-scores` to store patch-level attention scores:

```bash
python src/evaluate.py (associated arguments)
```

## Testing

Unit tests cover dataset loading and model forward passes. Run them with:

```bash
pytest
```
