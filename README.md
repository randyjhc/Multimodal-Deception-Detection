# Multimodal-Deception-Detection

## Training

### Basic single run

```bash
python run_training.py
```

Trains a BiLSTM on the train split with fixed hyperparameters (hidden=64, dropout=0.4, lr=1e-3). Saves the best checkpoint to `best_bilstm.pt` and generates `loss_curve.png` / `accuracy_curve.png`.

### Cross-validation + hyperparameter search (recommended)

```bash
python run_cv_training.py
```

Runs a 5-fold stratified CV grid search over 12 hyperparameter combinations (hidden, dropout, lr), trains a final model on all training samples using the best configuration, then evaluates once on the held-out test set. Saves the final checkpoint to `best_bilstm_final.pt`.

## Inference / Evaluation

```bash
# Evaluate the default checkpoint (best_bilstm.pt)
python test_model.py

# Evaluate a specific checkpoint
python test_model.py --ckpt best_bilstm_final.pt

# Specify data root and device
python test_model.py --ckpt best_bilstm_final.pt --root OpenFace_features --device cpu
```

Prints test loss, accuracy, precision, recall, F1, and a confusion matrix.