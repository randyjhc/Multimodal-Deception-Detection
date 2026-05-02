# Multimodal Deception Detection

Deception detection on the UR-LYING Dataset (private).

Three modalities are supported individually or in combination:
- **Visual**: facial action units and head pose from [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)
- **Audio**: prosodic features (formants) from [OpenSMILE](https://github.com/audeering/opensmile)
- **Text**: token-level embeddings from Whisper transcriptions via RoBERTa

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install core dependencies and activate virtual environment
uv sync
source .venv/bin/activate
```

## Dataset

Download the extracted features [(link)](https://drive.google.com/file/d/12T4O18JYbdWt7LM3Sx23fxsht6vn_8_w/view?usp=drive_link) and organize it as follows:

```
dataset/UR_LYING_Deception_Dataset/
├── openface_raw/
│   ├── Train/
│   │   ├── Truthful/   *.csv
│   │   └── Deceptive/  *.csv
│   └── Test/
│       ├── Truthful/   *.csv
│       └── Deceptive/  *.csv
├── opensmile_raw/       # same layout, *.csv
└── whisper_raw/         # same layout, *.txt transcriptions
```

## Training

### Multimodal: Audio + Visual + Text (Late-Fusion BiGRU)

```bash
python run_training_avt.py --config configs/config_avt  # A+V+T
python run_training_avt.py --config configs/config_av   # A+V
python run_training_avt.py --config configs/config_at   # A+T
python run_training_avt.py --config configs/config_vt   # V+T
```

Each config is a JSON file specifying data roots, hyperparameters, and which modalities to activate (set a root to `null` to disable that modality). See [`configs/config_avt`](configs/config_avt) for a documented example.

### Visual Only — BiLSTM Baseline

```bash
# Single run with hardcoded hyperparameters
python run_training.py

# 5-fold CV hyperparameter search, then final training + test evaluation
python run_cv_training.py
```

## Inference / Evaluation

```bash
# Evaluate default checkpoint
uv run python test_model.py

# Evaluate a specific checkpoint
uv run python test_model.py --ckpt best_bigru_avt.pt

# Override data roots and device
uv run python test_model.py --ckpt best_bigru_avt.pt \
    --root dataset/UR_LYING_Deception_Dataset/openface_raw \
    --opensmile_root dataset/UR_LYING_Deception_Dataset/opensmile_raw \
    --whisper_root dataset/UR_LYING_Deception_Dataset/whisper_raw \
    --device cpu
```

The script auto-detects model type from the checkpoint and prints test loss, accuracy, precision, recall, F1, and a confusion matrix.

## Analysis Scripts

```bash
# Action Unit statistics and plots from OpenFace CSVs
uv run --group dataset python open_face/au_stats.py --visualize
```

## Project Structure

```
Multimodal-Deception-Detection/
├── dataset/
│   ├── openface_dataset.py     # Visual features
│   ├── opensmile_dataset.py    # Audio features
│   ├── whisper_dataset.py      # Text embeddings via RoBERTa
│   └── multimodal_dataset.py   # Combines modalities
├── model/
│   ├── BiLSTM.py               # Bidirectional LSTM classifier
│   ├── BiGRU.py                # Bidirectional GRU classifier
│   ├── LateFusionBiGRU.py      # Late-fusion multimodal architecture
│   ├── train.py                # Single-modality training loop
│   └── late_fusion_train.py    # Multimodal training loop
├── open_face/
│   └── au_stats.py             # Action Unit statistics and visualization
├── configs/                    # JSON configs for multimodal training
├── run_training.py             # Visual-only BiLSTM single run
├── run_cv_training.py          # Visual-only BiLSTM with 5-fold CV search
├── run_training_avt.py         # Multimodal training
└── test_model.py               # Evaluation for a saved checkpoint
```
