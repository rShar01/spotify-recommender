# Order Perturbation Evaluation and Training

This directory contains scripts for evaluating and training LSTM models that are robust to changes in the order of songs in playlists.

## Overview

The order perturbation scripts allow you to:

1. Evaluate how existing LSTM models perform when the order of songs in playlists is changed
2. Train new LSTM models that are robust to these order perturbations
3. Compare the performance of regular LSTM models versus order-robust LSTM models

## Types of Order Perturbations

The following types of order perturbations are supported:

- **Swap**: Randomly swap pairs of songs in the playlist
- **Shift**: Shift the entire playlist by a random amount (circular shift)
- **Reverse**: Reverse the order of songs in the playlist
- **Shuffle**: Completely shuffle the playlist

## Scripts

### 1. Order Evaluation (`order_eval.py`)

Evaluates an LSTM model's robustness to order perturbations.

```bash
python order_eval.py --model MODEL_PATH --playlists PLAYLISTS_PATH --song-data SONG_DATA_PATH [OPTIONS]
```

Options:
- `--perturbation-type`: Type of perturbation to apply (swap, shift, reverse, shuffle)
- `--num-swaps`: Number of swaps to perform (for 'swap' type)
- `--batch-size`: Batch size for evaluation
- `--output`: Path to save evaluation results
- `--device`: Device to use for evaluation (cuda/cpu)

### 2. Order-Robust Training (`order_train.py`)

Trains an LSTM model to be robust to order perturbations.

```bash
python order_train.py --train-playlists TRAIN_PLAYLISTS_PATH --val-playlists VAL_PLAYLISTS_PATH --song-data SONG_DATA_PATH --output-dir OUTPUT_DIR [OPTIONS]
```

Options:
- `--perturbation-prob`: Probability of applying perturbation
- `--perturbation-types`: Types of perturbation to use
- `--max-swaps`: Maximum number of swaps to perform
- `--batch-size`: Batch size for training
- `--num-epochs`: Number of training epochs
- `--learning-rate`: Learning rate for optimizer
- `--hidden-dim`: Hidden dimension of LSTM
- `--num-layers`: Number of LSTM layers
- `--dropout`: Dropout probability
- `--device`: Device to use for training (cuda/cpu)
- `--save-every`: Save model every N epochs

### 3. Comprehensive Evaluation (`run_order_eval.py`)

Runs a comprehensive evaluation with different perturbation types and intensities.

```bash
python run_order_eval.py --model MODEL_PATH --playlists PLAYLISTS_PATH --song-data SONG_DATA_PATH --output-dir OUTPUT_DIR [OPTIONS]
```

Options:
- `--perturbation-types`: Types of perturbation to evaluate
- `--max-swaps`: Maximum number of swaps to evaluate
- `--batch-size`: Batch size for evaluation
- `--device`: Device to use for evaluation (cuda/cpu)

### 4. Model Comparison (`compare_models.py`)

Compares the performance of regular LSTM models versus order-robust LSTM models.

```bash
python compare_models.py --results-dir RESULTS_DIR --model-names MODEL1 MODEL2 [OPTIONS]
```

Options:
- `--k-values`: k values to include in plots
- `--output-dir`: Directory to save plots

### 5. Playlist Loader (`playlist_loader.py`)

Utility for loading and processing playlist data from multiple JSON files.

```bash
python playlist_loader.py --data-dir DATA_DIR --output-dir OUTPUT_DIR [OPTIONS]
```

Options:
- `--num-files`: Number of files to load (None = all)
- `--min-tracks`: Minimum number of tracks per playlist
- `--max-tracks`: Maximum number of tracks per playlist
- `--min-unique-tracks`: Minimum number of unique tracks per playlist
- `--train-ratio`: Ratio of training data
- `--val-ratio`: Ratio of validation data
- `--test-ratio`: Ratio of test data
- `--random-seed`: Random seed for reproducibility

### 6. Complete Workflow (`run_order_perturbation_workflow.py`)

Runs the entire order perturbation workflow, from data preparation to model training and evaluation.

```bash
python run_order_perturbation_workflow.py --playlist-data-dir PLAYLIST_DATA_DIR --song-data SONG_DATA_PATH --output-dir OUTPUT_DIR [OPTIONS]
```

Options:
- `--num-playlist-files`: Number of playlist files to use
- `--skip-regular`: Skip training regular LSTM model
- `--skip-robust`: Skip training order-robust LSTM model
- `--skip-eval`: Skip model evaluation
- `--skip-compare`: Skip model comparison
- `--num-epochs`: Number of epochs for training
- `--perturbation-prob`: Probability of applying perturbation during training
- `--device`: Device to use for training/evaluation

## Example Usage

### Evaluate a model with different order perturbations

```bash
# Evaluate with 1 swap
python order_eval.py --model models/lstm_model.pt --playlists data/test_playlists.json --song-data data/song_lookup.h5 --perturbation-type swap --num-swaps 1 --output swap1_results.json

# Evaluate with reversed order
python order_eval.py --model models/lstm_model.pt --playlists data/test_playlists.json --song-data data/song_lookup.h5 --perturbation-type reverse --output reverse_results.json
```

### Train an order-robust model

```bash
python order_train.py --train-playlists data/train_playlists.json --val-playlists data/val_playlists.json --song-data data/song_lookup.h5 --output-dir models/order_robust --perturbation-prob 0.7 --num-epochs 15
```

### Run comprehensive evaluation

```bash
python run_order_eval.py --model models/lstm_model.pt --playlists data/test_playlists.json --song-data data/song_lookup.h5 --output-dir eval_results
```

### Compare models

```bash
python compare_models.py --results-dir eval_results --model-names regular_lstm order_robust_lstm --output-dir comparison_plots
```

### Process playlist data from multiple files

```bash
python playlist_loader.py --data-dir /data/user_data/rshar/downloads/spotify/data/ --output-dir data/ --num-files 10 --min-tracks 5 --max-tracks 50
```

### Run the complete workflow

```bash
# Run the complete workflow with default settings
python run_order_perturbation_workflow.py --playlist-data-dir /data/user_data/rshar/downloads/spotify/data/ --song-data data/song_lookup.h5 --output-dir order_perturbation_results/ --num-playlist-files 10

# Skip training and just evaluate existing models
python run_order_perturbation_workflow.py --playlist-data-dir /data/user_data/rshar/downloads/spotify/data/ --song-data data/song_lookup.h5 --output-dir order_perturbation_results/ --skip-regular --skip-robust
```

## Implementation Details

### Order Perturbation Dataset

The `OrderPerturbationDataset` class applies order perturbations to playlists during training or evaluation. It supports:

- Applying perturbations with a specified probability
- Multiple types of perturbations
- Controlling the intensity of perturbations (e.g., number of swaps)

### Order-Robust Training

The training process:

1. Loads playlists and song embeddings
2. Creates datasets with order perturbation capabilities
3. Trains the LSTM model with perturbed playlists
4. Evaluates on validation data
5. Saves checkpoints and the final model

### Evaluation Metrics

The evaluation scripts compute:

- Top-k accuracy for various k values
- Performance across different perturbation types
- Performance with varying perturbation intensities (for swaps)

## Requirements

- PyTorch
- NumPy
- Matplotlib (for comparison plots)
- h5py (for song embeddings)
- tqdm (for progress bars)