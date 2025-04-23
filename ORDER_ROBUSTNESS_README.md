# Order Robustness for Spotify Playlist LSTM

This extension adds functionality to evaluate and train LSTM models that are robust to changes in the order of songs in playlists.

## Overview

The order of songs in a playlist can be an important factor in recommendation systems. These scripts allow you to:

1. **Evaluate** how sensitive an existing LSTM model is to changes in song order
2. **Train** a new LSTM model that's robust to song order perturbations
3. **Compare** regular and order-robust LSTM models

## Types of Order Perturbations

The following types of order perturbations are supported:

- **Swap**: Swap adjacent pairs of songs
- **Shift**: Move a song from one position to another
- **Reverse**: Reverse a segment of the playlist
- **Shuffle**: Completely shuffle the playlist

## Scripts

### 1. Evaluating Order Sensitivity (`order_eval.py`)

This script evaluates how an existing LSTM model performs when the order of songs in playlists is perturbed.

```bash
python order_eval.py \
  --model /path/to/model.pt \
  --data /path/to/playlist/data \
  --embeddings /path/to/embeddings.h5 \
  --songs /path/to/song_popularity.csv \
  --perturbation-type swap \
  --num-perturbations 2 \
  --output results/swap_2_eval.json
```

Options:
- `--perturbation-type`: Type of perturbation (`swap`, `shift`, `reverse`, `shuffle`)
- `--num-perturbations`: Number of perturbations to apply
- `--min-songs`: Minimum songs per playlist (default: 3)
- `--max-files`: Maximum number of data files to process (default: 20)
- `--batch-size`: Batch size for evaluation (default: 32)
- `--device`: Device to run on (`cuda` or `cpu`, default: auto-detect)

### 2. Training an Order-Robust LSTM (`order_train.py`)

This script trains an LSTM model that's robust to song order perturbations by applying random perturbations during training.

```bash
python order_train.py \
  --data /path/to/playlist/data \
  --embeddings /path/to/embeddings.h5 \
  --output models/order_robust \
  --perturbation-type swap \
  --perturbation-prob 0.5 \
  --max-perturbations 2 \
  --epochs 10
```

Options:
- `--perturbation-type`: Type of perturbation (`swap`, `shift`, `reverse`, `shuffle`)
- `--perturbation-prob`: Probability of applying perturbation to a playlist (default: 0.5)
- `--max-perturbations`: Maximum number of perturbations to apply (default: 2)
- `--hidden-dim`: Hidden dimension of LSTM (default: 1536)
- `--num-layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout rate (default: 0.3)
- `--batch-size`: Training batch size (default: 64)
- `--epochs`: Number of training epochs (default: 10)
- `--learning-rate`: Learning rate (default: 0.001)
- `--weight-decay`: Weight decay for regularization (default: 1e-5)
- `--max-files`: Maximum number of data files to use (default: all)
- `--min-songs`: Minimum songs per playlist (default: 4)
- `--top-songs`: Path to top songs list (optional)
- `--device`: Device to train on (`cuda` or `cpu`, default: auto-detect)
- `--checkpoint-interval`: Save model every N epochs (default: 1)

### 3. Comparing Models (`compare_order_robustness.py`)

This script compares a regular LSTM model with an order-robust LSTM model across different perturbation types and intensities.

```bash
python compare_order_robustness.py \
  --regular-model /path/to/regular_model.pt \
  --robust-model /path/to/robust_model.pt \
  --data /path/to/playlist/data \
  --embeddings /path/to/embeddings.h5 \
  --songs /path/to/song_popularity.csv \
  --output results/comparison
```

Options:
- `--perturbation-types`: Types of perturbations to test (default: all)
- `--max-perturbations`: Maximum number of perturbations to test (default: 3)
- `--batch-size`: Batch size (default: 32)
- `--max-files`: Maximum number of data files to process (default: 20)
- `--min-songs`: Minimum songs per playlist (default: 3)
- `--device`: Device to run on (default: auto-detect)

## Example Workflow

1. First, evaluate how sensitive your existing LSTM model is to order changes:

```bash
python order_eval.py \
  --model models/original_lstm.pt \
  --data data/playlists \
  --embeddings data/song_lookup.h5 \
  --songs data/song_popularity.csv \
  --perturbation-type swap \
  --num-perturbations 2 \
  --output results/original_swap_2.json
```

2. Train an order-robust LSTM model:

```bash
python order_train.py \
  --data data/playlists \
  --embeddings data/song_lookup.h5 \
  --output models/order_robust \
  --perturbation-type swap \
  --perturbation-prob 0.5 \
  --max-perturbations 2 \
  --epochs 10
```

3. Compare the original and order-robust models:

```bash
python compare_order_robustness.py \
  --regular-model models/original_lstm.pt \
  --robust-model models/order_robust/lstm_order_robust_final.pt \
  --data data/playlists \
  --embeddings data/song_lookup.h5 \
  --songs data/song_popularity.csv \
  --output results/comparison
```

4. Check the generated plots in `results/comparison/plots` to see how the models compare.

## Notes

- The order-robust model should perform better when playlist order is perturbed, but might have slightly lower performance on unperturbed playlists.
- Training with multiple perturbation types (by running training multiple times with different perturbation types) can create a more generally robust model.
- The comparison script generates plots that make it easy to visualize the differences in performance between the models.