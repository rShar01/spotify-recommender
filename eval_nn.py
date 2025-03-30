import torch
import torch.nn as nn
import numpy as np
import h5py
from tqdm import tqdm
import json
import os
import gc
from pathlib import Path
import random
from datetime import datetime
import time
import pandas as pd
from typing import List, Dict, Tuple, Optional
import sys

# Import the model class from your original code
class PlaylistRecommenderModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dims: List[int] = [512, 256]):
        """Neural network for playlist similarity prediction"""
        super().__init__()
        
        layers = []
        prev_dim = embedding_dim * 2  # Concatenated embeddings
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass with two playlist embeddings"""
        x = torch.cat([x1, x2], dim=1)
        return self.model(x)


def load_model(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load a trained recommender model"""
    print(f"Loading model from {model_path}")
    model_data = torch.load(model_path, map_location=device)
    
    # Create model
    model = PlaylistRecommenderModel(
        embedding_dim=model_data['embedding_dim'],
        hidden_dims=model_data.get('hidden_dims', [512, 256])
    )
    
    # Load weights
    model.load_state_dict(model_data['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, model_data


def explore_h5_file(file_path: str):
    """Explore the structure of the H5 file and print results"""
    print(f"Exploring H5 file: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        print("Top-level groups:", list(f.keys()))
        
        for group_name in f.keys():
            print(f"\nKeys in {group_name} group:", list(f[group_name].keys()))
            
            # Check a few keys to understand the structure
            for key in list(f[group_name].keys())[:3]:  # Look at first 3 keys
                item = f[group_name][key]
                if isinstance(item, h5py.Group):
                    print(f"  {key} is a group with keys: {list(item.keys())}")
                else:
                    print(f"  {key} is a dataset with shape: {item.shape}")


def load_embeddings_flexible(embedding_store_path: str, split: str = 'train', max_batches: int = None):
    """
    Load embeddings from H5 file with flexible structure detection
    This function will attempt to determine the structure automatically
    """
    print(f"Loading {split} embeddings from {embedding_store_path}")
    
    all_embeddings = []
    all_pids = []
    
    with h5py.File(embedding_store_path, 'r') as f:
        # Check if the specified split exists
        if split not in f:
            # List available splits
            available_splits = list(f.keys())
            print(f"Error: Split '{split}' not found in embedding store.")
            print(f"Available splits: {available_splits}")
            
            # Fall back to first available split if any
            if available_splits:
                split = available_splits[0]
                print(f"Falling back to split '{split}'")
            else:
                raise ValueError(f"No usable splits found in {embedding_store_path}")
        
        group = f[split]
        loaded_batches = 0
        
        # Check the structure - case 1: batch_X datasets directly in the split group
        batch_datasets = [k for k in group.keys() if k.startswith('batch_') and not k.endswith('_pids')]
        
        if batch_datasets:
            print(f"Found direct batch datasets in '{split}' group")
            
            # Sort to ensure consistent order
            batch_datasets = sorted(batch_datasets)
            
            # Limit if requested
            if max_batches:
                batch_datasets = batch_datasets[:max_batches]
            
            # Load each batch
            for batch_name in tqdm(batch_datasets, desc=f"Loading {split} batches"):
                pid_batch_name = f"{batch_name}_pids"
                
                if pid_batch_name in group:
                    embeddings = group[batch_name][:]
                    pids = group[pid_batch_name][:]
                    
                    all_embeddings.append(embeddings)
                    all_pids.append(pids)
                    loaded_batches += 1
                else:
                    print(f"Warning: Missing {pid_batch_name} for {batch_name}")
        
        # Case 2: slice_X groups containing batch datasets
        slice_groups = [k for k in group.keys() if k.startswith('slice_')]
        
        if slice_groups and loaded_batches == 0:
            print(f"Found slice groups in '{split}' group")
            
            # Sort to ensure consistent order
            slice_groups = sorted(slice_groups)
            
            total_batches_needed = max_batches or float('inf')
            
            for slice_name in tqdm(slice_groups, desc=f"Loading {split} slices"):
                slice_group = group[slice_name]
                
                # Find batch datasets in this slice
                batch_datasets = [k for k in slice_group.keys() if k.startswith('batch_') and not k.endswith('_pids')]
                batch_datasets = sorted(batch_datasets)
                
                for batch_name in batch_datasets:
                    if loaded_batches >= total_batches_needed:
                        break
                        
                    pid_batch_name = f"{batch_name}_pids"
                    
                    if pid_batch_name in slice_group:
                        embeddings = slice_group[batch_name][:]
                        pids = slice_group[pid_batch_name][:]
                        
                        all_embeddings.append(embeddings)
                        all_pids.append(pids)
                        loaded_batches += 1
                    else:
                        print(f"Warning: Missing {pid_batch_name} for {batch_name} in {slice_name}")
                
                if loaded_batches >= total_batches_needed:
                    break
        
        if loaded_batches == 0:
            print(f"No embeddings found in '{split}' group. Structure:")
            for key in group.keys():
                print(f"  {key}")
            raise ValueError(f"No usable embeddings found for split '{split}'")
        
        print(f"Loaded {loaded_batches} batches from split '{split}'")
    
    # Concatenate all loaded embeddings and PIDs
    if all_embeddings:
        return np.concatenate(all_embeddings), np.concatenate(all_pids)
    else:
        raise ValueError(f"No embeddings loaded from {embedding_store_path}")


def load_top_songs(count_song_path: str, top_n: int = 3000) -> List[str]:
    """Load the top N songs from the count_song.csv file"""
    print(f"Loading top {top_n} songs from {count_song_path}")
    df = pd.read_csv(count_song_path)
    
    # Sort by count in descending order and take top N
    top_songs = df.sort_values(by='count', ascending=False).head(top_n)
    
    # Get the track URIs
    track_uris = top_songs['track_uri'].tolist()
    
    print(f"Loaded {len(track_uris)} top songs")
    return track_uris


def create_playlist_song_matrix(data_path: str, pids: np.ndarray, top_songs: List[str]) -> np.ndarray:
    """Create one-hot encoded matrix mapping playlists to songs"""
    print(f"Creating playlist-song matrix for {len(pids)} playlists")
    
    # Convert to set for faster lookup
    pid_set = set(map(int, pids))
    track_to_idx = {track: idx for idx, track in enumerate(top_songs)}
    
    # Initialize matrix
    matrix = np.zeros((len(pids), len(top_songs)))
    pid_to_idx = {int(pid): idx for idx, pid in enumerate(pids)}
    
    # Find all slice files
    slice_files = list(Path(data_path).glob("mpd.slice.*.json"))
    
    # Track progress
    total_playlists_found = 0
    
    for slice_file in tqdm(slice_files, desc="Loading playlist data"):
        with open(slice_file, 'r') as f:
            data = json.load(f)
        
        for playlist in data.get('playlists', []):
            pid = playlist.get('pid')
            
            if pid in pid_set:
                row_idx = pid_to_idx[pid]
                total_playlists_found += 1
                
                # Add tracks to matrix
                for track in playlist.get('tracks', []):
                    track_uri = track.get('track_uri')
                    if track_uri in track_to_idx:
                        matrix[row_idx, track_to_idx[track_uri]] = 1
    
    print(f"Found {total_playlists_found}/{len(pids)} playlists in data")
    
    # Check how many playlists have at least one song
    non_empty_playlists = np.sum(np.sum(matrix, axis=1) > 0)
    print(f"Playlists with at least one song: {non_empty_playlists}/{len(pids)}")
    
    return matrix

def evaluate_model_with_advanced_metrics(
    model,
    test_embeddings,
    train_embeddings,
    test_pids,
    train_pids,
    test_playlist_matrix,
    train_playlist_matrix,
    top_songs,
    max_rows=500,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Enhanced evaluation with better metrics for sparse playlist data"""
    print(f"Running evaluation on up to {max_rows} test playlists")
    
    # Find test playlists with at least one song
    test_song_counts = np.sum(test_playlist_matrix, axis=1)
    valid_indices = np.where(test_song_counts > 0)[0]
    
    if len(valid_indices) == 0:
        print("No test playlists with songs found. Cannot evaluate.")
        return {"error": "No test playlists with songs found."}
    
    # Limit to max_rows for speed
    if len(valid_indices) > max_rows:
        valid_indices = np.random.choice(valid_indices, max_rows, replace=False)
    
    print(f"Evaluating on {len(valid_indices)} test playlists")
    
    # Track metrics
    all_true_labels = []
    all_predictions = []
    
    # Convert embeddings to tensor format
    test_tensors = torch.FloatTensor(test_embeddings[valid_indices]).to(device)
    train_tensors = torch.FloatTensor(train_embeddings).to(device)
    
    # Process in batches for efficiency
    batch_size = 32
    
    for batch_start in tqdm(range(0, len(valid_indices), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(valid_indices))
        batch_indices = valid_indices[batch_start:batch_end]
        batch_test_tensors = test_tensors[batch_start:batch_end]
        
        # For each test playlist in this batch
        for i, test_idx in enumerate(range(batch_start, batch_end)):
            test_idx = valid_indices[test_idx]
            test_pid = test_pids[test_idx]
            test_songs = test_playlist_matrix[test_idx]
            
            # Create two sets of evaluations:
            # 1. Mask songs that are in the playlist (1s) - for recall on positives
            # 2. Mask songs that are not in the playlist (0s) - for specificity
            
            # 1. Evaluate on positives (1s)
            positive_indices = np.where(test_songs > 0)[0]
            if len(positive_indices) > 0:
                # Choose one random positive song to mask
                mask_idx = np.random.choice(positive_indices)
                true_label = 1  # It's a positive example
                
                # Create test tensor
                test_tensor = batch_test_tensors[i].unsqueeze(0)
                
                # Score against all training playlists
                similarity_scores = []
                train_batch_size = 128
                
                for train_batch_start in range(0, len(train_tensors), train_batch_size):
                    train_batch_end = min(train_batch_start + train_batch_size, len(train_tensors))
                    train_batch = train_tensors[train_batch_start:train_batch_end]
                    
                    test_tensor_repeated = test_tensor.repeat(len(train_batch), 1)
                    
                    with torch.no_grad():
                        batch_scores = model(test_tensor_repeated, train_batch).cpu().numpy()
                    
                    similarity_scores.append(batch_scores)
                
                all_scores = np.concatenate(similarity_scores)
                most_similar_idx = np.argmax(all_scores)
                
                # Check prediction
                prediction = int(train_playlist_matrix[most_similar_idx, mask_idx] > 0)
                
                all_true_labels.append(true_label)
                all_predictions.append(prediction)
            
            # 2. Evaluate on negatives (0s)
            negative_indices = np.where(test_songs == 0)[0]
            if len(negative_indices) > 0:
                # Choose one random negative song to mask
                mask_idx = np.random.choice(negative_indices)
                true_label = 0  # It's a negative example
                
                # Create test tensor
                test_tensor = batch_test_tensors[i].unsqueeze(0)
                
                # Score against all training playlists
                similarity_scores = []
                train_batch_size = 128
                
                for train_batch_start in range(0, len(train_tensors), train_batch_size):
                    train_batch_end = min(train_batch_start + train_batch_size, len(train_tensors))
                    train_batch = train_tensors[train_batch_start:train_batch_end]
                    
                    test_tensor_repeated = test_tensor.repeat(len(train_batch), 1)
                    
                    with torch.no_grad():
                        batch_scores = model(test_tensor_repeated, train_batch).cpu().numpy()
                    
                    similarity_scores.append(batch_scores)
                
                all_scores = np.concatenate(similarity_scores)
                most_similar_idx = np.argmax(all_scores)
                
                # Check prediction
                prediction = int(train_playlist_matrix[most_similar_idx, mask_idx] > 0)
                
                all_true_labels.append(true_label)
                all_predictions.append(prediction)
    
    # Calculate metrics
    all_true_labels = np.array(all_true_labels)
    all_predictions = np.array(all_predictions)
    
    # Basic metrics
    accuracy = np.mean(all_true_labels == all_predictions)
    
    # Class-specific metrics
    true_positives = np.sum((all_true_labels == 1) & (all_predictions == 1))
    false_positives = np.sum((all_true_labels == 0) & (all_predictions == 1))
    true_negatives = np.sum((all_true_labels == 0) & (all_predictions == 0))
    false_negatives = np.sum((all_true_labels == 1) & (all_predictions == 0))
    
    # Calculate precision, recall, f1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # For the negative class (specificity)
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    # Balanced accuracy
    balanced_acc = (recall + specificity) / 2
    
    # Confusion matrix
    confusion_matrix = {
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'true_negatives': int(true_negatives),
        'false_negatives': int(false_negatives)
    }
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'balanced_accuracy': float(balanced_acc),
        'confusion_matrix': confusion_matrix,
        'total_samples': len(all_true_labels),
        'positive_samples': int(np.sum(all_true_labels == 1)),
        'negative_samples': int(np.sum(all_true_labels == 0))
    }

def evaluate_model_simple(
    model,
    test_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    test_pids: np.ndarray,
    train_pids: np.ndarray,
    test_playlist_matrix: np.ndarray,
    train_playlist_matrix: np.ndarray,
    top_songs: List[str],
    max_rows: int = 500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Simple evaluation: For each test playlist, find the most similar training playlist,
    and check if its song profile matches the test playlist.
    
    Args:
        model: Trained model
        test_embeddings: Test playlist embeddings
        train_embeddings: Training playlist embeddings
        test_pids, train_pids: Playlist IDs
        test_playlist_matrix, train_playlist_matrix: Song matrices
        top_songs: List of song URIs
        max_rows: Maximum number of test playlists to evaluate
    
    Returns:
        Dictionary with evaluation results
    """
    print(f"Running simple evaluation on up to {max_rows} test playlists")
    
    # Find test playlists with at least one song
    test_song_counts = np.sum(test_playlist_matrix, axis=1)
    valid_indices = np.where(test_song_counts > 0)[0]
    
    if len(valid_indices) == 0:
        print("No test playlists with songs found. Cannot evaluate.")
        return {"error": "No test playlists with songs found."}
    
    # Limit to max_rows for speed
    if len(valid_indices) > max_rows:
        valid_indices = np.random.choice(valid_indices, max_rows, replace=False)
    
    print(f"Evaluating on {len(valid_indices)} test playlists")
    
    # Define evaluation structures
    results = []
    
    # Convert embeddings to tensor format
    test_tensors = torch.FloatTensor(test_embeddings[valid_indices]).to(device)
    train_tensors = torch.FloatTensor(train_embeddings).to(device)
    
    # Process in batches for efficiency
    batch_size = 32
    correct_total = 0
    total_predictions = 0
    
    for batch_start in tqdm(range(0, len(valid_indices), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(valid_indices))
        batch_indices = valid_indices[batch_start:batch_end]
        batch_test_tensors = test_tensors[batch_start:batch_end]
        
        # For each test playlist in this batch
        for i, test_idx in enumerate(range(batch_start, batch_end)):
            test_idx = valid_indices[test_idx]
            test_pid = test_pids[test_idx]
            test_songs = test_playlist_matrix[test_idx]
            
            # Find songs in this playlist
            test_song_indices = np.where(test_songs > 0)[0]
            
            if len(test_song_indices) == 0:
                continue
            
            # Choose one random song to mask
            mask_idx = np.random.choice(test_song_indices)
            masked_song_uri = top_songs[mask_idx]
            
            # Create test tensor
            test_tensor = batch_test_tensors[i].unsqueeze(0)
            
            # Score against all training playlists
            # We'll process this in sub-batches to avoid OOM errors
            similarity_scores = []
            train_batch_size = 128
            
            for train_batch_start in range(0, len(train_tensors), train_batch_size):
                train_batch_end = min(train_batch_start + train_batch_size, len(train_tensors))
                train_batch = train_tensors[train_batch_start:train_batch_end]
                
                # Repeat test tensor for each training playlist in batch
                test_tensor_repeated = test_tensor.repeat(len(train_batch), 1)
                
                # Get similarity scores
                with torch.no_grad():
                    batch_scores = model(test_tensor_repeated, train_batch).cpu().numpy()
                
                similarity_scores.append(batch_scores)
            
            # Concatenate scores and find most similar
            all_scores = np.concatenate(similarity_scores)
            most_similar_idx = np.argmax(all_scores)
            
            # Get the most similar training playlist
            similar_pid = train_pids[most_similar_idx]
            similar_songs = train_playlist_matrix[most_similar_idx]
            
            # Check if the masked song is in the similar playlist
            is_correct = similar_songs[mask_idx] > 0
            
            # Update counters
            correct_total += int(is_correct)
            total_predictions += 1
            
            # Record result
            results.append({
                'test_pid': int(test_pid),
                'similar_pid': int(similar_pid),
                'masked_song': masked_song_uri,
                'masked_idx': int(mask_idx),
                'is_correct': bool(is_correct)
            })
    
    # Calculate overall metrics
    accuracy = correct_total / total_predictions if total_predictions > 0 else 0
    
    summary = {
        'accuracy': accuracy,
        'correct': correct_total,
        'total': total_predictions,
        'detailed_results': results[:20]  # Include a sample of detailed results
    }
    
    return summary


def run_evaluation(
    embedding_store_path: str,
    data_path: str,
    count_song_path: str,
    model_dir: str = "models",
    max_train_batches: int = 10,
    max_test_batches: int = 5,
    max_eval_rows: int = 500,
    top_n_songs: int = 3000
):
    """Main evaluation function that handles the entire process"""
    
    # First, check H5 file structure
    print("Exploring H5 file structure...")
    explore_h5_file(embedding_store_path)
    
    # Load embeddings with flexible loader
    print("\nLoading embeddings...")
    train_embeddings, train_pids = load_embeddings_flexible(
        embedding_store_path, 
        'train', 
        max_batches=max_train_batches
    )
    
    test_embeddings, test_pids = load_embeddings_flexible(
        embedding_store_path, 
        'test', 
        max_batches=max_test_batches
    )
    
    print(f"Loaded {len(train_embeddings)} train and {len(test_embeddings)} test embeddings")
    
    # Load top songs
    top_songs = load_top_songs(count_song_path, top_n=top_n_songs)
    
    # Create playlist-song matrices
    print("\nCreating playlist-song matrices...")
    train_playlist_matrix = create_playlist_song_matrix(data_path, train_pids, top_songs)
    test_playlist_matrix = create_playlist_song_matrix(data_path, test_pids, top_songs)
    
    # Find all model files
    model_paths = list(Path(model_dir).glob("playlist_recommender_fast_*.pt"))
    
    if not model_paths:
        print(f"No models found in {model_dir}")
        return
    
    print(f"\nFound {len(model_paths)} models to evaluate")
    
    # Evaluate each model
    all_results = []
    
    for model_path in model_paths:
        print(f"\n==== Evaluating {model_path} ====")
        model_name = model_path.name
        
        # Extract learning rate from filename
        lr = float(model_name.split('_')[-1].replace('.pt', ''))
        
        try:
            # Load model
            model, model_data = load_model(str(model_path))
            
            # Start evaluation timer
            start_time = time.time()
            
            # Run evaluation
            # results = evaluate_model_simple(
            #     model,
            #     test_embeddings,
            #     train_embeddings,
            #     test_pids,
            #     train_pids,
            #     test_playlist_matrix,
            #     train_playlist_matrix,
            #     top_songs,
            #     max_rows=max_eval_rows
            # )
            results = evaluate_model_with_advanced_metrics(
                model,
                test_embeddings,
                train_embeddings,
                test_pids,
                train_pids,
                test_playlist_matrix,
                train_playlist_matrix,
                top_songs,
                max_rows=max_eval_rows
            )

            
            # Record evaluation time
            eval_time = time.time() - start_time
            
            # Add model metadata
            model_results = {
                'model_name': model_name,
                'learning_rate': lr,
                'embedding_dim': model_data['embedding_dim'],
                'hidden_dims': model_data.get('hidden_dims', [512, 256]),
                'train_losses': [float(x) for x in model_data.get('epoch_losses', [])],
                **results,
                'eval_time': eval_time
            }
            
            all_results.append(model_results)
            
            # Print summary with enhanced metrics
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1 Score: {results['f1_score']:.4f}")
            print(f"  Balanced Accuracy: {results['balanced_accuracy']:.4f}")
            print(f"  Confusion Matrix: {results['confusion_matrix']}")
            print(f"  Evaluation time: {eval_time:.2f} seconds")
            
            # Clean up to avoid GPU memory issues
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error evaluating {model_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'nn_evaluation_results_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nEvaluation results saved to: {results_path}")
    
    # Print comparative results
    print("\nComparative Results:")
    print("-" * 60)
    print(f"{'Model':<30} {'Learning Rate':<15} {'Accuracy':<10}")
    print("-" * 60)
    
    for result in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{result['model_name']:<30} {result['learning_rate']:<15.5f} {result['accuracy']:<10.4f}")
    
    return all_results


def main():
    # Configuration
    embedding_store_path = "/data/user_data/rshar/downloads/spotify/playlist_embeddings.h5"
    data_path = "/data/user_data/rshar/downloads/spotify/data"
    count_song_path = "data/count_songs.csv"
    model_dir = "models"
    
    # Run evaluation with more robust loading
    run_evaluation(
        embedding_store_path=embedding_store_path,
        data_path=data_path,
        count_song_path=count_song_path,
        model_dir=model_dir,
        max_train_batches=10,  # Limit number of training batches
        max_test_batches=5,    # Limit number of test batches
        max_eval_rows=500,     # Limit number of evaluation rows
        top_n_songs=3000       # Number of top songs to use
    )


if __name__ == "__main__":
    main()