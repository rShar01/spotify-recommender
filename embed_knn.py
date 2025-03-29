import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import gc
import json
import time
import h5py
import os
from datetime import datetime
import warnings
from typing import List, Dict, Tuple
from sklearn.neighbors import NearestNeighbors


def load_embeddings(embedding_store_path: str, split: str, n_batches: int = 3):
    """Load embeddings from H5 file"""
    print(f"Loading {split} embeddings...")
    
    with h5py.File(embedding_store_path, 'r') as f:
        if split not in f:
            raise ValueError(f"Split '{split}' not found in embedding store")
            
        split_group = f[split]
        all_embeddings = []
        all_pids = []
        
        # Check for direct batch datasets
        batch_keys = [k for k in split_group.keys() if k.startswith('batch_') and not k.endswith('_pids')]
        
        if batch_keys:
            # Handle case with direct batch datasets
            batch_keys = sorted(batch_keys)[:n_batches]
            
            for batch_key in batch_keys:
                pid_key = f"{batch_key}_pids"
                if pid_key in split_group:
                    embeddings = split_group[batch_key][:]
                    pids = split_group[pid_key][:]
                    all_embeddings.append(embeddings)
                    all_pids.append(pids)
        else:
            # Handle case with slice groups
            slice_keys = [k for k in split_group.keys() if k.startswith('slice_')]
            
            if slice_keys:
                batch_count = 0
                for slice_key in sorted(slice_keys):
                    slice_group = split_group[slice_key]
                    
                    # Find batch datasets in slice
                    sub_batch_keys = [k for k in slice_group.keys() if k.startswith('batch_') and not k.endswith('_pids')]
                    
                    for batch_key in sorted(sub_batch_keys):
                        if batch_count >= n_batches:
                            break
                            
                        pid_key = f"{batch_key}_pids"
                        if pid_key in slice_group:
                            embeddings = slice_group[batch_key][:]
                            pids = slice_group[pid_key][:]
                            all_embeddings.append(embeddings)
                            all_pids.append(pids)
                            batch_count += 1
                    
                    if batch_count >= n_batches:
                        break
    
    if not all_embeddings:
        raise ValueError(f"No embeddings loaded from {split}")
    
    # Concatenate loaded embeddings
    all_embeddings = np.concatenate(all_embeddings)
    all_pids = np.concatenate(all_pids)
    
    # Create pid to index mapping
    pid_to_idx = {int(pid): i for i, pid in enumerate(all_pids)}
    
    print(f"Loaded {len(all_embeddings)} {split} embeddings")
    return all_embeddings, all_pids, pid_to_idx


def load_single_file(path):
    """Load a single CSV file and preprocess it"""
    df = pd.read_csv(path, index_col=0)
    if any(col in df.columns for col in ["pid", "collaborative", "num_followers", "num_tracks"]):
        df = df.drop(["pid", "collaborative", "num_followers", "num_tracks"], axis=1)
    return df


class EmbeddingKNNPredictor:
    """KNN predictor that works directly on embeddings"""
    
    def __init__(self, train_embeddings, train_csv_files, n_neighbors=5):
        self.train_embeddings = train_embeddings
        self.train_csv_files = train_csv_files
        self.n_neighbors = n_neighbors
        self.train_data = None
        
        # Initialize the KNN model
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        self.knn.fit(train_embeddings)
        
        # Load a small subset of training data for lookup
        self.load_train_sample()
    
    def load_train_sample(self, max_files=10):
        """Load a sample of training data for imputation"""
        print(f"Loading sample of {max_files} training files for lookup...")
        
        sample_files = random.sample(self.train_csv_files, min(max_files, len(self.train_csv_files)))
        
        dfs = []
        for path in tqdm(sample_files, desc="Loading train sample"):
            df = load_single_file(path)
            dfs.append(df)
        
        self.train_data = pd.concat(dfs)
        print(f"Loaded {len(self.train_data)} training rows for lookup")
    
    def transform(self, test_masked):
        """Transform method mimicking KNNImputer.transform"""
        print("Performing embedding-based KNN imputation...")
        
        # Make a copy of the test data
        test_imputed = test_masked.copy()
        
        # For each row with NaNs
        for i in tqdm(range(len(test_masked)), desc="Imputing rows"):
            row = test_masked[i]
            nan_indices = np.where(np.isnan(row))[0]
            
            if len(nan_indices) == 0:
                continue
            
            # Get test embedding for this row
            # Since we don't have direct mapping, use a dummy embedding for demonstration
            # In a real scenario, you would need to map test rows to embeddings
            # This is a placeholder that needs to be adapted to your data
            test_embedding = np.zeros((1, self.train_embeddings.shape[1]))
            
            # Find k nearest neighbors
            distances, indices = self.knn.kneighbors(test_embedding)
            
            # Use values from similar playlists to impute NaNs
            for col_idx in nan_indices:
                # Get values from nearest neighbors
                neighbor_values = []
                for neighbor_idx in indices[0]:
                    if self.train_data is not None and col_idx < self.train_data.shape[1]:
                        # Get value from training data
                        col_name = self.train_data.columns[col_idx]
                        value = self.train_data.iloc[neighbor_idx % len(self.train_data)][col_name]
                        neighbor_values.append(value)
                
                # Use majority vote or mean for imputation
                if neighbor_values:
                    # For binary data (0/1), use majority vote
                    if all(val in [0, 1] for val in neighbor_values):
                        imputed_value = int(np.round(np.mean(neighbor_values)))
                    else:
                        # For continuous data, use mean
                        imputed_value = np.mean(neighbor_values)
                    
                    test_imputed[i, col_idx] = imputed_value
                else:
                    # Fallback to binary imputation (0 or 1)
                    test_imputed[i, col_idx] = np.random.choice([0, 1])
        
        return test_imputed


def fast_predict_embedding_knn(train_embeddings, test_df, train_csv_files, n_neighbors=5, n_masks=5, max_rows=500):
    """Make optimized predictions using embedding KNN approach"""
    # Limit number of rows for speed
    if len(test_df) > max_rows:
        test_df = test_df.iloc[:max_rows]
        
    n_rows = len(test_df)
    n_cols = test_df.shape[1]
    results = []
    
    # Create fixed mask indices
    mask_indices = np.arange(0, n_cols, n_cols // n_masks)[:n_masks]
    mask_indices = np.tile(mask_indices, (n_rows, 1))
    
    # Create embedding KNN predictor
    knn_predictor = EmbeddingKNNPredictor(train_embeddings, train_csv_files, n_neighbors=n_neighbors)
    
    for mask_idx in range(n_masks):
        # Create float array for NaN support
        test_masked = test_df.values.astype(np.float64).copy()
        
        # Get true values
        true_vals = test_masked[np.arange(n_rows), mask_indices[:, mask_idx]]
        
        # Mask values
        test_masked[np.arange(n_rows), mask_indices[:, mask_idx]] = np.nan
        
        # Make predictions
        predictions = knn_predictor.transform(test_masked)
        
        # Get and round predictions
        pred_vals = predictions[np.arange(n_rows), mask_indices[:, mask_idx]]
        pred_vals = np.round(pred_vals).astype(int)
        
        # Store only overall accuracy data for speed
        correct = np.sum(true_vals == pred_vals)
        total = len(true_vals)
        
        results.append({
            'mask_position': mask_idx,
            'correct': int(correct),
            'total': total,
            'accuracy': float(correct/total if total > 0 else 0)
        })
    
    return results


def evaluate_embedding_knn(train_embeddings, test_paths, train_csv_files, n_neighbors=5, n_masks=5, max_files=10, max_rows=500):
    """Perform a fast evaluation on a small subset of test data using embedding-based KNN"""
    print(f"Fast evaluation on max {max_files} files, {max_rows} rows per file using K={n_neighbors}")
    
    # Use only a subset of test files
    test_paths = random.sample(test_paths, min(max_files, len(test_paths)))
    
    all_results = []
    for path in tqdm(test_paths, desc="Evaluating files"):
        test_df = load_single_file(path)
        batch_results = fast_predict_embedding_knn(
            train_embeddings, 
            test_df, 
            train_csv_files, 
            n_neighbors=n_neighbors,
            n_masks=n_masks, 
            max_rows=max_rows
        )
        
        for result in batch_results:
            result['file'] = path.name
            result['k'] = n_neighbors
            all_results.append(result)
    
    # Aggregate results
    total_correct = sum(r['correct'] for r in all_results)
    total_predictions = sum(r['total'] for r in all_results)
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    
    return {
        'k': n_neighbors,
        'detailed_results': all_results,
        'accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_predictions': total_predictions
    }


def run_embedding_knn_analysis(train_csv_files, test_csv_files, embedding_store_path,
                               k_values=[1, 5, 10, 20], n_masks=5, train_batches=3, test_max_files=10, max_rows=300):
    """Run analysis with KNN on embeddings for different K values"""
    metrics = []
    
    # Load embeddings
    train_embeddings, train_pids, _ = load_embeddings(embedding_store_path, 'train', n_batches=train_batches)
    
    # Evaluate for each K value
    for k in k_values:
        print(f"\nEvaluating embedding KNN with k={k}")
        start_time = time.time()
        
        try:
            # Evaluate
            results = evaluate_embedding_knn(
                train_embeddings, 
                test_csv_files, 
                train_csv_files,
                n_neighbors=k,
                n_masks=n_masks, 
                max_files=test_max_files, 
                max_rows=max_rows
            )
            
            eval_time = time.time() - start_time
            
            # Add timing information
            results['eval_time'] = eval_time
            metrics.append(results)
            
            print(f"Accuracy for k={k}: {results['accuracy']:.2%}")
            print(f"Evaluation completed in {eval_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error evaluating K={k}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save metrics with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_path = f'embedding_knn_metrics_{timestamp}.json'
    
    # Convert to JSON-serializable format
    json_metrics = []
    for m in metrics:
        json_metric = {k: v for k, v in m.items() if k != 'detailed_results'}
        json_metrics.append(json_metric)
        
    with open(metrics_path, 'w') as f:
        json.dump(json_metrics, f, indent=4)
    
    print(f"Metrics saved to: {metrics_path}")
    
    # Print final summary
    print("\nFinal Results Summary:")
    for metric in sorted(metrics, key=lambda x: x['accuracy'], reverse=True):
        print(f"k={metric['k']}: {metric['accuracy']:.2%} (eval: {metric['eval_time']:.1f}s)")
    
    return metrics


if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Set random seed
    np.random.seed(42)
    
    # Setup paths
    base_path = Path("/data/user_data/rshar/downloads/spotify/simple_vec")
    embedding_store_path = "/data/user_data/rshar/downloads/spotify/playlist_embeddings.h5"
    
    # Find CSV files
    all_csvs = sorted(list(base_path.glob("*")))
    
    # Split into train/test
    train_set = random.sample(all_csvs, int(len(all_csvs) * 0.70))
    test_set = [x for x in all_csvs if x not in train_set]
    
    print(f"Found {len(all_csvs)} CSV files, {len(train_set)} train, {len(test_set)} test")
    
    # Define k values to test
    k_values = [1, 5, 10, 20]
    
    # Run analysis
    metrics = run_embedding_knn_analysis(
        train_csv_files=train_set,
        test_csv_files=test_set,
        embedding_store_path=embedding_store_path,
        k_values=k_values,
        n_masks=1,              # number of predictions per row
        train_batches=3,        # number of embedding batches to load
        test_max_files=10,      # limit testing to 10 files
        max_rows=300            # limit to 300 rows per file
    )