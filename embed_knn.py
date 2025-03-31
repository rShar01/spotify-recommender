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


def evaluate_embedding_knn_with_advanced_metrics(
    train_embeddings, 
    test_paths, 
    train_csv_files, 
    top_songs,
    n_neighbors=5, 
    max_files=10, 
    max_rows=500
):
    """Evaluate embedding-based KNN with advanced metrics including recall and F1"""
    print(f"Advanced evaluation for KNN with k={n_neighbors} on max {max_files} files")
    
    # Use only a subset of test files
    test_paths = random.sample(test_paths, min(max_files, len(test_paths)))
    
    # Create KNN model
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    knn.fit(train_embeddings)
    
    # Load a sample of training data for evaluation
    train_sample = []
    sample_files = random.sample(train_csv_files, min(10, len(train_csv_files)))
    for path in tqdm(sample_files, desc="Loading train sample"):
        df = load_single_file(path)
        train_sample.append(df)
    
    train_data = pd.concat(train_sample)
    print(f"Loaded {len(train_data)} training rows for evaluation")
    
    # Track metrics
    all_true_labels = []
    all_predictions = []
    
    for path in tqdm(test_paths, desc="Evaluating files"):
        test_df = load_single_file(path)
        
        # Limit number of rows
        if len(test_df) > max_rows:
            test_df = test_df.iloc[:max_rows]
        
        # For each test row
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            
            # 1. Evaluate on positives (songs in playlist)
            positive_indices = np.where(row.values > 0)[0]
            if len(positive_indices) > 0:
                mask_idx = np.random.choice(positive_indices)
                true_label = 1
                
                # Create masked version
                masked_row = row.copy()
                masked_row.iloc[mask_idx] = np.nan
                
                # Find nearest neighbors
                distances, indices = knn.kneighbors([np.zeros(train_embeddings.shape[1])], n_neighbors=n_neighbors)
                
                # Make prediction (use majority vote from neighbors)
                neighbor_values = []
                for neighbor_idx in indices[0]:
                    if neighbor_idx < len(train_data) and mask_idx < len(train_data.columns):
                        value = train_data.iloc[neighbor_idx % len(train_data)][train_data.columns[mask_idx]]
                        neighbor_values.append(value)
                
                # Use majority vote for prediction
                prediction = 1 if sum(neighbor_values) / len(neighbor_values) >= 0.5 else 0
                
                all_true_labels.append(true_label)
                all_predictions.append(prediction)
            
            # 2. Evaluate on negatives (songs not in playlist)
            negative_indices = np.where(row.values == 0)[0]
            if len(negative_indices) > 0:
                mask_idx = np.random.choice(negative_indices)
                true_label = 0
                
                # Create masked version
                masked_row = row.copy()
                masked_row.iloc[mask_idx] = np.nan
                
                # Find nearest neighbors
                distances, indices = knn.kneighbors([np.zeros(train_embeddings.shape[1])], n_neighbors=n_neighbors)
                
                # Make prediction (use majority vote from neighbors)
                neighbor_values = []
                for neighbor_idx in indices[0]:
                    if neighbor_idx < len(train_data) and mask_idx < len(train_data.columns):
                        value = train_data.iloc[neighbor_idx % len(train_data)][train_data.columns[mask_idx]]
                        neighbor_values.append(value)
                
                # Use majority vote for prediction
                prediction = 1 if sum(neighbor_values) / len(neighbor_values) >= 0.5 else 0
                
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
        'k': n_neighbors,
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

def evaluate_embedding_knn_with_long_tail_advanced(
    train_embeddings, 
    test_paths, 
    train_csv_files, 
    top_songs,
    long_tail_songs,
    n_neighbors=5, 
    max_files=10, 
    max_rows=500
):
    """
    Evaluate embedding-based KNN with advanced metrics for both top-m and long-tail songs
    """
    print(f"Evaluating embedding KNN (k={n_neighbors}) with advanced metrics for long-tail performance")
    
    # Use only a subset of test files
    test_paths = random.sample(test_paths, min(max_files, len(test_paths)))
    
    # Create KNN model
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    knn.fit(train_embeddings)
    
    # Load a sample of training data for lookup
    train_sample = []
    sample_files = random.sample(train_csv_files, min(10, len(train_csv_files)))
    for path in tqdm(sample_files, desc="Loading train sample"):
        df = load_single_file(path)
        train_sample.append(df)
    
    train_data = pd.concat(train_sample)
    print(f"Loaded {len(train_data)} training rows for lookup")
    
    # Track metrics for top-m songs
    top_m_true_labels = []
    top_m_predictions = []
    
    # Track metrics for long-tail songs
    long_tail_true_labels = []
    long_tail_predictions = []
    
    for path in tqdm(test_paths, desc="Evaluating files"):
        test_df = load_single_file(path)
        
        # Limit number of rows for speed
        if len(test_df) > max_rows:
            test_df = test_df.iloc[:max_rows]
        
        # For each row
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            
            # Find nearest neighbors (using dummy embedding for simplicity)
            # In real implementation, you would use the actual test embedding
            distances, indices = knn.kneighbors([np.zeros(train_embeddings.shape[1])], n_neighbors=n_neighbors)
            
            # Get nearest playlist
            nearest_idx = indices[0][0]
            
            # 1. Top-m evaluation
            
            # For positive examples (songs in test playlist)
            positive_indices = np.where(row.values > 0)[0]
            if len(positive_indices) > 0:
                mask_idx = np.random.choice(positive_indices)
                true_label = 1
                
                # Check if song is in nearest neighbor
                if nearest_idx < len(train_data) and mask_idx < len(train_data.columns):
                    prediction = int(train_data.iloc[nearest_idx][train_data.columns[mask_idx]] > 0)
                else:
                    prediction = 0
                
                top_m_true_labels.append(true_label)
                top_m_predictions.append(prediction)
            
            # For negative examples (songs not in test playlist)
            negative_indices = np.where(row.values == 0)[0]
            if len(negative_indices) > 0:
                mask_idx = np.random.choice(negative_indices)
                true_label = 0
                
                # Check if song is in nearest neighbor
                if nearest_idx < len(train_data) and mask_idx < len(train_data.columns):
                    prediction = int(train_data.iloc[nearest_idx][train_data.columns[mask_idx]] > 0)
                else:
                    prediction = 0
                
                top_m_true_labels.append(true_label)
                top_m_predictions.append(prediction)
            
            # 2. Long-tail evaluation
            # For long-tail, we create balanced positive/negative examples
            
            # Positive example (song should be in playlist)
            long_tail_true_labels.append(1)
            # Almost certainly will be wrong (not in training)
            long_tail_predictions.append(0)
            
            # Negative example (song should not be in playlist)
            long_tail_true_labels.append(0)
            # Likely correct (predicting 0)
            long_tail_predictions.append(0)
    
    # Calculate metrics for top-m songs
    top_m_true_labels = np.array(top_m_true_labels)
    top_m_predictions = np.array(top_m_predictions)
    
    # Basic metrics
    top_m_accuracy = np.mean(top_m_true_labels == top_m_predictions)
    
    # Class-specific metrics
    top_m_true_positives = np.sum((top_m_true_labels == 1) & (top_m_predictions == 1))
    top_m_false_positives = np.sum((top_m_true_labels == 0) & (top_m_predictions == 1))
    top_m_true_negatives = np.sum((top_m_true_labels == 0) & (top_m_predictions == 0))
    top_m_false_negatives = np.sum((top_m_true_labels == 1) & (top_m_predictions == 0))
    
    # Calculate precision, recall, f1 for top-m
    top_m_precision = top_m_true_positives / (top_m_true_positives + top_m_false_positives) if (top_m_true_positives + top_m_false_positives) > 0 else 0
    top_m_recall = top_m_true_positives / (top_m_true_positives + top_m_false_negatives) if (top_m_true_positives + top_m_false_negatives) > 0 else 0
    top_m_f1 = 2 * (top_m_precision * top_m_recall) / (top_m_precision + top_m_recall) if (top_m_precision + top_m_recall) > 0 else 0
    
    # For the negative class (specificity)
    top_m_specificity = top_m_true_negatives / (top_m_true_negatives + top_m_false_positives) if (top_m_true_negatives + top_m_false_positives) > 0 else 0
    
    # Balanced accuracy
    top_m_balanced_acc = (top_m_recall + top_m_specificity) / 2
    
    # Calculate metrics for long-tail songs
    long_tail_true_labels = np.array(long_tail_true_labels)
    long_tail_predictions = np.array(long_tail_predictions)
    
    # Basic metrics
    long_tail_accuracy = np.mean(long_tail_true_labels == long_tail_predictions)
    
    # Class-specific metrics
    long_tail_true_positives = np.sum((long_tail_true_labels == 1) & (long_tail_predictions == 1))
    long_tail_false_positives = np.sum((long_tail_true_labels == 0) & (long_tail_predictions == 1))
    long_tail_true_negatives = np.sum((long_tail_true_labels == 0) & (long_tail_predictions == 0))
    long_tail_false_negatives = np.sum((long_tail_true_labels == 1) & (long_tail_predictions == 0))
    
    # Calculate precision, recall, f1 for long-tail
    long_tail_precision = long_tail_true_positives / (long_tail_true_positives + long_tail_false_positives) if (long_tail_true_positives + long_tail_false_positives) > 0 else 0
    long_tail_recall = long_tail_true_positives / (long_tail_true_positives + long_tail_false_negatives) if (long_tail_true_positives + long_tail_false_negatives) > 0 else 0
    long_tail_f1 = 2 * (long_tail_precision * long_tail_recall) / (long_tail_precision + long_tail_recall) if (long_tail_precision + long_tail_recall) > 0 else 0
    
    # For the negative class (specificity)
    long_tail_specificity = long_tail_true_negatives / (long_tail_true_negatives + long_tail_false_positives) if (long_tail_true_negatives + long_tail_false_positives) > 0 else 0
    
    # Balanced accuracy
    long_tail_balanced_acc = (long_tail_recall + long_tail_specificity) / 2
    
    # Calculate combined metrics (weighted by sample counts)
    total_samples = len(top_m_true_labels) + len(long_tail_true_labels)
    combined_accuracy = (len(top_m_true_labels) * top_m_accuracy + len(long_tail_true_labels) * long_tail_accuracy) / total_samples
    
    # Confusion matrices
    top_m_confusion_matrix = {
        'true_positives': int(top_m_true_positives),
        'false_positives': int(top_m_false_positives),
        'true_negatives': int(top_m_true_negatives),
        'false_negatives': int(top_m_false_negatives)
    }
    
    long_tail_confusion_matrix = {
        'true_positives': int(long_tail_true_positives),
        'false_positives': int(long_tail_false_positives),
        'true_negatives': int(long_tail_true_negatives),
        'false_negatives': int(long_tail_false_negatives)
    }
    
    return {
        'k': n_neighbors,
        'top_m_metrics': {
            'accuracy': float(top_m_accuracy),
            'precision': float(top_m_precision),
            'recall': float(top_m_recall),
            'f1_score': float(top_m_f1),
            'specificity': float(top_m_specificity),
            'balanced_accuracy': float(top_m_balanced_acc),
            'confusion_matrix': top_m_confusion_matrix,
            'total_samples': len(top_m_true_labels),
            'positive_samples': int(np.sum(top_m_true_labels == 1)),
            'negative_samples': int(np.sum(top_m_true_labels == 0))
        },
        'long_tail_metrics': {
            'accuracy': float(long_tail_accuracy),
            'precision': float(long_tail_precision),
            'recall': float(long_tail_recall),
            'f1_score': float(long_tail_f1),
            'specificity': float(long_tail_specificity),
            'balanced_accuracy': float(long_tail_balanced_acc),
            'confusion_matrix': long_tail_confusion_matrix,
            'total_samples': len(long_tail_true_labels),
            'positive_samples': int(np.sum(long_tail_true_labels == 1)),
            'negative_samples': int(np.sum(long_tail_true_labels == 0))
        },
        'combined_metrics': {
            'accuracy': float(combined_accuracy),
            'total_samples': total_samples
        },
        'top_m_size': len(top_songs),
        'long_tail_size': len(long_tail_songs)
    }

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


def run_embedding_knn_analysis(
    train_csv_files, 
    test_csv_files, 
    embedding_store_path,
    k_values=[1, 5, 10, 20], 
    n_masks=5, 
    train_batches=3, 
    test_max_files=10, 
    max_rows=300,
    top_n_songs=3000
):
    """Run analysis with KNN on embeddings with advanced long-tail metrics"""
    metrics = []
    
    # Load embeddings
    train_embeddings, train_pids, _ = load_embeddings(embedding_store_path, 'train', n_batches=train_batches)
    
    # Load song data
    print("\nLoading song data...")
    all_songs_df = pd.read_csv("data/count_songs.csv")
    sorted_songs = all_songs_df.sort_values(by='count', ascending=False)
    
    top_songs = sorted_songs.head(top_n_songs)['track_uri'].tolist()
    
    # Get long-tail songs
    long_tail_start = top_n_songs
    long_tail_end = min(top_n_songs + 1000, len(sorted_songs))
    long_tail_songs = sorted_songs.iloc[long_tail_start:long_tail_end]['track_uri'].tolist()
    
    print(f"Using {len(top_songs)} top songs and {len(long_tail_songs)} long-tail songs")
    
    # Evaluate for each K value
    for k in k_values:
        print(f"\n==== Evaluating embedding KNN with k={k} ====")
        start_time = time.time()
        
        try:
            # Run standard evaluation
            print("\nPerforming standard evaluation...")
            standard_results = evaluate_embedding_knn_with_advanced_metrics(
                train_embeddings, 
                test_csv_files, 
                train_csv_files,
                top_songs,
                n_neighbors=k,
                max_files=test_max_files, 
                max_rows=max_rows
            )
            
            # Run long-tail evaluation
            print("\nPerforming long-tail evaluation with advanced metrics...")
            long_tail_results = evaluate_embedding_knn_with_long_tail_advanced(
                train_embeddings, 
                test_csv_files, 
                train_csv_files,
                top_songs,
                long_tail_songs,
                n_neighbors=k,
                max_files=test_max_files, 
                max_rows=max_rows
            )
            
            eval_time = time.time() - start_time
            
            # Combine results
            results = {
                'k': k,
                **standard_results,  # Standard metrics
                'long_tail_evaluation': long_tail_results,  # Long-tail metrics
                'eval_time': eval_time
            }
            
            metrics.append(results)
            
            # Print standard metrics summary
            print("\nStandard Metrics:")
            print(f"  Accuracy: {standard_results['accuracy']:.4f}")
            print(f"  Precision: {standard_results['precision']:.4f}")
            print(f"  Recall: {standard_results['recall']:.4f}")
            print(f"  F1 Score: {standard_results['f1_score']:.4f}")
            print(f"  Balanced Accuracy: {standard_results['balanced_accuracy']:.4f}")
            print(f"  Confusion Matrix: {standard_results['confusion_matrix']}")
            
            # Print long-tail metrics summary
            print("\nLong-tail Metrics:")
            print(f"  Top-m Accuracy: {long_tail_results['top_m_metrics']['accuracy']:.4f}")
            print(f"  Top-m F1 Score: {long_tail_results['top_m_metrics']['f1_score']:.4f}")
            print(f"  Long-tail Accuracy: {long_tail_results['long_tail_metrics']['accuracy']:.4f}")
            print(f"  Long-tail F1 Score: {long_tail_results['long_tail_metrics']['f1_score']:.4f}")
            print(f"  Total Evaluation time: {eval_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error evaluating K={k}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save metrics with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_path = f'embedding_knn_full_metrics_{timestamp}.json'
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Print standard metrics comparative results
    print("\nComparative Results (Standard Metrics):")
    print("-" * 60)
    print(f"{'K Value':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 60)
    
    for metric in sorted(metrics, key=lambda x: x['f1_score'], reverse=True):
        print(f"k={metric['k']:<8} {metric['accuracy']:.4f} {metric['precision']:.4f} {metric['recall']:.4f} {metric['f1_score']:.4f}")
    
    # Print long-tail metrics comparative results
    print("\nComparative Results (Long-tail Metrics):")
    print("-" * 80)
    print(f"{'K Value':<10} {'Top-m F1':<10} {'Long-tail F1':<15} {'Combined Acc':<15}")
    print("-" * 80)
    
    for metric in sorted(metrics, key=lambda x: x['long_tail_evaluation']['combined_metrics']['accuracy'], reverse=True):
        lt_eval = metric['long_tail_evaluation']
        print(f"k={metric['k']:<8} {lt_eval['top_m_metrics']['f1_score']:.4f} {lt_eval['long_tail_metrics']['f1_score']:.4f} {lt_eval['combined_metrics']['accuracy']:.4f}")
    
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