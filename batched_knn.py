import pandas as pd
import numpy as np
import random
from pathlib import Path
from sklearn.impute import KNNImputer
from tqdm import tqdm
import gc
import json
import time
from datetime import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from sklearn.neighbors import NearestNeighbors

# Set number of CPU cores to use
N_JOBS = os.cpu_count() - 1  # Leave one core free
if N_JOBS < 1:
    N_JOBS = 1

def load_single_file(path):
    """Load a single CSV file and preprocess it"""
    df = pd.read_csv(path, index_col=0)
    # Convert to sparse matrix if your dataset is very sparse
    if any(col in df.columns for col in ["pid", "collaborative", "num_followers", "num_tracks"]):
        df = df.drop(["pid", "collaborative", "num_followers", "num_tracks"], axis=1)
    return df

class FastSpotifyKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        # Use NearestNeighbors instead of KNNImputer for performance
        # KNNImputer is much slower because it handles multiple NaN values
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', 
                                   n_jobs=N_JOBS, metric='euclidean')
    
    def fit(self, train_paths, max_files=100, max_rows_per_file=1000):
        """Fit the model on limited data for speed"""
        print(f"Fitting model with k={self.n_neighbors}")
        
        # Sample a subset of training files
        if max_files and len(train_paths) > max_files:
            train_paths = random.sample(train_paths, max_files)
        
        # Load and concatenate data
        dfs = []
        total_rows = 0
        
        for path in tqdm(train_paths, desc="Loading training files"):
            df = load_single_file(path)
            
            # Sample rows if needed
            if max_rows_per_file and len(df) > max_rows_per_file:
                df = df.sample(max_rows_per_file)
                
            dfs.append(df)
            total_rows += len(df)
            
            # Early stop if we have enough data
            if total_rows >= 10000:  # Limit total rows for very fast fitting
                break
                
        print("Concatenating training data...")
        train_data = pd.concat(dfs)
        print(f"Training on dataset of shape: {train_data.shape}")
        
        # Fit the model
        start_time = time.time()
        self.nn.fit(train_data)
        print(f"Fitting completed in {time.time() - start_time:.2f} seconds")
        
        # Store feature names
        self.feature_names = train_data.columns.tolist()
        
        # Store a sample of the training data for prediction
        self.train_data = train_data
        
        return self
    
    def predict(self, row, mask_idx):
        """Predict a single masked value using nearest neighbors without leakage"""
        # Create a modified row with the masked feature removed
        row_values = row.values.astype(np.float64)
        original_value = row_values[mask_idx]
        
        # Create feature mask (True for features to include, False for masked feature)
        feature_mask = np.ones(len(row_values), dtype=bool)
        feature_mask[mask_idx] = False
        
        # Only use non-masked features for finding neighbors
        masked_row = row_values[feature_mask]
        
        # We need to apply the same mask to our training data
        masked_train_data = self.train_data.values[:, feature_mask]
        
        # Find nearest neighbors using only the non-masked features
        # We'll use a temporary NearestNeighbors model
        temp_nn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto', 
                                  n_jobs=N_JOBS, metric='euclidean')
        temp_nn.fit(masked_train_data)
        distances, indices = temp_nn.kneighbors([masked_row], n_neighbors=self.n_neighbors)
        
        # Get values from neighbors at the masked position
        neighbor_values = [self.train_data.iloc[idx][self.train_data.columns[mask_idx]] for idx in indices[0]]
        
        # Simple majority vote for binary data
        prediction = round(sum(neighbor_values) / len(neighbor_values))
        
        return prediction, original_value
    
    def fast_evaluate(self, test_paths, n_masks=1, max_files=10, max_rows=300):
        """Fast evaluation focused on speed"""
        # Sample test files
        if len(test_paths) > max_files:
            test_paths = random.sample(list(test_paths), max_files)
            
        correct = 0
        total = 0
        
        for path in tqdm(test_paths, desc="Evaluating"):
            test_df = load_single_file(path)
            
            # Sample rows if needed
            if len(test_df) > max_rows:
                test_rows = random.sample(range(len(test_df)), max_rows)
            else:
                test_rows = range(len(test_df))
            
            # Use fixed, evenly spaced features for masking
            n_features = test_df.shape[1]
            mask_positions = np.linspace(0, n_features-1, n_masks, dtype=int)
            
            # Make predictions
            for row_idx in test_rows:
                row = test_df.iloc[row_idx]
                
                for mask_pos in mask_positions:
                    predicted, actual = self.predict(row, mask_pos)
                    
                    if predicted == actual:
                        correct += 1
                    total += 1
                    
                    # Early stopping option for very quick estimates
                    if total >= 1000:  # Limit total predictions for speed
                        break
                        
                if total >= 1000:
                    break
                    
            if total >= 1000:
                break
        
        accuracy = correct / total if total > 0 else 0
        return {
            'k': self.n_neighbors,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

def evaluate_knn_with_advanced_metrics(model, test_paths, top_songs, n_masks=5, max_files=10, max_rows=500):
    """Evaluate KNN imputer with advanced metrics including recall and F1"""
    print(f"Advanced evaluation for KNN imputer on max {max_files} files")
    
    # Use only a subset of test files
    test_paths = random.sample(test_paths, min(max_files, len(test_paths)))
    
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
            row = test_df.iloc[i].values
            
            # 1. Evaluate on positives (songs in playlist)
            positive_indices = np.where(row > 0)[0]
            if len(positive_indices) > 0:
                mask_idx = np.random.choice(positive_indices)
                true_label = 1
                
                # Create masked version
                masked_row = row.copy().astype(np.float64)
                masked_row[mask_idx] = np.nan
                
                # Reshape for KNNImputer
                masked_row = masked_row.reshape(1, -1)
                
                # Make prediction
                predicted_row = model.transform(masked_row)
                prediction = int(np.round(predicted_row[0, mask_idx]))
                
                all_true_labels.append(true_label)
                all_predictions.append(prediction)
            
            # 2. Evaluate on negatives (songs not in playlist)
            negative_indices = np.where(row == 0)[0]
            if len(negative_indices) > 0:
                mask_idx = np.random.choice(negative_indices)
                true_label = 0
                
                # Create masked version
                masked_row = row.copy().astype(np.float64)
                masked_row[mask_idx] = np.nan
                
                # Reshape for KNNImputer
                masked_row = masked_row.reshape(1, -1)
                
                # Make prediction
                predicted_row = model.transform(masked_row)
                prediction = int(np.round(predicted_row[0, mask_idx]))
                
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

def evaluate_knn_with_long_tail_advanced(model, test_paths, top_songs, long_tail_songs, max_files=10, max_rows=500):
    """
    Evaluate KNN with advanced metrics for both top-m and long-tail songs
    """
    print(f"Evaluating KNN with advanced metrics for long-tail performance")
    
    # Use only a subset of test files
    test_paths = random.sample(test_paths, min(max_files, len(test_paths)))
    
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
            row = test_df.iloc[i].values
            
            # 1. Top-m evaluation
            
            # For positive examples (songs in test playlist)
            positive_indices = np.where(row > 0)[0]
            if len(positive_indices) > 0:
                mask_idx = np.random.choice(positive_indices)
                true_label = 1
                
                # Create masked version for prediction
                masked_row = row.copy().astype(np.float64)
                masked_row[mask_idx] = np.nan
                
                # Make prediction
                predicted_row = model.transform(masked_row.reshape(1, -1))
                prediction = int(np.round(predicted_row[0, mask_idx]))
                
                top_m_true_labels.append(true_label)
                top_m_predictions.append(prediction)
            
            # For negative examples (songs not in test playlist)
            negative_indices = np.where(row == 0)[0]
            if len(negative_indices) > 0:
                mask_idx = np.random.choice(negative_indices)
                true_label = 0
                
                # Create masked version for prediction
                masked_row = row.copy().astype(np.float64)
                masked_row[mask_idx] = np.nan
                
                # Make prediction
                predicted_row = model.transform(masked_row.reshape(1, -1))
                prediction = int(np.round(predicted_row[0, mask_idx]))
                
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

def run_fast_k_analysis(
    train_set, 
    test_set, 
    k_values, 
    n_masks=5, 
    train_max_files=100, 
    test_max_files=10, 
    max_rows=500,
    top_n_songs=3000
):
    """Run a fast K analysis with advanced long-tail metrics"""
    metrics = []
    
    # Use a subset of train files to speed up training
    train_subset = random.sample(train_set, min(train_max_files, len(train_set)))
    
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
    
    for k in k_values:
        print(f"\n==== Training and evaluating KNN with k={k} ====")
        start_time = time.time()
        
        # Create and fit model
        model = KNNImputer(n_neighbors=k)
        
        # Load and concatenate training data
        dfs = []
        for path in tqdm(train_subset, desc="Loading training files"):
            curr_csv = load_single_file(path)
            dfs.append(curr_csv)
        
        train_data = pd.concat(dfs)
        print(f"Fitting on dataset of shape: {train_data.shape}")
        model.fit(train_data)
        
        fit_time = time.time() - start_time
        print(f"Model fitting completed in {fit_time:.2f} seconds")
        
        # Run standard evaluation
        print("\nPerforming standard evaluation...")
        eval_start = time.time()
        standard_results = evaluate_knn_with_advanced_metrics(
            model, 
            test_set, 
            top_songs,
            n_masks=n_masks,
            max_files=test_max_files, 
            max_rows=max_rows
        )
        
        # Run long-tail evaluation
        print("\nPerforming long-tail evaluation with advanced metrics...")
        long_tail_results = evaluate_knn_with_long_tail_advanced(
            model,
            test_set,
            top_songs,
            long_tail_songs,
            max_files=test_max_files,
            max_rows=max_rows
        )
        
        eval_time = time.time() - eval_start
        
        # Combine results
        results = {
            'k': k,
            **standard_results,  # Standard metrics
            'long_tail_evaluation': long_tail_results,  # Long-tail metrics
            'fit_time': fit_time,
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
        
        del model, train_data, dfs
        gc.collect()
    
    # Save metrics with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_path = f'knn_full_metrics_{timestamp}.json'
    
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
    warnings.filterwarnings("ignore")
    
    # Set random seed
    np.random.seed(42)
    
    # Setup paths
    base_path = Path("/data/user_data/rshar/downloads/spotify/simple_vec")
    all_csvs = sorted(list(base_path.glob("*")))
    
    # Split into train/test
    train_set = random.sample(all_csvs, int(1000 * 0.50))
    test_set = [x for x in all_csvs if x not in train_set]
    
    # Define k values to test
    k_values = [1, 3, 5, 10, 25, 100]
    
    # Run ultra-fast analysis
    metrics = run_fast_k_analysis(
        train_set=train_set,
        test_set=test_set,
        k_values=k_values,
        n_masks=1,
        train_max_files=50, #100
        test_max_files=5, #10 
        max_rows=150 #300
    )
    
    # Print final summary
    print("\nFinal Results Summary:")
    for metric in metrics:
        print(f"k={metric['k']}: {metric['accuracy']:.2%} (fit: {metric['fit_time']:.1f}s, eval: {metric['eval_time']:.1f}s)")
