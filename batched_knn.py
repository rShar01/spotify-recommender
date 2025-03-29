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

def run_fast_k_analysis(train_set, test_set, k_values, n_masks=1, 
                         max_train_files=100, max_test_files=10, max_rows=300):
    """Run a very fast k analysis optimized for speed"""
    metrics = []
    
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"Analyzing k={k}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # Create and fit model
        model = FastSpotifyKNN(n_neighbors=k)
        model.fit(train_set, max_files=max_train_files)
        
        fit_time = time.time() - start_time
        print(f"Model fitting completed in {fit_time:.2f} seconds")
        
        # Evaluate
        eval_start = time.time()
        results = model.fast_evaluate(
            test_set, 
            n_masks=n_masks,
            max_files=max_test_files, 
            max_rows=max_rows
        )
        eval_time = time.time() - eval_start
        
        # Add timing information
        results['fit_time'] = fit_time
        results['eval_time'] = eval_time
        metrics.append(results)
        
        print(f"Accuracy for k={k}: {results['accuracy']:.2%}")
        print(f"Evaluation completed in {eval_time:.2f} seconds")
        print(f"Total time: {fit_time + eval_time:.2f} seconds")
        
        # Clear memory
        del model
        gc.collect()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_path = f'knn_metrics_fast_{timestamp}.json'
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nMetrics saved to: {metrics_path}")
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
        max_train_files=100,
        max_test_files=10,
        max_rows=300
    )
    
    # Print final summary
    print("\nFinal Results Summary:")
    for metric in metrics:
        print(f"k={metric['k']}: {metric['accuracy']:.2%} (fit: {metric['fit_time']:.1f}s, eval: {metric['eval_time']:.1f}s)")