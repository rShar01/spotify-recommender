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

def load_single_file(path):
    """Load a single CSV file and preprocess it"""
    df = pd.read_csv(path, index_col=0)
    if any(col in df.columns for col in ["pid", "collaborative", "num_followers", "num_tracks"]):
        df = df.drop(["pid", "collaborative", "num_followers", "num_tracks"], axis=1)
    return df

class BatchedSpotifyKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
    
    def load_and_concatenate(self, file_paths, batch_size=50, max_files=None):
        """Load data in batches using parallel processing with optional limit"""
        if max_files is not None:
            file_paths = file_paths[:max_files]
            
        all_data = []
        
        # Sequential loading is often faster for local files due to disk bottlenecks
        for i in tqdm(range(0, len(file_paths), batch_size), desc="Loading data"):
            batch_paths = file_paths[i:i + batch_size]
            batch_dfs = []
            
            for path in batch_paths:
                curr_csv = load_single_file(path)
                batch_dfs.append(curr_csv)
            
            batch_concat = pd.concat(batch_dfs)
            all_data.append(batch_concat)
            
            del batch_dfs
            gc.collect()
        
        print("Concatenating all batches...")
        final_df = pd.concat(all_data)
        del all_data
        gc.collect()
        
        return final_df

    def fit(self, train_paths, batch_size=50, max_files=None):
        """Fit the model on data"""
        print(f"Fitting model with k={self.n_neighbors}")
        full_df = self.load_and_concatenate(train_paths, batch_size, max_files)
        
        print(f"Fitting on dataset of shape: {full_df.shape}")
        print("Fitting KNN imputer...")
        start_time = time.time()
        self.imputer.fit(full_df)
        print(f"Fitting completed in {time.time() - start_time:.2f} seconds")
        
        del full_df
        gc.collect()
        
        return self
    
    def fast_predict(self, test_df, n_masks=5, max_rows=500):
        """Make optimized predictions on a limited number of test rows"""
        # Limit number of rows for speed
        if len(test_df) > max_rows:
            test_df = test_df.iloc[:max_rows]
            
        n_rows = len(test_df)
        n_cols = test_df.shape[1]
        results = []
        
        # Create fixed mask indices instead of random to improve evaluation consistency
        mask_indices = np.arange(0, n_cols, n_cols // n_masks)[:n_masks]
        mask_indices = np.tile(mask_indices, (n_rows, 1))
        
        for mask_idx in range(n_masks):
            # Create float array for NaN support
            test_masked = test_df.values.astype(np.float64).copy()
            
            # Get true values
            true_vals = test_masked[np.arange(n_rows), mask_indices[:, mask_idx]]
            
            # Mask values
            test_masked[np.arange(n_rows), mask_indices[:, mask_idx]] = np.nan
            
            # Make predictions
            predictions = self.imputer.transform(test_masked)
            
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

def evaluate_fast(model, test_paths, n_masks=5, max_files=10, max_rows=500):
    """Perform a fast evaluation on a small subset of test data"""
    print(f"Fast evaluation on max {max_files} files, {max_rows} rows per file")
    
    # Use only a subset of test files
    # test_paths = test_paths[:max_files]
    test_paths = random.sample(test_paths, max_files)
    
    all_results = []
    for path in tqdm(test_paths, desc="Evaluating files"):
        test_df = load_single_file(path)
        batch_results = model.fast_predict(test_df, n_masks=n_masks, max_rows=max_rows)
        
        for result in batch_results:
            result['file'] = path.name
            result['k'] = model.n_neighbors
            all_results.append(result)
    
    # Aggregate results
    total_correct = sum(r['correct'] for r in all_results)
    total_predictions = sum(r['total'] for r in all_results)
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    
    return {
        'k': model.n_neighbors,
        'detailed_results': all_results,
        'accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_predictions': total_predictions
    }

def run_fast_k_analysis(train_set, test_set, k_values, n_masks=5, 
                         train_max_files=100, test_max_files=10, max_rows=500):
    """Run a fast K analysis with limited data for speed"""
    metrics = []
    
    # Use a subset of train files to speed up training
    train_subset = random.sample(train_set, min(train_max_files, len(train_set)))
    
    for k in k_values:
        print(f"\nTraining model with k={k} on {len(train_subset)} files")
        start_time = time.time()
        
        model = BatchedSpotifyKNN(n_neighbors=k)
        model.fit(train_subset, max_files=train_max_files)
        
        fit_time = time.time() - start_time
        print(f"Model fitting completed in {fit_time:.2f} seconds")
        
        eval_start = time.time()
        results = evaluate_fast(model, test_set, n_masks=n_masks, 
                                max_files=test_max_files, max_rows=max_rows)
        eval_time = time.time() - eval_start
        
        # Add timing information
        results['fit_time'] = fit_time
        results['eval_time'] = eval_time
        metrics.append(results)
        
        print(f"Accuracy for k={k}: {results['accuracy']:.2%}")
        print(f"Evaluation completed in {eval_time:.2f} seconds")
        
        del model
        gc.collect()
    
    # Save metrics with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_path = f'knn_metrics_fast_{timestamp}.json'
    
    # Convert to JSON-serializable format
    json_metrics = []
    for m in metrics:
        json_metric = {k: v for k, v in m.items() if k != 'detailed_results'}
        json_metrics.append(json_metric)
        
    with open(metrics_path, 'w') as f:
        json.dump(json_metrics, f, indent=4)
    
    print(f"Metrics saved to: {metrics_path}")
    return metrics

if __name__ == "__main__":
    # Suppress scikit-learn warnings about feature names
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Set random seed
    np.random.seed(42)
    
    # Setup paths
    base_path = Path("/data/user_data/rshar/downloads/spotify/simple_vec")
    all_csvs = sorted(list(base_path.glob("*")))
    
    # Split into train/test
    train_set = random.sample(all_csvs, int(1000 * 0.70))
    test_set = [x for x in all_csvs if x not in train_set]
    
    # Define k values to test
    k_values = [1, 3, 5, 10, 25, 100]
    
    # Run fast analysis
    metrics = run_fast_k_analysis(
        train_set=train_set,
        test_set=test_set,
        k_values=k_values,
        n_masks=1,              # number of predictions per row
        train_max_files=100,    # limit training to 100 files
        test_max_files=10,      # limit testing to 10 files
        max_rows=300            # limit to 500 rows per file
    )
    
    # Print final summary
    print("\nFinal Results Summary:")
    for metric in metrics:
        print(f"k={metric['k']}: {metric['accuracy']:.2%} (fit: {metric['fit_time']:.1f}s, eval: {metric['eval_time']:.1f}s)")