# noise_eval.py - Script to evaluate LSTM model robustness to noise

import torch
import numpy as np
import argparse
from pathlib import Path
import json
import time
from datetime import datetime
import os
import traceback

# Import the core evaluation functionality from lstm_eval_2.py
from lstm_eval_2 import (
    load_model, PlaylistLSTMModel, custom_collate, evaluate_top_k,
    load_song_data, SafeEmbeddingLookup, ImprovedEvalDataset,
    evaluate_with_dynamic_thresholds
)

class NoisyEmbeddingLookup:
    """Extended embedding lookup with noise injection capabilities"""
    
    def __init__(self, base_lookup, noise_type='gaussian', noise_level=0.1, noise_dims=None):
        """
        Initialize noisy embedding lookup
        
        Args:
            base_lookup: The original embedding lookup object
            noise_type: Type of noise ('gaussian', 'uniform', 'dropout', 'adversarial')
            noise_level: Intensity of noise (std dev for gaussian, range for uniform, prob for dropout)
            noise_dims: Dimensions to apply noise to (None=all)
        """
        self.base_lookup = base_lookup
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.noise_dims = noise_dims
        self.embedding_dim = 768  # ModernBERT dimension
        
        # Statistics tracking
        self.total_noise_added = 0
        self.total_embeddings = 0
        self.cache = {}  # Add a cache to improve performance
    
    def get_embedding(self, uri):
        """Get embedding with noise injection"""
        # Check cache first
        uri_str = str(uri)
        if uri_str in self.cache:
            return self.cache[uri_str]
        
        # Get original embedding
        try:
            embedding = self.base_lookup.get_embedding(uri)
            
            # Apply noise
            noisy_embedding = self._add_noise(embedding)
            
            # Track statistics
            self.total_embeddings += 1
            
            # Cache the result
            self.cache[uri_str] = noisy_embedding
            
            # Limit cache size
            if len(self.cache) > 5000:
                # Remove random entries to keep size manageable
                keys_to_remove = np.random.choice(list(self.cache.keys()), 1000, replace=False)
                for k in keys_to_remove:
                    del self.cache[k]
            
            return noisy_embedding
        except Exception as e:
            # Pass through any errors from base lookup
            raise e
    
    def get_batch_embeddings(self, uris):
        """Get batch embeddings with noise"""
        # Get original embeddings
        try:
            embeddings, valid_uris = self.base_lookup.get_batch_embeddings(uris)
            
            if len(embeddings) == 0:
                return np.array([]), []
            
            # Apply noise to each embedding
            noisy_embeddings = np.stack([self._add_noise(emb) for emb in embeddings])
            
            # Track statistics
            self.total_embeddings += len(embeddings)
            
            return noisy_embeddings, valid_uris
        except Exception as e:
            # Pass through any errors
            raise e
    
    def _add_noise(self, embedding):
        """Add noise to embedding based on specified type and level"""
        # Create a copy to avoid modifying original
        noisy_emb = embedding.copy()
        
        # Determine dimensions to apply noise to
        dims = range(self.embedding_dim) if self.noise_dims is None else self.noise_dims
        
        if self.noise_type == 'gaussian':
            # Gaussian noise: N(0, noise_level)
            noise = np.random.normal(0, self.noise_level, size=self.embedding_dim)
            # Apply only to selected dimensions
            if self.noise_dims is not None:
                mask = np.zeros(self.embedding_dim)
                mask[dims] = 1
                noise = noise * mask
            noisy_emb += noise
            self.total_noise_added += np.sum(np.abs(noise))
            
        elif self.noise_type == 'uniform':
            # Uniform noise: U(-noise_level, noise_level)
            noise = np.random.uniform(-self.noise_level, self.noise_level, size=self.embedding_dim)
            # Apply only to selected dimensions
            if self.noise_dims is not None:
                mask = np.zeros(self.embedding_dim)
                mask[dims] = 1
                noise = noise * mask
            noisy_emb += noise
            self.total_noise_added += np.sum(np.abs(noise))
            
        elif self.noise_type == 'dropout':
            # Dropout noise: randomly zero out dimensions with probability noise_level
            mask = np.random.binomial(1, 1-self.noise_level, size=self.embedding_dim)
            # Apply only to selected dimensions
            if self.noise_dims is not None:
                full_mask = np.ones(self.embedding_dim)
                full_mask[dims] = mask[dims]
                mask = full_mask
            noisy_emb = noisy_emb * mask
            self.total_noise_added += np.sum(noisy_emb * (1-mask))
            
        elif self.noise_type == 'adversarial':
            # Simple gradient approximation for adversarial noise
            # Add small random perturbations in random directions
            rand_dirs = np.random.randn(self.embedding_dim)
            rand_dirs = rand_dirs / np.linalg.norm(rand_dirs)
            # Apply only to selected dimensions
            if self.noise_dims is not None:
                mask = np.zeros(self.embedding_dim)
                mask[dims] = 1
                rand_dirs = rand_dirs * mask
            noisy_emb += self.noise_level * rand_dirs
            self.total_noise_added += self.noise_level
            
        else:
            # Unknown noise type - no noise added
            pass
        
        return noisy_emb
    
    def check_availability(self, uris, max_check=500):
        """Pass through to base lookup"""
        return self.base_lookup.check_availability(uris, max_check)
    
    def get_noise_stats(self):
        """Return statistics about applied noise"""
        return {
            'noise_type': self.noise_type,
            'noise_level': self.noise_level,
            'total_embeddings': self.total_embeddings,
            'avg_noise_magnitude': self.total_noise_added / max(1, self.total_embeddings)
        }


def evaluate_with_noise(
    model_path,
    data_dir,
    embedding_path,
    song_csv_path,
    noise_type='gaussian',
    noise_level=0.1,
    noise_dims=None,
    output_path=None,
    batch_size=32,
    max_files=20,
    min_songs=3,  # Added parameter to make dataset creation more flexible
    device="cuda" if torch.cuda.is_available() else "cpu",
    debug=True
):
    """Run evaluation with noise-injected embeddings"""
    start_time = time.time()
    
    print(f"Starting noisy evaluation with {noise_type} noise (level={noise_level})")
    print(f"Model: {model_path}")
    print(f"Data directory: {data_dir}")
    print(f"Embeddings: {embedding_path}")
    print(f"Device: {device}")
    
    # Load model
    try:
        model, model_info = load_model(model_path, device)
        
        # Create base embedding lookup
        base_lookup = SafeEmbeddingLookup(embedding_path)
        
        # Wrap with noisy lookup
        noisy_lookup = NoisyEmbeddingLookup(
            base_lookup, 
            noise_type=noise_type, 
            noise_level=noise_level,
            noise_dims=noise_dims
        )
        
        # Load song data
        top_songs, long_tail_songs, genre_map = load_song_data(song_csv_path)
        
        # Check embedding availability
        top_coverage = noisy_lookup.check_availability(top_songs)
        long_coverage = noisy_lookup.check_availability(long_tail_songs)
        print(f"Embedding coverage: {top_coverage:.1%} of top songs, {long_coverage:.1%} of long-tail songs")
        
        # Inspect the data directory - print first few files
        data_path = Path(data_dir)
        print(f"\nInspecting data directory: {data_path}")
        data_files = []
        for pattern in ["*.json", "*.csv"]:
            for file in data_path.glob(pattern):
                data_files.append(file)
        
        if data_files:
            print(f"Found {len(data_files)} data files. Sample files:")
            for i, file in enumerate(data_files[:5]):
                print(f"  {i+1}. {file}")
        else:
            print(f"WARNING: No .json or .csv files found in {data_path}")
            print(f"Directory contents: {[f.name for f in data_path.iterdir()][:10]}")
        
        # Create dataset with the ImprovedEvalDataset
        print(f"\nCreating dataset with min_songs={min_songs}, max_files={max_files}")
        dataset = ImprovedEvalDataset(
            data_dir=data_dir,
            embedding_lookup=noisy_lookup,
            top_songs=top_songs,
            long_tail_songs=long_tail_songs,
            genre_map=genre_map,
            min_songs=min_songs,  # Lower this value for more flexibility
            max_files=max_files
        )
        
        if len(dataset) == 0:
            print("ERROR: No viable playlists found for evaluation")
            # Try with a lower minimum songs requirement
            if min_songs > 2:
                min_songs = 2
                print(f"\nRetrying with min_songs={min_songs}")
                dataset = ImprovedEvalDataset(
                    data_dir=data_dir,
                    embedding_lookup=noisy_lookup,
                    top_songs=top_songs,
                    long_tail_songs=long_tail_songs,
                    genre_map=genre_map,
                    min_songs=min_songs,
                    max_files=max_files
                )
            
            if len(dataset) == 0:
                print("Still no viable playlists found. Exiting.")
                if output_path:
                    error_result = {
                        'error': 'No viable playlists found',
                        'model_path': model_path,
                        'noise_config': {
                            'type': noise_type,
                            'level': noise_level,
                            'dims': noise_dims if noise_dims is not None else "all"
                        },
                        'embedding_coverage': {
                            'top_songs': top_coverage,
                            'long_tail_songs': long_coverage
                        }
                    }
                    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                    with open(output_path, 'w') as f:
                        json.dump(error_result, f, indent=2)
                return None
        
        print(f"Created dataset with {len(dataset)} playlists")
        
        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=0  # Safer with custom embedding lookup
        )
        
        # Run dynamic threshold evaluation
        print("\nRunning dynamic threshold evaluation...")
        dynamic_results = evaluate_with_dynamic_thresholds(model, dataloader, device)
        
        # Run top-k evaluation
        print("\nRunning top-k evaluation...")
        topk_results = evaluate_top_k(model, dataloader, device=device)
        
        # Get noise statistics
        noise_stats = noisy_lookup.get_noise_stats()
        
        # Combine results
        results = {
            'model_path': model_path,
            'noise_config': {
                'type': noise_type,
                'level': noise_level,
                'dims': noise_dims if noise_dims is not None else "all",
                'stats': noise_stats
            },
            'dynamic_evaluation': dynamic_results,
            'topk_evaluation': topk_results,
            'embedding_coverage': {
                'top_songs': top_coverage,
                'long_tail_songs': long_coverage
            },
            'evaluation_time': time.time() - start_time,
            'model_info': {
                'hidden_dim': model.hidden_dim,
                'num_layers': model.num_layers
            },
            'dataset_info': {
                'size': len(dataset),
                'min_songs': min_songs
            }
        }
        
        # Save results if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            try:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {output_path}")
            except Exception as e:
                print(f"Error saving results: {e}")
                if debug:
                    traceback.print_exc()
        
        return results
        
    except Exception as e:
        print(f"ERROR: Failed to run evaluation: {e}")
        if debug:
            traceback.print_exc()
        return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Evaluate LSTM model with noisy embeddings")
    
    # Required arguments
    parser.add_argument("--model", required=True, help="Path to trained LSTM model")
    parser.add_argument("--data", required=True, help="Path to playlist data directory")
    parser.add_argument("--embeddings", required=True, help="Path to song embeddings file")
    parser.add_argument("--songs", required=True, help="Path to song popularity CSV")
    
    # Noise configuration
    parser.add_argument("--noise-type", choices=['gaussian', 'uniform', 'dropout', 'adversarial'], 
                        default='gaussian', help="Type of noise to add")
    parser.add_argument("--noise-level", type=float, default=0.1, 
                        help="Level of noise (std dev for gaussian, range for uniform, prob for dropout)")
    parser.add_argument("--noise-dims", type=str, default=None, 
                        help="Comma-separated list of dimensions to apply noise to (None=all)")
    
    # Optional arguments
    parser.add_argument("--output", default=None, help="Path to save results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-files", type=int, default=20, help="Maximum number of data files to process")
    parser.add_argument("--min-songs", type=int, default=3, help="Minimum songs per playlist")
    parser.add_argument("--device", default=None, help="Device to run on (default: auto-detect)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with full stack traces")
    
    args = parser.parse_args()
    
    # Process noise dimensions if provided
    noise_dims = None
    if args.noise_dims:
        try:
            noise_dims = [int(d) for d in args.noise_dims.split(',')]
        except:
            print(f"WARNING: Could not parse noise dimensions: {args.noise_dims}")
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run evaluation
    evaluate_with_noise(
        model_path=args.model,
        data_dir=args.data,
        embedding_path=args.embeddings,
        song_csv_path=args.songs,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        noise_dims=noise_dims,
        output_path=args.output,
        batch_size=args.batch_size,
        max_files=args.max_files,
        min_songs=args.min_songs,
        device=device,
        debug=args.debug
    )


if __name__ == "__main__":
    main()