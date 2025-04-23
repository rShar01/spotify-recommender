#!/usr/bin/env python3
# order_eval.py - Script to evaluate LSTM model robustness to song order changes

import torch
import numpy as np
import argparse
from pathlib import Path
import json
import time
from datetime import datetime
import os
import traceback
import random

# Import the core evaluation functionality from lstm_eval_2.py
from lstm_eval_2 import (
    load_model, PlaylistLSTMModel, custom_collate, evaluate_top_k,
    load_song_data, SafeEmbeddingLookup, ImprovedEvalDataset,
    evaluate_with_dynamic_thresholds
)

class OrderPerturbationEmbeddingLookup:
    """Embedding lookup with song order perturbation capabilities"""
    
    def __init__(self, base_lookup, perturbation_type='swap', num_perturbations=1):
        """
        Initialize order perturbation embedding lookup
        
        Args:
            base_lookup: The original embedding lookup object
            perturbation_type: Type of perturbation ('swap', 'shift', 'reverse', 'shuffle')
            num_perturbations: Number of perturbations to apply (e.g., number of swaps)
        """
        self.base_lookup = base_lookup
        self.perturbation_type = perturbation_type
        self.num_perturbations = num_perturbations
        
        # Statistics tracking
        self.total_perturbations = 0
        self.total_playlists = 0
        self.cache = {}  # Cache for individual embeddings
    
    def get_embedding(self, uri):
        """Get embedding for a single URI (pass-through to base lookup)"""
        # Check cache first
        uri_str = str(uri)
        if uri_str in self.cache:
            return self.cache[uri_str]
        
        # Get original embedding
        try:
            embedding = self.base_lookup.get_embedding(uri)
            
            # Cache the result
            self.cache[uri_str] = embedding
            
            # Limit cache size
            if len(self.cache) > 5000:
                # Remove random entries to keep size manageable
                keys_to_remove = np.random.choice(list(self.cache.keys()), 1000, replace=False)
                for k in keys_to_remove:
                    del self.cache[k]
            
            return embedding
        except Exception as e:
            # Pass through any errors from base lookup
            raise e
    
    def get_batch_embeddings(self, uris):
        """Get batch embeddings with order perturbation"""
        # Get original embeddings
        try:
            embeddings, valid_uris = self.base_lookup.get_batch_embeddings(uris)
            
            if len(embeddings) == 0:
                return np.array([]), []
            
            # Apply order perturbation
            perturbed_embeddings, perturbed_uris = self._perturb_order(embeddings, valid_uris)
            
            # Track statistics
            self.total_playlists += 1
            
            return perturbed_embeddings, perturbed_uris
        except Exception as e:
            # Pass through any errors
            raise e
    
    def _perturb_order(self, embeddings, uris):
        """Perturb the order of embeddings based on specified type and level"""
        # Create copies to avoid modifying originals
        perturbed_embeddings = embeddings.copy()
        perturbed_uris = uris.copy()
        
        # Skip perturbation if there are too few items
        if len(embeddings) <= 1:
            return perturbed_embeddings, perturbed_uris
        
        if self.perturbation_type == 'swap':
            # Swap pairs of adjacent songs
            for _ in range(self.num_perturbations):
                if len(perturbed_embeddings) < 2:
                    break
                    
                # Choose a random position to swap (not the last one)
                pos = random.randint(0, len(perturbed_embeddings) - 2)
                
                # Swap embeddings
                perturbed_embeddings[pos], perturbed_embeddings[pos+1] = \
                    perturbed_embeddings[pos+1].copy(), perturbed_embeddings[pos].copy()
                
                # Swap URIs
                perturbed_uris[pos], perturbed_uris[pos+1] = perturbed_uris[pos+1], perturbed_uris[pos]
                
                self.total_perturbations += 1
                
        elif self.perturbation_type == 'shift':
            # Shift a song to a different position
            for _ in range(self.num_perturbations):
                if len(perturbed_embeddings) < 2:
                    break
                    
                # Choose a random song to move
                from_pos = random.randint(0, len(perturbed_embeddings) - 1)
                
                # Choose a random position to move to (different from original)
                to_positions = list(range(len(perturbed_embeddings)))
                to_positions.remove(from_pos)
                to_pos = random.choice(to_positions)
                
                # Store the moved items
                moved_embedding = perturbed_embeddings[from_pos].copy()
                moved_uri = perturbed_uris[from_pos]
                
                # Remove from original position
                perturbed_embeddings = np.delete(perturbed_embeddings, from_pos, axis=0)
                del perturbed_uris[from_pos]
                
                # Insert at new position
                perturbed_embeddings = np.insert(perturbed_embeddings, to_pos, moved_embedding, axis=0)
                perturbed_uris.insert(to_pos, moved_uri)
                
                self.total_perturbations += 1
                
        elif self.perturbation_type == 'reverse':
            # Reverse a segment of the playlist
            for _ in range(self.num_perturbations):
                if len(perturbed_embeddings) < 3:  # Need at least 3 for meaningful reversal
                    break
                
                # Choose a segment length (at least 2)
                seg_len = min(random.randint(2, 3), len(perturbed_embeddings))
                
                # Choose a starting position
                start_pos = random.randint(0, len(perturbed_embeddings) - seg_len)
                end_pos = start_pos + seg_len
                
                # Reverse the segment
                perturbed_embeddings[start_pos:end_pos] = perturbed_embeddings[start_pos:end_pos][::-1].copy()
                perturbed_uris[start_pos:end_pos] = perturbed_uris[start_pos:end_pos][::-1]
                
                self.total_perturbations += 1
                
        elif self.perturbation_type == 'shuffle':
            # Completely shuffle the playlist
            indices = list(range(len(perturbed_embeddings)))
            random.shuffle(indices)
            
            perturbed_embeddings = perturbed_embeddings[indices].copy()
            perturbed_uris = [perturbed_uris[i] for i in indices]
            
            self.total_perturbations += 1
            
        return perturbed_embeddings, perturbed_uris
    
    def check_availability(self, uris, max_check=500):
        """Pass through to base lookup"""
        return self.base_lookup.check_availability(uris, max_check)
    
    def get_perturbation_stats(self):
        """Return statistics about applied perturbations"""
        return {
            'perturbation_type': self.perturbation_type,
            'num_perturbations': self.num_perturbations,
            'total_playlists': self.total_playlists,
            'total_perturbations': self.total_perturbations,
            'avg_perturbations_per_playlist': self.total_perturbations / max(1, self.total_playlists)
        }


def evaluate_with_order_perturbation(
    model_path,
    data_dir,
    embedding_path,
    song_csv_path,
    perturbation_type='swap',
    num_perturbations=1,
    output_path=None,
    batch_size=32,
    max_files=20,
    min_songs=3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    debug=True
):
    """Run evaluation with order-perturbed playlists"""
    start_time = time.time()
    
    print(f"Starting order perturbation evaluation with {perturbation_type} perturbation (count={num_perturbations})")
    print(f"Model: {model_path}")
    print(f"Data directory: {data_dir}")
    print(f"Embeddings: {embedding_path}")
    print(f"Device: {device}")
    
    # Load model
    try:
        model, model_info = load_model(model_path, device)
        
        # Create base embedding lookup
        base_lookup = SafeEmbeddingLookup(embedding_path)
        
        # Wrap with order perturbation lookup
        perturb_lookup = OrderPerturbationEmbeddingLookup(
            base_lookup, 
            perturbation_type=perturbation_type, 
            num_perturbations=num_perturbations
        )
        
        # Load song data
        top_songs, long_tail_songs, genre_map = load_song_data(song_csv_path)
        
        # Check embedding availability
        top_coverage = perturb_lookup.check_availability(top_songs)
        long_coverage = perturb_lookup.check_availability(long_tail_songs)
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
            embedding_lookup=perturb_lookup,
            top_songs=top_songs,
            long_tail_songs=long_tail_songs,
            genre_map=genre_map,
            min_songs=min_songs,
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
                    embedding_lookup=perturb_lookup,
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
                        'perturbation_config': {
                            'type': perturbation_type,
                            'num_perturbations': num_perturbations
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
        
        # Get perturbation statistics
        perturbation_stats = perturb_lookup.get_perturbation_stats()
        
        # Combine results
        results = {
            'model_path': model_path,
            'perturbation_config': {
                'type': perturbation_type,
                'num_perturbations': num_perturbations,
                'stats': perturbation_stats
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
    parser = argparse.ArgumentParser(description="Evaluate LSTM model with song order perturbations")
    
    # Required arguments
    parser.add_argument("--model", required=True, help="Path to trained LSTM model")
    parser.add_argument("--data", required=True, help="Path to playlist data directory")
    parser.add_argument("--embeddings", required=True, help="Path to song embeddings file")
    parser.add_argument("--songs", required=True, help="Path to song popularity CSV")
    
    # Perturbation configuration
    parser.add_argument("--perturbation-type", choices=['swap', 'shift', 'reverse', 'shuffle'], 
                        default='swap', help="Type of order perturbation to apply")
    parser.add_argument("--num-perturbations", type=int, default=1, 
                        help="Number of perturbations to apply (e.g., number of swaps)")
    
    # Optional arguments
    parser.add_argument("--output", default=None, help="Path to save results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-files", type=int, default=20, help="Maximum number of data files to process")
    parser.add_argument("--min-songs", type=int, default=3, help="Minimum songs per playlist")
    parser.add_argument("--device", default=None, help="Device to run on (default: auto-detect)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with full stack traces")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run evaluation
    evaluate_with_order_perturbation(
        model_path=args.model,
        data_dir=args.data,
        embedding_path=args.embeddings,
        song_csv_path=args.songs,
        perturbation_type=args.perturbation_type,
        num_perturbations=args.num_perturbations,
        output_path=args.output,
        batch_size=args.batch_size,
        max_files=args.max_files,
        min_songs=args.min_songs,
        device=device,
        debug=args.debug
    )


if __name__ == "__main__":
    main()