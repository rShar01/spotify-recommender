#!/usr/bin/env python3
"""
Order Perturbation Evaluation Script

Evaluates LSTM model robustness to changes in song order within playlists.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import json
import time
from datetime import datetime
import os
import random
import traceback
from typing import List, Dict, Tuple, Optional

# Import the core evaluation functionality from lstm_eval_2.py
from lstm_eval_2 import (
    load_model, PlaylistLSTMModel, custom_collate, evaluate_top_k,
    load_song_data, SafeEmbeddingLookup, ImprovedEvalDataset,
    evaluate_with_dynamic_thresholds
)

class OrderPerturbationEmbeddingLookup:
    """Embedding lookup with order perturbation capabilities"""

    def __init__(self, base_lookup, perturbation_type='swap', num_swaps=1):
        """
        Initialize order perturbation embedding lookup

        Args:
            base_lookup: The original embedding lookup object
            perturbation_type: Type of perturbation ('swap', 'shift', 'reverse', 'shuffle')
            num_swaps: Number of swaps to perform (for 'swap' type)
        """
        self.base_lookup = base_lookup
        self.perturbation_type = perturbation_type
        self.num_swaps = num_swaps
        self.cache = {}  # Cache for performance

    def get_embedding(self, uri):
        """Get embedding without perturbation (individual songs aren't perturbed)"""
        # Check cache first
        uri_str = str(uri)
        if uri_str in self.cache:
            return self.cache[uri_str]
            
        # Get the original embedding
        embedding = self.base_lookup.get_embedding(uri)
        
        # Cache and return
        self.cache[uri_str] = embedding
        return embedding
    
    def get_batch_embeddings(self, track_uris):
        """Get embeddings for a batch of track URIs without perturbation"""
        return self.base_lookup.get_batch_embeddings(track_uris)
    
    def get_track_info(self, track_uri):
        """Get track info without perturbation"""
        return self.base_lookup.get_track_info(track_uri)
    
    def perturb_playlist_order(self, playlist_embeddings):
        """
        Apply order perturbation to a playlist's embeddings
        
        Args:
            playlist_embeddings: Tensor of shape [seq_len, embedding_dim]
            
        Returns:
            Perturbed playlist embeddings
        """
        # Convert to numpy for easier manipulation
        embeddings = playlist_embeddings.clone().numpy()
        seq_len = embeddings.shape[0]
        
        # Skip perturbation if sequence is too short
        if seq_len <= 1:
            return playlist_embeddings
            
        if self.perturbation_type == 'swap':
            # Swap random pairs of songs
            for _ in range(min(self.num_swaps, seq_len // 2)):
                idx1, idx2 = random.sample(range(seq_len), 2)
                embeddings[idx1], embeddings[idx2] = embeddings[idx2].copy(), embeddings[idx1].copy()
                
        elif self.perturbation_type == 'shift':
            # Shift the playlist by a random amount
            shift = random.randint(1, seq_len - 1)
            embeddings = np.roll(embeddings, shift, axis=0)
            
        elif self.perturbation_type == 'reverse':
            # Reverse the order of songs
            embeddings = embeddings[::-1]
            
        elif self.perturbation_type == 'shuffle':
            # Completely shuffle the playlist
            indices = np.random.permutation(seq_len)
            embeddings = embeddings[indices]
            
        return torch.tensor(embeddings, dtype=playlist_embeddings.dtype)


class OrderPerturbationDataset(ImprovedEvalDataset):
    """Dataset with order perturbation capabilities"""
    
    def __init__(self, 
                 playlists, 
                 embedding_lookup, 
                 min_playlist_length=3, 
                 max_playlist_length=20,
                 perturbation_type='swap',
                 num_swaps=1):
        """Initialize dataset with order perturbation"""
        super().__init__(playlists, embedding_lookup, min_playlist_length, max_playlist_length)
        self.perturbation_type = perturbation_type
        self.num_swaps = num_swaps
        self.order_lookup = OrderPerturbationEmbeddingLookup(
            embedding_lookup, 
            perturbation_type=perturbation_type,
            num_swaps=num_swaps
        )
        
    def __getitem__(self, idx):
        """Get a playlist with order perturbation"""
        playlist = self.playlists[idx]
        track_uris = playlist['tracks']
        
        # Filter out any tracks not in our embedding lookup
        valid_tracks = []
        for uri in track_uris:
            try:
                # This will raise KeyError if the track is not in the lookup
                self.embedding_lookup.get_embedding(uri)
                valid_tracks.append(uri)
            except KeyError:
                continue
                
        # Ensure we have enough tracks
        if len(valid_tracks) < self.min_playlist_length:
            # If this playlist is too short, try another one
            return self.__getitem__((idx + 1) % len(self))
            
        # Truncate if needed
        if len(valid_tracks) > self.max_playlist_length:
            valid_tracks = valid_tracks[:self.max_playlist_length]
            
        # Get embeddings for all tracks
        embeddings = []
        for uri in valid_tracks:
            embedding = self.embedding_lookup.get_embedding(uri)
            embeddings.append(embedding)
            
        # Convert to tensor
        embeddings_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)
        
        # Apply order perturbation
        perturbed_embeddings = self.order_lookup.perturb_playlist_order(embeddings_tensor)
        
        # For evaluation, we'll predict the last track
        input_embeddings = perturbed_embeddings[:-1]
        target_embedding = embeddings_tensor[-1]  # Original last embedding (not perturbed)
        target_uri = valid_tracks[-1]
        
        return {
            'input_embeddings': input_embeddings,
            'target_embedding': target_embedding,
            'target_uri': target_uri,
            'playlist_id': playlist.get('pid', f"synthetic_{idx}"),
            'playlist_name': playlist.get('name', f"Playlist {idx}"),
            'original_length': len(valid_tracks)
        }


def evaluate_order_perturbation(model_path, 
                               playlists_path, 
                               song_data_path,
                               perturbation_type='swap',
                               num_swaps=1,
                               batch_size=32,
                               k_values=[1, 5, 10, 25, 50, 100],
                               device=None):
    """
    Evaluate model robustness to order perturbations
    
    Args:
        model_path: Path to the trained model
        playlists_path: Path to the playlists JSON file
        song_data_path: Path to the song data H5 file
        perturbation_type: Type of perturbation ('swap', 'shift', 'reverse', 'shuffle')
        num_swaps: Number of swaps to perform (for 'swap' type)
        batch_size: Batch size for evaluation
        k_values: List of k values for top-k evaluation
        device: Device to use for evaluation
        
    Returns:
        Dictionary of evaluation results
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
        
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    model.eval()
    
    # Load song data
    song_lookup = SafeEmbeddingLookup(song_data_path)
    
    # Load playlists
    with open(playlists_path, 'r') as f:
        playlists = json.load(f)
        
    # Create dataset with order perturbation
    dataset = OrderPerturbationDataset(
        playlists=playlists,
        embedding_lookup=song_lookup,
        perturbation_type=perturbation_type,
        num_swaps=num_swaps
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate
    )
    
    # Evaluate
    results = evaluate_top_k(model, dataloader, song_lookup, k_values, device)
    
    # Add metadata
    results['perturbation_type'] = perturbation_type
    results['num_swaps'] = num_swaps
    results['model_path'] = str(model_path)
    results['playlists_path'] = str(playlists_path)
    results['song_data_path'] = str(song_data_path)
    results['evaluation_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate LSTM model robustness to order perturbations")
    
    # Required arguments
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--playlists", required=True, help="Path to the playlists JSON file")
    parser.add_argument("--song-data", required=True, help="Path to the song data H5 file")
    
    # Optional arguments
    parser.add_argument("--perturbation-type", default="swap", 
                        choices=["swap", "shift", "reverse", "shuffle"],
                        help="Type of perturbation to apply")
    parser.add_argument("--num-swaps", type=int, default=1,
                        help="Number of swaps to perform (for 'swap' type)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--output", default="order_eval_results.json",
                        help="Path to save evaluation results")
    parser.add_argument("--device", default=None,
                        help="Device to use for evaluation (cuda/cpu)")
    
    args = parser.parse_args()
    
    try:
        # Evaluate
        results = evaluate_order_perturbation(
            model_path=args.model,
            playlists_path=args.playlists,
            song_data_path=args.song_data,
            perturbation_type=args.perturbation_type,
            num_swaps=args.num_swaps,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Evaluation results saved to {args.output}")
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Perturbation Type: {args.perturbation_type}")
        print(f"Number of Swaps: {args.num_swaps}")
        print(f"Number of Playlists: {results['num_playlists']}")
        print("\nTop-K Accuracy:")
        for k in sorted(results['top_k_accuracy'].keys()):
            print(f"  Top-{k}: {results['top_k_accuracy'][k]:.4f}")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        

if __name__ == "__main__":
    main()