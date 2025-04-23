#!/usr/bin/env python3
"""
Playlist Loader Utility

Utility functions for loading and processing playlist data from multiple JSON files.
"""

import json
import os
import glob
from pathlib import Path
import random
from tqdm import tqdm
import argparse
import traceback

def load_playlist_files(data_dir, num_files=None, random_seed=42):
    """
    Load playlist data from multiple JSON files
    
    Args:
        data_dir: Directory containing playlist JSON files
        num_files: Number of files to load (None = all)
        random_seed: Random seed for reproducibility
        
    Returns:
        List of playlists
    """
    # Set random seed
    random.seed(random_seed)
    
    # Find all playlist files
    pattern = os.path.join(data_dir, "mpd.slice.*.json")
    file_paths = glob.glob(pattern)
    
    if not file_paths:
        raise ValueError(f"No playlist files found matching pattern: {pattern}")
        
    print(f"Found {len(file_paths)} playlist files")
    
    # Sample files if requested
    if num_files is not None and num_files < len(file_paths):
        file_paths = random.sample(file_paths, num_files)
        print(f"Sampled {num_files} files")
        
    # Load playlists from files
    all_playlists = []
    
    for file_path in tqdm(file_paths, desc="Loading playlist files"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                playlists = data.get('playlists', [])
                all_playlists.extend(playlists)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            traceback.print_exc()
            
    print(f"Loaded {len(all_playlists)} playlists in total")
    
    return all_playlists


def filter_playlists(playlists, min_tracks=5, max_tracks=50, min_unique_tracks=5):
    """
    Filter playlists based on criteria
    
    Args:
        playlists: List of playlists
        min_tracks: Minimum number of tracks
        max_tracks: Maximum number of tracks
        min_unique_tracks: Minimum number of unique tracks
        
    Returns:
        Filtered list of playlists
    """
    filtered_playlists = []
    
    for playlist in tqdm(playlists, desc="Filtering playlists"):
        # Get tracks
        tracks = playlist.get('tracks', [])
        
        # Check number of tracks
        if len(tracks) < min_tracks or len(tracks) > max_tracks:
            continue
            
        # Check number of unique tracks
        unique_track_uris = set(track.get('track_uri', '') for track in tracks)
        if len(unique_track_uris) < min_unique_tracks:
            continue
            
        # Convert tracks to list of URIs for easier processing
        playlist['track_uris'] = [track.get('track_uri', '') for track in tracks]
        
        # Add to filtered list
        filtered_playlists.append(playlist)
        
    print(f"Filtered to {len(filtered_playlists)} playlists")
    
    return filtered_playlists


def split_playlists(playlists, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split playlists into train, validation, and test sets
    
    Args:
        playlists: List of playlists
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train, val, and test playlists
    """
    # Set random seed
    random.seed(random_seed)
    
    # Shuffle playlists
    shuffled_playlists = playlists.copy()
    random.shuffle(shuffled_playlists)
    
    # Calculate split indices
    n_playlists = len(shuffled_playlists)
    n_train = int(n_playlists * train_ratio)
    n_val = int(n_playlists * val_ratio)
    
    # Split playlists
    train_playlists = shuffled_playlists[:n_train]
    val_playlists = shuffled_playlists[n_train:n_train+n_val]
    test_playlists = shuffled_playlists[n_train+n_val:]
    
    print(f"Split into {len(train_playlists)} train, {len(val_playlists)} val, {len(test_playlists)} test playlists")
    
    return {
        'train': train_playlists,
        'val': val_playlists,
        'test': test_playlists
    }


def convert_to_simple_format(playlists):
    """
    Convert playlists to a simpler format for training/evaluation
    
    Args:
        playlists: List of playlists
        
    Returns:
        List of simplified playlists
    """
    simple_playlists = []
    
    for playlist in playlists:
        # Create simplified playlist
        simple_playlist = {
            'pid': playlist.get('pid', ''),
            'name': playlist.get('name', ''),
            'tracks': playlist.get('track_uris', [])
        }
        
        # Add to list
        simple_playlists.append(simple_playlist)
        
    return simple_playlists


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process playlist data for training/evaluation")
    
    # Required arguments
    parser.add_argument("--data-dir", required=True, help="Directory containing playlist JSON files")
    parser.add_argument("--output-dir", required=True, help="Directory to save processed playlists")
    
    # Optional arguments
    parser.add_argument("--num-files", type=int, default=None, help="Number of files to load (None = all)")
    parser.add_argument("--min-tracks", type=int, default=5, help="Minimum number of tracks per playlist")
    parser.add_argument("--max-tracks", type=int, default=50, help="Maximum number of tracks per playlist")
    parser.add_argument("--min-unique-tracks", type=int, default=5, help="Minimum number of unique tracks per playlist")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of training data")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of validation data")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio of test data")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load playlists
        playlists = load_playlist_files(args.data_dir, args.num_files, args.random_seed)
        
        # Filter playlists
        filtered_playlists = filter_playlists(
            playlists, 
            args.min_tracks, 
            args.max_tracks, 
            args.min_unique_tracks
        )
        
        # Split playlists
        split = split_playlists(
            filtered_playlists, 
            args.train_ratio, 
            args.val_ratio, 
            args.test_ratio, 
            args.random_seed
        )
        
        # Convert to simple format
        train_simple = convert_to_simple_format(split['train'])
        val_simple = convert_to_simple_format(split['val'])
        test_simple = convert_to_simple_format(split['test'])
        
        # Save to files
        train_path = os.path.join(args.output_dir, "train_playlists.json")
        val_path = os.path.join(args.output_dir, "val_playlists.json")
        test_path = os.path.join(args.output_dir, "test_playlists.json")
        
        with open(train_path, 'w') as f:
            json.dump(train_simple, f)
            
        with open(val_path, 'w') as f:
            json.dump(val_simple, f)
            
        with open(test_path, 'w') as f:
            json.dump(test_simple, f)
            
        print(f"Saved processed playlists to {args.output_dir}")
        
    except Exception as e:
        print(f"Error processing playlists: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()