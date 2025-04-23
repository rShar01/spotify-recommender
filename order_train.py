#!/usr/bin/env python3
# order_train.py - Modified from lstm.py to support song order perturbations during training

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import json
import os
import random
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from typing import List, Dict, Tuple, Optional
import pandas as pd
import gc
import time
from datetime import datetime
import argparse

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Constants
EMBEDDING_DIM = 768  # ModernBERT base dimension

class PlaylistLSTMModel(nn.Module):
    """LSTM model for playlist song prediction"""
    
    def __init__(self, 
                 input_dim: int = EMBEDDING_DIM, 
                 hidden_dim: int = 1536, 
                 output_dim: int = EMBEDDING_DIM,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        """
        Initialize LSTM model for playlist prediction
        
        Args:
            input_dim: Dimension of song embeddings
            hidden_dim: Hidden dimension of LSTM
            output_dim: Output dimension (same as song embeddings)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Main LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer to project from hidden dim to output dim
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
               x: torch.Tensor, 
               lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM
        
        Args:
            x: Batch of song sequence embeddings [batch_size, seq_len, input_dim]
            lengths: Length of each sequence in the batch
            
        Returns:
            Song embedding predictions [batch_size, output_dim]
        """
        # Pack padded sequences for efficiency
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Process through LSTM
        packed_output, (hidden, _) = self.lstm(packed_input)
        
        # Use the last hidden state from the top layer
        last_hidden = hidden[-1]
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Project to output dimension
        output = self.fc(last_hidden)
        
        return output

class OrderPerturbationSongEmbeddingLookup:
    """Song embedding lookup with order perturbation capabilities"""
    
    def __init__(self, embedding_store_path: str = "data/song_lookup.h5",
                 perturbation_type: str = 'swap',
                 perturbation_prob: float = 0.5,
                 max_perturbations: int = 2):
        """
        Initialize song embedding lookup with order perturbation
        
        Args:
            embedding_store_path: Path to H5 file with song embeddings
            perturbation_type: Type of perturbation ('swap', 'shift', 'reverse', 'shuffle')
            perturbation_prob: Probability of applying perturbation to a playlist
            max_perturbations: Maximum number of perturbations to apply
        """
        self.embedding_store_path = embedding_store_path
        self.uri_index_path = os.path.splitext(embedding_store_path)[0] + '_uri_index.npz'
        self.perturbation_type = perturbation_type
        self.perturbation_prob = perturbation_prob
        self.max_perturbations = max_perturbations
        self._load_uri_index()
        
        # Cache for embeddings to avoid repeated H5 file access
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Perturbation statistics
        self.perturbations_applied = 0
        self.playlists_processed = 0
        self.playlists_perturbed = 0
        
        print(f"Initialized order perturbation embedding lookup with {perturbation_type} perturbation "
              f"(prob={perturbation_prob}, max={max_perturbations})")
    
    def _load_uri_index(self):
        """Load the URI to index mapping"""
        try:
            data = np.load(self.uri_index_path, allow_pickle=True)
            uri_to_idx_list = data['uri_to_idx']
            self.uri_to_idx = {item[0]: item[1] for item in uri_to_idx_list}
            print(f"Loaded index with {len(self.uri_to_idx)} track URIs")
        except Exception as e:
            print(f"Error loading URI index: {e}")
            import traceback
            traceback.print_exc()
            self.uri_to_idx = {}
    
    def _perturb_order(self, uris):
        """Apply order perturbation to a list of URIs"""
        # Create a copy to avoid modifying the original
        perturbed_uris = uris.copy()
        
        # Skip perturbation if there are too few items
        if len(perturbed_uris) <= 1:
            return perturbed_uris
        
        # Determine number of perturbations to apply
        num_perturbations = random.randint(1, min(self.max_perturbations, len(perturbed_uris) // 2))
        
        if self.perturbation_type == 'swap':
            # Swap pairs of adjacent songs
            for _ in range(num_perturbations):
                if len(perturbed_uris) < 2:
                    break
                    
                # Choose a random position to swap (not the last one)
                pos = random.randint(0, len(perturbed_uris) - 2)
                
                # Swap URIs
                perturbed_uris[pos], perturbed_uris[pos+1] = perturbed_uris[pos+1], perturbed_uris[pos]
                
                self.perturbations_applied += 1
                
        elif self.perturbation_type == 'shift':
            # Shift a song to a different position
            for _ in range(num_perturbations):
                if len(perturbed_uris) < 2:
                    break
                    
                # Choose a random song to move
                from_pos = random.randint(0, len(perturbed_uris) - 1)
                
                # Choose a random position to move to (different from original)
                to_positions = list(range(len(perturbed_uris)))
                to_positions.remove(from_pos)
                to_pos = random.choice(to_positions)
                
                # Store the moved item
                moved_uri = perturbed_uris[from_pos]
                
                # Remove from original position
                del perturbed_uris[from_pos]
                
                # Insert at new position
                perturbed_uris.insert(to_pos, moved_uri)
                
                self.perturbations_applied += 1
                
        elif self.perturbation_type == 'reverse':
            # Reverse a segment of the playlist
            for _ in range(num_perturbations):
                if len(perturbed_uris) < 3:  # Need at least 3 for meaningful reversal
                    break
                
                # Choose a segment length (at least 2)
                seg_len = min(random.randint(2, 3), len(perturbed_uris))
                
                # Choose a starting position
                start_pos = random.randint(0, len(perturbed_uris) - seg_len)
                end_pos = start_pos + seg_len
                
                # Reverse the segment
                perturbed_uris[start_pos:end_pos] = perturbed_uris[start_pos:end_pos][::-1]
                
                self.perturbations_applied += 1
                
        elif self.perturbation_type == 'shuffle':
            # Completely shuffle the playlist
            random.shuffle(perturbed_uris)
            self.perturbations_applied += 1
            
        self.playlists_perturbed += 1
        return perturbed_uris
            
    def get_embedding(self, track_uri: str):
        """Get the embedding for a specific track URI"""
        # Try cache first
        if track_uri in self.cache:
            self.cache_hits += 1
            return self.cache[track_uri]
            
        self.cache_misses += 1
        
        if track_uri not in self.uri_to_idx:
            raise KeyError(f"Track URI {track_uri} not found in the index")
            
        idx = self.uri_to_idx[track_uri]
        
        with h5py.File(self.embedding_store_path, 'r') as f:
            embedding = f['embeddings'][idx].copy()
            
        # Add to cache
        self.cache[track_uri] = embedding
        
        # Limit cache size
        if len(self.cache) > 10000:
            # Remove random items
            to_remove = random.sample(list(self.cache.keys()), 1000)
            for k in to_remove:
                del self.cache[k]
            
        return embedding
    
    def get_batch_embeddings(self, track_uris: list):
        """Get embeddings for a batch of track URIs with potential order perturbation"""
        self.playlists_processed += 1
        
        # Decide whether to apply perturbation
        apply_perturbation = random.random() < self.perturbation_prob
        
        # Apply perturbation if selected
        if apply_perturbation:
            track_uris = self._perturb_order(track_uris)
        
        # Get embeddings for the (potentially perturbed) URIs
        embeddings = []
        valid_uris = []
        for uri in track_uris:
            try:
                embedding = self.get_embedding(uri)
                embeddings.append(embedding)
                valid_uris.append(uri)
            except KeyError:
                # Skip URIs that aren't found
                pass
                
        if not embeddings:
            return np.array([]), []
            
        return np.stack(embeddings), valid_uris
    
    def check_availability(self, uris, max_check=500):
        """Check what percentage of URIs have available embeddings"""
        if not uris:
            return 0.0
        
        sample_size = min(len(uris), max_check)
        check_uris = random.sample(uris, sample_size)
        
        available = 0
        for uri in check_uris:
            if uri in self.uri_to_idx:
                available += 1
        
        return available / sample_size
    
    def get_perturbation_stats(self):
        """Get statistics about the applied perturbations"""
        return {
            'type': self.perturbation_type,
            'probability': self.perturbation_prob,
            'max_perturbations': self.max_perturbations,
            'playlists_processed': self.playlists_processed,
            'playlists_perturbed': self.playlists_perturbed,
            'perturbation_rate': self.playlists_perturbed / max(1, self.playlists_processed),
            'total_perturbations': self.perturbations_applied,
            'avg_perturbations_per_perturbed': self.perturbations_applied / max(1, self.playlists_perturbed),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }

# Use the original SongEmbeddingLookup when no perturbation is needed
class SongEmbeddingLookup:
    """Utility class for looking up song embeddings from the H5 store"""
    
    def __init__(self, embedding_store_path: str = "data/song_lookup.h5"):
        self.embedding_store_path = embedding_store_path
        self.uri_index_path = os.path.splitext(embedding_store_path)[0] + '_uri_index.npz'
        self._load_uri_index()
        
        # Cache for embeddings to avoid repeated H5 file access
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
            
    def _load_uri_index(self):
        """Load the URI to index mapping"""
        try:
            data = np.load(self.uri_index_path, allow_pickle=True)
            uri_to_idx_list = data['uri_to_idx']
            self.uri_to_idx = {item[0]: item[1] for item in uri_to_idx_list}
            print(f"Loaded index with {len(self.uri_to_idx)} track URIs")
        except Exception as e:
            print(f"Error loading URI index: {e}")
            import traceback
            traceback.print_exc()
            self.uri_to_idx = {}
            
    def get_embedding(self, track_uri: str):
        """Get the embedding for a specific track URI"""
        # Try cache first
        if track_uri in self.cache:
            self.cache_hits += 1
            return self.cache[track_uri]
            
        self.cache_misses += 1
        
        if track_uri not in self.uri_to_idx:
            raise KeyError(f"Track URI {track_uri} not found in the index")
            
        idx = self.uri_to_idx[track_uri]
        
        with h5py.File(self.embedding_store_path, 'r') as f:
            embedding = f['embeddings'][idx].copy()
            
        # Add to cache
        self.cache[track_uri] = embedding
        
        # Limit cache size
        if len(self.cache) > 10000:
            # Remove random items
            to_remove = random.sample(list(self.cache.keys()), 1000)
            for k in to_remove:
                del self.cache[k]
            
        return embedding
    
    def get_batch_embeddings(self, track_uris: list):
        """Get embeddings for a batch of track URIs"""
        embeddings = []
        valid_uris = []
        for uri in track_uris:
            try:
                embedding = self.get_embedding(uri)
                embeddings.append(embedding)
                valid_uris.append(uri)
            except KeyError:
                # Skip URIs that aren't found
                pass
                
        if not embeddings:
            return np.array([]), []
            
        return np.stack(embeddings), valid_uris
    
    def check_availability(self, uris, max_check=500):
        """Check what percentage of URIs have available embeddings"""
        if not uris:
            return 0.0
        
        sample_size = min(len(uris), max_check)
        check_uris = random.sample(uris, sample_size)
        
        available = 0
        for uri in check_uris:
            if uri in self.uri_to_idx:
                available += 1
        
        return available / sample_size

class PlaylistDataset(Dataset):
    """Dataset for playlists with song embeddings"""
    
    def __init__(self, 
               data_dir: str,
               embedding_lookup,
               input_frac: float = 0.75,  # Use 75% of songs as input
               min_songs: int = 4,  # Minimum songs (at least 1 for testing)
               max_songs: int = 100,
               max_files: Optional[int] = None,
               top_songs_path: Optional[str] = None):
        """
        Initialize playlist dataset
        
        Args:
            data_dir: Directory containing playlist CSV files
            embedding_lookup: Song embedding lookup object
            input_frac: Fraction of playlist songs to use as input (rest for in-sample testing)
            min_songs: Minimum number of songs in a playlist to be included
            max_songs: Maximum number of songs in a playlist to be considered
            max_files: Maximum number of CSV files to load
            top_songs_path: Path to top songs list (if we want to filter)
        """
        self.data_dir = Path(data_dir)
        self.embedding_lookup = embedding_lookup
        self.input_frac = input_frac
        self.min_songs = min_songs
        self.max_songs = max_songs
        
        # Load top songs list if provided
        self.top_songs = None
        if top_songs_path:
            with open(top_songs_path, 'r') as f:
                self.top_songs = set(eval(f.read()))
                print(f"Loaded {len(self.top_songs)} top songs")
        
        # Find all CSV files
        print(f"Looking for CSV files in {self.data_dir}")
        csv_files = list(self.data_dir.glob("*.json"))
        if not csv_files:
            csv_files = list(self.data_dir.glob("*.csv"))
        
        if max_files and len(csv_files) > max_files:
            csv_files = random.sample(csv_files, max_files)
            
        print(f"Found {len(csv_files)} files, using max {max_files if max_files else 'all'}")
        
        # Process all files to collect valid playlists
        self.playlists = []
        self._load_playlists(csv_files)
        
        print(f"Dataset initialized with {len(self.playlists)} playlists")
    
    def _load_playlists(self, csv_files):
        """Load playlists from CSV files"""
        for file_path in tqdm(csv_files, desc="Loading playlist files"):
            try:
                if file_path.suffix.lower() == '.csv':
                    # Load CSV format
                    df = pd.read_csv(file_path, index_col=0)
                    self._process_csv_file(df)
                else:
                    # Load JSON format
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    self._process_json_file(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def _process_csv_file(self, df):
        """Process playlist CSV file"""
        # Drop metadata columns if they exist
        if any(col in df.columns for col in ["pid", "collaborative", "num_followers", "num_tracks"]):
            pid_column = df["pid"].tolist() if "pid" in df.columns else [-1] * len(df)
            df = df.drop(["pid", "collaborative", "num_followers", "num_tracks"], axis=1, errors='ignore')
            
        # Check each playlist (row)
        for i, row in df.iterrows():
            # Find songs that are in the playlist (value = 1)
            song_indices = np.where(row.values == 1)[0]
            
            # Skip if too few or too many songs
            if len(song_indices) < self.min_songs or len(song_indices) > self.max_songs:
                continue
                
            # Get the URIs of the songs
            song_uris = df.columns[song_indices].tolist()
            
            # If we have a top songs filter, check if at least one song is in the top list
            if self.top_songs and not any(uri in self.top_songs for uri in song_uris):
                continue
                
            # Add to playlists
            self.playlists.append({
                'pid': pid_column[i] if i < len(pid_column) else -1,
                'song_uris': song_uris
            })
    
    def _process_json_file(self, data):
        """Process playlist JSON file"""
        # Handle both single playlist and multiple playlist formats
        playlists = data.get('playlists', [data])
        
        for playlist in playlists:
            # Get tracks
            tracks = playlist.get('tracks', [])
            
            # Skip if too few or too many songs
            if len(tracks) < self.min_songs or len(tracks) > self.max_songs:
                continue
                
            # Extract URIs
            song_uris = []
            for track in tracks:
                uri = track.get('track_uri')
                if uri:
                    song_uris.append(uri)
            
            # Skip if too few valid URIs
            if len(song_uris) < self.min_songs:
                continue
                
            # If we have a top songs filter, check if at least one song is in the top list
            if self.top_songs and not any(uri in self.top_songs for uri in song_uris):
                continue
                
            # Add to playlists
            self.playlists.append({
                'pid': playlist.get('pid', -1),
                'song_uris': song_uris
            })
    
    def __len__(self):
        return len(self.playlists)
    
    def __getitem__(self, idx):
        """Get a playlist with input and target songs"""
        # Try multiple times if we encounter errors
        for _ in range(3):
            try:
                playlist = self.playlists[idx]
                song_uris = playlist['song_uris'].copy()
                
                # Shuffle for randomization
                random.shuffle(song_uris)
                
                # Split into input and target portions
                n_songs = len(song_uris)
                n_input = max(self.min_songs - 1, int(n_songs * self.input_frac))
                n_input = min(n_input, n_songs - 1)  # Ensure at least one song for target
                
                input_uris = song_uris[:n_input]
                target_uris = song_uris[n_input:]
                
                # Get embeddings for input section
                input_embeddings, valid_input_uris = self.embedding_lookup.get_batch_embeddings(input_uris)
                
                if len(input_embeddings) < self.min_songs - 1:
                    # Not enough valid embeddings, try another playlist
                    idx = random.randint(0, len(self.playlists) - 1)
                    continue
                
                # Get embeddings for target section
                target_embeddings, valid_target_uris = self.embedding_lookup.get_batch_embeddings(target_uris)
                
                if len(target_embeddings) == 0:
                    # No valid target songs, try another playlist
                    idx = random.randint(0, len(self.playlists) - 1)
                    continue
                
                # Choose a random target song
                target_idx = random.randint(0, len(target_embeddings) - 1)
                target_embedding = target_embeddings[target_idx]
                target_uri = valid_target_uris[target_idx]
                
                return {
                    'pid': playlist['pid'],
                    'input_embeddings': input_embeddings.astype(np.float32),
                    'input_length': len(input_embeddings),
                    'target_embedding': target_embedding.astype(np.float32),
                    'target_uri': target_uri
                }
                
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                idx = random.randint(0, len(self.playlists) - 1)
        
        # If we failed multiple times, return a dummy item
        print(f"Failed to get valid item after multiple attempts for index {idx}")
        dummy_embedding = np.zeros((EMBEDDING_DIM,), dtype=np.float32)
        return {
            'pid': -1,
            'input_embeddings': np.array([dummy_embedding, dummy_embedding]).astype(np.float32),
            'input_length': 2,
            'target_embedding': dummy_embedding,
            'target_uri': "dummy_uri"
        }

def custom_collate(batch):
    """Custom collate function for variable length sequences"""
    # Filter out any None items
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None
    
    # Extract components
    pids = [item['pid'] for item in batch]
    input_embeddings = [torch.tensor(item['input_embeddings']) for item in batch]
    input_lengths = torch.tensor([item['input_length'] for item in batch])
    target_embeddings = torch.stack([torch.tensor(item['target_embedding']) for item in batch])
    target_uris = [item['target_uri'] for item in batch]
    
    # Pad sequences
    padded_inputs = pad_sequence(input_embeddings, batch_first=True)
    
    return {
        'pids': pids,
        'input_embeddings': padded_inputs,
        'input_lengths': input_lengths,
        'target_embeddings': target_embeddings,
        'target_uris': target_uris
    }

def train_model(
    data_dir: str,
    embedding_path: str,
    output_dir: str,
    perturbation_type: str = 'swap',
    perturbation_prob: float = 0.5,
    max_perturbations: int = 2,
    hidden_dim: int = 1536,
    num_layers: int = 2,
    dropout: float = 0.3,
    batch_size: int = 64,
    epochs: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    max_files: Optional[int] = None,
    min_songs: int = 4,
    device: str = None,
    checkpoint_interval: int = 1,
    top_songs_path: Optional[str] = None
):
    """
    Train LSTM model with order perturbation for robustness
    
    Args:
        data_dir: Directory with playlist data
        embedding_path: Path to song embeddings H5 file
        output_dir: Directory to save model and logs
        perturbation_type: Type of order perturbation ('swap', 'shift', 'reverse', 'shuffle')
        perturbation_prob: Probability of applying perturbation to a playlist
        max_perturbations: Maximum number of perturbations to apply
        hidden_dim: Hidden dimension of LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        max_files: Maximum number of data files to use
        min_songs: Minimum songs per playlist
        device: Device to train on ('cuda' or 'cpu')
        checkpoint_interval: Save model every N epochs
        top_songs_path: Path to top songs list (optional)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize embedding lookup with order perturbation
    embedding_lookup = OrderPerturbationSongEmbeddingLookup(
        embedding_store_path=embedding_path,
        perturbation_type=perturbation_type,
        perturbation_prob=perturbation_prob,
        max_perturbations=max_perturbations
    )
    
    # Create dataset
    dataset = PlaylistDataset(
        data_dir=data_dir,
        embedding_lookup=embedding_lookup,
        min_songs=min_songs,
        max_files=max_files,
        top_songs_path=top_songs_path
    )
    
    if len(dataset) == 0:
        print("ERROR: No valid playlists found. Check data directory and minimum songs requirement.")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=0  # Safer with custom embedding lookup
    )
    
    # Initialize model
    model = PlaylistLSTMModel(
        input_dim=EMBEDDING_DIM,
        hidden_dim=hidden_dim,
        output_dim=EMBEDDING_DIM,
        num_layers=num_layers,
        dropout=dropout
    )
    model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Cosine similarity loss
    def cosine_similarity_loss(predictions, targets):
        # Normalize predictions and targets
        normalized_predictions = torch.nn.functional.normalize(predictions, p=2, dim=1)
        normalized_targets = torch.nn.functional.normalize(targets, p=2, dim=1)
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarities = torch.sum(normalized_predictions * normalized_targets, dim=1)
        
        # Loss is negative similarity (we want to maximize similarity)
        loss = -torch.mean(similarities)
        
        return loss
    
    # Training loop
    print(f"Starting training for {epochs} epochs")
    
    # Track metrics
    train_losses = []
    train_similarities = []
    
    # Training start time
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        epoch_similarities = []
        
        # Progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for batch in progress_bar:
            if batch is None:
                continue
                
            # Move data to device
            input_embeddings = batch['input_embeddings'].to(device)
            input_lengths = batch['input_lengths']
            target_embeddings = batch['target_embeddings'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(input_embeddings, input_lengths)
            
            # Compute loss
            loss = cosine_similarity_loss(predictions, target_embeddings)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Compute similarity for monitoring
            with torch.no_grad():
                normalized_predictions = torch.nn.functional.normalize(predictions, p=2, dim=1)
                normalized_targets = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)
                batch_similarity = torch.mean(torch.sum(normalized_predictions * normalized_targets, dim=1)).item()
            
            # Track metrics
            epoch_losses.append(loss.item())
            epoch_similarities.append(batch_similarity)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'sim': f"{batch_similarity:.4f}"
            })
        
        # Compute epoch metrics
        epoch_loss = np.mean(epoch_losses)
        epoch_similarity = np.mean(epoch_similarities)
        
        train_losses.append(epoch_loss)
        train_similarities.append(epoch_similarity)
        
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}, Similarity: {epoch_similarity:.4f}")
        
        # Save checkpoint
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            # Get perturbation stats
            perturbation_stats = embedding_lookup.get_perturbation_stats()
            
            # Create model info
            model_info = {
                'epoch': epoch,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout,
                'embedding_dim': EMBEDDING_DIM,
                'perturbation_config': {
                    'type': perturbation_type,
                    'probability': perturbation_prob,
                    'max_perturbations': max_perturbations,
                    'stats': perturbation_stats
                },
                'training_stats': {
                    'loss': float(epoch_loss),
                    'similarity': float(epoch_similarity),
                    'loss_history': [float(l) for l in train_losses],
                    'similarity_history': [float(s) for s in train_similarities]
                },
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'training_time': time.time() - start_time
            }
            
            # Save model
            checkpoint_path = output_dir / f"lstm_order_robust_e{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'model_info': model_info
            }, checkpoint_path)
            
            # Save model info separately
            info_path = output_dir / f"lstm_order_robust_e{epoch}_info.json"
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final model path
    final_model_path = output_dir / "lstm_order_robust_final.pt"
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_losses[-1],
        'model_info': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'embedding_dim': EMBEDDING_DIM,
            'perturbation_config': {
                'type': perturbation_type,
                'probability': perturbation_prob,
                'max_perturbations': max_perturbations,
                'stats': embedding_lookup.get_perturbation_stats()
            },
            'training_stats': {
                'loss_history': [float(l) for l in train_losses],
                'similarity_history': [float(s) for s in train_similarities]
            },
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'training_time': time.time() - start_time
        }
    }, final_model_path)
    
    print(f"Training completed. Final model saved to {final_model_path}")
    print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes")
    
    return final_model_path

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train LSTM model with song order perturbations")
    
    # Required arguments
    parser.add_argument("--data", required=True, help="Path to playlist data directory")
    parser.add_argument("--embeddings", required=True, help="Path to song embeddings file")
    parser.add_argument("--output", required=True, help="Directory to save model and logs")
    
    # Perturbation configuration
    parser.add_argument("--perturbation-type", choices=['swap', 'shift', 'reverse', 'shuffle'], 
                        default='swap', help="Type of order perturbation to apply")
    parser.add_argument("--perturbation-prob", type=float, default=0.5, 
                        help="Probability of applying perturbation to a playlist")
    parser.add_argument("--max-perturbations", type=int, default=2, 
                        help="Maximum number of perturbations to apply")
    
    # Model configuration
    parser.add_argument("--hidden-dim", type=int, default=1536, help="Hidden dimension of LSTM")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for regularization")
    
    # Data configuration
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of data files to use")
    parser.add_argument("--min-songs", type=int, default=4, help="Minimum songs per playlist")
    parser.add_argument("--top-songs", default=None, help="Path to top songs list (optional)")
    
    # Other options
    parser.add_argument("--device", default=None, help="Device to train on ('cuda' or 'cpu')")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Save model every N epochs")
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        data_dir=args.data,
        embedding_path=args.embeddings,
        output_dir=args.output,
        perturbation_type=args.perturbation_type,
        perturbation_prob=args.perturbation_prob,
        max_perturbations=args.max_perturbations,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_files=args.max_files,
        min_songs=args.min_songs,
        device=args.device,
        checkpoint_interval=args.checkpoint_interval,
        top_songs_path=args.top_songs
    )

if __name__ == "__main__":
    main()