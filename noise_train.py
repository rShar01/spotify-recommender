#!/usr/bin/env python3
# noise_train.py - Modified from lstm.py to support noise during training

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

class NoisySongEmbeddingLookup:
    """Song embedding lookup with noise injection"""
    
    def __init__(self, embedding_store_path: str = "data/song_lookup.h5",
                 noise_type: str = 'gaussian',
                 noise_level: float = 0.1,
                 noise_dims: List[int] = None):
        """
        Initialize song embedding lookup with noise
        
        Args:
            embedding_store_path: Path to H5 file with song embeddings
            noise_type: Type of noise to add ('gaussian', 'uniform', 'dropout')
            noise_level: Amount of noise to add
            noise_dims: Which dimensions to add noise to (None = all)
        """
        self.embedding_store_path = embedding_store_path
        self.uri_index_path = os.path.splitext(embedding_store_path)[0] + '_uri_index.npz'
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.noise_dims = noise_dims
        self._load_uri_index()
        
        # Cache for embeddings to avoid repeated H5 file access
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Noise statistics
        self.noise_applied_count = 0
        self.total_noise = 0.0
        
        print(f"Initialized noisy embedding lookup with {noise_type} noise (level={noise_level})")
    
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
    
    def _add_noise(self, embedding: np.ndarray) -> np.ndarray:
        """Add noise to an embedding based on the specified parameters"""
        # Make a copy to avoid modifying the original
        noisy_embedding = embedding.copy()
        dims = range(EMBEDDING_DIM) if self.noise_dims is None else self.noise_dims
        
        # Apply different types of noise
        if self.noise_type == 'gaussian':
            # Gaussian noise with specified standard deviation
            noise = np.random.normal(0, self.noise_level, size=EMBEDDING_DIM)
            if self.noise_dims is not None:
                # Only add noise to specified dimensions
                mask = np.zeros(EMBEDDING_DIM)
                mask[dims] = 1
                noise = noise * mask
            noisy_embedding += noise
            self.total_noise += np.sum(np.abs(noise))
            
        elif self.noise_type == 'uniform':
            # Uniform noise in range [-noise_level, noise_level]
            noise = np.random.uniform(-self.noise_level, self.noise_level, size=EMBEDDING_DIM)
            if self.noise_dims is not None:
                # Only add noise to specified dimensions
                mask = np.zeros(EMBEDDING_DIM)
                mask[dims] = 1
                noise = noise * mask
            noisy_embedding += noise
            self.total_noise += np.sum(np.abs(noise))
            
        elif self.noise_type == 'dropout':
            # Randomly set dimensions to zero with probability noise_level
            mask = np.random.binomial(1, 1-self.noise_level, size=EMBEDDING_DIM)
            if self.noise_dims is not None:
                # Only apply dropout to specified dimensions
                full_mask = np.ones(EMBEDDING_DIM)
                full_mask[dims] = mask[dims]
                mask = full_mask
            noisy_embedding = noisy_embedding * mask
            self.total_noise += np.sum(embedding * (1-mask))
        
        self.noise_applied_count += 1
        return noisy_embedding
            
    def get_embedding(self, track_uri: str):
        """Get the embedding for a specific track URI with noise"""
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
            
        # Add noise to the embedding
        noisy_embedding = self._add_noise(embedding)
            
        # Add to cache
        self.cache[track_uri] = noisy_embedding
        
        # Limit cache size
        if len(self.cache) > 10000:
            # Remove random items
            to_remove = random.sample(list(self.cache.keys()), 1000)
            for k in to_remove:
                del self.cache[k]
            
        return noisy_embedding
    
    def get_batch_embeddings(self, track_uris: list):
        """Get embeddings for a batch of track URIs with noise"""
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
    
    def get_noise_stats(self):
        """Get statistics about the applied noise"""
        return {
            'type': self.noise_type,
            'level': self.noise_level,
            'dims': self.noise_dims,
            'count': self.noise_applied_count,
            'avg_magnitude': self.total_noise / max(1, self.noise_applied_count),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }

# Use the original SongEmbeddingLookup when no noise is needed
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
            # Get song indices where row has 1 (songs in the playlist)
            song_indices = np.where(row.values == 1)[0]
            
            # Skip playlists with too few or too many songs
            if len(song_indices) < self.min_songs or len(song_indices) > self.max_songs:
                continue
                
            # Get the column names (song URIs)
            column_names = df.columns[song_indices].tolist()
            
            # Filter by top songs if needed
            if self.top_songs:
                column_names = [col for col in column_names if col in self.top_songs]
                
                # Re-check if we have enough songs after filtering
                if len(column_names) < self.min_songs:
                    continue
            
            # Store the playlist with its song URIs
            pid = pid_column[i] if i < len(pid_column) else -1
            self.playlists.append({
                'pid': pid,
                'song_uris': column_names
            })
    
    def _process_json_file(self, data):
        """Process playlist JSON file format from MPD"""
        # Check if this is MPD format
        if 'playlists' in data:
            playlists = data['playlists']
        else:
            playlists = [data]  # Assume it's a single playlist
        
        for playlist in playlists:
            try:
                # Skip playlists with too few or too many songs
                tracks = playlist.get('tracks', [])
                if len(tracks) < self.min_songs or len(tracks) > self.max_songs:
                    continue
                    
                # Extract track URIs
                song_uris = [track.get('track_uri') for track in tracks]
                
                # Filter by top songs if needed
                if self.top_songs:
                    song_uris = [uri for uri in song_uris if uri in self.top_songs]
                    
                    # Re-check if we have enough songs after filtering
                    if len(song_uris) < self.min_songs:
                        continue
                
                # Store the playlist with its song URIs
                self.playlists.append({
                    'pid': playlist.get('pid', -1),
                    'song_uris': song_uris
                })
            except Exception as e:
                print(f"Error processing playlist {playlist.get('pid', '?')}: {e}")
    
    def __len__(self):
        return len(self.playlists)
    
    def __getitem__(self, idx):
        playlist = self.playlists[idx]
        song_uris = playlist['song_uris']
        
        # Shuffle the songs
        random.shuffle(song_uris)
        
        # Calculate number of input songs (75% as specified)
        num_songs = len(song_uris)
        num_input = max(self.min_songs - 1, int(num_songs * self.input_frac))
        
        # Ensure we have at least one song for in-sample testing
        num_input = min(num_input, num_songs - 1)
        
        # Split into input and in-sample testing songs
        input_uris = song_uris[:num_input]
        test_uris = song_uris[num_input:]
        
        # Get embeddings
        try:
            input_embeddings, valid_input_uris = self.embedding_lookup.get_batch_embeddings(input_uris)
            test_embeddings, valid_test_uris = self.embedding_lookup.get_batch_embeddings(test_uris)
            
            if len(input_embeddings) == 0 or len(test_embeddings) == 0:
                # Fall back to a random valid playlist if we couldn't get embeddings
                return self.__getitem__(random.randint(0, len(self) - 1))
            
            # Choose a random test song embedding as target
            target_idx = random.randint(0, len(test_embeddings) - 1)
            target_embedding = test_embeddings[target_idx]
            target_uri = valid_test_uris[target_idx]
            
            return {
                'input_embeddings': input_embeddings.astype(np.float32),
                'lengths': len(input_embeddings),
                'target_embedding': target_embedding.astype(np.float32),
                'target_uri': target_uri,
                'test_uris': valid_test_uris,
                'test_embeddings': test_embeddings.astype(np.float32),
                'pid': playlist['pid']
            }
        except Exception as e:
            print(f"Error processing playlist {playlist['pid']}: {e}")
            # Fall back to a random valid playlist
            return self.__getitem__(random.randint(0, len(self) - 1))

def collate_playlist_batch(batch):
    """Custom collate function for variable-length playlists"""
    # Get max sequence length in this batch
    max_len = max(item['lengths'] for item in batch)
    
    # Prepare tensors
    batch_size = len(batch)
    input_seqs = []
    lengths = torch.zeros(batch_size, dtype=torch.long)
    target_embeddings = torch.zeros(batch_size, EMBEDDING_DIM)
    pids = []
    target_uris = []
    
    # Lists for in-sample testing data
    test_uris_list = []
    test_embeddings_list = []
    
    for i, item in enumerate(batch):
        # Pad sequence if needed
        seq_len = item['lengths']
        input_seq = item['input_embeddings']
        
        if seq_len < max_len:
            padding = np.zeros((max_len - seq_len, EMBEDDING_DIM), dtype=np.float32)
            input_seq = np.concatenate([input_seq, padding], axis=0)
        
        input_seqs.append(torch.tensor(input_seq, dtype=torch.float32))
        lengths[i] = seq_len
        target_embeddings[i] = torch.tensor(item['target_embedding'], dtype=torch.float32)
        
        # Store additional info
        pids.append(item['pid'])
        target_uris.append(item['target_uri'])
        test_uris_list.append(item['test_uris'])
        test_embeddings_list.append(item['test_embeddings'])
    
    # Stack to create batch
    input_seqs = torch.stack(input_seqs)
    
    return {
        'input_embeddings': input_seqs,
        'lengths': lengths,
        'target_embeddings': target_embeddings,
        'target_uris': target_uris,
        'test_uris': test_uris_list,
        'test_embeddings': test_embeddings_list,
        'pids': pids
    }

def train_model(
    model, 
    train_loader, 
    val_loader=None, 
    epochs=10, 
    lr=0.001, 
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="models/lstm_playlist_model.pt",
    noise_config=None  # Added to store noise configuration in the saved model
):
    """
    Train the LSTM model
    
    Args:
        model: LSTM model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs to train
        lr: Learning rate
        device: Device to train on
        save_path: Path to save the model
        noise_config: Noise configuration for tracking
    """
    print(f"Training model on {device} for {epochs} epochs")
    
    # Move model to device
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss function - custom cosine similarity loss for single target
    def cosine_similarity_loss(predictions, targets):
        # Normalize vectors
        pred_norm = torch.nn.functional.normalize(predictions, p=2, dim=1)
        target_norm = torch.nn.functional.normalize(targets, p=2, dim=1)
        
        # Cosine similarity (dot product of normalized vectors)
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        
        # Loss is 1 - similarity (so 0 is perfect match)
        loss = 1 - cos_sim
        
        return loss.mean()
    
    # Loss function that finds the most similar song in test set
    def in_sample_testing_loss(predictions, test_embeddings_list):
        batch_losses = []
        batch_similarities = []
        
        # Process each example in the batch
        for i, pred in enumerate(predictions):
            test_embeddings = test_embeddings_list[i]
            
            # Convert to tensor if needed
            if not isinstance(test_embeddings, torch.Tensor):
                test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32).to(device)
            
            # Normalize prediction
            pred_norm = torch.nn.functional.normalize(pred.unsqueeze(0), p=2, dim=1)
            
            # Normalize test embeddings
            test_norm = torch.nn.functional.normalize(test_embeddings, p=2, dim=1)
            
            # Compute similarities with all test songs
            similarities = torch.matmul(pred_norm, test_norm.t()).squeeze(0)
            
            # Get the highest similarity
            max_sim, max_idx = torch.max(similarities, dim=0)
            batch_similarities.append(max_sim.item())
            
            # Calculate loss (1 - similarity)
            loss = 1.0 - max_sim
            batch_losses.append(loss)
        
        # Average loss across the batch
        return torch.stack(batch_losses).mean(), np.mean(batch_similarities)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    # Create directory for model if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_losses = []
        
        # Training
        progress_bar = tqdm(train_loader)
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            input_embeddings = batch['input_embeddings'].to(device)
            lengths = batch['lengths']
            target_embeddings = batch['target_embeddings'].to(device)
            test_embeddings_list = batch['test_embeddings']
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_embeddings, lengths)
            
            # Calculate loss using in-sample testing approach
            loss, max_sim = in_sample_testing_loss(outputs, test_embeddings_list)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record loss
            train_losses.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item(), 'sim': max_sim})
            
            # Log cache stats occasionally
            if batch_idx % 100 == 0 and hasattr(train_loader.dataset, 'embedding_lookup'):
                lookup = train_loader.dataset.embedding_lookup
                if hasattr(lookup, 'cache_hits') and hasattr(lookup, 'cache_misses'):
                    hits = lookup.cache_hits
                    misses = lookup.cache_misses
                    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
                    print(f"Cache stats: hits={hits}, misses={misses}, hit_rate={hit_rate:.2%}")
                
                # Log noise stats if available
                if hasattr(lookup, 'get_noise_stats'):
                    noise_stats = lookup.get_noise_stats()
                    print(f"Noise stats: count={noise_stats['count']}, avg_magnitude={noise_stats['avg_magnitude']:.4f}")
        
        # Calculate average loss for epoch
        train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(train_loss)
        print(f"Train loss: {train_loss:.6f}")
        
        # Validation
        val_loss = None
        if val_loader:
            model.eval()
            val_losses = []
            val_similarities = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    # Move data to device
                    input_embeddings = batch['input_embeddings'].to(device)
                    lengths = batch['lengths']
                    test_embeddings_list = batch['test_embeddings']
                    
                    # Forward pass
                    outputs = model(input_embeddings, lengths)
                    
                    # Calculate loss using in-sample testing approach
                    loss, max_sim = in_sample_testing_loss(outputs, test_embeddings_list)
                    
                    # Record metrics
                    val_losses.append(loss.item())
                    val_similarities.append(max_sim)
            
            # Calculate average validation loss
            val_loss = sum(val_losses) / len(val_losses)
            history['val_loss'].append(val_loss)
            print(f"Validation loss: {val_loss:.6f}")
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': history,
                    'model_config': {
                        'input_dim': model.input_dim,
                        'hidden_dim': model.hidden_dim,
                        'output_dim': model.output_dim,
                        'num_layers': model.num_layers
                    }
                }
                
                # Add noise config if provided
                if noise_config:
                    model_data['noise_config'] = noise_config
                    
                torch.save(model_data, save_path)
                print(f"Saved best model with validation loss {val_loss:.6f}")
        
        # Record current learning rate
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
    
    # Save final model if no validation
    if not val_loader:
        model_data = {
            'epoch': epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history,
            'model_config': {
                'input_dim': model.input_dim,
                'hidden_dim': model.hidden_dim,
                'output_dim': model.output_dim,
                'num_layers': model.num_layers
            }
        }
        
        # Add noise config if provided
        if noise_config:
            model_data['noise_config'] = noise_config
            
        torch.save(model_data, save_path)
        print(f"Saved final model with training loss {train_loss:.6f}")
    
    return history


def train_with_noise(
    embedding_store_path: str,
    data_dir: str,
    output_dir: str,
    noise_type: str = 'gaussian',
    noise_level: float = 0.1,
    noise_dims: List[int] = None,
    top_songs_path: str = None,
    hidden_dim: int = 1536,
    dropout: float = 0.3,
    num_layers: int = 2,
    batch_size: int = 64,
    epochs: int = 20,
    learning_rate: float = 0.001,
    validation_split: float = 0.1,
    max_files: int = 100,
    min_songs: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Main function to train a model with noise"""
    start_time = time.time()
    
    # Format model filename based on noise parameters
    if noise_dims is None:
        noise_dim_str = "all"
    else:
        noise_dim_str = f"{len(noise_dims)}dims"
        
    model_filename = f"lstm_noise_{noise_type}_{noise_level}_{noise_dim_str}.pt"
    model_save_path = os.path.join(output_dir, model_filename)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create noise configuration
    noise_config = {
        'type': noise_type,
        'level': noise_level,
        'dims': noise_dims,
        'training_time': None
    }
    
    # Initialize noisy embedding lookup
    print(f"Initializing noisy embedding lookup with {noise_type} noise (level={noise_level})")
    embedding_lookup = NoisySongEmbeddingLookup(
        embedding_store_path=embedding_store_path,
        noise_type=noise_type,
        noise_level=noise_level,
        noise_dims=noise_dims
    )
    
    # Create dataset
    print(f"Creating dataset with noisy embeddings (min_songs={min_songs})")
    dataset = PlaylistDataset(
        data_dir=data_dir,
        embedding_lookup=embedding_lookup,
        input_frac=0.75,
        min_songs=min_songs,
        max_songs=100,
        top_songs_path=top_songs_path,
        max_files=max_files
    )
    
    if len(dataset) == 0:
        print(f"ERROR: No valid playlists found. Try lowering min_songs (current value: {min_songs})")
        return None
    
    # Split into train and validation
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Dataset split: {train_size} train, {val_size} validation")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_playlist_batch,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_playlist_batch,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    model = PlaylistLSTMModel(
        input_dim=EMBEDDING_DIM,
        hidden_dim=hidden_dim,
        output_dim=EMBEDDING_DIM,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Print model summary
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        device=device,
        save_path=model_save_path,
        noise_config=noise_config
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    noise_config['training_time'] = training_time
    
    # Save training history with noise information
    history_path = os.path.splitext(model_save_path)[0] + '_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']] if 'val_loss' in history else [],
            'learning_rates': [float(x) for x in history['learning_rates']],
            'noise_config': noise_config,
            'dataset_info': {
                'size': len(dataset),
                'train_size': train_size,
                'val_size': val_size,
                'min_songs': min_songs
            },
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"Training complete in {training_time/60:.1f} minutes")
    print(f"Model saved to {model_save_path}")
    print(f"History saved to {history_path}")
    
    return model_save_path


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description="Train LSTM model with noise augmentation")
    
    # Required arguments
    parser.add_argument("--embeddings", required=True, help="Path to song embeddings file")
    parser.add_argument("--data", required=True, help="Path to playlist data directory")
    parser.add_argument("--output-dir", required=True, help="Directory to save models")
    
    # Noise configuration
    parser.add_argument("--noise-type", choices=['gaussian', 'uniform', 'dropout'], 
                        default='gaussian', help="Type of noise to add")
    parser.add_argument("--noise-level", type=float, default=0.1, 
                        help="Level of noise (std dev for gaussian, range for uniform, prob for dropout)")
    parser.add_argument("--noise-dims", type=str, default=None, 
                        help="Comma-separated list of dimensions to apply noise to (None=all)")
    
    # Optional arguments
    parser.add_argument("--top-songs", default=None, help="Path to top songs list")
    parser.add_argument("--hidden-dim", type=int, default=1536, help="Hidden dimension size")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--max-files", type=int, default=100, help="Maximum number of data files")
    parser.add_argument("--min-songs", type=int, default=4, help="Minimum songs per playlist")
    parser.add_argument("--device", default=None, help="Device to run on (default: auto-detect)")
    
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
    
    # Run training
    train_with_noise(
        embedding_store_path=args.embeddings,
        data_dir=args.data,
        output_dir=args.output_dir,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        noise_dims=noise_dims,
        top_songs_path=args.top_songs,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        validation_split=args.val_split,
        max_files=args.max_files,
        min_songs=args.min_songs,
        device=device
    )


if __name__ == "__main__":
    main()