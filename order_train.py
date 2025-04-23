#!/usr/bin/env python3
"""
Order-Robust LSTM Training Script

Trains LSTM models to be robust to changes in song order within playlists.
"""

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
import traceback

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
        self.dropout_rate = dropout

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        """
        Forward pass

        Args:
            x: Batch of sequences [batch_size, seq_len, input_dim]
            lengths: Length of each sequence in the batch

        Returns:
            Predicted next song embedding [batch_size, output_dim]
        """
        # Pack padded sequences
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Pass through LSTM
        packed_output, (hidden, _) = self.lstm(packed_input)
        
        # Get the last hidden state
        last_hidden = hidden[-1]  # [batch_size, hidden_dim]
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Pass through output layer
        output = self.fc(last_hidden)  # [batch_size, output_dim]
        
        return output


class SongEmbeddingLookup:
    """Utility class for looking up song embeddings from the H5 store"""
    
    def __init__(self, embedding_store_path: str = "data/song_lookup.h5"):
        self.embedding_store_path = embedding_store_path
        self.uri_index_path = os.path.splitext(embedding_store_path)[0] + '_uri_index.npz'
        self._load_uri_index()
        
    def _load_uri_index(self):
        """Load the URI to index mapping"""
        try:
            data = np.load(self.uri_index_path, allow_pickle=True)
            uri_to_idx_list = data['uri_to_idx']
            self.uri_to_idx = {item[0]: item[1] for item in uri_to_idx_list}
            print(f"Loaded index with {len(self.uri_to_idx)} track URIs")
        except Exception as e:
            print(f"Error loading URI index: {e}")
            print(traceback.format_exc())
            self.uri_to_idx = {}
        
    def get_embedding(self, track_uri: str):
        """Get the embedding for a specific track URI"""
        if track_uri not in self.uri_to_idx:
            raise KeyError(f"Track URI {track_uri} not found in the index")
            
        idx = self.uri_to_idx[track_uri]
        
        with h5py.File(self.embedding_store_path, 'r') as f:
            embedding = f['embeddings'][idx]
            
        return embedding
    
    def get_batch_embeddings(self, track_uris: list):
        """Get embeddings for a batch of track URIs"""
        indices = [self.uri_to_idx[uri] for uri in track_uris if uri in self.uri_to_idx]
        
        if not indices:
            return np.array([])
            
        with h5py.File(self.embedding_store_path, 'r') as f:
            embeddings = f['embeddings'][indices]
            
        return embeddings


class OrderPerturbationDataset(Dataset):
    """Dataset with order perturbation for training"""
    
    def __init__(self, 
                 playlists, 
                 embedding_lookup,
                 min_playlist_length=3,
                 max_playlist_length=20,
                 perturbation_prob=0.5,
                 perturbation_types=None,
                 max_swaps=2):
        """
        Initialize dataset with order perturbation
        
        Args:
            playlists: List of playlist dictionaries
            embedding_lookup: SongEmbeddingLookup instance
            min_playlist_length: Minimum playlist length to include
            max_playlist_length: Maximum playlist length to use
            perturbation_prob: Probability of applying perturbation
            perturbation_types: List of perturbation types to use
            max_swaps: Maximum number of swaps to perform
        """
        self.playlists = playlists
        self.embedding_lookup = embedding_lookup
        self.min_playlist_length = min_playlist_length
        self.max_playlist_length = max_playlist_length
        self.perturbation_prob = perturbation_prob
        
        # Default perturbation types if none provided
        if perturbation_types is None:
            self.perturbation_types = ['swap', 'shift', 'reverse', 'shuffle']
        else:
            self.perturbation_types = perturbation_types
            
        self.max_swaps = max_swaps
        
        # Filter playlists by length
        self.valid_playlist_indices = []
        for i, playlist in enumerate(playlists):
            if len(playlist.get('tracks', [])) >= min_playlist_length:
                self.valid_playlist_indices.append(i)
                
        print(f"Found {len(self.valid_playlist_indices)} valid playlists out of {len(playlists)}")
        
    def __len__(self):
        return len(self.valid_playlist_indices)
    
    def perturb_playlist_order(self, track_uris):
        """
        Apply order perturbation to a playlist's track URIs
        
        Args:
            track_uris: List of track URIs
            
        Returns:
            Perturbed list of track URIs
        """
        # Make a copy to avoid modifying the original
        perturbed_uris = track_uris.copy()
        
        # Skip perturbation if sequence is too short
        if len(perturbed_uris) <= 1:
            return perturbed_uris
            
        # Randomly select perturbation type
        perturbation_type = random.choice(self.perturbation_types)
        
        if perturbation_type == 'swap':
            # Swap random pairs of songs
            num_swaps = random.randint(1, min(self.max_swaps, len(perturbed_uris) // 2))
            for _ in range(num_swaps):
                idx1, idx2 = random.sample(range(len(perturbed_uris)), 2)
                perturbed_uris[idx1], perturbed_uris[idx2] = perturbed_uris[idx2], perturbed_uris[idx1]
                
        elif perturbation_type == 'shift':
            # Shift the playlist by a random amount
            shift = random.randint(1, len(perturbed_uris) - 1)
            perturbed_uris = perturbed_uris[shift:] + perturbed_uris[:shift]
            
        elif perturbation_type == 'reverse':
            # Reverse the order of songs
            perturbed_uris = perturbed_uris[::-1]
            
        elif perturbation_type == 'shuffle':
            # Completely shuffle the playlist
            random.shuffle(perturbed_uris)
            
        return perturbed_uris
    
    def __getitem__(self, idx):
        """Get a playlist with potential order perturbation"""
        playlist_idx = self.valid_playlist_indices[idx]
        playlist = self.playlists[playlist_idx]
        track_uris = playlist.get('tracks', [])
        
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
            
        # Apply order perturbation with probability
        if random.random() < self.perturbation_prob:
            perturbed_tracks = self.perturb_playlist_order(valid_tracks[:-1])
            # Keep the target track (last one) unchanged
            input_tracks = perturbed_tracks
            target_track = valid_tracks[-1]
        else:
            # No perturbation
            input_tracks = valid_tracks[:-1]
            target_track = valid_tracks[-1]
            
        # Get embeddings
        input_embeddings = []
        for uri in input_tracks:
            try:
                embedding = self.embedding_lookup.get_embedding(uri)
                input_embeddings.append(embedding)
            except KeyError:
                # Skip if embedding not found
                continue
                
        # Get target embedding
        try:
            target_embedding = self.embedding_lookup.get_embedding(target_track)
        except KeyError:
            # If target embedding not found, try another playlist
            return self.__getitem__((idx + 1) % len(self))
            
        # Convert to tensors
        input_embeddings = torch.tensor(np.array(input_embeddings), dtype=torch.float32)
        target_embedding = torch.tensor(target_embedding, dtype=torch.float32)
        
        return {
            'input_embeddings': input_embeddings,
            'target_embedding': target_embedding,
            'input_length': len(input_embeddings),
            'playlist_id': playlist.get('pid', f"synthetic_{idx}")
        }


def custom_collate(batch):
    """
    Custom collate function for variable length sequences
    
    Args:
        batch: List of dictionaries from dataset
        
    Returns:
        Batched tensors with padding
    """
    # Sort batch by sequence length (descending)
    batch = sorted(batch, key=lambda x: x['input_length'], reverse=True)
    
    # Get input embeddings and lengths
    input_embeddings = [item['input_embeddings'] for item in batch]
    input_lengths = torch.tensor([item['input_length'] for item in batch])
    
    # Pad sequences
    padded_inputs = pad_sequence(input_embeddings, batch_first=True)
    
    # Get target embeddings
    target_embeddings = torch.stack([item['target_embedding'] for item in batch])
    
    # Get playlist IDs
    playlist_ids = [item['playlist_id'] for item in batch]
    
    return {
        'input_embeddings': padded_inputs,
        'input_lengths': input_lengths,
        'target_embeddings': target_embeddings,
        'playlist_ids': playlist_ids
    }


def cosine_similarity_loss(predictions, targets):
    """
    Cosine similarity loss function
    
    Args:
        predictions: Predicted embeddings [batch_size, embedding_dim]
        targets: Target embeddings [batch_size, embedding_dim]
        
    Returns:
        Loss value (1 - cosine similarity)
    """
    # Normalize embeddings
    predictions_norm = torch.nn.functional.normalize(predictions, p=2, dim=1)
    targets_norm = torch.nn.functional.normalize(targets, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = torch.sum(predictions_norm * targets_norm, dim=1)
    
    # Loss is 1 - cosine similarity
    loss = 1 - cosine_sim
    
    return loss.mean()


def train_order_robust_model(train_playlists_path,
                            val_playlists_path,
                            song_data_path,
                            output_dir,
                            perturbation_prob=0.5,
                            perturbation_types=None,
                            max_swaps=2,
                            batch_size=64,
                            num_epochs=10,
                            learning_rate=0.001,
                            hidden_dim=1536,
                            num_layers=2,
                            dropout=0.3,
                            device=None,
                            save_every=1):
    """
    Train an order-robust LSTM model
    
    Args:
        train_playlists_path: Path to training playlists JSON
        val_playlists_path: Path to validation playlists JSON
        song_data_path: Path to song data H5 file
        output_dir: Directory to save model checkpoints
        perturbation_prob: Probability of applying perturbation
        perturbation_types: List of perturbation types to use
        max_swaps: Maximum number of swaps to perform
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        hidden_dim: Hidden dimension of LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        device: Device to use for training
        save_every: Save model every N epochs
        
    Returns:
        Trained model and training history
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
        
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load song data
    song_lookup = SongEmbeddingLookup(song_data_path)
    
    # Load playlists
    with open(train_playlists_path, 'r') as f:
        train_playlists = json.load(f)
        
    with open(val_playlists_path, 'r') as f:
        val_playlists = json.load(f)
        
    # Create datasets
    train_dataset = OrderPerturbationDataset(
        playlists=train_playlists,
        embedding_lookup=song_lookup,
        perturbation_prob=perturbation_prob,
        perturbation_types=perturbation_types,
        max_swaps=max_swaps
    )
    
    val_dataset = OrderPerturbationDataset(
        playlists=val_playlists,
        embedding_lookup=song_lookup,
        perturbation_prob=perturbation_prob,
        perturbation_types=perturbation_types,
        max_swaps=max_swaps
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        # Progress bar for training
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in train_pbar:
            # Move batch to device
            input_embeddings = batch['input_embeddings'].to(device)
            input_lengths = batch['input_lengths']
            target_embeddings = batch['target_embeddings'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(input_embeddings, input_lengths)
            
            # Compute loss
            loss = cosine_similarity_loss(predictions, target_embeddings)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Record loss
            train_losses.append(loss.item())
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Compute average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        
        # Progress bar for validation
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_pbar:
                # Move batch to device
                input_embeddings = batch['input_embeddings'].to(device)
                input_lengths = batch['input_lengths']
                target_embeddings = batch['target_embeddings'].to(device)
                
                # Forward pass
                predictions = model(input_embeddings, input_lengths)
                
                # Compute loss
                loss = cosine_similarity_loss(predictions, target_embeddings)
                
                # Record loss
                val_losses.append(loss.item())
                
                # Update progress bar
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        # Compute average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(output_dir, f"best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'perturbation_prob': perturbation_prob,
                'perturbation_types': perturbation_types,
                'max_swaps': max_swaps,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout
            }, model_path)
            print(f"Saved best model to {model_path}")
            
        # Save checkpoint every N epochs
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'perturbation_prob': perturbation_prob,
                'perturbation_types': perturbation_types,
                'max_swaps': max_swaps,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': history['train_loss'][-1],
        'val_loss': history['val_loss'][-1],
        'perturbation_prob': perturbation_prob,
        'perturbation_types': perturbation_types,
        'max_swaps': max_swaps,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")
    
    return model, history


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train order-robust LSTM model")
    
    # Required arguments
    parser.add_argument("--train-playlists", required=True, help="Path to training playlists JSON")
    parser.add_argument("--val-playlists", required=True, help="Path to validation playlists JSON")
    parser.add_argument("--song-data", required=True, help="Path to song data H5 file")
    parser.add_argument("--output-dir", required=True, help="Directory to save model checkpoints")
    
    # Optional arguments
    parser.add_argument("--perturbation-prob", type=float, default=0.5,
                        help="Probability of applying perturbation")
    parser.add_argument("--perturbation-types", nargs='+',
                        choices=["swap", "shift", "reverse", "shuffle"],
                        default=["swap", "shift", "reverse", "shuffle"],
                        help="Types of perturbation to use")
    parser.add_argument("--max-swaps", type=int, default=2,
                        help="Maximum number of swaps to perform")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    parser.add_argument("--hidden-dim", type=int, default=1536,
                        help="Hidden dimension of LSTM")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability")
    parser.add_argument("--device", default=None,
                        help="Device to use for training (cuda/cpu)")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save model every N epochs")
    
    args = parser.parse_args()
    
    try:
        # Train model
        train_order_robust_model(
            train_playlists_path=args.train_playlists,
            val_playlists_path=args.val_playlists,
            song_data_path=args.song_data,
            output_dir=args.output_dir,
            perturbation_prob=args.perturbation_prob,
            perturbation_types=args.perturbation_types,
            max_swaps=args.max_swaps,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=args.device,
            save_every=args.save_every
        )
        
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        

if __name__ == "__main__":
    main()