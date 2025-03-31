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
from sklearn.metrics.pairwise import cosine_similarity

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
            import traceback
            traceback.print_exc()
            self.uri_to_idx = {}
            
        # Cache for embeddings to avoid repeated H5 file access
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
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
        for uri in track_uris:
            try:
                embedding = self.get_embedding(uri)
                embeddings.append(embedding)
            except KeyError:
                # Skip URIs that aren't found
                pass
                
        if not embeddings:
            return np.array([])
            
        return np.stack(embeddings)
    
class PlaylistDataset(Dataset):
    """Dataset for playlists with song embeddings"""
    
    def __init__(self, 
                data_dir: str,
                embedding_lookup: SongEmbeddingLookup,
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
        # self.min_frac_input = min_frac_input
        # self.max_frac_input = max_frac_input
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
            if file_path.suffix.lower() == '.csv':
                # Load CSV format
                df = pd.read_csv(file_path, index_col=0)
                self._process_csv_file(df)
            else:
                # Load JSON format
                try:
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
            input_embeddings = self.embedding_lookup.get_batch_embeddings(input_uris)
            test_embeddings = self.embedding_lookup.get_batch_embeddings(test_uris)
            
            if len(input_embeddings) == 0 or len(test_embeddings) == 0:
                # Fall back to a random valid playlist if we couldn't get embeddings
                return self.__getitem__(random.randint(0, len(self) - 1))
            
            # Choose a random test song embedding as target
            target_idx = random.randint(0, len(test_embeddings) - 1)
            target_embedding = test_embeddings[target_idx]
            target_uri = test_uris[target_idx]
            
            return {
                'input_embeddings': input_embeddings.astype(np.float32),
                'lengths': len(input_embeddings),
                'target_embedding': target_embedding.astype(np.float32),
                'target_uri': target_uri,
                'test_uris': test_uris,
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

def train_model(model, train_loader, val_loader=None, epochs=10, lr=0.001, 
                device="cuda" if torch.cuda.is_available() else "cpu",
                save_path="models/lstm_playlist_model.pt"):
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
                torch.save({
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
                }, save_path)
                print(f"Saved best model with validation loss {val_loss:.6f}")
        
        # Record current learning rate
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
    
    # Save final model if no validation
    if not val_loader:
        torch.save({
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
        }, save_path)
        print(f"Saved final model with training loss {train_loss:.6f}")
    
    return history

def evaluate_model(model, test_loader, 
                  device="cuda" if torch.cuda.is_available() else "cpu",
                  embedding_lookup=None):
    """
    Evaluate model performance on test data using in-sample testing
    
    Args:
        model: Trained LSTM model
        test_loader: Test data loader
        device: Device to evaluate on
        embedding_lookup: Song embedding lookup for finding most similar songs
    """
    print(f"Evaluating model on {device}")
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Metrics
    total_loss = 0.0
    avg_similarities = []
    all_similarities = []
    correct_at_k = {1: 0, 5: 0, 10: 0}
    
    # In-sample testing function
    def in_sample_testing_evaluation(predictions, test_embeddings_list, test_uris_list, target_uris):
        batch_losses = []
        batch_similarities = []
        batch_correct_at_k = {1: 0, 5: 0, 10: 0}
        
        # Process each example in the batch
        for i, pred in enumerate(predictions):
            test_embeddings = test_embeddings_list[i]
            test_uris = test_uris_list[i]
            target_uri = target_uris[i]
            
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
            sim_values, sim_indices = torch.sort(similarities, descending=True)
            max_sim = sim_values[0].item()
            all_similarities.append(max_sim)
            batch_similarities.append(max_sim)
            
            # Calculate loss (1 - similarity)
            loss = 1.0 - max_sim
            batch_losses.append(loss)
            
            # Check if target is in top-k
            for k in correct_at_k.keys():
                if k <= len(test_uris):
                    top_k_indices = sim_indices[:k].cpu().numpy()
                    top_k_uris = [test_uris[idx] for idx in top_k_indices]
                    if target_uri in top_k_uris:
                        batch_correct_at_k[k] += 1
        
        # Return metrics
        return (
            torch.stack(batch_losses).mean(), 
            np.mean(batch_similarities),
            batch_correct_at_k
        )
    
    # Evaluation loop
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move data to device
            input_embeddings = batch['input_embeddings'].to(device)
            lengths = batch['lengths']
            test_embeddings_list = batch['test_embeddings']
            test_uris_list = batch['test_uris']
            target_uris = batch['target_uris']
            
            # Forward pass
            predictions = model(input_embeddings, lengths)
            
            # Evaluate using in-sample testing
            loss, batch_sim, batch_correct = in_sample_testing_evaluation(
                predictions, 
                test_embeddings_list, 
                test_uris_list, 
                target_uris
            )
            
            # Update metrics
            total_loss += loss.item() * len(predictions)
            avg_similarities.append(batch_sim)
            
            # Update top-k counts
            for k in correct_at_k.keys():
                correct_at_k[k] += batch_correct[k]
            
            total_samples += len(predictions)
    
    # Calculate average loss
    avg_loss = total_loss / total_samples
    
    # Calculate average similarity
    avg_sim = sum(avg_similarities) / len(avg_similarities) if avg_similarities else 0
    
    # Calculate top-K accuracy
    top_k_accuracy = {k: correct_at_k[k] / total_samples for k in correct_at_k.keys()}
    
    print(f"Test loss: {avg_loss:.6f}")
    print(f"Average similarity: {avg_sim:.6f}")
    print("Top-K accuracy (in-sample testing):")
    for k, acc in top_k_accuracy.items():
        print(f"  Top-{k}: {acc:.4f}")
    
    # Return results
    return {
        'loss': avg_loss,
        'avg_similarity': avg_sim,
        'top_k_accuracy': top_k_accuracy,
        'similarity_scores': all_similarities
    }

def main():
    """Main training function"""
    # Configuration
    config = {
        # 'data_dir': "/data/user_data/rshar/downloads/spotify/simple_vec",
        'data_dir': "/data/user_data/rshar/downloads/spotify/data",
        'embedding_store_path': "data/song_lookup.h5",
        # 'top_songs_path': "data/3000_songs_list.txt",
        'top_songs_path': None,
        'model_save_path': "models/lstm_playlist_model.pt",
        'hidden_dim': 1536,
        'dropout': 0.3,
        'num_layers': 2,
        'batch_size': 64,
        'epochs': 20,
        'learning_rate': 0.001,
        'validation_split': 0.1,
        'max_files': 100,  # Limit for development
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print(f"Using device: {config['device']}")
    
    # Initialize embedding lookup
    embedding_lookup = SongEmbeddingLookup(config['embedding_store_path'])
    
    # Create dataset
    print("Creating dataset...")
    dataset = PlaylistDataset(
        data_dir=config['data_dir'],
        embedding_lookup=embedding_lookup,
        input_frac=0.75,  # Use 75% of songs as input as requested
        min_songs=4,  # At least 4 songs (3 input, 1 for testing)
        top_songs_path=config['top_songs_path'],
        max_files=config['max_files']
    )
    
    # Split into train and validation
    val_size = int(len(dataset) * config['validation_split'])
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Dataset split: {train_size} train, {val_size} validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_playlist_batch,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_playlist_batch,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Create model
    model = PlaylistLSTMModel(
        input_dim=EMBEDDING_DIM,
        hidden_dim=config['hidden_dim'],
        output_dim=EMBEDDING_DIM,
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Print model summary
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        lr=config['learning_rate'],
        device=config['device'],
        save_path=config['model_save_path']
    )
    
    # Save training history
    history_path = os.path.splitext(config['model_save_path'])[0] + '_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']] if 'val_loss' in history else [],
            'learning_rates': [float(x) for x in history['learning_rates']]
        }, f, indent=2)
    
    print(f"Training history saved to {history_path}")
    
    # this didnt work so I am just ignoring for now
    # Evaluate on validation set
    # results = evaluate_model(
    #     model=model,
    #     test_loader=val_loader,
    #     device=config['device'],
    #     embedding_lookup=embedding_lookup
    # )
    
    # Save evaluation results
    # results_path = os.path.splitext(config['model_save_path'])[0] + '_eval_results.json'
    # with open(results_path, 'w') as f:
    #     json.dump({
    #         'loss': float(results['loss']),
    #         'avg_similarity': float(results['avg_similarity']),
    #         'top_k_accuracy': {str(k): float(v) for k, v in results['top_k_accuracy'].items()},
    #         'similarity_histogram': np.histogram(results['similarity_scores'], bins=10, range=(0, 1))[0].tolist()
    #     }, f, indent=2)
    
    # print(f"Evaluation results saved to {results_path}")

if __name__ == "__main__":
    main()