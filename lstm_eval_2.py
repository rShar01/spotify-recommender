#!/usr/bin/env python3
"""
Improved LSTM Model Evaluation Script

Evaluates LSTM model for playlist prediction with:
1. Dynamic thresholds based on similarity distributions
2. Better negative examples
3. Consistent handling between top and long-tail songs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import json
import os
import random
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import pandas as pd
import time
from datetime import datetime
import argparse
import traceback
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Constants
EMBEDDING_DIM = 768  # ModernBERT base dimension


class PlaylistLSTMModel(nn.Module):
    """LSTM model for playlist song prediction"""
    
    def __init__(self, 
                 input_dim=EMBEDDING_DIM, 
                 hidden_dim=1536, 
                 output_dim=EMBEDDING_DIM,
                 num_layers=2,
                 dropout=0.3):
        """Initialize LSTM model for playlist prediction"""
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
    
    def forward(self, x, lengths):
        """Forward pass through the LSTM"""
        # Pack padded sequences for efficiency
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Process through LSTM
        _, (hidden, _) = self.lstm(packed_input)
        
        # Use the last hidden state from the top layer
        last_hidden = hidden[-1]
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Project to output dimension
        output = self.fc(last_hidden)
        
        return output


def load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load a trained LSTM model"""
    print(f"Loading model from {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get('model_config', {
            'input_dim': EMBEDDING_DIM,
            'hidden_dim': 1536,
            'output_dim': EMBEDDING_DIM,
            'num_layers': 2
        })
        
        model = PlaylistLSTMModel(
            input_dim=model_config.get('input_dim', EMBEDDING_DIM),
            hidden_dim=model_config.get('hidden_dim', 1536),
            output_dim=model_config.get('output_dim', EMBEDDING_DIM),
            num_layers=model_config.get('num_layers', 2)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully. Hidden dim: {model.hidden_dim}, Layers: {model.num_layers}")
        return model, checkpoint
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        raise


class SafeEmbeddingLookup:
    """Safe lookup for song embeddings with error handling"""
    
    def __init__(self, embedding_path):
        self.embedding_path = embedding_path
        self.uri_to_idx = {}
        self.cache = {}
        self.error_count = 0
        self._load_index()
    
    def _load_index(self):
        """Load URI to index mapping with verification"""
        try:
            index_path = os.path.splitext(self.embedding_path)[0] + '_uri_index.npz'
            if not os.path.exists(index_path):
                print(f"WARNING: Could not find URI index at {index_path}")
                return
            
            data = np.load(index_path, allow_pickle=True)
            uri_list = data['uri_to_idx']
            
            # Convert to a safer dictionary with string keys and int values
            for item in uri_list:
                try:
                    # Handle potential encoding issues
                    uri = str(item[0])
                    idx = int(item[1])
                    self.uri_to_idx[uri] = idx
                except Exception:
                    continue
            
            print(f"Loaded index with {len(self.uri_to_idx)} URIs")
            
            # Verify against H5 file
            with h5py.File(self.embedding_path, 'r') as f:
                if 'embeddings' in f:
                    embedding_count = f['embeddings'].shape[0]
                    print(f"Found {embedding_count} embeddings in H5 file")
                    
                    # Check for indices beyond array bounds
                    self.uri_to_idx = {uri: idx for uri, idx in self.uri_to_idx.items() 
                                      if idx < embedding_count}
                    print(f"Filtered to {len(self.uri_to_idx)} valid URIs")
        except Exception as e:
            print(f"Error loading URI index: {e}")
            traceback.print_exc()
    
    def get_embedding(self, uri):
        """Get embedding for a song URI with error handling"""
        uri = str(uri)  # Ensure string
        
        # Check cache first
        if uri in self.cache:
            return self.cache[uri]
        
        if uri not in self.uri_to_idx:
            raise KeyError(f"URI {uri} not found in index")
        
        idx = self.uri_to_idx[uri]
        
        try:
            with h5py.File(self.embedding_path, 'r') as f:
                embedding = f['embeddings'][idx].copy()
                
                # Add to cache (limit size)
                self.cache[uri] = embedding
                if len(self.cache) > 5000:
                    # Remove random items
                    to_remove = random.sample(list(self.cache.keys()), 1000)
                    for key in to_remove:
                        del self.cache[key]
                
                return embedding
        except Exception as e:
            self.error_count += 1
            if self.error_count % 100 == 0:
                print(f"Embedding lookup errors: {self.error_count}")
            raise KeyError(f"Error accessing embedding: {str(e)}")
    
    def get_batch_embeddings(self, uris):
        """Get embeddings for multiple URIs, skipping any that fail"""
        embeddings = []
        valid_uris = []
        
        for uri in uris:
            try:
                embedding = self.get_embedding(uri)
                embeddings.append(embedding)
                valid_uris.append(uri)
            except KeyError:
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


def load_song_data(count_song_path, top_n=3000):
    """Load top songs and long-tail songs with genre data if available"""
    try:
        df = pd.read_csv(count_song_path)
        
        # Sort by popularity
        sorted_df = df.sort_values(by='count', ascending=False)
        
        # Get top songs
        top_songs = sorted_df.head(top_n)['track_uri'].tolist()
        
        # Get long-tail songs (next 1000 after top_n)
        long_tail_songs = sorted_df.iloc[top_n:top_n+1000]['track_uri'].tolist()
        
        # Try to get genre info if available
        genre_map = {}
        if 'genre' in sorted_df.columns:
            for _, row in sorted_df.iterrows():
                uri = row['track_uri']
                genre = row['genre']
                if not pd.isna(genre):
                    genre_map[uri] = genre
        
        print(f"Loaded {len(top_songs)} top songs and {len(long_tail_songs)} long-tail songs")
        return top_songs, long_tail_songs, genre_map
    except Exception as e:
        print(f"Error loading song data: {e}")
        traceback.print_exc()
        return [], [], {}


class ImprovedEvalDataset(Dataset):
    """Dataset for more accurate evaluation with better negative examples"""
    
    def __init__(self, data_dir, embedding_lookup, top_songs=None, long_tail_songs=None, 
                 genre_map=None, input_fraction=0.75, min_songs=4, max_files=None):
        self.data_dir = Path(data_dir)
        self.embedding_lookup = embedding_lookup
        self.top_songs = set(top_songs) if top_songs else set()
        self.long_tail_songs = set(long_tail_songs) if long_tail_songs else set()
        self.genre_map = genre_map or {}
        self.input_fraction = input_fraction
        self.min_songs = min_songs
        
        # Find and process data files
        print(f"Finding data files in {data_dir}")
        data_files = []
        for ext in ['*.json', '*.csv']:
            data_files.extend(list(self.data_dir.glob(ext)))
        
        if max_files and len(data_files) > max_files:
            data_files = random.sample(data_files, max_files)
        
        # Extract playlists
        self.playlists = []
        for file_path in tqdm(data_files, desc="Loading files"):
            try:
                self._process_file(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Create a hard negative pool for both top and long-tail songs
        self.all_top_embeddings = {}
        self.all_long_embeddings = {}
        
        # Precompute a sample of embeddings for creating hard negatives
        self._precompute_negative_pool()
        
        # Pre-verify valid embeddings
        self._verify_embeddings()
        
        print(f"Created dataset with {len(self.playlists)} playlists")
    
    def _precompute_negative_pool(self):
        """Precompute a pool of song embeddings for creating hard negatives"""
        print("Precomputing negative example pool...")
        
        # Process top songs (up to 500)
        top_sample = random.sample(list(self.top_songs), min(500, len(self.top_songs)))
        for uri in tqdm(top_sample, desc="Computing top song embeddings"):
            try:
                embedding = self.embedding_lookup.get_embedding(uri)
                genre = self.genre_map.get(uri, "unknown")
                if genre not in self.all_top_embeddings:
                    self.all_top_embeddings[genre] = []
                self.all_top_embeddings[genre].append((uri, embedding))
            except Exception:
                pass
        
        # Process long-tail songs (up to 200)
        long_sample = random.sample(list(self.long_tail_songs), min(200, len(self.long_tail_songs)))
        for uri in tqdm(long_sample, desc="Computing long-tail song embeddings"):
            try:
                embedding = self.embedding_lookup.get_embedding(uri)
                genre = self.genre_map.get(uri, "unknown")
                if genre not in self.all_long_embeddings:
                    self.all_long_embeddings[genre] = []
                self.all_long_embeddings[genre].append((uri, embedding))
            except Exception:
                pass
                
        print(f"Created negative pool with {sum(len(v) for v in self.all_top_embeddings.values())} top songs" 
              f" and {sum(len(v) for v in self.all_long_embeddings.values())} long-tail songs")
    
    def _process_file(self, file_path):
        """Process a playlist file (JSON or CSV)"""
        if file_path.suffix.lower() == '.csv':
            try:
                df = pd.read_csv(file_path, index_col=0)
                self._process_csv(df)
            except Exception:
                pass
        else:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if 'playlists' in data:
                    playlists = data['playlists']
                else:
                    playlists = [data]  # Single playlist
                    
                for playlist in playlists:
                    self._process_playlist(playlist)
            except Exception:
                pass
    
    def _process_csv(self, df):
        """Process CSV playlist data"""
        # Get pid column if it exists
        pid_column = df.get("pid", [-1] * len(df))
        
        # Drop metadata columns
        meta_cols = ["pid", "collaborative", "num_followers", "num_tracks"]
        for col in meta_cols:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Process each row
        for i, row in df.iterrows():
            song_indices = np.where(row.values == 1)[0]
            if len(song_indices) < self.min_songs:
                continue
                
            # Get URIs
            song_uris = df.columns[song_indices].tolist()
            
            # Categorize
            top_uris = [uri for uri in song_uris if uri in self.top_songs]
            long_tail_uris = [uri for uri in song_uris if uri in self.long_tail_songs]
            
            # Extract genre info if available
            genres = set()
            for uri in song_uris:
                if uri in self.genre_map:
                    genres.add(self.genre_map[uri])
            
            # Store
            self.playlists.append({
                'pid': pid_column[i] if i < len(pid_column) else -1,
                'song_uris': song_uris,
                'top_uris': top_uris,
                'long_tail_uris': long_tail_uris,
                'genres': list(genres)
            })
    
    def _process_playlist(self, playlist):
        """Process a JSON playlist"""
        tracks = playlist.get('tracks', [])
        if len(tracks) < self.min_songs:
            return
            
        # Get URIs
        song_uris = [track.get('track_uri') for track in tracks]
        
        # Categorize
        top_uris = [uri for uri in song_uris if uri in self.top_songs]
        long_tail_uris = [uri for uri in song_uris if uri in self.long_tail_songs]
        
        # Extract genre info if available
        genres = set()
        for uri in song_uris:
            if uri in self.genre_map:
                genres.add(self.genre_map[uri])
        
        # Store
        self.playlists.append({
            'pid': playlist.get('pid', -1),
            'song_uris': song_uris,
            'top_uris': top_uris,
            'long_tail_uris': long_tail_uris,
            'genres': list(genres)
        })
    
    def _verify_embeddings(self):
        """Pre-verify which playlists have enough valid embeddings"""
        verified_playlists = []
        
        for playlist in tqdm(self.playlists, desc="Verifying embeddings"):
            # Check embeddings for all songs
            embeddings, valid_uris = self.embedding_lookup.get_batch_embeddings(playlist['song_uris'])
            
            if len(embeddings) >= self.min_songs:
                # Update playlist with only valid URIs
                playlist['song_uris'] = valid_uris
                playlist['top_uris'] = [uri for uri in valid_uris if uri in self.top_songs]
                playlist['long_tail_uris'] = [uri for uri in valid_uris if uri in self.long_tail_songs]
                verified_playlists.append(playlist)
        
        print(f"Filtered from {len(self.playlists)} to {len(verified_playlists)} playlists with valid embeddings")
        self.playlists = verified_playlists
    
    def _get_hard_negative_embeddings(self, positive_uri, is_top_song=True, same_genre=False):
        """Get challenging negative example embeddings"""
        # Choose song pool
        pool = self.all_top_embeddings if is_top_song else self.all_long_embeddings
        
        # Get genre if possible
        genre = self.genre_map.get(positive_uri, "unknown")
        
        # Choose potential candidates
        candidates = []
        
        # If we want same genre and have songs in that genre, use them
        if same_genre and genre in pool and len(pool[genre]) > 1:
            for uri, emb in pool[genre]:
                if uri != positive_uri:  # Avoid the positive example itself
                    candidates.append((uri, emb))
        
        # If not enough, use random ones
        if len(candidates) < 5:
            # Flatten the pool
            all_candidates = [(uri, emb) for genre_list in pool.values() 
                             for uri, emb in genre_list 
                             if uri != positive_uri]
            
            # Sample random candidates
            if all_candidates:
                candidates = random.sample(all_candidates, min(5, len(all_candidates)))
        
        # If still no candidates, create random embeddings
        if not candidates:
            random_emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            random_emb = random_emb / np.linalg.norm(random_emb)
            return "random_negative", random_emb
        
        # Return a random candidate
        uri, emb = random.choice(candidates)
        return uri, emb
    
    def set_epoch(self, epoch):
        """Support for dataloader reset between evaluation passes"""
        pass
    
    def __len__(self):
        return len(self.playlists)
    
    def __getitem__(self, idx):
        """Get a playlist with properly masked songs and hard negative examples"""
        # Try multiple times if we encounter errors
        for _ in range(3):
            try:
                playlist = self.playlists[idx]
                song_uris = playlist['song_uris'].copy()
                
                # Shuffle for randomization
                random.shuffle(song_uris)
                
                # Split into input and test portions
                n_songs = len(song_uris)
                n_input = max(self.min_songs - 1, int(n_songs * self.input_fraction))
                n_input = min(n_input, n_songs - 1)  # Ensure at least one song for testing
                
                input_uris = song_uris[:n_input]
                test_uris = song_uris[n_input:]
                
                # Get embeddings for input section
                input_embeddings, valid_input_uris = self.embedding_lookup.get_batch_embeddings(input_uris)
                
                if len(input_embeddings) < self.min_songs - 1:
                    # Not enough valid embeddings, try another playlist
                    idx = random.randint(0, len(self.playlists) - 1)
                    continue
                
                # Get embeddings for test section
                test_embeddings, valid_test_uris = self.embedding_lookup.get_batch_embeddings(test_uris)
                
                if len(test_embeddings) == 0:
                    # No valid test songs, try another playlist
                    idx = random.randint(0, len(self.playlists) - 1)
                    continue
                
                # Base result dictionary
                result = {
                    'pid': playlist['pid'],
                    'input_embeddings': input_embeddings.astype(np.float32),
                    'input_length': len(input_embeddings),
                    'test_embeddings': test_embeddings.astype(np.float32),
                    'test_uris': valid_test_uris,
                    'hard_negative_embeddings': [],
                    'hard_negative_uris': []
                }
                
                # Find a top song to mask if possible
                top_test_uris = [uri for uri in valid_test_uris if uri in self.top_songs]
                if top_test_uris:
                    # Choose a random top song to mask
                    top_uri = random.choice(top_test_uris)
                    top_idx = valid_test_uris.index(top_uri)
                    result['has_top_mask'] = True
                    result['top_mask_uri'] = top_uri
                    result['top_mask_embedding'] = test_embeddings[top_idx].astype(np.float32)
                    result['top_mask_idx'] = top_idx
                    
                    # Create hard negative examples - both same genre and different genre
                    # Same genre (semantically similar)
                    neg_uri, neg_emb = self._get_hard_negative_embeddings(top_uri, is_top_song=True, same_genre=True)
                    result['hard_negative_embeddings'].append(neg_emb)
                    result['hard_negative_uris'].append(neg_uri)
                    
                    # Different genre
                    neg_uri, neg_emb = self._get_hard_negative_embeddings(top_uri, is_top_song=True, same_genre=False)
                    result['hard_negative_embeddings'].append(neg_emb)
                    result['hard_negative_uris'].append(neg_uri)
                else:
                    result['has_top_mask'] = False
                
                # Find a long-tail song to mask if possible
                long_test_uris = [uri for uri in valid_test_uris if uri in self.long_tail_songs]
                if long_test_uris:
                    # Choose a random long-tail song to mask
                    long_uri = random.choice(long_test_uris)
                    long_idx = valid_test_uris.index(long_uri)
                    result['has_long_mask'] = True
                    result['long_mask_uri'] = long_uri
                    result['long_mask_embedding'] = test_embeddings[long_idx].astype(np.float32)
                    result['long_mask_idx'] = long_idx
                    
                    # Create hard negative examples - both same genre and different genre
                    # Same genre (semantically similar)
                    neg_uri, neg_emb = self._get_hard_negative_embeddings(long_uri, is_top_song=False, same_genre=True)
                    result['hard_negative_embeddings'].append(neg_emb)
                    result['hard_negative_uris'].append(neg_uri)
                    
                    # Different genre
                    neg_uri, neg_emb = self._get_hard_negative_embeddings(long_uri, is_top_song=False, same_genre=False)
                    result['hard_negative_embeddings'].append(neg_emb)
                    result['hard_negative_uris'].append(neg_uri)
                else:
                    result['has_long_mask'] = False
                
                # Convert hard negatives to numpy arrays if any exist
                if result['hard_negative_embeddings']:
                    result['hard_negative_embeddings'] = np.stack(result['hard_negative_embeddings']).astype(np.float32)
                
                # Return if we have at least one valid mask
                if result['has_top_mask'] or result['has_long_mask']:
                    return result
                
                # If we don't have any masks, try another playlist
                idx = random.randint(0, len(self.playlists) - 1)
            
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                idx = random.randint(0, len(self.playlists) - 1)
        
        # Fallback to default values if all attempts fail
        return {
            'pid': -1,
            'input_embeddings': np.zeros((self.min_songs, EMBEDDING_DIM), dtype=np.float32),
            'input_length': self.min_songs,
            'test_embeddings': np.zeros((1, EMBEDDING_DIM), dtype=np.float32),
            'test_uris': ['dummy'],
            'has_top_mask': False,
            'has_long_mask': False,
            'hard_negative_embeddings': np.zeros((1, EMBEDDING_DIM), dtype=np.float32),
            'hard_negative_uris': ['dummy_neg']
        }


def custom_collate(batch):
    """Custom collate function with error handling"""
    # Handle empty batch
    if not batch:
        return {
            'input_embeddings': torch.zeros((1, 1, EMBEDDING_DIM)),
            'lengths': torch.ones(1),
            'pids': [-1],
            'has_top_mask': [False],
            'has_long_mask': [False],
            'test_embeddings': [],
            'test_uris': [],
            'hard_negative_embeddings': [],
            'hard_negative_uris': []
        }
    
    # Get batch size and max sequence length
    batch_size = len(batch)
    max_length = max(item.get('input_length', 1) for item in batch)
    
    # Initialize tensors and lists
    input_embeddings = []
    lengths = []
    pids = []
    has_top_mask = []
    top_mask_uris = []
    top_mask_embeddings = []
    has_long_mask = []
    long_mask_uris = []
    long_mask_embeddings = []
    test_embeddings_list = []
    test_uris_list = []
    hard_negative_embeddings_list = []
    hard_negative_uris_list = []
    
    # Process each item
    for item in batch:
        try:
            # Get input sequence
            seq = item.get('input_embeddings', np.zeros((1, EMBEDDING_DIM), dtype=np.float32))
            seq_len = item.get('input_length', 1)
            
            # Pad if needed
            if seq_len < max_length:
                padding = np.zeros((max_length - seq_len, EMBEDDING_DIM), dtype=np.float32)
                seq = np.concatenate([seq, padding], axis=0)
            
            # Add to batch
            input_embeddings.append(torch.tensor(seq, dtype=torch.float32))
            lengths.append(seq_len)
            pids.append(item.get('pid', -1))
            
            # Handle top mask
            has_top = item.get('has_top_mask', False)
            has_top_mask.append(has_top)
            
            if has_top:
                uri = item.get('top_mask_uri', '')
                emb = item.get('top_mask_embedding', np.zeros(EMBEDDING_DIM, dtype=np.float32))
                top_mask_uris.append(uri)
                top_mask_embeddings.append(torch.tensor(emb, dtype=torch.float32))
            
            # Handle long-tail mask
            has_long = item.get('has_long_mask', False)
            has_long_mask.append(has_long)
            
            if has_long:
                uri = item.get('long_mask_uri', '')
                emb = item.get('long_mask_embedding', np.zeros(EMBEDDING_DIM, dtype=np.float32))
                long_mask_uris.append(uri)
                long_mask_embeddings.append(torch.tensor(emb, dtype=torch.float32))
            
            # Store test data
            test_embeddings_list.append(item.get('test_embeddings', np.array([])))
            test_uris_list.append(item.get('test_uris', []))
            
            # Store hard negative examples
            hard_negative_embeddings_list.append(item.get('hard_negative_embeddings', np.array([])))
            hard_negative_uris_list.append(item.get('hard_negative_uris', []))
            
        except Exception as e:
            print(f"Error in collate function: {e}")
            # Add default values
            input_embeddings.append(torch.zeros((max_length, EMBEDDING_DIM), dtype=torch.float32))
            lengths.append(1)
            pids.append(-1)
            has_top_mask.append(False)
            has_long_mask.append(False)
            test_embeddings_list.append(np.array([]))
            test_uris_list.append([])
            hard_negative_embeddings_list.append(np.array([]))
            hard_negative_uris_list.append([])
    
    # Stack tensors
    result = {
        'input_embeddings': torch.stack(input_embeddings),
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'pids': pids,
        'has_top_mask': has_top_mask,
        'has_long_mask': has_long_mask,
        'test_embeddings': test_embeddings_list,
        'test_uris': test_uris_list,
        'hard_negative_embeddings': hard_negative_embeddings_list,
        'hard_negative_uris': hard_negative_uris_list
    }
    
    # Add mask tensors if any
    if any(has_top_mask):
        result['top_mask_uris'] = top_mask_uris
        result['top_mask_embeddings'] = torch.stack(top_mask_embeddings) if top_mask_embeddings else None
    
    if any(has_long_mask):
        result['long_mask_uris'] = long_mask_uris
        result['long_mask_embeddings'] = torch.stack(long_mask_embeddings) if long_mask_embeddings else None
    
    return result


def evaluate_with_dynamic_thresholds(model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate model with dynamic thresholds for top and long-tail songs
    and better negative examples
    """
    print("Evaluating model with dynamic thresholds and challenging negatives...")
    model.to(device)
    model.eval()
    
    # Track similarity scores for finding dynamic thresholds
    all_similarity_data = {
        'top': {'positive': [], 'negative': []},
        'long': {'positive': [], 'negative': []}
    }
    
    # Stats
    stats = {
        'processed': 0,
        'errors': 0,
        'empty_batches': 0
    }
    
    # First pass: collect all similarity scores to determine thresholds
    print("Pass 1: Collecting similarity scores for threshold determination...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting similarities")):
            try:
                # Skip empty batches
                if len(batch['input_embeddings']) == 0:
                    stats['empty_batches'] += 1
                    continue
                
                # Get predictions
                input_embeddings = batch['input_embeddings'].to(device)
                lengths = batch['lengths']
                predictions = model(input_embeddings, lengths)
                
                # Process each item in batch
                for i in range(len(predictions)):
                    try:
                        # Normalize prediction
                        pred = F.normalize(predictions[i].unsqueeze(0), p=2, dim=1)
                        
                        # Process top-song mask
                        if batch['has_top_mask'][i]:
                            # Get positive example similarity
                            pos_emb = batch['top_mask_embeddings'][i].to(device)
                            pos_norm = F.normalize(pos_emb.unsqueeze(0), p=2, dim=1)
                            pos_sim = torch.mm(pred, pos_norm.t()).item()
                            all_similarity_data['top']['positive'].append(pos_sim)
                            
                            # Get hard negative similarities
                            if i < len(batch['hard_negative_embeddings']) and len(batch['hard_negative_embeddings'][i]) > 0:
                                for neg_idx in range(len(batch['hard_negative_embeddings'][i])):
                                    neg_emb = torch.tensor(batch['hard_negative_embeddings'][i][neg_idx], dtype=torch.float32).to(device)
                                    neg_norm = F.normalize(neg_emb.unsqueeze(0), p=2, dim=1)
                                    neg_sim = torch.mm(pred, neg_norm.t()).item()
                                    all_similarity_data['top']['negative'].append(neg_sim)
                        
                        # Process long-tail mask
                        if batch['has_long_mask'][i]:
                            # Get positive example similarity
                            pos_emb = batch['long_mask_embeddings'][i].to(device)
                            pos_norm = F.normalize(pos_emb.unsqueeze(0), p=2, dim=1)
                            pos_sim = torch.mm(pred, pos_norm.t()).item()
                            all_similarity_data['long']['positive'].append(pos_sim)
                            
                            # Get hard negative similarities 
                            if i < len(batch['hard_negative_embeddings']) and len(batch['hard_negative_embeddings'][i]) > 0:
                                for neg_idx in range(len(batch['hard_negative_embeddings'][i])):
                                    neg_emb = torch.tensor(batch['hard_negative_embeddings'][i][neg_idx], dtype=torch.float32).to(device)
                                    neg_norm = F.normalize(neg_emb.unsqueeze(0), p=2, dim=1)
                                    neg_sim = torch.mm(pred, neg_norm.t()).item()
                                    all_similarity_data['long']['negative'].append(neg_sim)
                            
                        stats['processed'] += 1
                    
                    except Exception as e:
                        stats['errors'] += 1
                        if stats['errors'] % 50 == 0:
                            print(f"Error processing item: {e}")
            
            except Exception as e:
                stats['errors'] += 1
                if stats['errors'] % 10 == 0:
                    print(f"Error processing batch: {e}")
    
    # Calculate optimal thresholds using precision-recall curve or ROC curve
    print("Determining optimal thresholds...")
    thresholds = {}
    for category in ['top', 'long']:
        try:
            positive_scores = np.array(all_similarity_data[category]['positive'])
            negative_scores = np.array(all_similarity_data[category]['negative'])
            
            # Skip if we don't have enough data
            if len(positive_scores) < 10 or len(negative_scores) < 10:
                print(f"Not enough data for {category} songs to determine threshold")
                # Use default
                thresholds[category] = 0.5
                continue
            
            # Create labels (1 for positive, 0 for negative)
            y_true = np.hstack([np.ones(len(positive_scores)), np.zeros(len(negative_scores))])
            y_scores = np.hstack([positive_scores, negative_scores])
            
            # Calculate optimal threshold using precision-recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
            
            # F1 score at each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            # Find threshold with best F1 score
            best_idx = np.argmax(f1_scores)
            best_threshold = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.5
            
            # Ensure threshold is not too extreme
            best_threshold = max(min(best_threshold, 0.95), 0.05)
            
            thresholds[category] = float(best_threshold)
            print(f"Optimal threshold for {category} songs: {best_threshold:.4f}")
        except Exception as e:
            print(f"Error calculating threshold for {category}: {e}")
            # Use default
            thresholds[category] = 0.5
    
    # Second pass: evaluate using the optimized thresholds
    print(f"\nPass 2: Evaluating with dynamic thresholds: top={thresholds['top']:.4f}, long={thresholds['long']:.4f}")
    
    # Reset dataloader iterator (create a new one with same dataset)
    # This is a workaround since we can't easily reset existing dataloaders
    new_dataloader = DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        collate_fn=dataloader.collate_fn,
        num_workers=dataloader.num_workers
    )
    
    # Track metrics for top songs
    top_metrics = {
        'true_pos': 0,
        'false_pos': 0,
        'true_neg': 0,
        'false_neg': 0,
        'similarities': []
    }
    
    # Track metrics for long-tail songs
    long_metrics = {
        'true_pos': 0,
        'false_pos': 0,
        'true_neg': 0,
        'false_neg': 0,
        'similarities': []
    }
    
    # Evaluation loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(new_dataloader, desc="Evaluating with thresholds")):
            try:
                # Skip empty batches
                if len(batch['input_embeddings']) == 0:
                    continue
                
                # Get predictions
                input_embeddings = batch['input_embeddings'].to(device)
                lengths = batch['lengths']
                predictions = model(input_embeddings, lengths)
                
                # Process each item in batch
                for i in range(len(predictions)):
                    try:
                        # Normalize prediction
                        pred = F.normalize(predictions[i].unsqueeze(0), p=2, dim=1)
                        
                        # Process top-song mask
                        if batch['has_top_mask'][i]:
                            # Get positive example similarity
                            pos_emb = batch['top_mask_embeddings'][i].to(device)
                            pos_norm = F.normalize(pos_emb.unsqueeze(0), p=2, dim=1)
                            pos_sim = torch.mm(pred, pos_norm.t()).item()
                            top_metrics['similarities'].append((pos_sim, 1))  # 1 = positive
                            
                            # Check against threshold
                            if pos_sim > thresholds['top']:
                                top_metrics['true_pos'] += 1
                            else:
                                top_metrics['false_neg'] += 1
                            
                            # Get hard negative similarities
                            if i < len(batch['hard_negative_embeddings']) and len(batch['hard_negative_embeddings'][i]) > 0:
                                for neg_idx in range(min(2, len(batch['hard_negative_embeddings'][i]))):  # Use at most 2 hard negatives
                                    neg_emb = torch.tensor(batch['hard_negative_embeddings'][i][neg_idx], dtype=torch.float32).to(device)
                                    neg_norm = F.normalize(neg_emb.unsqueeze(0), p=2, dim=1)
                                    neg_sim = torch.mm(pred, neg_norm.t()).item()
                                    top_metrics['similarities'].append((neg_sim, 0))  # 0 = negative
                                    
                                    # Check against threshold
                                    if neg_sim > thresholds['top']:
                                        top_metrics['false_pos'] += 1
                                    else:
                                        top_metrics['true_neg'] += 1
                        
                        # Process long-tail mask - USING SAME APPROACH as top songs
                        if batch['has_long_mask'][i]:
                            # Get positive example similarity
                            pos_emb = batch['long_mask_embeddings'][i].to(device)
                            pos_norm = F.normalize(pos_emb.unsqueeze(0), p=2, dim=1)
                            pos_sim = torch.mm(pred, pos_norm.t()).item()
                            long_metrics['similarities'].append((pos_sim, 1))  # 1 = positive
                            
                            # Check against threshold
                            if pos_sim > thresholds['long']:
                                long_metrics['true_pos'] += 1
                            else:
                                long_metrics['false_neg'] += 1
                            
                            # Get hard negative similarities 
                            if i < len(batch['hard_negative_embeddings']) and len(batch['hard_negative_embeddings'][i]) > 0:
                                for neg_idx in range(min(2, len(batch['hard_negative_embeddings'][i]))):  # Use at most 2 hard negatives
                                    neg_emb = torch.tensor(batch['hard_negative_embeddings'][i][neg_idx], dtype=torch.float32).to(device)
                                    neg_norm = F.normalize(neg_emb.unsqueeze(0), p=2, dim=1)
                                    neg_sim = torch.mm(pred, neg_norm.t()).item()
                                    long_metrics['similarities'].append((neg_sim, 0))  # 0 = negative
                                    
                                    # Check against threshold
                                    if neg_sim > thresholds['long']:
                                        long_metrics['false_pos'] += 1
                                    else:
                                        long_metrics['true_neg'] += 1
                    
                    except Exception as e:
                        stats['errors'] += 1
                        if stats['errors'] % 50 == 0:
                            print(f"Error processing item: {e}")
            
            except Exception as e:
                stats['errors'] += 1
                if stats['errors'] % 10 == 0:
                    print(f"Error processing batch: {e}")
    
    # Calculate metrics for top songs
    top_tp = top_metrics['true_pos']
    top_fp = top_metrics['false_pos']
    top_tn = top_metrics['true_neg']
    top_fn = top_metrics['false_neg']
    
    if (top_tp + top_fp) > 0:
        top_precision = top_tp / (top_tp + top_fp)
    else:
        top_precision = 0.0
        
    if (top_tp + top_fn) > 0:
        top_recall = top_tp / (top_tp + top_fn)
    else:
        top_recall = 0.0
        
    if (top_precision + top_recall) > 0:
        top_f1 = 2 * (top_precision * top_recall) / (top_precision + top_recall)
    else:
        top_f1 = 0.0
        
    top_total = top_tp + top_fp + top_tn + top_fn
    if top_total > 0:
        top_accuracy = (top_tp + top_tn) / top_total
    else:
        top_accuracy = 0.0
    
    # Calculate metrics for long-tail songs
    long_tp = long_metrics['true_pos']
    long_fp = long_metrics['false_pos']
    long_tn = long_metrics['true_neg']
    long_fn = long_metrics['false_neg']
    
    if (long_tp + long_fp) > 0:
        long_precision = long_tp / (long_tp + long_fp)
    else:
        long_precision = 0.0
        
    if (long_tp + long_fn) > 0:
        long_recall = long_tp / (long_tp + long_fn)
    else:
        long_recall = 0.0
        
    if (long_precision + long_recall) > 0:
        long_f1 = 2 * (long_precision * long_recall) / (long_precision + long_recall)
    else:
        long_f1 = 0.0
        
    long_total = long_tp + long_fp + long_tn + long_fn
    if long_total > 0:
        long_accuracy = (long_tp + long_tn) / long_total
    else:
        long_accuracy = 0.0
    
    # Process similarity statistics
    top_sim_values = [sim for sim, _ in top_metrics['similarities']]
    if top_sim_values:
        top_sim_stats = {
            'mean': float(np.mean(top_sim_values)),
            'median': float(np.median(top_sim_values)),
            'min': float(np.min(top_sim_values)),
            'max': float(np.max(top_sim_values))
        }
    else:
        top_sim_stats = {'mean': 0, 'median': 0, 'min': 0, 'max': 0}
    
    long_sim_values = [sim for sim, _ in long_metrics['similarities']]
    if long_sim_values:
        long_sim_stats = {
            'mean': float(np.mean(long_sim_values)),
            'median': float(np.median(long_sim_values)),
            'min': float(np.min(long_sim_values)),
            'max': float(np.max(long_sim_values))
        }
    else:
        long_sim_stats = {'mean': 0, 'median': 0, 'min': 0, 'max': 0}
    
    # Compile final results
    results = {
        'top_metrics': {
            'accuracy': float(top_accuracy),
            'precision': float(top_precision),
            'recall': float(top_recall),
            'f1_score': float(top_f1),
            'confusion_matrix': {
                'true_positives': int(top_tp),
                'false_positives': int(top_fp),
                'true_negatives': int(top_tn),
                'false_negatives': int(top_fn)
            },
            'sample_count': int(top_total),
            'similarity_stats': top_sim_stats,
            'threshold': float(thresholds['top'])
        },
        'long_metrics': {
            'accuracy': float(long_accuracy),
            'precision': float(long_precision),
            'recall': float(long_recall),
            'f1_score': float(long_f1),
            'confusion_matrix': {
                'true_positives': int(long_tp),
                'false_positives': int(long_fp),
                'true_negatives': int(long_tn),
                'false_negatives': int(long_fn)
            },
            'sample_count': int(long_total),
            'similarity_stats': long_sim_stats,
            'threshold': float(thresholds['long'])
        },
        'processing_stats': stats,
        'thresholds': thresholds
    }
    
    # Print summary
    print("\n--- Evaluation Results with Dynamic Thresholds ---")
    
    print("\nTop Songs Metrics:")
    print(f"  Threshold: {thresholds['top']:.4f}")
    print(f"  Accuracy: {top_accuracy:.4f}")
    print(f"  Precision: {top_precision:.4f}")
    print(f"  Recall: {top_recall:.4f}")
    print(f"  F1 Score: {top_f1:.4f}")
    print(f"  Confusion Matrix: TP={top_tp}, FP={top_fp}, TN={top_tn}, FN={top_fn}")
    print(f"  Mean Similarity: {top_sim_stats['mean']:.4f}")
    
    print("\nLong-tail Songs Metrics:")
    print(f"  Threshold: {thresholds['long']:.4f}")
    print(f"  Accuracy: {long_accuracy:.4f}")
    print(f"  Precision: {long_precision:.4f}")
    print(f"  Recall: {long_recall:.4f}")
    print(f"  F1 Score: {long_f1:.4f}")
    print(f"  Confusion Matrix: TP={long_tp}, FP={long_fp}, TN={long_tn}, FN={long_fn}")
    print(f"  Mean Similarity: {long_sim_stats['mean']:.4f}")
    
    print("\nProcessing Stats:")
    print(f"  Items Processed: {stats['processed']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Empty Batches: {stats['empty_batches']}")
    
    return results


def evaluate_top_k(model, dataloader, k_values=[1, 5, 10, 20], device="cuda" if torch.cuda.is_available() else "cpu"):
    """Compute top-k accuracy for retrieval evaluation with improved handling"""
    print(f"Computing top-k accuracy for k={k_values}")
    model.to(device)
    model.eval()
    
    # Track metrics for each k value
    results = {k: {"correct": 0, "total": 0} for k in k_values}
    errors = 0
    
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing top-k"):
            try:
                # Skip empty batches
                if len(batch['input_embeddings']) == 0:
                    continue
                
                # Get predictions for all items in batch
                input_embeddings = batch['input_embeddings'].to(device)
                lengths = batch['lengths']
                predictions = model(input_embeddings, lengths)
                
                # Process each item
                for i in range(len(predictions)):
                    try:
                        # Skip if no test data
                        if not isinstance(batch['test_embeddings'][i], np.ndarray) or batch['test_embeddings'][i].size == 0:
                            continue
                        
                        # Find target URI from masks
                        target_uri = None
                        if batch['has_top_mask'][i]:
                            target_uri = batch['top_mask_uris'][i]
                        elif batch['has_long_mask'][i]:
                            target_uri = batch['long_mask_uris'][i]
                        
                        if not target_uri:
                            continue
                        
                        # Get and normalize prediction
                        pred = F.normalize(predictions[i].unsqueeze(0), p=2, dim=1)
                        
                        # Get and normalize test embeddings
                        test_embs = torch.tensor(batch['test_embeddings'][i], dtype=torch.float32).to(device)
                        test_uris = batch['test_uris'][i]
                        test_norm = F.normalize(test_embs, p=2, dim=1)
                        
                        # Calculate similarities to all test songs
                        similarities = torch.mm(pred, test_norm.t()).squeeze(0)
                        
                        # Sort by similarity (highest first)
                        sorted_indices = torch.argsort(similarities, descending=True)
                        
                        # Convert to list for easier processing
                        sorted_indices = sorted_indices.cpu().tolist()
                        
                        # Check for each k value
                        for k in k_values:
                            # Only consider k values that make sense for test set size
                            if k <= len(test_uris):
                                # Get the top k URIs by similarity
                                top_k_indices = sorted_indices[:k]
                                top_k_uris = [test_uris[idx] for idx in top_k_indices]
                                
                                # Check if target is in top-k
                                if target_uri in top_k_uris:
                                    results[k]["correct"] += 1
                                
                                results[k]["total"] += 1
                    
                    except Exception as e:
                        errors += 1
                        if errors % 50 == 0:
                            print(f"Error in top-k item {i}: {e}")
            
            except Exception as e:
                errors += 1
                if errors % 10 == 0:
                    print(f"Error in top-k batch: {e}")
    
    # Calculate accuracies for each k value
    accuracies = {}
    for k in k_values:
        if results[k]["total"] > 0:
            accuracies[k] = results[k]["correct"] / results[k]["total"]
        else:
            accuracies[k] = 0.0
    
    # Print results
    print("\nTop-K Accuracy Results:")
    for k in k_values:
        print(f"  Top-{k}: {accuracies[k]:.4f} ({results[k]['correct']}/{results[k]['total']})")
    
    return {
        'accuracies': accuracies,
        'correct': {k: results[k]['correct'] for k in k_values},
        'total': results[k_values[0]]['total'] if results[k_values[0]]['total'] > 0 else 0,
        'errors': errors
    }


def run_improved_evaluation(
    model_path,
    data_dir,
    embedding_path,
    song_csv_path,
    output_path=None,
    batch_size=32,
    max_files=20,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Run evaluation with dynamic thresholds and better negative examples"""
    start_time = time.time()
    
    print(f"Starting improved evaluation of {model_path}")
    print(f"Data directory: {data_dir}")
    print(f"Embeddings: {embedding_path}")
    print(f"Song data: {song_csv_path}")
    print(f"Device: {device}")
    
    # Load model
    try:
        model, model_info = load_model(model_path, device)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return None
    
    # Load embedding lookup
    embedding_lookup = SafeEmbeddingLookup(embedding_path)
    
    # Load song data with genre information if available
    top_songs, long_tail_songs, genre_map = load_song_data(song_csv_path)
    
    # Check embedding availability
    top_coverage = embedding_lookup.check_availability(top_songs)
    long_coverage = embedding_lookup.check_availability(long_tail_songs)
    print(f"Embedding coverage: {top_coverage:.1%} of top songs, {long_coverage:.1%} of long-tail songs")
    
    # Create dataset with improved negative examples
    dataset = ImprovedEvalDataset(
        data_dir=data_dir,
        embedding_lookup=embedding_lookup,
        top_songs=top_songs,
        long_tail_songs=long_tail_songs,
        genre_map=genre_map,
        max_files=max_files
    )
    
    if len(dataset) == 0:
        print("ERROR: No viable playlists found for evaluation")
        return {
            'error': 'No viable playlists found',
            'embedding_coverage': {
                'top_songs': top_coverage,
                'long_tail_songs': long_coverage
            }
        }
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=0  # Safer with custom embedding lookup
    )
    
    # Run evaluation with dynamic thresholds
    dynamic_results = evaluate_with_dynamic_thresholds(model, dataloader, device)
    
    # Run top-k evaluation
    topk_results = evaluate_top_k(model, dataloader, device=device)
    
    # Combine results
    results = {
        'model_path': model_path,
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
        }
    }
    
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Evaluate LSTM model with improved metrics")
    
    # Required arguments
    parser.add_argument("--model", required=True, help="Path to trained LSTM model")
    parser.add_argument("--data", required=True, help="Path to playlist data directory")
    parser.add_argument("--embeddings", required=True, help="Path to song embeddings file")
    parser.add_argument("--songs", required=True, help="Path to song popularity CSV")
    
    # Optional arguments
    parser.add_argument("--output", default=None, help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_files", type=int, default=20, help="Maximum number of data files to process")
    parser.add_argument("--device", default=None, help="Device to run on (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run evaluation
    run_improved_evaluation(
        model_path=args.model,
        data_dir=args.data,
        embedding_path=args.embeddings,
        song_csv_path=args.songs,
        output_path=args.output,
        batch_size=args.batch_size,
        max_files=args.max_files,
        device=device
    )


if __name__ == "__main__":
    main()