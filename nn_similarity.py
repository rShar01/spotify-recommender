import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
import h5py
from tqdm import tqdm
import random
import os
import json
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

class PlaylistRecommenderModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dims: List[int] = [512, 256]):
        """Neural network for playlist similarity prediction"""
        super().__init__()
        
        layers = []
        prev_dim = embedding_dim * 2  # Concatenated embeddings
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass with two playlist embeddings"""
        x = torch.cat([x1, x2], dim=1)
        return self.model(x)

class PlaylistPairDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings: np.ndarray, pids: np.ndarray, positive_pairs: List[Tuple[int, int]]):
        """Dataset for training with playlist pairs"""
        self.embeddings = torch.FloatTensor(embeddings)
        self.pids = pids
        self.positive_pairs = positive_pairs
        self.pid_to_idx = {pid: idx for idx, pid in enumerate(pids)}
        
    def __len__(self):
        return len(self.positive_pairs) * 2  # Include negative pairs
        
    def __getitem__(self, idx):
        is_positive = idx < len(self.positive_pairs)
        
        if is_positive:
            pid1, pid2 = self.positive_pairs[idx]
        else:
            # Create negative pair
            idx1 = random.randint(0, len(self.pids) - 1)
            idx2 = random.randint(0, len(self.pids) - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, len(self.pids) - 1)
            pid1, pid2 = self.pids[idx1], self.pids[idx2]
        
        # Get embeddings
        emb1 = self.embeddings[self.pid_to_idx[pid1]]
        emb2 = self.embeddings[self.pid_to_idx[pid2]]
        
        return emb1, emb2, torch.tensor([float(is_positive)])

def find_similar_playlists_fast(embeddings: np.ndarray, pids: np.ndarray, 
                              max_pairs: int = 50000, 
                              k_neighbors: int = 10,
                              sample_size: int = None) -> List[Tuple[int, int]]:
    """
    Find similar playlist pairs using KNN for much faster processing
    
    Args:
        embeddings: Playlist embeddings
        pids: Playlist IDs
        max_pairs: Maximum number of pairs to return
        k_neighbors: Number of neighbors to find for each playlist
        sample_size: Number of playlists to sample (None = use all)
    """
    # Sample data if requested
    if sample_size is not None and sample_size < len(embeddings):
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
        sample_pids = pids[indices]
    else:
        sample_embeddings = embeddings
        sample_pids = pids
    
    print(f"Finding similar playlists among {len(sample_embeddings)} playlists...")
    
    # Use scikit-learn's NearestNeighbors for fast KNN
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree').fit(sample_embeddings)
    distances, indices = nbrs.kneighbors(sample_embeddings)
    
    # Create pairs from neighbors
    similar_pairs = []
    for i, neighbors in enumerate(indices):
        # Skip the first neighbor (self)
        for j in neighbors[1:]:
            # Ensure i < j to avoid duplicates
            if i < j:
                similar_pairs.append((int(sample_pids[i]), int(sample_pids[j])))
            
            # Break if we have enough pairs
            if len(similar_pairs) >= max_pairs:
                break
        
        # Break if we have enough pairs
        if len(similar_pairs) >= max_pairs:
            break
    
    print(f"Found {len(similar_pairs)} similar playlist pairs")
    return similar_pairs

def load_embeddings(embedding_store_path: str, split: str = 'train', max_slices: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings from H5 file"""
    all_embeddings = []
    all_pids = []
    
    with h5py.File(embedding_store_path, 'r') as f:
        if split not in f:
            raise ValueError(f"Split '{split}' not found in embedding store")
            
        split_group = f[split]
        
        # Get slice names and limit if requested
        slice_names = list(split_group.keys())
        if max_slices is not None:
            slice_names = slice_names[:max_slices]
        
        # Load slices
        for slice_name in tqdm(slice_names, desc=f"Loading {split} embeddings"):
            slice_group = split_group[slice_name]
            
            # Load all batches in slice
            batch_names = [n for n in slice_group.keys() if n.startswith('batch_') and not n.endswith('_pids')]
            for batch_name in batch_names:
                embeddings = slice_group[batch_name][:]
                pids = slice_group[f"{batch_name}_pids"][:]
                
                all_embeddings.append(embeddings)
                all_pids.append(pids)
    
    if not all_embeddings:
        raise ValueError(f"No embeddings found for split '{split}'")
        
    return np.concatenate(all_embeddings), np.concatenate(all_pids)

def train_and_save_model(
    embedding_store_path: str,
    model_save_path: str,
    max_slices: int = 5,  # Limit to 5 slices for speed
    sample_size: int = 10000,  # Sample 10k playlists for pairs
    batch_size: int = 128,
    num_epochs: int = 5,
    learning_rate: float = 0.00005,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train and save the recommendation model with fast processing"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Load embeddings
    print("Loading embeddings...")
    train_embeddings, train_pids = load_embeddings(
        embedding_store_path, 
        'train',
        max_slices=max_slices
    )
    
    print(f"Loaded {len(train_embeddings)} embeddings with dimension {train_embeddings.shape[1]}")
    
    # Find similar playlist pairs using fast method
    positive_pairs = find_similar_playlists_fast(
        train_embeddings, 
        train_pids,
        max_pairs=50000,
        k_neighbors=5,
        sample_size=sample_size
    )
    
    # Create dataset and dataloader
    dataset = PlaylistPairDataset(train_embeddings, train_pids, positive_pairs)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    # Save embedding dimension for model loading
    embedding_dim = train_embeddings.shape[1]
    
    # Initialize model
    model = PlaylistRecommenderModel(embedding_dim=embedding_dim)
    model.to(device)
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track losses
    epoch_losses = []
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        batch_losses = []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for emb1, emb2, labels in progress_bar:
            # Move to device
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(emb1, emb2)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record loss
            batch_losses.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate and store epoch loss
        epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
        epoch_losses.append(epoch_loss)
        
        # Print epoch loss
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed - Average loss: {epoch_loss:.6f}")
    
    # Print loss summary
    print("\nTraining complete!")
    print("Loss by epoch:")
    for i, loss in enumerate(epoch_losses):
        print(f"  Epoch {i+1}: {loss:.6f}")
    
    # Save model and metadata
    print(f"\nSaving model to {model_save_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': embedding_dim,
        'hidden_dims': [512, 256],
        'epoch_losses': epoch_losses,
    }, model_save_path)
    
    # Save metadata
    metadata = {
        'embedding_dim': int(embedding_dim),
        'num_playlists_used': int(len(train_pids)),
        'num_similar_pairs': len(positive_pairs),
        'device_used': device,
        'epochs': num_epochs,
        'batch_size': batch_size,
        'epoch_losses': [float(loss) for loss in epoch_losses],
    }
    
    metadata_path = os.path.splitext(model_save_path)[0] + '_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training complete. Model saved to {model_save_path}")
    print(f"Metadata saved to {metadata_path}")

def main():
    # Parameter settings for faster training
    params = {
        'max_slices': 12,           # Use only 5 slices
        'sample_size': 30000,      # Sample 10k playlists for finding pairs
        'batch_size': 128,         # Larger batch size
        'num_epochs': 20,           # Fewer epochs
        'learning_rate': 0.00005
    }

    # Configuration
    embedding_store_path = "/data/user_data/rshar/downloads/spotify/playlist_embeddings.h5"
    model_save_path = f"models/playlist_recommender_fast_{params['learning_rate']}.pt"
   
    # Train and save model
    train_and_save_model(
        embedding_store_path=embedding_store_path,
        model_save_path=model_save_path,
        **params
    )
    
    # Verify model loading
    try:
        print("\nVerifying model loading...")
        model_data = torch.load(model_save_path)
        model = PlaylistRecommenderModel(
            embedding_dim=model_data['embedding_dim'],
            hidden_dims=model_data['hidden_dims']
        )
        model.load_state_dict(model_data['model_state_dict'])
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()