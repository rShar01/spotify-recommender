import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Tuple, Iterator
import json
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import glob
import os
from pathlib import Path

class PlaylistSliceLoader:
    """Handles loading and iteration over MPD slice files"""
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.slice_files = sorted(glob.glob(str(self.data_dir / "mpd.slice.*.json")))
        
        if not self.slice_files:
            raise ValueError(f"No MPD slice files found in {data_dir}")
            
        print(f"Found {len(self.slice_files)} slice files")
    
    def load_slice(self, slice_file: str) -> List[Dict]:
        """Load a single slice file"""
        with open(slice_file, 'r') as f:
            data = json.load(f)
            return data['playlists']
    
    def __iter__(self) -> Iterator[List[Dict]]:
        """Iterate over slices, yielding playlists from each slice"""
        for slice_file in self.slice_files:
            yield self.load_slice(slice_file)

class PlaylistDataset(Dataset):
    """Dataset class for handling playlists"""
    def __init__(self, playlists: List[Dict]):
        self.playlists = playlists
    
    def __len__(self):
        return len(self.playlists)
    
    def __getitem__(self, idx):
        return self.playlists[idx]

def custom_collate(batch):
    """Custom collate function that handles variable-sized playlists"""
    return batch

class PlaylistEmbedder:
    def __init__(self, 
                 model_name: str = "answerdotai/ModernBERT-base", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 32,
                 embedding_store_path: str = "/data/user_data/rshar/downloads/spotify/playlist_embeddings.h5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.batch_size = batch_size
        self.embedding_store_path = embedding_store_path
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def _get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts"""
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True,
                              max_length=512, padding=True).to(self.device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings

    def process_playlist(self, playlist: Dict) -> np.ndarray:
        """Process a single playlist and return its embedding"""
        # Get track embeddings
        track_texts = [
            f"This is the song {track['track_name']} performed by {track['artist_name']} from their album {track['album_name']}"
            for track in playlist['tracks']
        ]
        
        # Process tracks in smaller batches to avoid memory issues
        track_batch_size = 16
        track_embeddings = []
        
        for i in range(0, len(track_texts), track_batch_size):
            batch_texts = track_texts[i:i + track_batch_size]
            batch_embeddings = self._get_batch_embeddings(batch_texts)
            track_embeddings.append(batch_embeddings)
            
        track_embeddings = np.concatenate(track_embeddings) if len(track_embeddings) > 1 else track_embeddings[0]
        avg_track_embedding = np.mean(track_embeddings, axis=0)
        
        # Process metadata
        metadata_features = np.array([
            playlist['num_tracks'],
            playlist['num_albums'],
            playlist['num_artists'],
            playlist['num_followers'],
            playlist['num_edits'],
            playlist['duration_ms']
        ])
        metadata_features = (metadata_features - np.mean(metadata_features)) / np.std(metadata_features)
        
        # Project and combine
        metadata_projection = np.random.randn(len(metadata_features), len(avg_track_embedding))
        metadata_embedding = metadata_features @ metadata_projection
        final_embedding = 0.7 * avg_track_embedding + 0.3 * metadata_embedding
        
        return final_embedding

    def process_slice(self, playlists: List[Dict], slice_idx: int, split: str):
        """Process a single slice of playlists and store embeddings"""
        dataset = PlaylistDataset(playlists)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=custom_collate
        )
        
        with h5py.File(self.embedding_store_path, 'a') as f:
            # Create groups if they don't exist
            if split not in f:
                f.create_group(split)
            if f'slice_{slice_idx}' not in f[split]:
                f[split].create_group(f'slice_{slice_idx}')
            
            slice_group = f[split][f'slice_{slice_idx}']
            
            # Process in batches
            for batch_idx, batch_playlists in enumerate(tqdm(dataloader, desc=f"Processing {split} slice {slice_idx}")):
                batch_embeddings = []
                batch_pids = []
                
                # Process each playlist in the batch
                for playlist in batch_playlists:
                    embedding = self.process_playlist(playlist)
                    batch_embeddings.append(embedding)
                    batch_pids.append(playlist['pid'])
                
                # Store batch embeddings and playlist IDs
                batch_embeddings = np.stack(batch_embeddings)
                slice_group.create_dataset(
                    f'batch_{batch_idx}',
                    data=batch_embeddings,
                    compression='gzip'
                )
                slice_group.create_dataset(
                    f'batch_{batch_idx}_pids',
                    data=batch_pids,
                    compression='gzip'
                )

def process_dataset(data_dir: str, train_ratio: float = 0.7):
    """Process the entire MPD dataset slice by slice"""
    loader = PlaylistSliceLoader(data_dir)
    embedder = PlaylistEmbedder()
    
    # Split slice indices into train and test
    num_slices = len(loader.slice_files)
    num_train_slices = int(num_slices * train_ratio)
    
    print(f"Processing {num_train_slices} training slices and {num_slices - num_train_slices} test slices")
    
    # Process slices
    for slice_idx, playlists in enumerate(tqdm(loader, desc="Processing slices")):
        if slice_idx < num_train_slices:
            embedder.process_slice(playlists, slice_idx, 'train')
        else:
            embedder.process_slice(playlists, slice_idx, 'test')

def main():
    # Directory containing the MPD slice files
    data_dir = "/data/user_data/rshar/downloads/spotify/data"
    
    # Process the dataset
    process_dataset(data_dir, train_ratio=0.7)
    
    print("Completed processing all slices!")

if __name__ == "__main__":
    main()