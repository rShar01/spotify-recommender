import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import h5py
from tqdm import tqdm
import os
from pathlib import Path
import argparse
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('song_embedder.log')
    ]
)
logger = logging.getLogger('song_embedder')

class SongEmbedder:
    def __init__(self, 
                model_name: str = "answerdotai/ModernBERT-base", 
                device: str = "cuda" if torch.cuda.is_available() else "cpu",
                batch_size: int = 32,
                embedding_store_path: str = "data/song_lookup.h5"):
        """Initialize ModernBERT and other settings for song embedding"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.batch_size = batch_size
        self.embedding_store_path = embedding_store_path
        self.model.to(device)
        self.model.eval()
        
        # Create storage directory if needed
        os.makedirs(os.path.dirname(embedding_store_path), exist_ok=True)
        
        logger.info(f"Initialized ModernBERT on {device}")
    
    def _clean_string_value(self, value):
        """Ensure value is a valid string for H5 storage"""
        if value is None:
            return ""
        try:
            # Convert to string and ensure it's valid
            return str(value).encode('utf-8', errors='replace').decode('utf-8')
        except:
            # Return empty string for any errors
            return ""
    
    @torch.no_grad()
    def embed_songs(self, songs_df: pd.DataFrame, max_songs: int = None):
        """Generate embeddings for all songs in the dataframe"""
        # Sample songs if requested (though now we prefer to do this outside)
        if max_songs is not None and max_songs < len(songs_df):
            logger.info(f"Sampling {max_songs} out of {len(songs_df)} songs")
            songs_df = songs_df.sample(max_songs, random_state=42)
        
        # Make a copy to prevent modifying the original
        songs_df = songs_df.copy()
        
        # Sort by popularity (count) to prioritize most common songs
        songs_df = songs_df.sort_values(by='count', ascending=False)
        
        # Fill NaN values
        songs_df = songs_df.fillna('')
        
        # Ensure all string columns are actually strings
        for col in ['track_name', 'artist_name', 'album_name', 'track_uri', 'artist_uri', 'album_uri']:
            if col in songs_df.columns:
                songs_df[col] = songs_df[col].astype(str)
        
        # Prepare H5 file for storing embeddings
        with h5py.File(self.embedding_store_path, 'w') as f:
            # Store metadata
            f.attrs['embedding_dim'] = 768  # ModernBERT base dimension
            f.attrs['num_songs'] = len(songs_df)
            f.attrs['model_name'] = "answerdotai/ModernBERT-base"
            
            # Create datasets with explicit string dtypes for string data
            embedding_dataset = f.create_dataset('embeddings', 
                                               shape=(len(songs_df), 768),
                                               dtype=np.float32,
                                               compression='gzip')
            
            # Create datasets for song info with explicit string dtypes
            # h5py special_dtype for variable length strings
            str_dt = h5py.special_dtype(vlen=str)
            
            track_uri_dataset = f.create_dataset('track_uri', 
                                              shape=(len(songs_df),),
                                              dtype=str_dt)
            
            track_name_dataset = f.create_dataset('track_name', 
                                               shape=(len(songs_df),),
                                               dtype=str_dt)
            
            artist_name_dataset = f.create_dataset('artist_name', 
                                                shape=(len(songs_df),),
                                                dtype=str_dt)
            
            album_name_dataset = f.create_dataset('album_name', 
                                               shape=(len(songs_df),),
                                               dtype=str_dt)
            
            popularity_dataset = f.create_dataset('count', 
                                               shape=(len(songs_df),),
                                               dtype=np.int32)
            
            # Process songs in batches
            for i in tqdm(range(0, len(songs_df), self.batch_size), desc="Embedding songs"):
                batch_df = songs_df.iloc[i:i+self.batch_size]
                
                # Create text representations for each song
                texts = []
                for _, row in batch_df.iterrows():
                    try:
                        # Use clean values with error handling
                        track_name = self._clean_string_value(row.get('track_name', ''))
                        artist_name = self._clean_string_value(row.get('artist_name', ''))
                        album_name = self._clean_string_value(row.get('album_name', ''))
                        
                        text = f"This is the song {track_name} performed by {artist_name} from their album {album_name}"
                        texts.append(text)
                    except Exception as e:
                        logger.error(f"Error creating text for song: {e}")
                        # Add a placeholder to maintain alignment
                        texts.append("Unknown song")
                
                try:
                    # Generate embeddings
                    inputs = self.tokenizer(texts, return_tensors="pt", truncation=True,
                                          max_length=512, padding=True).to(self.device)
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    
                    # Store embeddings and metadata
                    batch_idx = slice(i, min(i + self.batch_size, len(songs_df)))
                    
                    # Store embeddings
                    embedding_dataset[batch_idx] = embeddings
                    
                    # Process and store string data carefully
                    for j, row in enumerate(batch_df.itertuples()):
                        idx = i + j
                        if idx >= len(songs_df):
                            break
                            
                        # Clean and store strings
                        try:
                            track_uri_dataset[idx] = self._clean_string_value(getattr(row, 'track_uri', ''))
                            track_name_dataset[idx] = self._clean_string_value(getattr(row, 'track_name', ''))
                            artist_name_dataset[idx] = self._clean_string_value(getattr(row, 'artist_name', ''))
                            album_name_dataset[idx] = self._clean_string_value(getattr(row, 'album_name', ''))
                            
                            # For count, ensure it's an integer
                            try:
                                count = int(getattr(row, 'count', 0))
                            except (ValueError, TypeError):
                                count = 0
                            popularity_dataset[idx] = count
                            
                        except Exception as e:
                            logger.error(f"Error storing data for song at index {idx}: {e}")
                            logger.error(traceback.format_exc())
                            
                except Exception as e:
                    logger.error(f"Error processing batch starting at index {i}: {e}")
                    logger.error(traceback.format_exc())
        
        logger.info(f"Embeddings saved to {self.embedding_store_path}")
        
        # Create a lookup index file for fast URI to index mapping
        self._create_uri_index(songs_df)
    
    def _create_uri_index(self, songs_df: pd.DataFrame):
        """Create a lookup file for fast URI to index mapping"""
        # Clean URIs and create mapping
        uri_to_idx = {}
        for idx, uri in enumerate(songs_df['track_uri']):
            try:
                # Ensure URI is a clean string
                clean_uri = self._clean_string_value(uri)
                if clean_uri:  # Only add non-empty URIs
                    uri_to_idx[clean_uri] = idx
            except Exception as e:
                logger.error(f"Error processing URI at index {idx}: {e}")
        
        # Save as a compressed numpy file
        uri_index_path = os.path.splitext(self.embedding_store_path)[0] + '_uri_index.npz'
        
        # Ensure the data is properly converted for serialization
        uri_items = [(str(uri), int(idx)) for uri, idx in uri_to_idx.items()]
        
        np.savez_compressed(uri_index_path, uri_to_idx=np.array(uri_items, dtype=object))
        
        logger.info(f"URI index saved to {uri_index_path} with {len(uri_to_idx)} entries")


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
            logger.info(f"Loaded index with {len(self.uri_to_idx)} track URIs")
        except Exception as e:
            logger.error(f"Error loading URI index: {e}")
            logger.error(traceback.format_exc())
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
    
    def get_track_info(self, track_uri: str):
        """Get all stored information about a track"""
        if track_uri not in self.uri_to_idx:
            raise KeyError(f"Track URI {track_uri} not found in the index")
            
        idx = self.uri_to_idx[track_uri]
        
        with h5py.File(self.embedding_store_path, 'r') as f:
            info = {
                'track_uri': f['track_uri'][idx],
                'track_name': f['track_name'][idx],
                'artist_name': f['artist_name'][idx],
                'album_name': f['album_name'][idx],
                'count': int(f['count'][idx]),
                'embedding': f['embeddings'][idx]
            }
            
        return info
    
    def get_most_similar_songs(self, query_embedding, n=10):
        """Find the most similar songs to a given embedding"""
        with h5py.File(self.embedding_store_path, 'r') as f:
            all_embeddings = f['embeddings'][:]
            all_track_uris = f['track_uri'][:]
            
            # Compute cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            all_norms = np.linalg.norm(all_embeddings, axis=1)
            
            dot_products = np.dot(all_embeddings, query_embedding)
            similarities = dot_products / (all_norms * query_norm)
            
            # Get top indices
            top_indices = np.argsort(similarities)[-n:][::-1]
            
            # Get track URIs and similarity scores
            results = [(all_track_uris[idx], similarities[idx]) for idx in top_indices]
            
        return results


def main():
    parser = argparse.ArgumentParser(description="Generate song embeddings with ModernBERT")
    parser.add_argument("--input", default="data/count_songs.csv", help="Path to song data CSV")
    parser.add_argument("--output", default="data/song_lookup.h5", help="Output path for embeddings")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding")
    parser.add_argument("--max-songs", type=int, default=None, help="Maximum number of songs to embed")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), 
                       help="Device to use for embedding (cuda/cpu)")
    parser.add_argument("--resume", action="store_true", help="Resume from a previous run")
    parser.add_argument("--chunk-size", type=int, default=10000, 
                       help="Process data in chunks of this size to save memory")
    
    args = parser.parse_args()
    
    # Load song data
    logger.info(f"Loading song data from {args.input}")
    try:
        songs_df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(songs_df)} songs")
    except Exception as e:
        logger.error(f"Error loading song data: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Initialize embedder
    embedder = SongEmbedder(
        batch_size=args.batch_size,
        device=args.device,
        embedding_store_path=args.output
    )
    
    # Sample if requested
    if args.max_songs is not None and args.max_songs < len(songs_df):
        logger.info(f"Sampling {args.max_songs} out of {len(songs_df)} songs")
        sampled_df = songs_df.sample(args.max_songs, random_state=42)
    else:
        sampled_df = songs_df
        
    # Process in chunks if needed (for very large datasets)
    if args.chunk_size and len(sampled_df) > args.chunk_size:
        logger.info(f"Processing {len(sampled_df)} songs in chunks of {args.chunk_size}")
        
        # First chunk with a different output path
        first_chunk = sampled_df.iloc[:args.chunk_size]
        temp_output = args.output + ".temp"
        embedder.embedding_store_path = temp_output
        
        # Embed first chunk
        try:
            embedder.embed_songs(first_chunk, max_songs=None)
            logger.info(f"Completed first chunk of {len(first_chunk)} songs")
        except Exception as e:
            logger.error(f"Error processing first chunk: {e}")
            logger.error(traceback.format_exc())
            return
            
        # Process remaining chunks (future implementation)
        logger.info("Processing in chunks is not fully implemented yet. First chunk completed successfully.")
        
        # Rename temp file to final output
        try:
            if os.path.exists(temp_output):
                if os.path.exists(args.output):
                    os.remove(args.output)
                os.rename(temp_output, args.output)
                
                # Also rename the URI index
                temp_index = os.path.splitext(temp_output)[0] + '_uri_index.npz'
                final_index = os.path.splitext(args.output)[0] + '_uri_index.npz'
                
                if os.path.exists(temp_index):
                    if os.path.exists(final_index):
                        os.remove(final_index)
                    os.rename(temp_index, final_index)
        except Exception as e:
            logger.error(f"Error renaming output files: {e}")
            logger.error(traceback.format_exc())
    else:
        # Process entire dataset at once
        try:
            embedder.embed_songs(sampled_df, max_songs=None)
        except Exception as e:
            logger.error(f"Error embedding songs: {e}")
            logger.error(traceback.format_exc())
            return
    
    # Test the lookup functionality
    logger.info("\nTesting lookup functionality...")
    try:
        lookup = SongEmbeddingLookup(args.output)
        
        # Get a sample URI from the SAMPLED dataframe
        sample_uri = sampled_df.iloc[0]['track_uri']
        logger.info(f"Sample URI: {sample_uri}")
        
        # Get and print info
        try:
            info = lookup.get_track_info(sample_uri)
            logger.info(f"Track: {info['track_name']}")
            logger.info(f"Artist: {info['artist_name']}")
            logger.info(f"Album: {info['album_name']}")
            logger.info(f"Popularity count: {info['count']}")
            logger.info(f"Embedding shape: {info['embedding'].shape}")
        except Exception as e:
            logger.error(f"Error testing lookup: {e}")
            logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error initializing lookup: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()