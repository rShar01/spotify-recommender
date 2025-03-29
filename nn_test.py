import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
import random
from tqdm import tqdm
import json

# Import the model class from your original code
class PlaylistRecommenderModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dims: list[int] = [512, 256]):
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


def load_model(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load a trained recommender model"""
    print(f"Loading model from {model_path}")
    model_data = torch.load(model_path, map_location=device)
    
    # Create model
    model = PlaylistRecommenderModel(
        embedding_dim=model_data['embedding_dim'],
        hidden_dims=model_data.get('hidden_dims', [512, 256])
    )
    
    # Load weights
    model.load_state_dict(model_data['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, model_data


def create_synthetic_data(embedding_dim=768, n_train=100, n_test=50, n_songs=1000, seed=42):
    """
    Create synthetic data for testing:
    - Random embeddings
    - Random song matrices with "clusters" of similar playlists
    - Guaranteed overlap between test and train for evaluation
    """
    print(f"Creating synthetic data with {n_train} train and {n_test} test playlists")
    np.random.seed(seed)
    
    # Create 5 "template" playlists with specific song patterns
    templates = np.zeros((5, n_songs))
    for i in range(5):
        # Each template has 50-100 songs
        n_template_songs = np.random.randint(50, 100)
        song_indices = np.random.choice(n_songs, n_template_songs, replace=False)
        templates[i, song_indices] = 1
    
    # Generate train playlists based on templates (with some noise)
    train_matrix = np.zeros((n_train, n_songs))
    train_embeddings = np.zeros((n_train, embedding_dim))
    
    for i in range(n_train):
        # Pick a template
        template_idx = i % 5
        template = templates[template_idx]
        
        # Add noise (flip 5-10% of bits)
        noise_factor = np.random.uniform(0.05, 0.1)
        n_flips = int(noise_factor * n_songs)
        flip_indices = np.random.choice(n_songs, n_flips, replace=False)
        
        # Copy template and apply noise
        playlist = template.copy()
        for idx in flip_indices:
            playlist[idx] = 1 - playlist[idx]  # Flip the bit
        
        train_matrix[i] = playlist
        
        # Create corresponding embedding
        # Base component from template
        base_embedding = np.random.normal(0, 1, embedding_dim)
        # Add playlist-specific noise
        noise = np.random.normal(0, 0.2, embedding_dim)
        train_embeddings[i] = base_embedding + noise
    
    # Generate test playlists
    test_matrix = np.zeros((n_test, n_songs))
    test_embeddings = np.zeros((n_test, embedding_dim))
    
    for i in range(n_test):
        # Pick a template
        template_idx = i % 5
        template = templates[template_idx]
        
        # Add more noise for test playlists
        noise_factor = np.random.uniform(0.1, 0.15)
        n_flips = int(noise_factor * n_songs)
        flip_indices = np.random.choice(n_songs, n_flips, replace=False)
        
        # Copy template and apply noise
        playlist = template.copy()
        for idx in flip_indices:
            playlist[idx] = 1 - playlist[idx]  # Flip the bit
        
        test_matrix[i] = playlist
        
        # Create corresponding embedding
        # Similar to train embeddings from same template, but with more noise
        template_idx_in_train = template_idx  # Find a train playlist with same template
        base_embedding = train_embeddings[template_idx_in_train].copy()
        noise = np.random.normal(0, 0.3, embedding_dim)
        test_embeddings[i] = base_embedding + noise
    
    print("Synthetic data created.")
    return train_embeddings, test_embeddings, train_matrix, test_matrix


def run_synthetic_test(model, train_embeddings, test_embeddings, train_matrix, test_matrix, n_masks=100, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run evaluation with synthetic data
    """
    print(f"Running synthetic test with {n_masks} masks")
    
    # Find test playlists with songs
    row_sums = np.sum(test_matrix, axis=1)
    valid_rows = np.where(row_sums > 0)[0]
    
    print(f"Found {len(valid_rows)} test playlists with songs")
    
    # Create masks
    masks = []
    mask_row_indices = np.random.choice(valid_rows, min(n_masks, len(valid_rows)), replace=False)
    
    for row_idx in mask_row_indices:
        # Find songs in this playlist
        song_indices = np.where(test_matrix[row_idx] > 0)[0]
        if len(song_indices) > 0:
            # Choose random song
            col_idx = np.random.choice(song_indices)
            masks.append((row_idx, col_idx))
    
    print(f"Created {len(masks)} masks")
    
    # Convert embeddings to tensors
    train_tensors = torch.FloatTensor(train_embeddings).to(device)
    
    # Track results
    correct = 0
    total = 0
    
    # Process masks
    for row_idx, col_idx in tqdm(masks, desc="Evaluating masks"):
        # Get test embedding
        test_embedding = test_embeddings[row_idx]
        test_tensor = torch.FloatTensor(test_embedding).unsqueeze(0).to(device)
        
        # Calculate similarity scores
        with torch.no_grad():
            # Create repeated tensor for batch processing
            test_repeated = test_tensor.repeat(len(train_tensors), 1)
            similarity_scores = model(test_repeated, train_tensors).cpu().numpy().flatten()
        
        # Find most similar playlist
        most_similar_idx = np.argmax(similarity_scores)
        
        # Check if the masked song is in the similar playlist
        if train_matrix[most_similar_idx, col_idx] > 0:
            correct += 1
        
        total += 1
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def test_models_with_synthetic_data(model_dir="models", embedding_dim=768, n_masks=100):
    """Test all models with synthetic data"""
    print("Starting synthetic data test")
    
    # Find models
    model_paths = list(Path(model_dir).glob("playlist_recommender_fast_*.pt"))
    
    if not model_paths:
        print("No models found!")
        return
    
    print(f"Found {len(model_paths)} models")
    
    # Create synthetic data
    train_embeddings, test_embeddings, train_matrix, test_matrix = create_synthetic_data(
        embedding_dim=embedding_dim,
        n_train=100,
        n_test=50,
        n_songs=1000
    )
    
    # Test each model
    results = []
    
    for model_path in model_paths:
        print(f"\n==== Testing {model_path.name} ====")
        
        # Extract learning rate from filename
        lr = float(model_path.name.split('_')[-1].replace('.pt', ''))
        
        try:
            # Load model
            model, model_data = load_model(str(model_path))
            
            # Check if embedding dimension matches
            model_embed_dim = model_data.get('embedding_dim', 768)
            if model_embed_dim != embedding_dim:
                print(f"Warning: Model embedding dim ({model_embed_dim}) doesn't match synthetic data ({embedding_dim})")
                print("Regenerating synthetic data with correct dimension...")
                train_embeddings, test_embeddings, train_matrix, test_matrix = create_synthetic_data(
                    embedding_dim=model_embed_dim,
                    n_train=100,
                    n_test=50,
                    n_songs=1000
                )
            
            # Run test
            eval_results = run_synthetic_test(
                model,
                train_embeddings,
                test_embeddings,
                train_matrix,
                test_matrix,
                n_masks=n_masks
            )
            
            # Record results
            result = {
                'model_name': model_path.name,
                'learning_rate': lr,
                'accuracy': eval_results['accuracy'],
                'correct': eval_results['correct'],
                'total': eval_results['total'],
            }
            
            results.append(result)
            
            # Print summary
            print(f"  Accuracy: {eval_results['accuracy']:.4f} ({eval_results['correct']}/{eval_results['total']})")
            
        except Exception as e:
            print(f"Error testing {model_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    with open('synthetic_test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print comparative results
    print("\nSynthetic Test Results:")
    print("-" * 60)
    print(f"{'Model':<30} {'Learning Rate':<15} {'Accuracy':<10}")
    print("-" * 60)
    
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{result['model_name']:<30} {result['learning_rate']:<15.5f} {result['accuracy']:<10.4f}")
    
    return results


def main():
    model_dir = "models"
    test_models_with_synthetic_data(model_dir=model_dir, n_masks=100)


if __name__ == "__main__":
    main()