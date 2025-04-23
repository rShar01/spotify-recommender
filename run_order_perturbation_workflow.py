#!/usr/bin/env python3
"""
Order Perturbation Workflow Script

Runs the entire order perturbation workflow, from data preparation to model training and evaluation.
"""

import argparse
import os
import subprocess
import time
from datetime import datetime
import traceback
import json

def run_command(cmd, description=None):
    """
    Run a command and print output
    
    Args:
        cmd: Command to run (list of strings)
        description: Description of the command
        
    Returns:
        True if successful, False otherwise
    """
    if description:
        print(f"\n=== {description} ===\n")
        
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def run_workflow(playlist_data_dir, 
                song_data_path,
                output_dir,
                num_playlist_files=10,
                train_regular=True,
                train_robust=True,
                evaluate_models=True,
                compare_models=True,
                num_epochs=10,
                perturbation_prob=0.7,
                device=None):
    """
    Run the entire order perturbation workflow
    
    Args:
        playlist_data_dir: Directory containing playlist JSON files
        song_data_path: Path to song data H5 file
        output_dir: Directory to save all outputs
        num_playlist_files: Number of playlist files to use
        train_regular: Whether to train a regular LSTM model
        train_robust: Whether to train an order-robust LSTM model
        evaluate_models: Whether to evaluate the models
        compare_models: Whether to compare the models
        num_epochs: Number of epochs for training
        perturbation_prob: Probability of applying perturbation during training
        device: Device to use for training/evaluation
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    data_dir = os.path.join(output_dir, "data")
    models_dir = os.path.join(output_dir, "models")
    eval_dir = os.path.join(output_dir, "eval_results")
    comparison_dir = os.path.join(output_dir, "comparisons")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Define paths
    train_playlists_path = os.path.join(data_dir, "train_playlists.json")
    val_playlists_path = os.path.join(data_dir, "val_playlists.json")
    test_playlists_path = os.path.join(data_dir, "test_playlists.json")
    
    regular_model_dir = os.path.join(models_dir, "regular_lstm")
    robust_model_dir = os.path.join(models_dir, "order_robust_lstm")
    
    regular_model_path = os.path.join(regular_model_dir, "best_model.pt")
    robust_model_path = os.path.join(robust_model_dir, "best_model.pt")
    
    # Step 1: Process playlist data
    if not (os.path.exists(train_playlists_path) and 
            os.path.exists(val_playlists_path) and 
            os.path.exists(test_playlists_path)):
        
        success = run_command(
            [
                "python", "playlist_loader.py",
                "--data-dir", playlist_data_dir,
                "--output-dir", data_dir,
                "--num-files", str(num_playlist_files),
                "--min-tracks", "5",
                "--max-tracks", "50"
            ],
            "Processing playlist data"
        )
        
        if not success:
            print("Failed to process playlist data. Exiting.")
            return
    else:
        print("Playlist data already processed. Skipping.")
    
    # Step 2: Train regular LSTM model
    if train_regular and not os.path.exists(regular_model_path):
        os.makedirs(regular_model_dir, exist_ok=True)
        
        cmd = [
            "python", "order_train.py",
            "--train-playlists", train_playlists_path,
            "--val-playlists", val_playlists_path,
            "--song-data", song_data_path,
            "--output-dir", regular_model_dir,
            "--perturbation-prob", "0.0",  # No perturbation for regular model
            "--num-epochs", str(num_epochs),
            "--batch-size", "64"
        ]
        
        if device:
            cmd.extend(["--device", device])
            
        success = run_command(cmd, "Training regular LSTM model")
        
        if not success:
            print("Failed to train regular LSTM model. Exiting.")
            return
    elif train_regular:
        print("Regular LSTM model already trained. Skipping.")
    
    # Step 3: Train order-robust LSTM model
    if train_robust and not os.path.exists(robust_model_path):
        os.makedirs(robust_model_dir, exist_ok=True)
        
        cmd = [
            "python", "order_train.py",
            "--train-playlists", train_playlists_path,
            "--val-playlists", val_playlists_path,
            "--song-data", song_data_path,
            "--output-dir", robust_model_dir,
            "--perturbation-prob", str(perturbation_prob),
            "--perturbation-types", "swap", "shift", "reverse", "shuffle",
            "--num-epochs", str(num_epochs),
            "--batch-size", "64"
        ]
        
        if device:
            cmd.extend(["--device", device])
            
        success = run_command(cmd, "Training order-robust LSTM model")
        
        if not success:
            print("Failed to train order-robust LSTM model. Exiting.")
            return
    elif train_robust:
        print("Order-robust LSTM model already trained. Skipping.")
    
    # Step 4: Evaluate models
    if evaluate_models:
        # Evaluate regular model
        if os.path.exists(regular_model_path):
            regular_eval_dir = os.path.join(eval_dir, "regular_lstm")
            os.makedirs(regular_eval_dir, exist_ok=True)
            
            cmd = [
                "python", "run_order_eval.py",
                "--model", regular_model_path,
                "--playlists", test_playlists_path,
                "--song-data", song_data_path,
                "--output-dir", regular_eval_dir,
                "--perturbation-types", "swap", "shift", "reverse", "shuffle",
                "--max-swaps", "3",
                "--batch-size", "32"
            ]
            
            if device:
                cmd.extend(["--device", device])
                
            success = run_command(cmd, "Evaluating regular LSTM model")
            
            if not success:
                print("Failed to evaluate regular LSTM model.")
        else:
            print("Regular LSTM model not found. Skipping evaluation.")
        
        # Evaluate robust model
        if os.path.exists(robust_model_path):
            robust_eval_dir = os.path.join(eval_dir, "order_robust_lstm")
            os.makedirs(robust_eval_dir, exist_ok=True)
            
            cmd = [
                "python", "run_order_eval.py",
                "--model", robust_model_path,
                "--playlists", test_playlists_path,
                "--song-data", song_data_path,
                "--output-dir", robust_eval_dir,
                "--perturbation-types", "swap", "shift", "reverse", "shuffle",
                "--max-swaps", "3",
                "--batch-size", "32"
            ]
            
            if device:
                cmd.extend(["--device", device])
                
            success = run_command(cmd, "Evaluating order-robust LSTM model")
            
            if not success:
                print("Failed to evaluate order-robust LSTM model.")
        else:
            print("Order-robust LSTM model not found. Skipping evaluation.")
    
    # Step 5: Compare models
    if compare_models and os.path.exists(os.path.join(eval_dir, "regular_lstm")) and os.path.exists(os.path.join(eval_dir, "order_robust_lstm")):
        cmd = [
            "python", "compare_models.py",
            "--results-dir", eval_dir,
            "--model-names", "regular_lstm", "order_robust_lstm",
            "--output-dir", comparison_dir
        ]
        
        success = run_command(cmd, "Comparing models")
        
        if not success:
            print("Failed to compare models.")
    elif compare_models:
        print("Evaluation results not found for both models. Skipping comparison.")
    
    # Create summary file
    summary = {
        'workflow_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'playlist_data_dir': playlist_data_dir,
        'song_data_path': song_data_path,
        'output_dir': output_dir,
        'num_playlist_files': num_playlist_files,
        'num_epochs': num_epochs,
        'perturbation_prob': perturbation_prob,
        'device': device,
        'steps_completed': {
            'data_processing': os.path.exists(train_playlists_path),
            'regular_model_training': os.path.exists(regular_model_path),
            'robust_model_training': os.path.exists(robust_model_path),
            'regular_model_evaluation': os.path.exists(os.path.join(eval_dir, "regular_lstm")),
            'robust_model_evaluation': os.path.exists(os.path.join(eval_dir, "order_robust_lstm")),
            'model_comparison': os.path.exists(os.path.join(comparison_dir))
        }
    }
    
    summary_path = os.path.join(output_dir, "workflow_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nWorkflow summary saved to {summary_path}")
    print("\nWorkflow completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run order perturbation workflow")
    
    # Required arguments
    parser.add_argument("--playlist-data-dir", required=True, 
                        help="Directory containing playlist JSON files")
    parser.add_argument("--song-data", required=True, 
                        help="Path to song data H5 file")
    parser.add_argument("--output-dir", required=True, 
                        help="Directory to save all outputs")
    
    # Optional arguments
    parser.add_argument("--num-playlist-files", type=int, default=10,
                        help="Number of playlist files to use")
    parser.add_argument("--skip-regular", action="store_true",
                        help="Skip training regular LSTM model")
    parser.add_argument("--skip-robust", action="store_true",
                        help="Skip training order-robust LSTM model")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip model evaluation")
    parser.add_argument("--skip-compare", action="store_true",
                        help="Skip model comparison")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of epochs for training")
    parser.add_argument("--perturbation-prob", type=float, default=0.7,
                        help="Probability of applying perturbation during training")
    parser.add_argument("--device", default=None,
                        help="Device to use for training/evaluation (cuda/cpu)")
    
    args = parser.parse_args()
    
    try:
        # Run workflow
        run_workflow(
            playlist_data_dir=args.playlist_data_dir,
            song_data_path=args.song_data,
            output_dir=args.output_dir,
            num_playlist_files=args.num_playlist_files,
            train_regular=not args.skip_regular,
            train_robust=not args.skip_robust,
            evaluate_models=not args.skip_eval,
            compare_models=not args.skip_compare,
            num_epochs=args.num_epochs,
            perturbation_prob=args.perturbation_prob,
            device=args.device
        )
        
    except Exception as e:
        print(f"Error during workflow: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()