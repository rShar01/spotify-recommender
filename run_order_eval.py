#!/usr/bin/env python3
"""
Comprehensive Order Perturbation Evaluation Script

Evaluates LSTM model robustness to different types and intensities of order perturbations.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime
import traceback

def run_evaluation(model_path, 
                  playlists_path, 
                  song_data_path,
                  output_dir,
                  perturbation_types=None,
                  max_swaps=3,
                  batch_size=32,
                  device=None):
    """
    Run comprehensive evaluation with different perturbation types and intensities
    
    Args:
        model_path: Path to the trained model
        playlists_path: Path to the playlists JSON file
        song_data_path: Path to the song data H5 file
        output_dir: Directory to save evaluation results
        perturbation_types: List of perturbation types to evaluate
        max_swaps: Maximum number of swaps to evaluate
        batch_size: Batch size for evaluation
        device: Device to use for evaluation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default perturbation types if none provided
    if perturbation_types is None:
        perturbation_types = ['swap', 'shift', 'reverse', 'shuffle']
        
    # Get model name from path
    model_name = os.path.basename(model_path).split('.')[0]
    
    # Create summary file
    summary_path = os.path.join(output_dir, f"{model_name}_order_eval_summary.json")
    summary = {
        'model_path': str(model_path),
        'playlists_path': str(playlists_path),
        'song_data_path': str(song_data_path),
        'evaluation_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'results': {}
    }
    
    # Run evaluations for each perturbation type and intensity
    for perturbation_type in perturbation_types:
        summary['results'][perturbation_type] = {}
        
        # For swap, evaluate different numbers of swaps
        if perturbation_type == 'swap':
            for num_swaps in range(1, max_swaps + 1):
                print(f"\n=== Evaluating {perturbation_type} with {num_swaps} swaps ===\n")
                
                # Output file for this configuration
                output_file = os.path.join(
                    output_dir, 
                    f"{model_name}_{perturbation_type}_{num_swaps}_swaps.json"
                )
                
                # Build command
                cmd = [
                    "python", "order_eval.py",
                    "--model", str(model_path),
                    "--playlists", str(playlists_path),
                    "--song-data", str(song_data_path),
                    "--perturbation-type", perturbation_type,
                    "--num-swaps", str(num_swaps),
                    "--batch-size", str(batch_size),
                    "--output", output_file
                ]
                
                if device:
                    cmd.extend(["--device", device])
                    
                # Run evaluation
                try:
                    subprocess.run(cmd, check=True)
                    
                    # Load results
                    with open(output_file, 'r') as f:
                        results = json.load(f)
                        
                    # Add to summary
                    summary['results'][perturbation_type][f"{num_swaps}_swaps"] = {
                        'top_k_accuracy': results['top_k_accuracy'],
                        'num_playlists': results['num_playlists']
                    }
                    
                except Exception as e:
                    print(f"Error evaluating {perturbation_type} with {num_swaps} swaps: {e}")
                    traceback.print_exc()
        else:
            # For other perturbation types, just run once
            print(f"\n=== Evaluating {perturbation_type} ===\n")
            
            # Output file for this configuration
            output_file = os.path.join(
                output_dir, 
                f"{model_name}_{perturbation_type}.json"
            )
            
            # Build command
            cmd = [
                "python", "order_eval.py",
                "--model", str(model_path),
                "--playlists", str(playlists_path),
                "--song-data", str(song_data_path),
                "--perturbation-type", perturbation_type,
                "--batch-size", str(batch_size),
                "--output", output_file
            ]
            
            if device:
                cmd.extend(["--device", device])
                
            # Run evaluation
            try:
                subprocess.run(cmd, check=True)
                
                # Load results
                with open(output_file, 'r') as f:
                    results = json.load(f)
                    
                # Add to summary
                summary['results'][perturbation_type] = {
                    'top_k_accuracy': results['top_k_accuracy'],
                    'num_playlists': results['num_playlists']
                }
                
            except Exception as e:
                print(f"Error evaluating {perturbation_type}: {e}")
                traceback.print_exc()
    
    # Save summary
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nEvaluation summary saved to {summary_path}")
    
    # Print overall summary
    print("\nOverall Evaluation Summary:")
    for perturbation_type, results in summary['results'].items():
        print(f"\n{perturbation_type.upper()}:")
        
        if perturbation_type == 'swap':
            for num_swaps, swap_results in results.items():
                print(f"  {num_swaps}:")
                for k, acc in swap_results['top_k_accuracy'].items():
                    print(f"    Top-{k}: {acc:.4f}")
        else:
            for k, acc in results['top_k_accuracy'].items():
                print(f"  Top-{k}: {acc:.4f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive order perturbation evaluation"
    )
    
    # Required arguments
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--playlists", required=True, help="Path to the playlists JSON file")
    parser.add_argument("--song-data", required=True, help="Path to the song data H5 file")
    parser.add_argument("--output-dir", required=True, help="Directory to save evaluation results")
    
    # Optional arguments
    parser.add_argument("--perturbation-types", nargs='+',
                        choices=["swap", "shift", "reverse", "shuffle"],
                        default=["swap", "shift", "reverse", "shuffle"],
                        help="Types of perturbation to evaluate")
    parser.add_argument("--max-swaps", type=int, default=3,
                        help="Maximum number of swaps to evaluate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--device", default=None,
                        help="Device to use for evaluation (cuda/cpu)")
    
    args = parser.parse_args()
    
    try:
        # Run evaluation
        run_evaluation(
            model_path=args.model,
            playlists_path=args.playlists,
            song_data_path=args.song_data,
            output_dir=args.output_dir,
            perturbation_types=args.perturbation_types,
            max_swaps=args.max_swaps,
            batch_size=args.batch_size,
            device=args.device
        )
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()