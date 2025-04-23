#!/usr/bin/env python3
# compare_order_robustness.py - Compare regular LSTM vs order-robust LSTM

import argparse
import json
import os
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch

from order_eval import evaluate_with_order_perturbation

def run_comparison(
    regular_model_path,
    robust_model_path,
    data_dir,
    embedding_path,
    song_csv_path,
    output_dir,
    perturbation_types=None,
    max_perturbations=3,
    batch_size=32,
    max_files=20,
    min_songs=3,
    device=None
):
    """
    Run comprehensive comparison between regular and order-robust LSTM models
    
    Args:
        regular_model_path: Path to regular LSTM model
        robust_model_path: Path to order-robust LSTM model
        data_dir: Directory with playlist data
        embedding_path: Path to song embeddings
        song_csv_path: Path to song popularity CSV
        output_dir: Directory to save results
        perturbation_types: List of perturbation types to test
        max_perturbations: Maximum number of perturbations to test
        batch_size: Batch size for evaluation
        max_files: Maximum number of data files to process
        min_songs: Minimum songs per playlist
        device: Device to run on ('cuda' or 'cpu')
    """
    # Set default perturbation types if not provided
    if perturbation_types is None:
        perturbation_types = ['swap', 'shift', 'reverse', 'shuffle']
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start time
    start_time = time.time()
    
    # Results dictionary
    results = {
        'regular_model': regular_model_path,
        'robust_model': robust_model_path,
        'perturbation_results': {},
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'comparison_time': None
    }
    
    # Run evaluations for each perturbation type and count
    for perturbation_type in perturbation_types:
        results['perturbation_results'][perturbation_type] = {
            'regular_model': {},
            'robust_model': {}
        }
        
        for num_perturbations in range(1, max_perturbations + 1):
            print(f"\n{'='*80}")
            print(f"Evaluating with {perturbation_type} perturbation (count={num_perturbations})")
            print(f"{'='*80}")
            
            # Evaluate regular model
            print(f"\nEvaluating regular model: {regular_model_path}")
            regular_output_path = output_dir / f"regular_{perturbation_type}_{num_perturbations}.json"
            
            regular_result = evaluate_with_order_perturbation(
                model_path=regular_model_path,
                data_dir=data_dir,
                embedding_path=embedding_path,
                song_csv_path=song_csv_path,
                perturbation_type=perturbation_type,
                num_perturbations=num_perturbations,
                output_path=regular_output_path,
                batch_size=batch_size,
                max_files=max_files,
                min_songs=min_songs,
                device=device
            )
            
            # Evaluate robust model
            print(f"\nEvaluating robust model: {robust_model_path}")
            robust_output_path = output_dir / f"robust_{perturbation_type}_{num_perturbations}.json"
            
            robust_result = evaluate_with_order_perturbation(
                model_path=robust_model_path,
                data_dir=data_dir,
                embedding_path=embedding_path,
                song_csv_path=song_csv_path,
                perturbation_type=perturbation_type,
                num_perturbations=num_perturbations,
                output_path=robust_output_path,
                batch_size=batch_size,
                max_files=max_files,
                min_songs=min_songs,
                device=device
            )
            
            # Store results
            if regular_result:
                results['perturbation_results'][perturbation_type]['regular_model'][num_perturbations] = {
                    'dynamic_evaluation': regular_result['dynamic_evaluation'],
                    'topk_evaluation': regular_result['topk_evaluation']
                }
            
            if robust_result:
                results['perturbation_results'][perturbation_type]['robust_model'][num_perturbations] = {
                    'dynamic_evaluation': robust_result['dynamic_evaluation'],
                    'topk_evaluation': robust_result['topk_evaluation']
                }
    
    # Record total time
    results['comparison_time'] = time.time() - start_time
    
    # Save overall results
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComparison completed in {results['comparison_time'] / 60:.2f} minutes")
    print(f"Results saved to {summary_path}")
    
    # Generate comparison plots
    generate_comparison_plots(results, output_dir)
    
    return results

def generate_comparison_plots(results, output_dir):
    """Generate plots comparing regular and robust models"""
    output_dir = Path(output_dir)
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Extract perturbation types
    perturbation_types = list(results['perturbation_results'].keys())
    
    # Plot top-k metrics
    for k in [1, 5, 10]:
        plt.figure(figsize=(15, 10))
        
        for i, p_type in enumerate(perturbation_types):
            # Get data for regular model
            regular_data = []
            perturbation_counts = []
            
            for count, data in results['perturbation_results'][p_type]['regular_model'].items():
                if f'top_{k}_accuracy' in data['topk_evaluation']:
                    regular_data.append(data['topk_evaluation'][f'top_{k}_accuracy'])
                    perturbation_counts.append(int(count))
            
            # Get data for robust model
            robust_data = []
            for count in perturbation_counts:
                if count in results['perturbation_results'][p_type]['robust_model']:
                    robust_data.append(
                        results['perturbation_results'][p_type]['robust_model'][count]['topk_evaluation'][f'top_{k}_accuracy']
                    )
            
            # Plot if we have data
            if regular_data and robust_data and len(regular_data) == len(robust_data):
                plt.subplot(2, 2, i+1)
                plt.plot(perturbation_counts, regular_data, 'o-', label=f'Regular Model')
                plt.plot(perturbation_counts, robust_data, 's-', label=f'Order-Robust Model')
                plt.title(f'{p_type.capitalize()} Perturbation')
                plt.xlabel('Number of Perturbations')
                plt.ylabel(f'Top-{k} Accuracy')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
        
        plt.suptitle(f'Top-{k} Accuracy Comparison: Regular vs Order-Robust LSTM', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(plots_dir / f'top_{k}_comparison.png', dpi=300)
        plt.close()
    
    # Plot MRR (Mean Reciprocal Rank)
    plt.figure(figsize=(15, 10))
    
    for i, p_type in enumerate(perturbation_types):
        # Get data for regular model
        regular_data = []
        perturbation_counts = []
        
        for count, data in results['perturbation_results'][p_type]['regular_model'].items():
            if 'mrr' in data['topk_evaluation']:
                regular_data.append(data['topk_evaluation']['mrr'])
                perturbation_counts.append(int(count))
        
        # Get data for robust model
        robust_data = []
        for count in perturbation_counts:
            if count in results['perturbation_results'][p_type]['robust_model']:
                robust_data.append(
                    results['perturbation_results'][p_type]['robust_model'][count]['topk_evaluation']['mrr']
                )
        
        # Plot if we have data
        if regular_data and robust_data and len(regular_data) == len(robust_data):
            plt.subplot(2, 2, i+1)
            plt.plot(perturbation_counts, regular_data, 'o-', label=f'Regular Model')
            plt.plot(perturbation_counts, robust_data, 's-', label=f'Order-Robust Model')
            plt.title(f'{p_type.capitalize()} Perturbation')
            plt.xlabel('Number of Perturbations')
            plt.ylabel('Mean Reciprocal Rank (MRR)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
    
    plt.suptitle('Mean Reciprocal Rank Comparison: Regular vs Order-Robust LSTM', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plots_dir / 'mrr_comparison.png', dpi=300)
    plt.close()
    
    # Plot dynamic threshold metrics
    for metric in ['precision', 'recall', 'f1']:
        plt.figure(figsize=(15, 10))
        
        for i, p_type in enumerate(perturbation_types):
            # Get data for regular model
            regular_data = []
            perturbation_counts = []
            
            for count, data in results['perturbation_results'][p_type]['regular_model'].items():
                if metric in data['dynamic_evaluation']['best_metrics']:
                    regular_data.append(data['dynamic_evaluation']['best_metrics'][metric])
                    perturbation_counts.append(int(count))
            
            # Get data for robust model
            robust_data = []
            for count in perturbation_counts:
                if count in results['perturbation_results'][p_type]['robust_model']:
                    robust_data.append(
                        results['perturbation_results'][p_type]['robust_model'][count]['dynamic_evaluation']['best_metrics'][metric]
                    )
            
            # Plot if we have data
            if regular_data and robust_data and len(regular_data) == len(robust_data):
                plt.subplot(2, 2, i+1)
                plt.plot(perturbation_counts, regular_data, 'o-', label=f'Regular Model')
                plt.plot(perturbation_counts, robust_data, 's-', label=f'Order-Robust Model')
                plt.title(f'{p_type.capitalize()} Perturbation')
                plt.xlabel('Number of Perturbations')
                plt.ylabel(f'{metric.capitalize()}')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
        
        plt.suptitle(f'{metric.capitalize()} Comparison: Regular vs Order-Robust LSTM', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(plots_dir / f'{metric}_comparison.png', dpi=300)
        plt.close()
    
    print(f"Comparison plots saved to {plots_dir}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Compare regular LSTM vs order-robust LSTM")
    
    # Required arguments
    parser.add_argument("--regular-model", required=True, help="Path to regular LSTM model")
    parser.add_argument("--robust-model", required=True, help="Path to order-robust LSTM model")
    parser.add_argument("--data", required=True, help="Path to playlist data directory")
    parser.add_argument("--embeddings", required=True, help="Path to song embeddings file")
    parser.add_argument("--songs", required=True, help="Path to song popularity CSV")
    parser.add_argument("--output", required=True, help="Directory to save results")
    
    # Optional arguments
    parser.add_argument("--perturbation-types", nargs='+', 
                        choices=['swap', 'shift', 'reverse', 'shuffle'], 
                        default=['swap', 'shift', 'reverse', 'shuffle'],
                        help="Types of perturbations to test")
    parser.add_argument("--max-perturbations", type=int, default=3, 
                        help="Maximum number of perturbations to test")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-files", type=int, default=20, help="Maximum number of data files to process")
    parser.add_argument("--min-songs", type=int, default=3, help="Minimum songs per playlist")
    parser.add_argument("--device", default=None, help="Device to run on (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Run comparison
    run_comparison(
        regular_model_path=args.regular_model,
        robust_model_path=args.robust_model,
        data_dir=args.data,
        embedding_path=args.embeddings,
        song_csv_path=args.songs,
        output_dir=args.output,
        perturbation_types=args.perturbation_types,
        max_perturbations=args.max_perturbations,
        batch_size=args.batch_size,
        max_files=args.max_files,
        min_songs=args.min_songs,
        device=args.device
    )

if __name__ == "__main__":
    main()