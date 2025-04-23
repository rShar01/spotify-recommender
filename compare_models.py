#!/usr/bin/env python3
"""
Model Comparison Script

Compares the performance of regular LSTM models versus order-robust LSTM models
on different types of order perturbations.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import subprocess
import traceback

def load_evaluation_results(results_dir, model_names):
    """
    Load evaluation results for multiple models
    
    Args:
        results_dir: Directory containing evaluation results
        model_names: List of model names to compare
        
    Returns:
        Dictionary of evaluation results by model
    """
    results = {}
    
    for model_name in model_names:
        summary_path = os.path.join(results_dir, f"{model_name}_order_eval_summary.json")
        
        try:
            with open(summary_path, 'r') as f:
                model_results = json.load(f)
                results[model_name] = model_results
                print(f"Loaded results for {model_name}")
        except Exception as e:
            print(f"Error loading results for {model_name}: {e}")
            traceback.print_exc()
            
    return results


def plot_comparison(results, model_names, k_values=None, output_dir=None):
    """
    Plot comparison of model performance
    
    Args:
        results: Dictionary of evaluation results by model
        model_names: List of model names to compare
        k_values: List of k values to include in plots
        output_dir: Directory to save plots
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    if k_values is None:
        # Use all k values from the first model's first perturbation type
        first_model = model_names[0]
        first_pert = next(iter(results[first_model]['results']))
        
        if first_pert == 'swap':
            first_swap = next(iter(results[first_model]['results'][first_pert]))
            k_values = list(map(int, results[first_model]['results'][first_pert][first_swap]['top_k_accuracy'].keys()))
        else:
            k_values = list(map(int, results[first_model]['results'][first_pert]['top_k_accuracy'].keys()))
            
    # Sort k values
    k_values = sorted(k_values)
    
    # Define perturbation types
    perturbation_types = ['swap', 'shift', 'reverse', 'shuffle']
    
    # Plot for each perturbation type
    for pert_type in perturbation_types:
        if pert_type == 'swap':
            # For swap, plot for each number of swaps
            max_swaps = 0
            for model_name in model_names:
                if pert_type in results[model_name]['results']:
                    num_swaps = len(results[model_name]['results'][pert_type])
                    max_swaps = max(max_swaps, num_swaps)
                    
            for num_swaps in range(1, max_swaps + 1):
                swap_key = f"{num_swaps}_swaps"
                
                # Create figure
                plt.figure(figsize=(10, 6))
                
                # Plot for each model
                for model_name in model_names:
                    if (pert_type in results[model_name]['results'] and 
                        swap_key in results[model_name]['results'][pert_type]):
                        
                        # Get accuracies for each k
                        accuracies = []
                        for k in k_values:
                            acc = results[model_name]['results'][pert_type][swap_key]['top_k_accuracy'].get(str(k), 0)
                            accuracies.append(acc)
                            
                        # Plot
                        plt.plot(k_values, accuracies, marker='o', label=model_name)
                
                # Set labels and title
                plt.xlabel('k (Top-k Accuracy)')
                plt.ylabel('Accuracy')
                plt.title(f'Model Comparison - {pert_type.capitalize()} ({num_swaps} swaps)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                # Save or show
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f"comparison_{pert_type}_{num_swaps}_swaps.png"))
                    plt.close()
                else:
                    plt.show()
        else:
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot for each model
            for model_name in model_names:
                if pert_type in results[model_name]['results']:
                    # Get accuracies for each k
                    accuracies = []
                    for k in k_values:
                        acc = results[model_name]['results'][pert_type]['top_k_accuracy'].get(str(k), 0)
                        accuracies.append(acc)
                        
                    # Plot
                    plt.plot(k_values, accuracies, marker='o', label=model_name)
            
            # Set labels and title
            plt.xlabel('k (Top-k Accuracy)')
            plt.ylabel('Accuracy')
            plt.title(f'Model Comparison - {pert_type.capitalize()}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Save or show
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"comparison_{pert_type}.png"))
                plt.close()
            else:
                plt.show()
    
    # Create a summary bar chart for a specific k value
    k_for_summary = 10  # Use top-10 accuracy for summary
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define width of bars
    bar_width = 0.35 / len(model_names)
    
    # Define x positions
    x_pos = np.arange(len(perturbation_types))
    
    # Plot for each model
    for i, model_name in enumerate(model_names):
        # Get accuracies for each perturbation type
        accuracies = []
        for pert_type in perturbation_types:
            if pert_type == 'swap':
                # For swap, use 1 swap
                swap_key = "1_swaps"
                if (pert_type in results[model_name]['results'] and 
                    swap_key in results[model_name]['results'][pert_type]):
                    acc = results[model_name]['results'][pert_type][swap_key]['top_k_accuracy'].get(str(k_for_summary), 0)
                else:
                    acc = 0
            else:
                if pert_type in results[model_name]['results']:
                    acc = results[model_name]['results'][pert_type]['top_k_accuracy'].get(str(k_for_summary), 0)
                else:
                    acc = 0
                    
            accuracies.append(acc)
            
        # Plot
        plt.bar(x_pos + i * bar_width, accuracies, width=bar_width, label=model_name)
    
    # Set labels and title
    plt.xlabel('Perturbation Type')
    plt.ylabel(f'Top-{k_for_summary} Accuracy')
    plt.title(f'Model Comparison - Top-{k_for_summary} Accuracy by Perturbation Type')
    plt.xticks(x_pos + bar_width * (len(model_names) - 1) / 2, [p.capitalize() for p in perturbation_types])
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend()
    
    # Save or show
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"comparison_summary_top{k_for_summary}.png"))
        plt.close()
    else:
        plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Compare performance of regular vs order-robust LSTM models"
    )
    
    # Required arguments
    parser.add_argument("--results-dir", required=True, 
                        help="Directory containing evaluation results")
    parser.add_argument("--model-names", nargs='+', required=True,
                        help="Names of models to compare")
    
    # Optional arguments
    parser.add_argument("--k-values", nargs='+', type=int,
                        default=[1, 5, 10, 25, 50, 100],
                        help="k values to include in plots")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save plots")
    
    args = parser.parse_args()
    
    try:
        # Load results
        results = load_evaluation_results(args.results_dir, args.model_names)
        
        # Plot comparison
        plot_comparison(results, args.model_names, args.k_values, args.output_dir)
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()