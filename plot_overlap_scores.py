#!/usr/bin/env python3
"""
Script to plot real and approximate overlap scores, and recall values from binary files.
Each iteration is plotted as a separate line for overlap scores.
Recall values are plotted with separate lines for each probe combination.
"""

import os
import re
import struct
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def read_binary_doubles(filepath):
    """Read binary file containing double values."""
    try:
        # Use numpy's fromfile for efficient binary reading
        data = np.fromfile(filepath, dtype=np.float64)
        return data
    except Exception as e:
        # Fallback to struct method
        with open(filepath, 'rb') as f:
            data = f.read()
        num_doubles = len(data) // 8
        doubles = struct.unpack(f'{num_doubles}d', data)
        return np.array(doubles)

def find_iteration_files(pattern_prefix):
    """Find all files matching the pattern and extract iteration numbers."""
    files = {}
    for filepath in Path('.').glob(f'{pattern_prefix}*.bin'):
        match = re.search(r'iter_(\d+)', filepath.name)
        if match:
            iter_num = int(match.group(1))
            files[iter_num] = filepath
    return files

def plot_overlap_scores(approx_files, real_files, output_dir='.'):
    """Plot approximate and real overlap scores."""
    
    # Read approximate overlap scores
    approx_data = {}
    for iter_num, filepath in sorted(approx_files.items()):
        try:
            scores = read_binary_doubles(filepath)
            approx_data[iter_num] = scores
            print(f"Read approx overlap scores for iteration {iter_num}: {len(scores)} values")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    # Read real overlap scores
    real_data = {}
    for iter_num, filepath in sorted(real_files.items()):
        try:
            scores = read_binary_doubles(filepath)
            real_data[iter_num] = scores
            print(f"Read real overlap scores for iteration {iter_num}: {len(scores)} values")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    if not approx_data and not real_data:
        print("No overlap score files found!")
        return
    
    # Determine maximum cluster ID
    max_cluster_id = 0
    for scores in list(approx_data.values()) + list(real_data.values()):
        max_cluster_id = max(max_cluster_id, len(scores) - 1)
    
    # Plot approximate overlap scores
    if approx_data:
        plt.figure(figsize=(12, 6))
        for iter_num in sorted(approx_data.keys()):
            scores = approx_data[iter_num]
            cluster_ids = np.arange(len(scores))
            plt.plot(cluster_ids, scores, label=f'Iteration {iter_num}', marker='o', markersize=2, linewidth=1)
        
        plt.xlabel('Mega Cluster ID', fontsize=12)
        plt.ylabel('Approximate Overlap Score', fontsize=12)
        plt.title('Approximate Overlap Scores by Iteration', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'approx_overlap_scores.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved approximate overlap scores plot to {output_path}")
        plt.close()
    
    # Plot real overlap scores
    if real_data:
        plt.figure(figsize=(12, 6))
        for iter_num in sorted(real_data.keys()):
            scores = real_data[iter_num]
            cluster_ids = np.arange(len(scores))
            plt.plot(cluster_ids, scores, label=f'Iteration {iter_num}', marker='o', markersize=2, linewidth=1)
        
        plt.xlabel('Mega Cluster ID', fontsize=12)
        plt.ylabel('Real Overlap Score', fontsize=12)
        plt.title('Real Overlap Scores by Iteration', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'real_overlap_scores.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved real overlap scores plot to {output_path}")
        plt.close()

def plot_recall_scores(recall_files, output_dir='.'):
    """Plot recall scores with separate line for each iteration.
    X-axis: query index (0-9), Y-axis: recall.
    Each recall file contains 10 doubles (one recall per query)."""
    
    # Read recall scores
    recall_data = {}
    expected_num_queries = 10  # Fixed: 10 queries per file
    
    for iter_num, filepath in sorted(recall_files.items()):
        try:
            recalls = read_binary_doubles(filepath)
            num_values = len(recalls)
            
            print(f"Read recall file for iteration {iter_num} ({filepath}): {num_values} values")
            if num_values > 0:
                print(f"  First 3 values: {recalls[:min(3, num_values)]}")
                print(f"  Value range: [{np.min(recalls):.4f}, {np.max(recalls):.4f}]")
            
            if num_values != expected_num_queries:
                print(f"  Warning: Expected {expected_num_queries} queries, but found {num_values} values")
                # If it's exactly 10, use it; otherwise take first 10 or all if less than 10
                if num_values == expected_num_queries:
                    recall_data[iter_num] = recalls
                elif num_values > expected_num_queries:
                    print(f"  Taking first {expected_num_queries} values")
                    recall_data[iter_num] = recalls[:expected_num_queries]
                else:
                    print(f"  Using all {num_values} values (less than expected)")
                    recall_data[iter_num] = recalls
            else:
                recall_data[iter_num] = recalls
                print(f"  Successfully loaded {expected_num_queries} query recall values")
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    if not recall_data:
        print("No recall score files found!")
        return
    
    # Determine actual number of queries from data (use minimum to ensure consistency)
    if not recall_data:
        print("No valid recall data found!")
        return
    
    num_queries = min(len(recalls) for recalls in recall_data.values())
    
    print(f"\nPlotting recall data: {len(recall_data)} iterations, {num_queries} queries per iteration")
    
    # Verify all iterations have the same number of queries
    for iter_num, recalls in recall_data.items():
        if len(recalls) != num_queries:
            print(f"  Warning: Iteration {iter_num} has {len(recalls)} values, will use first {num_queries}")
    
    # Plot recall vs query index with separate line for each iteration
    plt.figure(figsize=(14, 8))
    
    iterations = sorted(recall_data.keys())
    query_indices = np.arange(num_queries)
    
    for iter_num in iterations:
        if iter_num in recall_data:
            recalls = recall_data[iter_num]
            # Take only the first num_queries values
            valid_recalls = recalls[:num_queries]
            print(f"  Plotting iteration {iter_num}: {len(valid_recalls)} values, range [{np.min(valid_recalls):.2f}, {np.max(valid_recalls):.2f}]")
            plt.plot(query_indices, valid_recalls, label=f'Iteration {iter_num}', 
                    marker='o', markersize=5, linewidth=2)
    
    plt.xlabel('Query Index', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title(f'Recall Scores by Query (Each Line = One Iteration, {num_queries} Queries)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(query_indices)  # Show all query indices
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'recall_scores.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved recall scores plot to {output_path}")
    plt.close()
    
    # Also create a heatmap if we have multiple iterations
    if len(iterations) > 1:
        # Create matrix: rows = iterations, cols = queries
        recall_matrix = np.zeros((len(iterations), num_queries))
        for i, iter_num in enumerate(iterations):
            if iter_num in recall_data:
                recalls = recall_data[iter_num]
                recall_matrix[i, :] = recalls[:num_queries]
        
        plt.figure(figsize=(max(12, num_queries * 0.8), max(8, len(iterations) * 0.5)))
        im = plt.imshow(recall_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(im, label='Recall')
        plt.xlabel('Query Index', fontsize=12)
        plt.ylabel('Iteration', fontsize=12)
        plt.title('Recall Scores Heatmap', fontsize=14, fontweight='bold')
        plt.yticks(range(len(iterations)), [f'Iter {iter}' for iter in iterations])
        plt.xticks(range(num_queries), range(num_queries))
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'recall_scores_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved recall scores heatmap to {output_path}")
        plt.close()

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot overlap scores from binary files')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory for plots (default: current directory)')
    parser.add_argument('--input-dir', '-i', default='data/scores', help='Input directory to search for files (default: data/scores)')
    args = parser.parse_args()
    
    # Change to input directory
    original_dir = os.getcwd()
    if os.path.exists(args.input_dir):
        os.chdir(args.input_dir)
    else:
        print(f"Warning: Input directory '{args.input_dir}' does not exist. Using current directory.")
    
    # Find files
    approx_files = find_iteration_files('approx_overlap_scores_iter_')
    real_files = find_iteration_files('real_overlap_scores_iter_')
    recall_files = find_iteration_files('recall_iter_')
    
    print(f"Found {len(approx_files)} approximate overlap score files")
    print(f"Found {len(real_files)} real overlap score files")
    print(f"Found {len(recall_files)} recall score files")
    
    if not approx_files and not real_files and not recall_files:
        print("No score files found in current directory!")
        return
    
    # Create output directory if it doesn't exist (use absolute path from original directory)
    output_path = os.path.join(original_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_path, exist_ok=True)
    
    # Plot (use absolute path for output)
    if approx_files or real_files:
        plot_overlap_scores(approx_files, real_files, output_path)
    
    if recall_files:
        plot_recall_scores(recall_files, output_path)
    
    # Change back to original directory
    os.chdir(original_dir)

if __name__ == '__main__':
    main()
