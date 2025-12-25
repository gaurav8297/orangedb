#!/usr/bin/env python3
"""
Script to plot real and approximate overlap scores from binary files.
Each iteration is plotted as a separate line.
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
    
    print(f"Found {len(approx_files)} approximate overlap score files")
    print(f"Found {len(real_files)} real overlap score files")
    
    if not approx_files and not real_files:
        print("No overlap score files found in current directory!")
        return
    
    # Create output directory if it doesn't exist (use absolute path from original directory)
    output_path = os.path.join(original_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_path, exist_ok=True)
    
    # Plot (use absolute path for output)
    plot_overlap_scores(approx_files, real_files, output_path)
    
    # Change back to original directory
    os.chdir(original_dir)

if __name__ == '__main__':
    main()
