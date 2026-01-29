#!/usr/bin/env python3
"""
Print L1 cluster size histogram (devided by 100 gap size)

This script loads:
1. The cluster sizes histogram file
2. Prints the histogram for all iterations
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('/Users/gilli.hadayo/orangedb/src/include')
from visualize_helpers import read_cluster_sizes_histogram

# File paths
CLUSTER_SIZES_HISTOGRAM_FILE = 'cluster_sizes_histogram.bin'

# Load the cluster sizes histogram (returns tuple: (hard_cluster_size_limit, num_iterations, {iteration: histogram_array}))
hard_cluster_size_limit, num_iterations, cluster_sizes_histograms = read_cluster_sizes_histogram(CLUSTER_SIZES_HISTOGRAM_FILE)

if hard_cluster_size_limit is None or num_iterations is None or len(cluster_sizes_histograms) == 0:
    print("No histogram data found or file is empty!")
    exit(1)

# Print all iterations
print(f"Hard Cluster Size Limit: {hard_cluster_size_limit}")
print(f"Number of iterations (from header): {num_iterations}")
print(f"Found {len(cluster_sizes_histograms)} iterations: {sorted(cluster_sizes_histograms.keys())}")

# Create subplots for each iteration
num_iterations = len(cluster_sizes_histograms)
fig, axes = plt.subplots(1, num_iterations, figsize=(5*num_iterations, 4))
if num_iterations == 1:
    axes = [axes]

sorted_iterations = sorted(cluster_sizes_histograms.keys())
for idx, iteration in enumerate(sorted_iterations):
    histogram = cluster_sizes_histograms[iteration]
    
    # Create bin labels (0-99, 100-199, etc.)
    bin_labels = [f"{i*100}-{(i+1)*100-1}" if i < len(histogram)-1 else f"{i*100}+" 
                 for i in range(len(histogram))]
    
    # Plot histogram
    axes[idx].bar(range(len(histogram)), histogram, edgecolor='black')
    axes[idx].set_title(f'L1 Cluster Size Histogram - Iteration {iteration}')
    axes[idx].set_xlabel('Cluster Size Range')
    axes[idx].set_ylabel('Number of Clusters')
    axes[idx].set_xticks(range(len(histogram)))
    axes[idx].set_xticklabels(bin_labels, rotation=45, ha='right')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()