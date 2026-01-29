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
CLUSTER_SIZES_HISTOGRAM_FILE_HARD_LIMIT = 'cluster_sizes_histogram_hard_limit.bin'
# CLUSTER_SIZES_HISTOGRAM_FILE_REBALANCE_CENTROIDS = 'cluster_sizes_histogram_rebalance_centroids.bin'
# CLUSTER_SIZES_HISTOGRAM_FILE_REBALANCE_VECTORS = 'cluster_sizes_histogram_rebalance_vectors.bin'
CLUSTER_SIZES_HISTOGRAM_FILE_DOUBLE_KMEANS = 'cluster_sizes_histogram_double_kmeans.bin'

# Load the cluster sizes histogram (returns tuple: (hard_cluster_size_limit, num_iterations, {iteration: histogram_array}))
hard_cluster_size_limit_hard_limit, num_iterations_hard_limit, cluster_sizes_histograms_hard_limit = read_cluster_sizes_histogram(CLUSTER_SIZES_HISTOGRAM_FILE_HARD_LIMIT)
hard_cluster_size_limit_double_kmeans, num_iterations_double_kmeans, cluster_sizes_histograms_double_kmeans = read_cluster_sizes_histogram(CLUSTER_SIZES_HISTOGRAM_FILE_DOUBLE_KMEANS)


if hard_cluster_size_limit_hard_limit is None or num_iterations_hard_limit is None or len(cluster_sizes_histograms_hard_limit) == 0 or hard_cluster_size_limit_double_kmeans is None or num_iterations_double_kmeans is None or len(cluster_sizes_histograms_double_kmeans) == 0:
    print("No histogram data found or file is empty!")
    exit(1)


# Print all iterations
print(f"Hard Cluster Size Limit for Hard Limit: {hard_cluster_size_limit_hard_limit}")
print(f"Number of iterations (from header): {num_iterations_hard_limit}")
print(f"Hard Cluster Size Limit for Double K-Means: {hard_cluster_size_limit_double_kmeans}")
print(f"Number of iterations (from header): {num_iterations_double_kmeans}")
print(f"Found {len(cluster_sizes_histograms_hard_limit)} iterations: {sorted(cluster_sizes_histograms_hard_limit.keys())}")
print(f"Found {len(cluster_sizes_histograms_double_kmeans)} iterations: {sorted(cluster_sizes_histograms_double_kmeans.keys())}")

# Create subplots for each iteration
num_iterations = len(cluster_sizes_histograms_hard_limit)
fig, axes = plt.subplots(2, num_iterations, figsize=(5*num_iterations, 4))
# Flatten axes array to handle both single and multiple iterations
axes = axes.flatten()

sorted_iterations_hard_limit = sorted(cluster_sizes_histograms_hard_limit.keys())
sorted_iterations_double_kmeans = sorted(cluster_sizes_histograms_double_kmeans.keys())
for idx, iteration in enumerate(sorted_iterations_hard_limit):
    histogram_hard_limit = cluster_sizes_histograms_hard_limit[iteration]
    histogram_double_kmeans = cluster_sizes_histograms_double_kmeans[iteration]
    # Create bin labels (0-99, 100-199, etc.)
    bin_labels = [f"{i*100}-{(i+1)*100-1}" if i < len(histogram_hard_limit)-1 else f"{i*100}+" 
                 for i in range(len(histogram_hard_limit))]
    
    # Plot histogram for hard limit (top row)
    axes[idx].bar(range(len(histogram_hard_limit)), histogram_hard_limit, edgecolor='black')
    axes[idx].set_title(f'Hard Limit - Iteration {iteration}')
    axes[idx].set_xlabel('Cluster Size Range')
    axes[idx].set_ylabel('Number of Clusters')
    axes[idx].set_xticks(range(len(histogram_hard_limit)))
    axes[idx].set_xticklabels(bin_labels, rotation=45, ha='right')
    axes[idx].grid(axis='y', alpha=0.3)

    # Plot histogram for double k-means (bottom row)
    axes[idx+num_iterations].bar(range(len(histogram_double_kmeans)), histogram_double_kmeans, edgecolor='black')
    axes[idx+num_iterations].set_title(f'Double K-Means - Iteration {iteration}')
    axes[idx+num_iterations].set_xlabel('Cluster Size Range')
    axes[idx+num_iterations].set_ylabel('Number of Clusters')
    axes[idx+num_iterations].set_xticks(range(len(histogram_double_kmeans)))
    axes[idx+num_iterations].set_xticklabels(bin_labels, rotation=45, ha='right')
    axes[idx+num_iterations].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()