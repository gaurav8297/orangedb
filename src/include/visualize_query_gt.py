#!/usr/bin/env python3
"""
Visualize Query and its ground truth vectors in 3D UMAP space.

This script loads:
1. The UMAP 3D projection of the base dataset + queries
2. The ground truth file to find nearest neighbors
3. Displays them in an interactive 3D plot

How it works:
- Base vectors: ROW_ID 0 to NUM_BASE_VECTORS-1
- Query vectors: ROW_ID NUM_BASE_VECTORS to NUM_BASE_VECTORS+NUM_QUERIES-1
- Ground truth: Contains indices of nearest base vectors for each query

Color scheme:
- Query vector: RED (large star marker)
- Ground truth vectors: Colored by cluster assignment
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('/Users/gilli.hadayo/orangedb/src/include')
from visualize_helpers import read_binary_umap_no_clustering, read_ground_truth, read_binary_clustering

# Configuration
NUM_BASE_VECTORS = 10000  # Number of base vectors (before adding queries)
NUM_QUERIES = 50
K = 100  # Number of ground truth neighbors per query

# File paths
UMAP_FILE = 'umap_3D_with_query_without_clustering.bin'
GT_FILE = '/Users/gilli.hadayo/orangedb/data/siftsmall/gt.bin'

# Load the UMAP 3D projection (base vectors + queries)
df_umap = read_binary_umap_no_clustering(UMAP_FILE, is_3d=True)

# Load ground truth data (contains indices into base vectors)
gt_data = read_ground_truth(GT_FILE)

# Interactive clustering selection
print(f"Choose clustering method\n 1 - Hard Limit\n 2 - Rebalanced Centroids:")
viz_choice = input("Your choice: ").strip()

try:
    clustering_method = int(viz_choice)
    if clustering_method < 1 or clustering_method > 2:
        print(f"Invalid choice. Defaulting to hard limit.")
        clustering_method = 1
except ValueError:
    print("Invalid input. Defaulting to hard limit.")
    clustering_method = 1

# Interactive cluster hirarchy selection
print(f"Choose clustering hirarchy\n 1 - L1\n 2 - L2:")
viz_choice = input("Your choice: ").strip()

try:
    clustering_type = int(viz_choice)
    if clustering_type < 1 or clustering_type > 2:
        print(f"Invalid choice. Defaulting to L1.")
        clustering_type = 1
except ValueError:
    print("Invalid input. Defaulting to L1.")
    clustering_type = 1

if clustering_method == 1 and clustering_type == 1:
    CLUSTERING_FILE = 'HARD_LIMIT_clustering_data_l1_iter_10.bin'
elif clustering_method == 1 and clustering_type == 2:
    CLUSTERING_FILE = 'HARD_LIMIT_clustering_data_l2_iter_10.bin'
elif clustering_method == 2 and clustering_type == 1:
    CLUSTERING_FILE = 'REBALANCE_CENTROIDS_clustering_data_l1_iter_10.bin'
elif clustering_method == 2 and clustering_type == 2:
    CLUSTERING_FILE = 'REBALANCE_CENTROIDS_clustering_data_l2_iter_10.bin'
else:
    print("Invalid choice. Defaulting to hard limit L1.")
    CLUSTERING_FILE = 'HARD_LIMIT_clustering_data_l1.bin'

# Load clustering data (ROW_ID -> Cluster_ID mapping)
df_clustering = read_binary_clustering(CLUSTERING_FILE)

# Interactive query selection
print(f"Choose query id (0-{NUM_QUERIES-1}):")
viz_choice = input("Your choice: ").strip()

try:
    query_id = int(viz_choice)
    if query_id < 0 or query_id >= NUM_QUERIES:
        print(f"Invalid choice. Must be 0-{NUM_QUERIES-1}. Defaulting to 0.")
        query_id = 0
except ValueError:
    print("Invalid input. Defaulting to 0.")
    query_id = 0

print(f"\nVisualizing Query #{query_id}")

# Get query vector coordinates
# Query vectors are appended after base vectors: ROW_ID = NUM_BASE_VECTORS + query_id
query_row_id = NUM_BASE_VECTORS + query_id
query_coords = df_umap[df_umap['ROW_ID'] == query_row_id]

# Get ground truth indices for this query
# Ground truth file contains indices of nearest neighbors from the base vectors
gt_start_idx = query_id * K
gt_end_idx = gt_start_idx + K
ground_truth_ids = gt_data[gt_start_idx:gt_end_idx]

ground_truth_coords = df_umap[df_umap['ROW_ID'].isin(ground_truth_ids)].copy()

# Merge with clustering data to get cluster assignments
ground_truth_coords = ground_truth_coords.merge(
    df_clustering[['ROW_ID', 'Cluster_ID']], 
    on='ROW_ID', 
    how='left'
)

# Validation: Check cluster assignments
num_with_clusters = ground_truth_coords['Cluster_ID'].notna().sum()
num_without_clusters = ground_truth_coords['Cluster_ID'].isna().sum()
unique_cluster_ids = ground_truth_coords['Cluster_ID'].dropna().unique()

print(f"\nCluster Assignment Summary:")
print(f"  Vectors with cluster assignment: {num_with_clusters}/{len(ground_truth_coords)}")
print(f"  Vectors without cluster assignment: {num_without_clusters}")
print(f"  Unique clusters represented: {len(unique_cluster_ids)}")
print(f"  Cluster IDs: {sorted([int(x) for x in unique_cluster_ids])}")

if num_without_clusters > 0:
    missing_cluster_ids = ground_truth_coords[ground_truth_coords['Cluster_ID'].isna()]['ROW_ID'].values
    print(f"  WARNING: {num_without_clusters} ground truth vectors have no cluster assignment!")
    print(f"  Missing ROW_IDs: {missing_cluster_ids[:10]}{'...' if len(missing_cluster_ids) > 10 else ''}")

if len(query_coords) == 0:
    print(f"ERROR: Query vector with ROW_ID={query_row_id} not found in UMAP data!")
    sys.exit(1)

if len(ground_truth_coords) == 0:
    print("ERROR: No ground truth vectors found in UMAP data!")
    sys.exit(1)

# Create 3D plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

"""
# Plot all vectors in light gray (optional - comment out for cleaner view)
ax.scatter(df_umap['UMAP_1'], 
          df_umap['UMAP_2'], 
          df_umap['UMAP_3'],
          c='lightgray', 
          s=1, 
          alpha=0.1,
          label='All vectors')
"""

# Plot ground truth vectors colored by cluster
# Generate colors for each cluster
cluster_ids = ground_truth_coords['Cluster_ID'].fillna(-1).astype(int)
unique_clusters = sorted(cluster_ids.unique())
n_colors = len(unique_clusters)

# Use a colormap (updated for matplotlib 3.7+)
colormap = plt.colormaps.get_cmap('tab20' if n_colors <= 20 else 'hsv')
colors = [colormap(i / n_colors) for i in range(n_colors)]

# Create cluster to color mapping
cluster_to_color = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}

# Plot each cluster separately for legend
total_plotted = 0
for cluster_id in unique_clusters:
    mask = cluster_ids == cluster_id
    num_in_cluster = mask.sum()
    if num_in_cluster > 0:
        cluster_data = ground_truth_coords[mask]
        label = f'Cluster {cluster_id} ({num_in_cluster})' if cluster_id >= 0 else f'Unassigned ({num_in_cluster})'
        ax.scatter(cluster_data['UMAP_1'], 
                  cluster_data['UMAP_2'], 
                  cluster_data['UMAP_3'],
                  c=[cluster_to_color[cluster_id]], 
                  s=50, 
                  alpha=0.8,
                  label=label,
                  edgecolors='black',
                  linewidths=0.5)
        total_plotted += num_in_cluster

print(f"\nPlotting Summary:")
print(f"  Total ground truth vectors plotted: {total_plotted}/{len(ground_truth_coords)}")

ax.scatter(query_coords['UMAP_1'], 
          query_coords['UMAP_2'], 
          query_coords['UMAP_3'],
          c='red', 
          s=200, 
          alpha=1.0,
          label=f'Query #{query_id}',
          edgecolors='darkred',
          linewidths=2,
          marker='*')

ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9, markerscale=1.5)
ax.set_xlabel('UMAP_1')
ax.set_ylabel('UMAP_2')
ax.set_zlabel('UMAP_3')
ax.set_title(f'Query #{query_id} Ground Truth Vectors by Cluster Assignment')

plt.tight_layout()
plt.show()

