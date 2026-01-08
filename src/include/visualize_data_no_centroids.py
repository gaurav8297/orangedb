import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from visualize_helpers import read_binary_umap_no_clustering, read_binary_clustering

interactive_mode = True

"""
Right now the 2D option is disabled 

# visualization type
if interactive_mode:
    print("\nChoose visualization type:")
    print("  1 - 2D")
    print("  2 - 3D")
    viz_choice = input("\nYour choice (1 or 2): ").strip()
    
    if viz_choice == '1':
        flag_2D = True
        flag_3D = False
    elif viz_choice == '2':
        flag_2D = False
        flag_3D = True
    else:
        print("Invalid choice. Defaulting to 3D.")
        flag_2D = False
        flag_3D = True
else:
    flag_2D = False
    flag_3D = True
"""

# cluster hirarchy
if interactive_mode:
    print("\nChoose cluster hirarchy:")
    print("  1 - L1")
    print("  2 - L2")
    viz_choice = input("\nYour choice (1 or 2): ").strip()
    
    if viz_choice == '1':
        flag_L1 = True
        flag_L2 = False
    elif viz_choice == '2':
        flag_L1 = False
        flag_L2 = True
    else:
        print("Invalid choice. Defaulting to 3D.")
        flag_L1 = True
        flag_L2 = False
else:
    flag_L1 = True
    flag_L2 = False


# clustering mode
if interactive_mode:
    print("\nChoose clustering mode:")
    print("  1 - Hard Limit")
    print("  2 - Rebalanced Centroids")
    viz_choice = input("\nYour choice (1 or 2): ").strip()
    
    if viz_choice == '1':
        flag_HardLimit = True
        flag_RebalancedCentroids = False
    elif viz_choice == '2':
        flag_HardLimit = False
        flag_RebalancedCentroids = True
    else:
        print("Invalid choice. Defaulting to Hard Limit.")
        flag_HardLimit = True
        flag_RebalancedCentroids = False
else:
    flag_HardLimit = True
    flag_RebalancedCentroids = False


# Load binary files
#if flag_2D:
#    df_umap = read_binary_umap_no_clustering('umap_2D_without_clustering.bin', is_3d=False)
#elif flag_3D:
#    df_umap = read_binary_umap_no_clustering('umap_3D_without_clustering.bin', is_3d=True)
#else:
#    print("Error: Invalid input choise")
#    exit(1)
df_umap = read_binary_umap_no_clustering('umap_3D_without_clustering.bin', is_3d=True)

if flag_L1 and flag_HardLimit:
    df_clustering = read_binary_clustering('HARD_LIMIT_clustering_data_l1.bin')
elif flag_L2 and flag_HardLimit:
    df_clustering = read_binary_clustering('HARD_LIMIT_clustering_data_l2.bin')
elif flag_L1 and flag_RebalancedCentroids:
    df_clustering = read_binary_clustering('REBALANCE_CENTROIDS_clustering_data_l1.bin')
elif flag_L2 and flag_RebalancedCentroids:
    df_clustering = read_binary_clustering('REBALANCE_CENTROIDS_clustering_data_l2.bin')
else:
    print("Error: Invalid input choise")
    exit(1)

# print available clusters
unique_clusters = sorted(df_clustering['Cluster_ID'].unique())
cluster_counts = df_clustering['Cluster_ID'].value_counts().sort_index()

print(f"\nLoaded {len(df_umap)} vectors")

# color palette
n_clusters = len(unique_clusters)
if n_clusters <= 10:
    cmap = cm.get_cmap('tab10')
    colors = [cmap(i) for i in range(n_clusters)]
elif n_clusters <= 20:
    cmap = cm.get_cmap('tab20')
    colors = [cmap(i) for i in range(n_clusters)]
else:
    colors = [cm.hsv(i / n_clusters) for i in range(n_clusters)]

cluster_color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}
palette_dict = {cluster_id: cluster_color_map[cluster_id] for cluster_id in unique_clusters}

print("\n" + "="*60)
print("Available Clusters:")
print("="*60)
for cluster_id in unique_clusters:
    print(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} vectors")
print("="*60)

# Select Clusters
if interactive_mode:
    print("\nEnter cluster IDs to visualize:")
    print("  - Single cluster: 5")
    print("  - Multiple clusters: 1,3,5")
    print("  - All clusters: all")
    user_input = input("\nYour choice: ").strip()
    
    if user_input.lower() == 'all':
        cluster_filter = None
    else:
        try:
            cluster_filter = [int(x.strip()) for x in user_input.split(',')]
            invalid_clusters = [c for c in cluster_filter if c not in unique_clusters]
            if invalid_clusters:
                print(f"Warning: Invalid cluster IDs {invalid_clusters} will be ignored")
                cluster_filter = [c for c in cluster_filter if c in unique_clusters]
        except ValueError:
            print("Invalid input. Showing all clusters.")
            cluster_filter = None
else:
    cluster_filter = None

print(f"\nVisualizing clusters: {cluster_filter if cluster_filter else 'ALL'}")
print()

# ------------------------------------------------------------

"""
Right now the 2D option is disabled 

if(flag_2D):
    df_vectors_index_filtered = df_clustering.copy()
    df_vectors_filtered = df_umap.copy()

    if cluster_filter is not None:
        df_vectors_index_filtered = df_vectors_index_filtered[df_vectors_index_filtered['Cluster_ID'].isin(cluster_filter)]
        title = f'UMAP Projection - Clusters {cluster_filter}'
    else:
        title = 'UMAP Projection of Custom Clustering'

    fig, ax = plt.subplots(figsize=(12, 8))
    clusters_in_view = sorted(df_vectors_index_filtered['Cluster_ID'].unique())
    for cluster_id in clusters_in_view:
        vector_ids = df_vectors_index_filtered[df_vectors_index_filtered['Cluster_ID'] == cluster_id]['ROW_ID']     
        cluster_data = df_vectors_filtered[df_vectors_filtered['ROW_ID'].isin(vector_ids)]
        ax.scatter(cluster_data['UMAP_1'], 
                  cluster_data['UMAP_2'],
                  c=[cluster_color_map[cluster_id]], 
                  label=f'Cluster {cluster_id}',
                  s=10,
                  alpha=0.6)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, markerscale=2)
    ax.set_xlabel('UMAP_1')
    ax.set_ylabel('UMAP_2')
    ax.set_title(title)
    
    plt.tight_layout()
    print(f"Showing {len(df_vectors_index_filtered)} vectors from {len(clusters_in_view)} clusters")
    plt.show()
"""
# ------------------------------------------------------------

#if(flag_3D):
df_vectors_index_filtered = df_clustering.copy()
df_vectors_filtered = df_umap.copy()

if cluster_filter is not None:
    df_vectors_index_filtered = df_vectors_index_filtered[df_vectors_index_filtered['Cluster_ID'].isin(cluster_filter)]
    title = f'3D UMAP Projection - Clusters {cluster_filter}'
else:
    title = '3D UMAP Projection of Custom Clustering'

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

clusters_in_view = sorted(df_vectors_index_filtered['Cluster_ID'].unique())
for cluster_id in clusters_in_view:
    vector_ids = df_vectors_index_filtered[df_vectors_index_filtered['Cluster_ID'] == cluster_id]['ROW_ID']     
    cluster_data = df_vectors_filtered[df_vectors_filtered['ROW_ID'].isin(vector_ids)]
    ax.scatter(cluster_data['UMAP_1'], 
                cluster_data['UMAP_2'], 
                cluster_data['UMAP_3'],
                c=[cluster_color_map[cluster_id]], 
                label=f'Cluster {cluster_id}',
                s=10, 
                alpha=0.6)

ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9, markerscale=2)

ax.set_xlabel('UMAP_1')
ax.set_ylabel('UMAP_2')
ax.set_zlabel('UMAP_3')
ax.set_title(title)

plt.tight_layout()
print(f"Showing {len(df_vectors_index_filtered)} vectors from {len(clusters_in_view)} clusters")
plt.show()