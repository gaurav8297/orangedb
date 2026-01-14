import pandas as pd
import struct


def read_binary_umap(filepath, is_3d=False):
    """Read binary UMAP data with clustering information.
    
    Binary format:
    - Header: num_records (int)
    - For each record: UMAP_1 (float), UMAP_2 (float), [UMAP_3 (float)], Cluster_ID (int), Is_Centroid (int)
    """
    data = []
    with open(filepath, 'rb') as f:
        # Read header: number of records
        num_records = struct.unpack('i', f.read(4))[0]
        
        # Read records
        for _ in range(num_records):
            umap_1 = struct.unpack('f', f.read(4))[0]
            umap_2 = struct.unpack('f', f.read(4))[0]
            if is_3d:
                umap_3 = struct.unpack('f', f.read(4))[0]
            cluster_id = struct.unpack('i', f.read(4))[0]
            is_centroid = struct.unpack('i', f.read(4))[0]
            
            if is_3d:
                data.append([umap_1, umap_2, umap_3, cluster_id, is_centroid])
            else:
                data.append([umap_1, umap_2, cluster_id, is_centroid])
    
    # Create DataFrame
    if is_3d:
        df = pd.DataFrame(data, columns=['UMAP_1', 'UMAP_2', 'UMAP_3', 'Cluster_ID', 'Is_Centroid'])
    else:
        df = pd.DataFrame(data, columns=['UMAP_1', 'UMAP_2', 'Cluster_ID', 'Is_Centroid'])
    
    return df


def read_binary_umap_no_clustering(filepath, is_3d=False):
    """Read binary UMAP data without clustering information.
    
    Binary format:
    - Header: num_vectors (int)
    - For each vector: UMAP_1 (float), UMAP_2 (float), [UMAP_3 (float)], ROW_ID (int)
    """
    data = []
    with open(filepath, 'rb') as f:
        # Read header: number of vectors
        num_vectors = struct.unpack('i', f.read(4))[0]
        
        # Read records
        for _ in range(num_vectors):
            umap_1 = struct.unpack('f', f.read(4))[0]
            umap_2 = struct.unpack('f', f.read(4))[0]
            if is_3d:
                umap_3 = struct.unpack('f', f.read(4))[0]
            row_id = struct.unpack('i', f.read(4))[0]
            
            if is_3d:
                data.append([umap_1, umap_2, umap_3, row_id])
            else:
                data.append([umap_1, umap_2, row_id])
    
    # Create DataFrame
    if is_3d:
        df = pd.DataFrame(data, columns=['UMAP_1', 'UMAP_2', 'UMAP_3', 'ROW_ID'])
    else:
        df = pd.DataFrame(data, columns=['UMAP_1', 'UMAP_2', 'ROW_ID'])
    
    return df


def read_binary_clustering(filepath):
    """Read binary clustering data.
    
    Binary format:
    - Header: num_records (int)
    - For each record: ROW_ID (int), Cluster_ID (int)
    """
    data = []
    with open(filepath, 'rb') as f:
        # Read header: number of records
        num_records = struct.unpack('i', f.read(4))[0]
        
        # Read records
        for _ in range(num_records):
            row_id = struct.unpack('i', f.read(4))[0]
            cluster_id = struct.unpack('i', f.read(4))[0]
            data.append([row_id, cluster_id])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['ROW_ID', 'Cluster_ID'])
    return df


def read_ground_truth(filepath):
    """Read ground truth file (vector_idx_t format).
    
    Binary format:
    - Array of integers representing ground truth indices
    - For k=100 queries, this is a flat array: [q0_gt0, q0_gt1, ..., q0_gt99, q1_gt0, ...]
    
    Returns: raw array of integers
    """
    import numpy as np
    # Read as 64-bit unsigned integers (vector_idx_t is uint64_t in C++)
    gt_data = np.fromfile(filepath, dtype=np.uint64)
    return gt_data

