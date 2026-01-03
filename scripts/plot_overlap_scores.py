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

def read_binary_floats(filepath):
    """Read binary file containing float values."""
    try:
        data = np.fromfile(filepath, dtype=np.float32)
        return data
    except Exception as e:
        with open(filepath, 'rb') as f:
            data = f.read()
        num_floats = len(data) // 4
        floats = struct.unpack(f'{num_floats}f', data)
        return np.array(floats)

def read_nested_vector(filepath):
    """Read nested vector file (std::vector<std::vector<vector_idx_t>>).
    Format: [uint64 numOuter] [uint64 innerSize, uint64[] data] * numOuter"""
    data = []
    with open(filepath, 'rb') as f:
        # Read number of outer vectors
        num_outer_bytes = f.read(8)
        if not num_outer_bytes:
            return data
        num_outer = struct.unpack('Q', num_outer_bytes)[0]
        
        for _ in range(num_outer):
            # Read size of inner vector
            inner_size_bytes = f.read(8)
            if not inner_size_bytes:
                break
            inner_size = struct.unpack('Q', inner_size_bytes)[0]
            
            # Read inner vector data
            if inner_size > 0:
                inner_bytes = f.read(inner_size * 8)  # 8 bytes per uint64
                inner_data = struct.unpack(f'{inner_size}Q', inner_bytes)
                data.append(set(inner_data))  # Use set for efficient intersection
            else:
                data.append(set())
    
    return data

def cosine_distance(vec1, vec2):
    """Compute cosine distance (1 - cosine similarity) between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    cosine_sim = dot_product / (norm1 * norm2)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    return 1.0 - cosine_sim

def angular_distance(vec1, vec2):
    """Compute angular distance (in radians) between two vectors."""
    cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    return 1-cosine_sim

def match_centroids_angular_wrt_mean(prev_centroids, new_centroids, prev_mean_centroid, dim):
    """Match centroids between iterations using angular distance relative to previous mean centroid.
    The angular distance is measured between vectors from mean to each centroid.
    Returns: mapping from new_centroid_idx to (prev_centroid_idx, angular_distance)"""
    num_prev = len(prev_centroids) // dim
    num_new = len(new_centroids) // dim
    
    matches = {}
    used_prev = set()
    
    # For each new centroid, find closest unmatched previous centroid
    # Distance is measured as angle between (prev_centroid - mean) and (new_centroid - mean)
    for new_idx in range(num_new):
        new_centroid = new_centroids[new_idx * dim:(new_idx + 1) * dim]
        # Vector from mean to new centroid
        # new_vec = new_centroid - prev_mean_centroid
        new_vec = new_centroid
        new_vec_norm = np.linalg.norm(new_vec)
        if new_vec_norm < 1e-9:
            continue
        new_vec = new_vec / new_vec_norm
        
        min_dist = float('inf')
        best_prev_idx = -1
        
        for prev_idx in range(num_prev):
            if prev_idx in used_prev:
                continue
            prev_centroid = prev_centroids[prev_idx * dim:(prev_idx + 1) * dim]
            # Vector from mean to previous centroid
            # prev_vec = prev_centroid - prev_mean_centroid
            prev_vec = prev_centroid
            prev_vec_norm = np.linalg.norm(prev_vec)
            if prev_vec_norm < 1e-9:
                continue
            prev_vec = prev_vec / prev_vec_norm
            
            # Angular distance between these vectors
            dist = angular_distance(new_vec, prev_vec)
            if dist < min_dist:
                min_dist = dist
                best_prev_idx = prev_idx
        
        if best_prev_idx != -1:
            matches[new_idx] = (best_prev_idx, min_dist)
            used_prev.add(best_prev_idx)
    
    return matches

def calculate_mean_centroid(centroids, dim):
    """Calculate the mean centroid from all centroids."""
    num_centroids = len(centroids) // dim
    mean_centroid = np.zeros(dim)
    
    for i in range(num_centroids):
        centroid = centroids[i * dim:(i + 1) * dim]
        mean_centroid += centroid
    
    mean_centroid /= num_centroids
    
    # Normalize for angular distance
    norm = np.linalg.norm(mean_centroid)
    if norm > 1e-9:
        mean_centroid = mean_centroid / norm
    
    return mean_centroid

def filter_centroids_by_change(centroid_files, dim, max_angular_change=0.1):
    """Filter centroids based on angular change between matched centroids.
    Matching is done using angular distance relative to previous iteration's mean centroid.
    Returns: set of centroid indices that have minimal angular change after matching."""
    if len(centroid_files) < 2:
        # Need at least 2 iterations to compare
        return None
    
    iterations = sorted(centroid_files.keys())
    all_stable_centroids = set()
    
    # Compare consecutive iterations
    for i in range(len(iterations) - 1):
        prev_iter = iterations[i]
        new_iter = iterations[i + 1]
        
        try:
            prev_centroids = read_binary_floats(centroid_files[prev_iter])
            new_centroids = read_binary_floats(centroid_files[new_iter])
            
            # Calculate mean centroid of previous iteration (not normalized for vector calculation)
            num_prev = len(prev_centroids) // dim
            prev_mean_centroid = np.zeros(dim)
            for prev_idx in range(num_prev):
                prev_centroid = prev_centroids[prev_idx * dim:(prev_idx + 1) * dim]
                prev_mean_centroid += prev_centroid
            prev_mean_centroid /= num_prev
            
            # Match centroids using angular distance relative to prev mean
            matches = match_centroids_angular_wrt_mean(prev_centroids, new_centroids, prev_mean_centroid, dim)
            
            # Now calculate actual angular change between matched centroids
            stable_in_transition = []
            angular_changes = []
            
            for new_idx, (prev_idx, match_angle) in matches.items():
                prev_centroid = prev_centroids[prev_idx * dim:(prev_idx + 1) * dim]
                new_centroid = new_centroids[new_idx * dim:(new_idx + 1) * dim]
                
                # Normalize centroids for angular distance calculation
                prev_norm = np.linalg.norm(prev_centroid)
                new_norm = np.linalg.norm(new_centroid)
                if prev_norm > 1e-9 and new_norm > 1e-9:
                    prev_centroid_norm = prev_centroid / prev_norm
                    new_centroid_norm = new_centroid / new_norm
                    
                    # Calculate angular distance between matched centroids
                    angular_dist = angular_distance(prev_centroid_norm, new_centroid_norm)
                    angular_changes.append(angular_dist)
                    
                    if angular_dist <= max_angular_change:
                        all_stable_centroids.add(new_idx)
                        stable_in_transition.append(new_idx)
            
            num_new_centroids = len(new_centroids) // dim
            
            # Print distribution statistics
            if angular_changes:
                angular_changes = np.array(angular_changes)
                print(f"\n  Iteration {prev_iter} -> {new_iter}: Angular Change Distribution")
                print(f"    Total matched centroids: {len(angular_changes)}")
                print(f"    Min: {np.min(angular_changes):.6f}, Max: {np.max(angular_changes):.6f}")
                print(f"    Mean: {np.mean(angular_changes):.6f}, Median: {np.median(angular_changes):.6f}")
                print(f"    Std Dev: {np.std(angular_changes):.6f}")
                
                # Percentiles
                percentiles = [10, 25, 50, 75, 90, 95, 99]
                print(f"    Percentiles:")
                for p in percentiles:
                    val = np.percentile(angular_changes, p)
                    print(f"      {p}th: {val:.6f}")
                
                # Histogram bins
                bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, np.pi]
                hist, bin_edges = np.histogram(angular_changes, bins=bins)
                print(f"    Distribution:")
                for i in range(len(hist)):
                    count = hist[i]
                    percentage = 100.0 * count / len(angular_changes) if len(angular_changes) > 0 else 0
                    print(f"      [{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f}): {count} ({percentage:.1f}%)")
                
                print(f"    Centroids with change <= {max_angular_change:.4f}: {len(stable_in_transition)}/{num_new_centroids} ({100.0*len(stable_in_transition)/num_new_centroids:.1f}%)")
            else:
                print(f"  Iteration {prev_iter} -> {new_iter}: No valid matches found")
            
        except Exception as e:
            print(f"Error comparing centroids between iterations {prev_iter} and {new_iter}: {e}")
            import traceback
            traceback.print_exc()
    
    return all_stable_centroids if all_stable_centroids else None

def find_iteration_files(pattern_prefix, exclude_pattern=None, iter_regex=r'iter_(\d+)'):
    """Find all files matching the pattern and extract iteration numbers."""
    files = {}
    for filepath in Path('.').glob(f'{pattern_prefix}*.bin'):
        # Skip files that match the exclude pattern
        if exclude_pattern and exclude_pattern in filepath.name:
            continue
        match = re.search(iter_regex, filepath.name)
        if match:
            iter_num = int(match.group(1))
            files[iter_num] = filepath
    return files

def match_centroids_by_preserved_ids(prev_mega_mini, new_mega_mini):
    """Match centroids based on how much data/IDs are preserved from prev to new.
    Uses Jaccard similarity (intersection / union) to match.
    Returns: mapping from new_idx to (prev_idx, jaccard_similarity, intersection_size, prev_size, new_size)"""
    num_prev = len(prev_mega_mini)
    num_new = len(new_mega_mini)
    
    matches = {}
    used_prev = set()
    
    # For each new centroid, find the best matching previous centroid
    for new_idx in range(num_new):
        new_ids = new_mega_mini[new_idx]
        best_prev_idx = -1
        best_jaccard = -1
        best_intersection = 0
        
        for prev_idx in range(num_prev):
            if prev_idx in used_prev:
                continue
            prev_ids = prev_mega_mini[prev_idx]
            
            # Calculate Jaccard similarity
            intersection = len(prev_ids & new_ids)
            union = len(prev_ids | new_ids)
            jaccard = intersection / union if union > 0 else 0
            
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_prev_idx = prev_idx
                best_intersection = intersection
        
        if best_prev_idx != -1:
            prev_size = len(prev_mega_mini[best_prev_idx])
            new_size = len(new_ids)
            matches[new_idx] = (best_prev_idx, best_jaccard, best_intersection, prev_size, new_size)
            used_prev.add(best_prev_idx)
    
    return matches

def analyze_preserved_ids(prev_files, new_files, output_dir='.'):
    """Analyze and print distribution of preserved IDs between iterations."""
    iterations = sorted(set(prev_files.keys()) & set(new_files.keys()))
    
    if not iterations:
        print("No matching prev/new mega-mini centroid files found")
        return
    
    for iter_num in iterations:
        print(f"\n{'='*60}")
        print(f"Iteration {iter_num}: Preserved IDs Analysis")
        print(f"{'='*60}")
        
        try:
            prev_mega_mini = read_nested_vector(prev_files[iter_num])
            new_mega_mini = read_nested_vector(new_files[iter_num])
            
            print(f"  Previous: {len(prev_mega_mini)} mega centroids")
            print(f"  New: {len(new_mega_mini)} mega centroids")
            
            # Match centroids
            matches = match_centroids_by_preserved_ids(prev_mega_mini, new_mega_mini)
            
            # Collect statistics
            jaccard_similarities = []
            intersection_sizes = []
            preservation_ratios = []  # intersection / prev_size
            growth_ratios = []  # new_size / prev_size
            
            for new_idx, (prev_idx, jaccard, intersection, prev_size, new_size) in matches.items():
                jaccard_similarities.append(jaccard)
                intersection_sizes.append(intersection)
                if prev_size > 0:
                    preservation_ratios.append(intersection / prev_size)
                    growth_ratios.append(new_size / prev_size)
            
            jaccard_similarities = np.array(jaccard_similarities)
            intersection_sizes = np.array(intersection_sizes)
            preservation_ratios = np.array(preservation_ratios)
            growth_ratios = np.array(growth_ratios)
            
            # Print Jaccard similarity distribution
            print(f"\n  Jaccard Similarity Distribution:")
            print(f"    Min: {np.min(jaccard_similarities):.4f}, Max: {np.max(jaccard_similarities):.4f}")
            print(f"    Mean: {np.mean(jaccard_similarities):.4f}, Median: {np.median(jaccard_similarities):.4f}")
            print(f"    Std Dev: {np.std(jaccard_similarities):.4f}")
            
            # Percentiles
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            print(f"    Percentiles:")
            for p in percentiles:
                val = np.percentile(jaccard_similarities, p)
                print(f"      {p}th: {val:.4f}")
            
            # Histogram bins for Jaccard
            bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            hist, bin_edges = np.histogram(jaccard_similarities, bins=bins)
            print(f"    Distribution:")
            for i in range(len(hist)):
                count = hist[i]
                percentage = 100.0 * count / len(jaccard_similarities) if len(jaccard_similarities) > 0 else 0
                print(f"      [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {count} ({percentage:.1f}%)")
            
            # Print preservation ratio distribution
            print(f"\n  Preservation Ratio (intersection/prev_size) Distribution:")
            print(f"    Min: {np.min(preservation_ratios):.4f}, Max: {np.max(preservation_ratios):.4f}")
            print(f"    Mean: {np.mean(preservation_ratios):.4f}, Median: {np.median(preservation_ratios):.4f}")
            print(f"    Std Dev: {np.std(preservation_ratios):.4f}")
            
            hist, bin_edges = np.histogram(preservation_ratios, bins=bins)
            print(f"    Distribution:")
            for i in range(len(hist)):
                count = hist[i]
                percentage = 100.0 * count / len(preservation_ratios) if len(preservation_ratios) > 0 else 0
                print(f"      [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {count} ({percentage:.1f}%)")
            
            # Print intersection sizes
            print(f"\n  Intersection Sizes Distribution:")
            print(f"    Min: {np.min(intersection_sizes)}, Max: {np.max(intersection_sizes)}")
            print(f"    Mean: {np.mean(intersection_sizes):.1f}, Median: {np.median(intersection_sizes):.1f}")
            print(f"    Total IDs preserved: {np.sum(intersection_sizes)}")
            
            # Print growth ratios
            print(f"\n  Size Change Ratio (new_size/prev_size) Distribution:")
            print(f"    Min: {np.min(growth_ratios):.4f}, Max: {np.max(growth_ratios):.4f}")
            print(f"    Mean: {np.mean(growth_ratios):.4f}, Median: {np.median(growth_ratios):.4f}")
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Jaccard similarity histogram
            axes[0, 0].hist(jaccard_similarities, bins=20, edgecolor='black', alpha=0.7)
            axes[0, 0].set_xlabel('Jaccard Similarity')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title(f'Jaccard Similarity Distribution (Iter {iter_num})')
            axes[0, 0].axvline(np.mean(jaccard_similarities), color='red', linestyle='--', label=f'Mean: {np.mean(jaccard_similarities):.3f}')
            axes[0, 0].legend()
            
            # Preservation ratio histogram
            axes[0, 1].hist(preservation_ratios, bins=20, edgecolor='black', alpha=0.7)
            axes[0, 1].set_xlabel('Preservation Ratio')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title(f'Preservation Ratio Distribution (Iter {iter_num})')
            axes[0, 1].axvline(np.mean(preservation_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(preservation_ratios):.3f}')
            axes[0, 1].legend()
            
            # Intersection sizes histogram
            axes[1, 0].hist(intersection_sizes, bins=20, edgecolor='black', alpha=0.7)
            axes[1, 0].set_xlabel('Intersection Size')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title(f'Intersection Sizes Distribution (Iter {iter_num})')
            axes[1, 0].axvline(np.mean(intersection_sizes), color='red', linestyle='--', label=f'Mean: {np.mean(intersection_sizes):.1f}')
            axes[1, 0].legend()
            
            # Growth ratios histogram
            axes[1, 1].hist(growth_ratios, bins=20, edgecolor='black', alpha=0.7)
            axes[1, 1].set_xlabel('Size Change Ratio')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title(f'Size Change Ratio Distribution (Iter {iter_num})')
            axes[1, 1].axvline(np.mean(growth_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(growth_ratios):.3f}')
            axes[1, 1].axvline(1.0, color='green', linestyle=':', label='No change (1.0)')
            axes[1, 1].legend()
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'preserved_ids_distribution_iter_{iter_num}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n  Saved distribution plot to {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"Error analyzing preserved IDs for iteration {iter_num}: {e}")
            import traceback
            traceback.print_exc()

def plot_overlap_scores_by_closeness(approx_files, real_files, centroid_files, dim, output_dir='.'):
    """Plot overlap scores ordered by centroid closeness for each iteration pair."""
    if not centroid_files or len(centroid_files) < 2:
        print("Need at least 2 iterations with centroid files for closeness-based plotting")
        return
    
    iterations = sorted(set(approx_files.keys()) | set(real_files.keys()) | set(centroid_files.keys()))
    
    # Process each consecutive iteration pair
    for i in range(len(iterations) - 1):
        prev_iter = iterations[i]
        new_iter = iterations[i + 1]
        
        if prev_iter not in centroid_files or new_iter not in centroid_files:
            continue
        
        if (prev_iter not in approx_files and prev_iter not in real_files) or \
           (new_iter not in approx_files and new_iter not in real_files):
            continue
        
        try:
            # Read centroids
            prev_centroids = read_binary_floats(centroid_files[prev_iter])
            new_centroids = read_binary_floats(centroid_files[new_iter])
            
            # Calculate mean centroid of previous iteration
            num_prev = len(prev_centroids) // dim
            prev_mean_centroid = np.zeros(dim)
            for prev_idx in range(num_prev):
                prev_centroid = prev_centroids[prev_idx * dim:(prev_idx + 1) * dim]
                prev_mean_centroid += prev_centroid
            prev_mean_centroid /= num_prev
            
            # Match centroids
            matches = match_centroids_angular_wrt_mean(prev_centroids, new_centroids, prev_mean_centroid, dim)
            
            # Sort matches by angular distance (closest first)
            sorted_matches = sorted(matches.items(), key=lambda x: x[1][1])  # Sort by angular distance
            
            # Read overlap scores
            prev_approx_scores = None
            prev_real_scores = None
            new_approx_scores = None
            new_real_scores = None
            
            if prev_iter in approx_files:
                prev_approx_scores = read_binary_doubles(approx_files[prev_iter])
            if prev_iter in real_files:
                prev_real_scores = read_binary_doubles(real_files[prev_iter])
            if new_iter in approx_files:
                new_approx_scores = read_binary_doubles(approx_files[new_iter])
            if new_iter in real_files:
                new_real_scores = read_binary_doubles(real_files[new_iter])
            
            # Create plots ordered by closeness
            match_indices = []
            prev_approx_values = []
            prev_real_values = []
            new_approx_values = []
            new_real_values = []
            angular_distances = []
            
            for match_idx, (new_idx, (prev_idx, angular_dist)) in enumerate(sorted_matches):
                match_indices.append(match_idx)
                angular_distances.append(angular_dist)
                
                if prev_approx_scores is not None and prev_idx < len(prev_approx_scores):
                    prev_approx_values.append(prev_approx_scores[int(prev_idx)])
                else:
                    prev_approx_values.append(np.nan)
                
                if prev_real_scores is not None and prev_idx < len(prev_real_scores):
                    prev_real_values.append(prev_real_scores[int(prev_idx)])
                else:
                    prev_real_values.append(np.nan)
                
                if new_approx_scores is not None and new_idx < len(new_approx_scores):
                    new_approx_values.append(new_approx_scores[int(new_idx)])
                else:
                    new_approx_values.append(np.nan)
                
                if new_real_scores is not None and new_idx < len(new_real_scores):
                    new_real_values.append(new_real_scores[int(new_idx)])
                else:
                    new_real_values.append(np.nan)
            
            # Plot approximate overlap scores
            if prev_approx_scores is not None or new_approx_scores is not None:
                plt.figure(figsize=(14, 8))
                
                if prev_approx_scores is not None:
                    plt.plot(match_indices, prev_approx_values, label=f'Iteration {prev_iter}', 
                            marker='o', markersize=3, linewidth=1.5, alpha=0.7)
                if new_approx_scores is not None:
                    plt.plot(match_indices, new_approx_values, label=f'Iteration {new_iter}', 
                            marker='s', markersize=3, linewidth=1.5, alpha=0.7)
                
                plt.xlabel('Match Index (Ordered by Angular Closeness)', fontsize=12)
                plt.ylabel('Approximate Overlap Score', fontsize=12)
                plt.title(f'Approximate Overlap Scores: Iteration {prev_iter} -> {new_iter}\n(Ordered by Centroid Closeness)', 
                         fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                output_path = os.path.join(output_dir, f'approx_overlap_iter_{prev_iter}_to_{new_iter}_by_closeness.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Saved approximate overlap scores plot (by closeness) to {output_path}")
                plt.close()
            
            # Plot real overlap scores
            if prev_real_scores is not None or new_real_scores is not None:
                plt.figure(figsize=(14, 8))
                
                if prev_real_scores is not None:
                    plt.plot(match_indices, prev_real_values, label=f'Iteration {prev_iter}', 
                            marker='o', markersize=3, linewidth=1.5, alpha=0.7)
                if new_real_scores is not None:
                    plt.plot(match_indices, new_real_values, label=f'Iteration {new_iter}', 
                            marker='s', markersize=3, linewidth=1.5, alpha=0.7)
                
                plt.xlabel('Match Index (Ordered by Angular Closeness)', fontsize=12)
                plt.ylabel('Real Overlap Score', fontsize=12)
                plt.title(f'Real Overlap Scores: Iteration {prev_iter} -> {new_iter}\n(Ordered by Centroid Closeness)', 
                         fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                output_path = os.path.join(output_dir, f'real_overlap_iter_{prev_iter}_to_{new_iter}_by_closeness.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Saved real overlap scores plot (by closeness) to {output_path}")
                plt.close()
            
            # Also create a plot showing angular distances
            plt.figure(figsize=(14, 6))
            plt.plot(match_indices, angular_distances, marker='o', markersize=2, linewidth=1, alpha=0.7)
            plt.xlabel('Match Index (Ordered by Angular Closeness)', fontsize=12)
            plt.ylabel('Angular Distance (radians)', fontsize=12)
            plt.title(f'Angular Distances: Iteration {prev_iter} -> {new_iter}\n(Ordered by Closeness)', 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f'angular_distances_iter_{prev_iter}_to_{new_iter}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved angular distances plot to {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"Error plotting closeness-based scores for iterations {prev_iter} -> {new_iter}: {e}")
            import traceback
            traceback.print_exc()

def plot_overlap_scores(approx_files, real_files, output_dir='.', centroid_files=None, dim=None, max_angular_change=0.1):
    """Plot approximate and real overlap scores, optionally filtered by centroid change."""
    
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
    
    # Filter by centroid change if centroid files are provided
    valid_centroid_indices = None
    if centroid_files and dim:
        print(f"\nFiltering centroids by angular change (max={max_angular_change})...")
        valid_centroid_indices = filter_centroids_by_change(centroid_files, dim, max_angular_change)
        if valid_centroid_indices:
            print(f"  Found {len(valid_centroid_indices)} centroids with minimal change")
        else:
            print("  No valid centroids found, plotting all centroids")
    
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
            
            # Filter scores if valid_centroid_indices is provided
            if valid_centroid_indices is not None:
                filtered_ids = [cid for cid in cluster_ids if cid in valid_centroid_indices]
                filtered_scores = [scores[int(cid)] for cid in filtered_ids]
                cluster_ids = np.array(filtered_ids)
                scores = np.array(filtered_scores)
            
            label = f'Iteration {iter_num}'
            if iter_num == 0:
                label = "Initial clustering"
            plt.plot(cluster_ids, scores, label=label, marker='o', markersize=2, linewidth=1)
        
        title_suffix = " (Filtered: Minimal Centroid Change)" if valid_centroid_indices else ""
        plt.xlabel('Mega Cluster ID', fontsize=12)
        plt.ylabel('Approximate Overlap Score', fontsize=12)
        plt.title(f'Approximate Overlap Scores by Iteration{title_suffix}', fontsize=14, fontweight='bold')
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
            
            # Filter scores if valid_centroid_indices is provided
            if valid_centroid_indices is not None:
                filtered_ids = [cid for cid in cluster_ids if cid in valid_centroid_indices]
                filtered_scores = [scores[int(cid)] for cid in filtered_ids]
                cluster_ids = np.array(filtered_ids)
                scores = np.array(filtered_scores)
            
            label = f'Iteration {iter_num}'
            if iter_num == 0:
                label = "Initial clustering"
            plt.plot(cluster_ids, scores, label=label, marker='o', markersize=2, linewidth=1)
        
        title_suffix = " (Filtered: Minimal Centroid Change)" if valid_centroid_indices else ""
        plt.xlabel('Mega Cluster ID', fontsize=12)
        plt.ylabel('Real Overlap Score', fontsize=12)
        plt.title(f'Real Overlap Scores by Iteration{title_suffix}', fontsize=14, fontweight='bold')
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
            label = f'Iteration {iter_num}'
            if iter_num == 0:
                label = "Initial clustering"
            plt.plot(query_indices, valid_recalls, label=label,
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
    parser.add_argument('--output-dir', '-o', default='data/plots', help='Output directory for plots (default: current directory)')
    parser.add_argument('--input-dir', '-i', default='data/scores', help='Input directory to search for files (default: data/scores)')
    parser.add_argument('--dim', '-d', type=int, default=768, help='Dimension of vectors (required for centroid filtering)')
    parser.add_argument('--max-angular-change', '-m', type=float, default=0.5, help='Maximum angular change (in radians) for filtering centroids (default: 0.1)')
    parser.add_argument('--filter-centroids', '-f', action='store_true', help='Filter overlap scores to only show centroids with minimal change')
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
    centroid_files = find_iteration_files('mega_centroids_iter_')
    # Use specific regex for prev files: iter_prev_(\d+)
    mega_mini_prev_files = find_iteration_files('mega_mini_centroids_iter_prev_', iter_regex=r'iter_prev_(\d+)')
    # Exclude prev files when looking for the new files
    mega_mini_files = find_iteration_files('mega_mini_centroids_iter_', exclude_pattern='_prev_')
    
    print(f"Found {len(approx_files)} approximate overlap score files")
    print(f"Found {len(real_files)} real overlap score files")
    print(f"Found {len(recall_files)} recall score files")
    print(f"Found {len(centroid_files)} centroid files")
    print(f"Found {len(mega_mini_prev_files)} mega-mini centroid prev files")
    print(f"Found {len(mega_mini_files)} mega-mini centroid files")
    
    if not approx_files and not real_files and not recall_files:
        print("No score files found in current directory!")
        return
    
    # Create output directory if it doesn't exist (use absolute path from original directory)
    output_path = os.path.join(original_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_path, exist_ok=True)
    
    # Plot (use absolute path for output)
    if approx_files or real_files:
        # Plot closeness-based scores for each iteration pair
        if centroid_files and args.dim:
            print("\nCreating closeness-based plots for each iteration pair...")
            plot_overlap_scores_by_closeness(approx_files, real_files, centroid_files, args.dim, output_path)
        
        # Plot regular overlap scores (by cluster ID)
        centroid_files_for_filtering = None
        if not args.filter_centroids:
            if not args.dim:
                print("Warning: --dim is required for centroid filtering. Skipping filter.")
            elif not centroid_files:
                print("Warning: No centroid files found. Skipping filter.")
            else:
                centroid_files_for_filtering = centroid_files
                print(f"Using centroid filtering with max angular change = {args.max_angular_change}")
        
        plot_overlap_scores(approx_files, real_files, output_path, 
                          centroid_files=centroid_files_for_filtering,
                          dim=args.dim,
                          max_angular_change=args.max_angular_change)
    
    if recall_files:
        plot_recall_scores(recall_files, output_path)
    
    # Analyze preserved IDs if mega-mini centroid files are available
    if mega_mini_prev_files and mega_mini_files:
        print("\nAnalyzing preserved IDs between iterations...")
        analyze_preserved_ids(mega_mini_prev_files, mega_mini_files, output_path)
    
    # Change back to original directory
    os.chdir(original_dir)

if __name__ == '__main__':
    main()
