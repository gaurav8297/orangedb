#!/usr/bin/env python3
"""
Plot overlap scores for small, medium, and large mega clusters.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def read_mega_mini_centroids(filepath: str) -> list[list[int]]:
    """Read nested vector of mega -> mini centroid IDs."""
    with open(filepath, 'rb') as f:
        num_mega = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        mega_mini_ids = []
        for _ in range(num_mega):
            num_mini = np.frombuffer(f.read(8), dtype=np.uint64)[0]
            if num_mini > 0:
                mini_ids = np.frombuffer(f.read(num_mini * 8), dtype=np.uint64)
                mega_mini_ids.append(mini_ids.tolist())
            else:
                mega_mini_ids.append([])
        return mega_mini_ids


def read_overlap_scores(filepath: str) -> np.ndarray:
    """Read flat array of overlap scores (one per mini centroid)."""
    return np.fromfile(filepath, dtype=np.float64)


def pick_mega_clusters(mega_mini_ids: list[list[int]]) -> dict[str, int]:
    """Pick small, medium, and large mega clusters based on mini cluster count."""
    sizes = [(i, len(mini_ids)) for i, mini_ids in enumerate(mega_mini_ids)]
    sizes = [(i, s) for i, s in sizes if s > 0]  # Filter empty clusters
    sizes.sort(key=lambda x: x[1])
    
    n = len(sizes)
    small_idx = sizes[0][0]
    medium_idx = sizes[n // 2][0]
    large_idx = sizes[-1][0]
    
    print(f"Selected mega clusters:")
    print(f"  Small:  mega_id={small_idx}, num_mini={len(mega_mini_ids[small_idx])}")
    print(f"  Medium: mega_id={medium_idx}, num_mini={len(mega_mini_ids[medium_idx])}")
    print(f"  Large:  mega_id={large_idx}, num_mini={len(mega_mini_ids[large_idx])}")
    
    return {"small": small_idx, "medium": medium_idx, "large": large_idx}


def plot_overlap_scores(
    mega_mini_ids: list[list[int]],
    approx_scores: np.ndarray,
    real_scores: np.ndarray,
    selected_megas: dict[str, int],
    iteration: int,
    output_dir: str = "data/plots",
    sample_ratio: float = 0.3
):
    """Plot overlap scores for selected mega clusters."""
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for label, mega_id in selected_megas.items():
        mini_ids = np.array(mega_mini_ids[mega_id])
        
        # Sample 30% randomly
        n_sample = max(1, int(len(mini_ids) * sample_ratio))
        sample_idx = np.random.choice(len(mini_ids), size=n_sample, replace=False)
        sample_idx = np.sort(sample_idx)
        
        sampled_mini_ids = mini_ids[sample_idx]
        approx = approx_scores[sampled_mini_ids]
        real = real_scores[sampled_mini_ids]
        
        plt.figure(figsize=(10, 5))
        x = np.arange(len(sampled_mini_ids))
        
        plt.plot(x, approx, label='Approx', alpha=0.8, marker='o', markersize=3)
        plt.plot(x, real, label='Real', alpha=0.8, marker='o', markersize=3)
        
        plt.xlabel("Mini Cluster Index (sampled)")
        plt.ylabel("Overlap Score")
        plt.title(f"{label.capitalize()} Mega Cluster (id={mega_id}, sampled {n_sample}/{len(mini_ids)}) - Iter {iteration}")
        plt.legend()
        plt.tight_layout()
        
        out_path = Path(output_dir) / f"overlap_scores_iter{iteration}_{label}.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot overlap scores for mega clusters")
    parser.add_argument("--base-dir", type=str, default="data/scores/", help="Directory with score files")
    parser.add_argument("--iter", type=int, default=3, help="Iteration number to plot")
    parser.add_argument("--output-dir", type=str, default="data/plots/", help="Output directory for plots")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    iteration = args.iter
    
    # Read data files
    mega_mini_path = base_dir / f"mega_mini_centroids_iter_{iteration}.bin"
    approx_path = base_dir / f"approx_overlap_scores_iter_{iteration}.bin"
    real_path = base_dir / f"real_overlap_scores_iter_{iteration}.bin"
    
    print(f"Reading files for iteration {iteration}...")
    mega_mini_ids = read_mega_mini_centroids(mega_mini_path)
    approx_scores = read_overlap_scores(approx_path)
    real_scores = read_overlap_scores(real_path)
    
    print(f"Loaded {len(mega_mini_ids)} mega clusters, {len(approx_scores)} mini centroids")
    
    # Pick representative mega clusters
    selected = pick_mega_clusters(mega_mini_ids)
    
    # Plot
    plot_overlap_scores(
        mega_mini_ids, 
        approx_scores, 
        real_scores, 
        selected, 
        iteration,
        args.output_dir
    )

if __name__ == "__main__":
    main()
