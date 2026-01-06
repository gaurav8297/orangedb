#!/usr/bin/env python3
"""
Track overlap score evolution per LSH bucket across iterations.

1. Map mini centroids to LSH buckets using Faiss IndexLSH
2. Average overlap ratios per bucket
3. Plot evolution across iterations
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import faiss


def read_mini_centroids(filepath: str, num_centroids: int) -> np.ndarray:
    """Read mini centroids, infer dimension from file size."""
    data = np.fromfile(filepath, dtype=np.float32)
    dim = len(data) // num_centroids
    return data.reshape(num_centroids, dim)


def read_overlap_scores(filepath: str) -> np.ndarray:
    """Read flat array of overlap scores (one per mini centroid)."""
    return np.fromfile(filepath, dtype=np.float64)


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


def get_worst_mini_centroids(mega_mini_ids: list[list[int]], real_scores: np.ndarray, worst_k: int, min_mini_count: int = 10) -> np.ndarray:
    """Get worst_k mini centroids (lowest real overlap score) from each mega cluster."""
    worst_mini_ids = []
    
    for mini_ids in mega_mini_ids:
        # Skip mega clusters with fewer than min_mini_count mini centroids
        if len(mini_ids) < min_mini_count:
            continue
        
        mini_ids = np.array(mini_ids)
        scores = real_scores[mini_ids]
        
        # Get indices of worst_k lowest scores
        k = min(worst_k, len(mini_ids))
        worst_indices = np.argsort(scores)[:k]
        worst_mini_ids.extend(mini_ids[worst_indices].tolist())
    
    return np.array(worst_mini_ids, dtype=np.uint64)


def build_lsh_index(dim: int, nbits: int) -> faiss.IndexLSH:
    """Build LSH index for encoding."""
    index = faiss.IndexLSH(dim, nbits, True, False)  # rotate_data=True, train_thresholds=False
    return index


def encode_to_buckets(index: faiss.IndexLSH, centroids: np.ndarray) -> tuple[np.ndarray, int]:
    """Encode centroids to LSH bucket codes."""
    # Python API: sa_encode returns the codes directly
    n = centroids.shape[0]
    codes = index.sa_encode(centroids)
    return codes, n


def codes_to_bucket_ids(codes: np.ndarray, code_size: int, n: int) -> np.ndarray:
    """Convert byte codes to integer bucket IDs."""
    codes = codes.reshape(n, code_size)
    
    # Treat each code as a single integer (for grouping)
    # For small nbits (<=64), we can pack into uint64
    if code_size <= 8:
        bucket_ids = np.zeros(n, dtype=np.uint64)
        for i in range(code_size):
            bucket_ids |= codes[:, i].astype(np.uint64) << (8 * i)
        return bucket_ids
    else:
        # For larger codes, use tuple representation as string
        return np.array([codes[i].tobytes() for i in range(n)])


def aggregate_scores_by_bucket(bucket_ids: np.ndarray, scores: np.ndarray, worst_n: int = 10) -> dict:
    """Average of worst_n scores per bucket."""
    bucket_scores = {}
    unique_buckets = np.unique(bucket_ids)
    
    for bucket in unique_buckets:
        mask = bucket_ids == bucket
        bucket_scores_arr = scores[mask]
        
        # Take worst_n (lowest) scores and average them
        k = min(worst_n, len(bucket_scores_arr))
        worst_scores = np.sort(bucket_scores_arr)[:k]
        
        bucket_scores[bucket] = {
            'mean': np.mean(worst_scores),
            'count': np.sum(mask)
        }
    
    return bucket_scores


def load_iteration_data(base_dir: Path, iteration: int, index: faiss.IndexLSH, worst_k: int):
    """Load and process data for one iteration using provided LSH index."""
    approx_path = base_dir / f"approx_overlap_scores_iter_{iteration}.bin"
    real_path = base_dir / f"real_overlap_scores_iter_{iteration}.bin"
    mini_centroids_path = base_dir / f"mini_centroids_iter_{iteration}.bin"
    mega_mini_path = base_dir / f"mega_mini_centroids_iter_{iteration}.bin"
    
    approx_scores = read_overlap_scores(approx_path)
    real_scores = read_overlap_scores(real_path)
    num_centroids = len(approx_scores)
    
    mini_centroids = read_mini_centroids(mini_centroids_path, num_centroids)
    mega_mini_ids = read_mega_mini_centroids(mega_mini_path)
    
    # Get only worst_k mini centroids from each mega cluster
    worst_mini_ids = get_worst_mini_centroids(mega_mini_ids, real_scores, worst_k)
    
    # Filter to only worst mini centroids
    filtered_centroids = mini_centroids[worst_mini_ids]
    filtered_approx = approx_scores[worst_mini_ids]
    filtered_real = real_scores[worst_mini_ids]
    
    # Encode using shared LSH index
    codes, n = encode_to_buckets(index, filtered_centroids)
    bucket_ids = codes_to_bucket_ids(codes, index.code_size, n)
    
    # Aggregate scores
    approx_by_bucket = aggregate_scores_by_bucket(bucket_ids, filtered_approx)
    real_by_bucket = aggregate_scores_by_bucket(bucket_ids, filtered_real)
    
    return bucket_ids, approx_by_bucket, real_by_bucket


def collect_bucket_evolution(base_dir: Path, iterations: list[int], nbits: int, worst_k: int):
    """Collect bucket scores across iterations."""
    all_buckets = set()
    iteration_data = []
    
    # Build LSH index ONCE using first iteration to get dimension
    first_iter = iterations[0]
    approx_path = base_dir / f"approx_overlap_scores_iter_{first_iter}.bin"
    mini_centroids_path = base_dir / f"mini_centroids_iter_{first_iter}.bin"
    
    approx_scores = read_overlap_scores(approx_path)
    num_centroids = len(approx_scores)
    mini_centroids = read_mini_centroids(mini_centroids_path, num_centroids)
    dim = mini_centroids.shape[1]
    
    print(f"Building LSH index with dim={dim}, nbits={nbits}")
    print(f"Using worst {worst_k} mini centroids per mega cluster")
    index = build_lsh_index(dim, nbits)
    
    # Now process all iterations with the same index
    for it in iterations:
        bucket_ids, approx_by_bucket, real_by_bucket = load_iteration_data(base_dir, it, index, worst_k)
        all_buckets.update(approx_by_bucket.keys())
        iteration_data.append({
            'approx': approx_by_bucket,
            'real': real_by_bucket
        })
    
    return list(all_buckets), iteration_data


def plot_recall_evolution(
    base_dir: Path,
    iterations: list[int],
    output_dir: str,
    sample_ratio: float = 1.0
):
    """Plot recall per query, with each iteration as a separate line."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load recall data for all iterations
    recall_data = []
    for it in iterations:
        recall_path = base_dir / f"recall_iter_{it}.bin"
        if recall_path.exists():
            recalls = np.fromfile(recall_path, dtype=np.float64)
            recall_data.append(recalls)
        else:
            print(f"Warning: {recall_path} not found, skipping iteration {it}")
            recall_data.append(None)
    
    # Find valid iterations
    valid_iters = [(it, data) for it, data in zip(iterations, recall_data) if data is not None]
    if not valid_iters:
        print("No recall data found, skipping recall plot")
        return
    
    # Sample queries for readability
    num_queries = len(valid_iters[0][1])
    n_sample = max(1, int(num_queries * sample_ratio))
    sample_idx = np.sort(np.random.choice(num_queries, size=n_sample, replace=False))
    
    plt.figure(figsize=(14, 6))
    for iteration, recalls in valid_iters:
        sampled_recalls = recalls[sample_idx]
        plt.plot(np.arange(len(sample_idx)), sampled_recalls, marker='o', markersize=2, label=f"Iter {iteration}", alpha=0.7)
    
    plt.xlabel("Query ID (sampled)")
    plt.ylabel("Recall")
    plt.title(f"Per-Query Recall (sampled {n_sample}/{num_queries} queries)")
    plt.legend()
    plt.tight_layout()
    
    out_path = Path(output_dir) / "recall_per_query.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def plot_bucket_evolution(
    buckets: list,
    iteration_data: list[dict],
    iterations: list[int],
    output_dir: str,
    nbits: int,
    top_k: int = 10
):
    """Plot overlap scores per bucket, with each iteration as a separate line."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get bucket sizes from first iteration to pick representative buckets
    first_iter_approx = iteration_data[0]['approx']
    bucket_sizes = [(b, first_iter_approx[b]['count']) for b in buckets if b in first_iter_approx]
    bucket_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Pick top_k largest buckets
    selected_buckets = [b for b, _ in bucket_sizes[:top_k]]
    
    print(f"Selected {len(selected_buckets)} largest buckets for plotting")
    for b, size in bucket_sizes[:top_k]:
        print(f"  Bucket {b}: {size} mini centroids")
    
    x = np.arange(len(selected_buckets))
    
    # Plot approx scores - x=buckets, each line=iteration
    plt.figure(figsize=(14, 6))
    for it_idx, iteration in enumerate(iterations):
        scores = []
        for bucket in selected_buckets:
            data = iteration_data[it_idx]
            if bucket in data['approx']:
                scores.append(np.exp(-data['approx'][bucket]['mean']))
            else:
                scores.append(np.nan)
        plt.plot(x, scores, marker='o', markersize=3, label=f"Iter {iteration}")
    
    plt.xlabel("LSH Bucket")
    plt.ylabel("exp(-Avg Approx Overlap Score)")
    plt.title(f"Approx Overlap Score per LSH Bucket (nbits={nbits})")
    plt.xticks(x[::max(1, len(x)//20)], [selected_buckets[i] for i in range(0, len(x), max(1, len(x)//20))], rotation=45)
    plt.legend()
    plt.tight_layout()
    
    out_path = Path(output_dir) / f"lsh_bucket_approx_evolution_nbits{nbits}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()
    
    # Plot real scores - x=buckets, each line=iteration
    plt.figure(figsize=(14, 6))
    for it_idx, iteration in enumerate(iterations):
        scores = []
        for bucket in selected_buckets:
            data = iteration_data[it_idx]
            if bucket in data['real']:
                scores.append(np.exp(-data['real'][bucket]['mean']))
            else:
                scores.append(np.nan)
        plt.plot(x, scores, marker='o', markersize=3, label=f"Iter {iteration}")
    
    plt.xlabel("LSH Bucket")
    plt.ylabel("exp(-Avg Real Overlap Score)")
    plt.title(f"Real Overlap Score per LSH Bucket (nbits={nbits})")
    plt.xticks(x[::max(1, len(x)//20)], [selected_buckets[i] for i in range(0, len(x), max(1, len(x)//20))], rotation=45)
    plt.legend()
    plt.tight_layout()
    
    out_path = Path(output_dir) / f"lsh_bucket_real_evolution_nbits{nbits}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot LSH bucket overlap score evolution")
    parser.add_argument("--base-dir", type=str, default="../data/scores/", help="Directory with score files")
    parser.add_argument("--iterations", type=str, default="1,2,3,4,5", help="Comma-separated iteration numbers")
    parser.add_argument("--nbits", type=int, default=8, help="Number of LSH bits")
    parser.add_argument("--output-dir", type=str, default="../data/plots/", help="Output directory")
    parser.add_argument("--top-k", type=int, default=50, help="Number of largest buckets to plot")
    parser.add_argument("--worst-k", type=int, default=20, help="Number of worst mini centroids per mega cluster")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    iterations = [int(x) for x in args.iterations.split(",")]
    
    print(f"Processing iterations: {iterations}")
    print(f"LSH nbits: {args.nbits}")
    
    buckets, iteration_data = collect_bucket_evolution(base_dir, iterations, args.nbits, args.worst_k)
    print(f"Found {len(buckets)} unique LSH buckets")
    
    plot_bucket_evolution(
        buckets,
        iteration_data,
        iterations,
        args.output_dir,
        args.nbits,
        args.top_k
    )
    
    # Plot recall evolution
    plot_recall_evolution(base_dir, iterations, args.output_dir)


if __name__ == "__main__":
    main()
