#!/usr/bin/env python3
"""
Generate synthetic vectors from bigann_base.bvecs by adding random decimal perturbations.
Reads top 10M vectors and creates 100M synthetic vectors.
"""

import numpy as np
import struct
import os
from pathlib import Path

def read_bvecs(filename, num_vectors=None):
    """
    Read vectors from .bvecs file format.

    Format: each vector is [d, v1, v2, ..., vd] where d is dimension (int32)
    and v1...vd are unsigned bytes.
    """
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            # Read dimension
            dim_bytes = f.read(4)
            if not dim_bytes:
                break

            dim = struct.unpack('i', dim_bytes)[0]

            # Read vector data (unsigned bytes)
            vec_bytes = f.read(dim)
            if len(vec_bytes) != dim:
                break

            vec = np.frombuffer(vec_bytes, dtype=np.uint8)
            vectors.append(vec.astype(np.float32))

            if num_vectors and len(vectors) >= num_vectors:
                break

    return np.array(vectors)

def write_fvecs(filename, vectors):
    """
    Write vectors to .fvecs file format.

    Format: each vector is [d, v1, v2, ..., vd] where d is dimension (int32)
    and v1...vd are float32 values.
    """
    with open(filename, 'wb') as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack('i', dim))
            f.write(vec.astype(np.float32).tobytes())

def generate_synthetic_vectors(base_vectors, target_count, perturbation_range=(0.0, 0.9)):
    """
    Generate synthetic vectors by adding random decimal perturbations.

    Args:
        base_vectors: Base vectors to use as templates
        target_count: Number of synthetic vectors to generate
        perturbation_range: Range of random decimal values to add (min, max)

    Returns:
        Array of synthetic vectors
    """
    num_base = len(base_vectors)
    dim = base_vectors.shape[1]

    synthetic_vectors = np.zeros((target_count, dim), dtype=np.float32)

    print(f"Generating {target_count} synthetic vectors from {num_base} base vectors...")

    # Generate in batches to manage memory
    batch_size = 1000000
    for i in range(0, target_count, batch_size):
        end_idx = min(i + batch_size, target_count)
        batch_count = end_idx - i

        # Randomly select base vectors
        base_indices = np.random.randint(0, num_base, size=batch_count)
        batch_vectors = base_vectors[base_indices].copy()

        # Add random single decimal digit [0.0, 0.1, 0.2, ..., 0.9] to each dimension
        random_digits = np.random.randint(0, 10, size=(batch_count, dim))
        perturbations = (random_digits / 10.0).astype(np.float32)

        synthetic_vectors[i:end_idx] = batch_vectors + perturbations

        if (i // batch_size) % 10 == 0:
            print(f"  Generated {end_idx}/{target_count} vectors ({100*end_idx/target_count:.1f}%)")

    return synthetic_vectors

def main():
    input_file = "/home/centos/bigann_base.bvecs"
    output_file = "/home/centos/bigann_synthetic_50M.fvecs"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(f"Please ensure bigann_base.bvecs is in the data/ directory")
        return

    # Parameters
    num_base_vectors = 10_000_000  # 10M
    num_synthetic_vectors = 50_000_000  # 100M

    print(f"Reading {num_base_vectors:,} vectors from {input_file}...")
    base_vectors = read_bvecs(input_file, num_vectors=num_base_vectors)
    print(f"Read {len(base_vectors):,} vectors of dimension {base_vectors.shape[1]}")

    print(f"\nGenerating {num_synthetic_vectors:,} synthetic vectors...")
    synthetic_vectors = generate_synthetic_vectors(base_vectors, num_synthetic_vectors)

    print(f"\nWriting synthetic vectors to {output_file}...")
    write_fvecs(output_file, synthetic_vectors)

    file_size_gb = os.path.getsize(output_file) / (1024**3)
    print(f"Done! Generated {len(synthetic_vectors):,} vectors")
    print(f"Output file: {output_file} ({file_size_gb:.2f} GB)")
    print(f"\nExample original vector: {base_vectors[0][:10]}")
    print(f"Example synthetic vector: {synthetic_vectors[0][:10]}")

if __name__ == "__main__":
    main()
