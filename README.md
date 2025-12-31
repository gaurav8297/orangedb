# OrangeDB

A high-performance vector database written in C++ that implements advanced clustering-based indexing strategies for approximate nearest neighbor (ANN) search. OrangeDB focuses on dynamic reclustering techniques to maintain search quality as data distributions evolve.

## Features

- **Multiple Indexing Strategies**
  - HNSW (Hierarchical Navigable Small World) graphs
  - Partitioned indexes with hierarchical clustering
  - Incremental indexes for streaming data
  - Dynamic reclustering indexes for evolving datasets

- **Advanced Clustering**
  - K-means clustering with Faiss integration
  - Two-level hierarchical clustering (mega/mini centroids)
  - Dynamic reclustering based on cluster quality metrics
  - Configurable cluster size limits and sampling ratios

- **Performance Optimizations**
  - SIMD-accelerated distance computations (SimSIMD)
  - OpenMP parallelization
  - Scalar quantization support (8-bit)
  - Native CPU optimizations in release mode

- **Visualization & Analysis**
  - UMAP dimensionality reduction (2D/3D)
  - Live and offline visualization modes
  - Clustering quality metrics
  - Recall and overlap analysis

- **Distance Metrics**
  - L2 (Euclidean) distance
  - Inner Product (IP) similarity
  - Extensible distance computation framework

## Architecture

### Core Components

```
src/
├── include/
│   ├── hnsw.h                    # HNSW index implementation
│   ├── partitioned_index.h       # Partitioned indexing
│   ├── incremental_index.h       # Incremental updates
│   ├── reclustering_index.h      # Dynamic reclustering
│   ├── clustering.h              # Clustering algorithms
│   ├── storage/                  # Storage layer
│   ├── fastQ/                    # SIMD optimizations & quantization
│   └── common.h                  # Shared data structures
└── main.cpp                      # Benchmarks and experiments
```

### Third-Party Dependencies

- **[Faiss](https://github.com/facebookresearch/faiss)**: Meta's similarity search and clustering library
- **[SimSIMD](https://github.com/ashvardanian/SimSIMD)**: SIMD-optimized distance computations
- **[iRangeGraph](https://github.com/Jeffery-Meng/CanDE)**: Graph-based search algorithms
- **[spdlog](https://github.com/gabime/spdlog)**: Fast C++ logging
- **[umappp](https://github.com/LTLA/umappp)**: UMAP dimensionality reduction
- **[Apache Arrow/Parquet](https://arrow.apache.org/)**: Columnar data processing
- **[nlohmann/json](https://github.com/nlohmann/json)**: JSON parsing
- **[Backward-cpp](https://github.com/bombela/backward-cpp)**: Stack trace debugging

## Building

### Prerequisites

**macOS:**
```bash
brew install cmake libomp apache-arrow
```

**Linux (CentOS/RHEL):**
```bash
# See setup_centos_7.sh for detailed setup
sudo yum install cmake3 gcc-c++ openmp-devel
```

### Build Commands

```bash
# Release build (optimized)
make build

# Debug build (with AddressSanitizer)
make debug

# Parallel build with custom thread count
NUM_THREADS=8 make build
```

The build system uses CMake with C++20 standard. Binaries are placed in `build/release/bin/` or `build/debug/bin/`.

### Build Configuration

- **Compiler**: GCC/Clang with C++20 support
- **Optimization**: `-O3` with optional `-march=native`
- **Debug**: AddressSanitizer enabled for memory safety
- **Parallelism**: OpenMP for multi-threading

## Usage

### Running Benchmarks

The main executable (`orangedb_main`) supports various benchmark modes:

```bash
./build/release/bin/orangedb_main -run benchmarkFastReclustering \
  -baseVectorPath data/siftsmall/base.fvecs \
  -queryVectorPath data/siftsmall/query.fvecs \
  -groundTruthPath data/siftsmall/gt.bin \
  -k 100 \
  -numVectors 10000 \
  -numIters 20 \
  -megaCentroidSize 10 \
  -miniCentroidSize 500 \
  -nMegaProbes 4 \
  -nMiniProbes 40 \
  -numQueries 50 \
  -hardClusterSizeLimit 600 \
  -numThreads 1
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-run` | Benchmark mode to execute | - |
| `-baseVectorPath` | Path to base vectors (`.fvecs` format) | - |
| `-queryVectorPath` | Path to query vectors | - |
| `-groundTruthPath` | Path to ground truth results | - |
| `-k` | Number of nearest neighbors | 100 |
| `-numVectors` | Number of vectors to index | - |
| `-numIters` | Number of iteration/insertion rounds | 20 |
| `-megaCentroidSize` | First-level cluster count | 10 |
| `-miniCentroidSize` | Second-level cluster size | 500 |
| `-nMegaProbes` | First-level probes during search | 4 |
| `-nMiniProbes` | Second-level probes during search | 40 |
| `-hardClusterSizeLimit` | Max cluster size before split | 10000 |
| `-numThreads` | Thread count for parallel operations | 1 |
| `-useIP` | Use Inner Product instead of L2 | 0 |
| `-umap_mode` | Visualization mode (0=none, 1=live, 2=offline) | 0 |

### Visualization

Generate UMAP projections of your dataset:

```bash
# 2D projection
./build/release/bin/orangedb_main -run run_umap_2D_without_clustering \
  -baseVectorPath data/siftsmall/base.fvecs \
  -numVectors 10000 \
  -outputPath umap_2D.bin

# 3D projection
./build/release/bin/orangedb_main -run run_umap_3D_without_clustering \
  -baseVectorPath data/siftsmall/base.fvecs \
  -numVectors 10000 \
  -outputPath umap_3D.bin
```

Use the Python visualization scripts in `src/include/`:
- `visualize_data_with_centroids.py`: Visualize with cluster centroids
- `visualize_data_no_centroids.py`: Visualize raw embeddings
- `visualize_helpers.py`: Helper utilities

## Data Formats

### Vector Formats

- **`.fvecs`**: Standard vector format used in ANN benchmarks
  - 4-byte integer (dimension) followed by dimension × 4-byte floats
  - Multiple vectors concatenated
  
- **`.bin`**: Binary format for serialized index state

- **`.csv`**: Text format for embeddings and analysis

### Example Datasets

The `data/siftsmall/` directory contains the SIFT-Small dataset:
- `base.fvecs`: 10K base vectors (128-dim)
- `query.fvecs`: 100 query vectors
- `groundtruth.ivecs`: True k-nearest neighbors

## Development

### Project Structure

```
orangedb/
├── src/                    # Source code
│   ├── include/            # Header files
│   ├── *.cpp               # Implementation files
│   └── main.cpp            # Entry point with benchmarks
├── data/                   # Test datasets
├── third_party/            # External dependencies
├── build/                  # Build output (generated)
├── CMakeLists.txt          # CMake configuration
├── Makefile                # Build shortcuts
└── README.md               # This file
```

### Adding New Benchmarks

Edit `src/main.cpp` and add a new benchmark function. Follow the pattern of existing benchmarks like `benchmarkFastReclustering()`.

### Code Style

- C++20 features encouraged
- Header-only when possible for templates
- Use `spdlog` for logging
- Follow existing naming conventions

## Research & Experimentation

The `exp_gilli.sh` and `exp.sh` scripts contain experimental configurations for research on:
- Cluster size limits and their impact on recall
- Live vs. offline UMAP visualization
- Reclustering strategies and thresholds
- Multi-threaded performance analysis

Results are output as:
- PNG plots for visual analysis
- Binary files for metric storage
- CSV files for external analysis

## Performance Metrics

OrangeDB tracks:
- **Recall@k**: Search quality metric
- **Distance Computations**: Query efficiency
- **Cluster Overlap**: Reclustering quality
- **Angular Distances**: Centroid movement
- **Write Amplification**: Storage efficiency

## Contributing

When contributing:
1. Follow the existing code structure
2. Add tests for new indexing strategies
3. Update documentation for new parameters
4. Run benchmarks to verify performance
5. Use `make debug` to check for memory issues

## License

[Add your license here]

## References

- HNSW paper: [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)
- Faiss: [A Library for Efficient Similarity Search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
- UMAP: [Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426)

## Contact

For questions, issues, or contributions, please [add contact information or link to issues].

## TODO
- [ ] Benchmark Clustering + with equal distribution (10M Dataset)
- [ ] Implement Disk Storage
- [ ] Implement Transactions [Local Storage + Final Storage + WAL]
