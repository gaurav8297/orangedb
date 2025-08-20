# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OrangeDB is a high-performance vector database written in C++ that implements multiple indexing strategies for approximate nearest neighbor search. The project focuses on clustering-based approaches and reclustering techniques for efficient vector similarity search.

## Core Architecture

### Main Components
- **Vector Indexes**: HNSW, partitioned indexes, incremental indexes, and reclustering indexes
- **Clustering**: K-means and advanced clustering algorithms for data partitioning
- **Storage Layer**: File handling and data chunk management with planned disk storage support
- **FastQ**: SIMD-optimized distance computation library for scalar quantization
- **Third-party integrations**: Faiss (Meta's similarity search library), SimSIMD, iRangeGraph, and Arrow/Parquet

### Key Modules
- `src/reclustering_index.cpp` - Main reclustering implementation (currently in development)
- `src/hnsw.cpp` - HNSW (Hierarchical Navigable Small World) index implementation  
- `src/partitioned_index.cpp` - Partitioned indexing strategy
- `src/clustering.cpp` - Clustering algorithms and utilities
- `src/incremental_index.cpp` - Support for incremental index updates
- `src/file_handle.cpp` - Low-level file I/O operations

## Build System

### Building the Project
```bash
# Release build
make build

# Debug build (with AddressSanitizer)
make debug

# With custom thread count
NUM_THREADS=8 make build
```

### Build Configuration
- Uses CMake with C++20 standard
- Supports both Debug (with AddressSanitizer) and Release configurations
- OpenMP enabled for parallel processing
- Native CPU optimization in release mode (`-march=native`)
- Requires Arrow and Parquet libraries

### Dependencies
- **OpenMP**: For parallel processing
- **Apache Arrow/Parquet**: For columnar data processing
- **Faiss**: Meta's similarity search and clustering library
- **SimSIMD**: SIMD-optimized distance computations
- **spdlog**: Fast C++ logging library
- **nlohmann/json**: JSON processing
- **Backward-cpp**: Stack trace generation for debugging

## Development Workflow

### Project Structure
```
src/
├── include/           # Header files
│   ├── storage/       # Storage layer headers
│   └── fastQ/         # SIMD optimization headers
├── *.cpp             # Implementation files
└── main.cpp          # Entry point

third_party/          # External dependencies
├── faiss/            # Meta's similarity search library
├── simsimd/          # SIMD distance computations
├── iRangeGraph/      # Graph-based search
└── liburing/         # Async I/O (not currently used)
```

### Current Development Focus
The project is actively working on reclustering approaches as indicated by recent commits. The main areas of development are:
- Reclustering algorithms for improved search performance
- Integration with Faiss for clustering operations
- Performance optimization using SIMD instructions
- Storage layer improvements with planned disk storage support

### Key Files for Understanding
- `src/main.cpp` - Entry point with various index implementations and benchmarks
- `src/include/reclustering_index.h` - Core reclustering interface (recently modified)
- `src/include/common.h` - Shared data structures and constants
- `CMakeLists.txt` - Build configuration and dependencies
- `Makefile` - Simple build interface

### Running and Testing
The main executable is `orangedb_main` built from `src/main.cpp`. The project includes datasets in the `data/` directory (gist and siftsmall) for testing and benchmarking.


