#pragma once

#include <cstdint>

void benchmark_cuvs_balanced_kmeans(
    const float* hostData,
    int64_t numVectors,
    int64_t dimension,
    int64_t numClusters,
    uint32_t nIter,
    bool useIP);
