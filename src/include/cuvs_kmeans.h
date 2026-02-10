#pragma once

#include <cstdint>

void cuvs_kmeans_fit(
    const float* hostData,
    int64_t numVectors,
    int64_t dimension,
    int64_t numClusters,
    uint32_t nIter,
    bool useIP,
    float* outCentroids,
    int gpuDevice = 0);

void cuvs_kmeans_predict(
    const float* hostData,
    int64_t numVectors,
    int64_t dimension,
    const float* hostCentroids,
    int64_t numClusters,
    uint32_t nIter,
    bool useIP,
    uint32_t* outLabels,
    int gpuDevice = 0);
