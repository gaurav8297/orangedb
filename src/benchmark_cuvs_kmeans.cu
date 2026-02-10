#include "benchmark_cuvs_kmeans.h"

#include <raft/core/resources.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

#include <chrono>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

void benchmark_cuvs_balanced_kmeans(
    const float* hostData,
    int64_t numVectors,
    int64_t dimension,
    int64_t numClusters,
    uint32_t nIter,
    bool useIP)
{
    printf("=== cuVS Balanced K-Means Benchmark ===\n");
    printf("  Vectors: %ld, Dimension: %ld, Clusters: %ld, Iterations: %u\n",
           (long)numVectors, (long)dimension, (long)numClusters, nIter);
    printf("  Metric: %s\n", useIP ? "InnerProduct" : "L2Expanded");

    raft::resources handle;
    auto stream = raft::resource::get_cuda_stream(handle);

    // Copy host data to device
    auto d_dataset = raft::make_device_matrix<float, int64_t>(handle, numVectors, dimension);
    raft::copy(d_dataset.data_handle(), hostData, numVectors * dimension, stream);
    raft::resource::sync_stream(handle);

    // Allocate outputs
    auto centroids = raft::make_device_matrix<float, int64_t>(handle, numClusters, dimension);
    auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, numVectors);

    // Configure balanced k-means params
    cuvs::cluster::kmeans::balanced_params params;
    params.metric = useIP ? cuvs::distance::DistanceType::InnerProduct
                          : cuvs::distance::DistanceType::L2Expanded;
    params.n_iters = nIter;

    // Benchmark fit_predict (combined training + assignment)
    raft::resource::sync_stream(handle);
    auto t0 = std::chrono::high_resolution_clock::now();

    cuvs::cluster::kmeans::fit_predict(
        handle,
        params,
        raft::make_const_mdspan(d_dataset.view()),
        centroids.view(),
        labels.view());

    raft::resource::sync_stream(handle);
    auto t1 = std::chrono::high_resolution_clock::now();

    double fitPredictMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  fit_predict time: %.2f ms\n", fitPredictMs);

    // Benchmark predict-only (assignment with existing centroids)
    raft::resource::sync_stream(handle);
    auto t2 = std::chrono::high_resolution_clock::now();

    cuvs::cluster::kmeans::predict(
        handle,
        params,
        raft::make_const_mdspan(d_dataset.view()),
        raft::make_const_mdspan(centroids.view()),
        labels.view());

    raft::resource::sync_stream(handle);
    auto t3 = std::chrono::high_resolution_clock::now();

    double predictMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    printf("  predict time: %.2f ms\n", predictMs);

    // Copy labels back to host and compute cluster balance statistics
    std::vector<uint32_t> hostLabels(numVectors);
    raft::copy(hostLabels.data(), labels.data_handle(), numVectors, stream);
    raft::resource::sync_stream(handle);

    std::vector<int> histogram(numClusters, 0);
    for (int64_t i = 0; i < numVectors; ++i) {
        if (hostLabels[i] < (uint32_t)numClusters) {
            histogram[hostLabels[i]]++;
        }
    }

    int minSize = *std::min_element(histogram.begin(), histogram.end());
    int maxSize = *std::max_element(histogram.begin(), histogram.end());
    double avgSize = static_cast<double>(numVectors) / numClusters;

    double sumSquaredDiff = 0.0;
    for (int count : histogram) {
        double diff = count - avgSize;
        sumSquaredDiff += diff * diff;
    }
    double stdDev = std::sqrt(sumSquaredDiff / numClusters);

    int emptyCount = std::count(histogram.begin(), histogram.end(), 0);

    printf("  Cluster balance statistics:\n");
    printf("    Min cluster size: %d\n", minSize);
    printf("    Max cluster size: %d\n", maxSize);
    printf("    Average cluster size: %.2f\n", avgSize);
    printf("    Standard deviation: %.2f\n", stdDev);
    printf("    CV (stddev/avg): %.4f\n", stdDev / avgSize);
    printf("    Empty clusters: %d\n", emptyCount);
}
