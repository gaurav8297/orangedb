#include "cuvs_kmeans.h"

#include <raft/core/resources.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

#include <chrono>
#include <cstdio>

void cuvs_kmeans_fit(
    const float* hostData,
    int64_t numVectors,
    int64_t dimension,
    int64_t numClusters,
    uint32_t nIter,
    bool useIP,
    float* outCentroids,
    uint32_t* outLabels)
{
    printf("[cuvs_kmeans_fit] numVectors=%ld, dimension=%ld, numClusters=%ld, nIter=%u, metric=%s\n",
           (long)numVectors, (long)dimension, (long)numClusters, nIter,
           useIP ? "InnerProduct" : "L2Expanded");

    raft::resources handle;
    auto stream = raft::resource::get_cuda_stream(handle);

    printf("[cuvs_kmeans_fit] Copying data to device...\n");
    auto d_dataset = raft::make_device_matrix<float, int64_t>(handle, numVectors, dimension);
    raft::copy(d_dataset.data_handle(), hostData, numVectors * dimension, stream);
    raft::resource::sync_stream(handle);

    auto centroids = raft::make_device_matrix<float, int64_t>(handle, numClusters, dimension);
    auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, numVectors);

    cuvs::cluster::kmeans::balanced_params params;
    params.metric = useIP ? cuvs::distance::DistanceType::InnerProduct
                          : cuvs::distance::DistanceType::L2Expanded;
    params.n_iters = nIter;

    printf("[cuvs_kmeans_fit] Running fit_predict...\n");
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

    double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[cuvs_kmeans_fit] fit_predict completed in %.2f ms\n", elapsedMs);

    printf("[cuvs_kmeans_fit] Copying results to host...\n");
    raft::copy(outCentroids, centroids.data_handle(), numClusters * dimension, stream);
    raft::copy(outLabels, labels.data_handle(), numVectors, stream);
    raft::resource::sync_stream(handle);

    printf("[cuvs_kmeans_fit] Done.\n");
}

void cuvs_kmeans_predict(
    const float* hostData,
    int64_t numVectors,
    int64_t dimension,
    const float* hostCentroids,
    int64_t numClusters,
    uint32_t nIter,
    bool useIP,
    uint32_t* outLabels)
{
    printf("[cuvs_kmeans_predict] numVectors=%ld, dimension=%ld, numClusters=%ld, nIter=%u, metric=%s\n",
           (long)numVectors, (long)dimension, (long)numClusters, nIter,
           useIP ? "InnerProduct" : "L2Expanded");

    raft::resources handle;
    auto stream = raft::resource::get_cuda_stream(handle);

    printf("[cuvs_kmeans_predict] Copying data to device...\n");
    auto d_dataset = raft::make_device_matrix<float, int64_t>(handle, numVectors, dimension);
    raft::copy(d_dataset.data_handle(), hostData, numVectors * dimension, stream);

    auto d_centroids = raft::make_device_matrix<float, int64_t>(handle, numClusters, dimension);
    raft::copy(d_centroids.data_handle(), hostCentroids, numClusters * dimension, stream);

    auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, numVectors);
    raft::resource::sync_stream(handle);

    cuvs::cluster::kmeans::balanced_params params;
    params.metric = useIP ? cuvs::distance::DistanceType::InnerProduct
                          : cuvs::distance::DistanceType::L2Expanded;
    params.n_iters = nIter;

    printf("[cuvs_kmeans_predict] Running predict...\n");
    raft::resource::sync_stream(handle);
    auto t0 = std::chrono::high_resolution_clock::now();

    cuvs::cluster::kmeans::predict(
        handle,
        params,
        raft::make_const_mdspan(d_dataset.view()),
        raft::make_const_mdspan(d_centroids.view()),
        labels.view());

    raft::resource::sync_stream(handle);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[cuvs_kmeans_predict] predict completed in %.2f ms\n", elapsedMs);

    printf("[cuvs_kmeans_predict] Copying labels to host...\n");
    raft::copy(outLabels, labels.data_handle(), numVectors, stream);
    raft::resource::sync_stream(handle);

    printf("[cuvs_kmeans_predict] Done.\n");
}
