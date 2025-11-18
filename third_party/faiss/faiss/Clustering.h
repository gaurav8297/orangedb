/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/** Implementation of k-means clustering with many variants. */

#ifndef FAISS_CLUSTERING_H
#define FAISS_CLUSTERING_H
#include <faiss/Index.h>

#include <vector>
#include <atomic>
#include <array>
#include <cmath>
#include <limits>

namespace faiss {

static constexpr int MAX_VECTOR_INDEX_NUM_CLUSTERS = 50000;

/**
 * This struct is used for balanced k-means clustering. It is passed to the
 * index.search() function to add weights to the distance values to
 * equalize the number of points assigned to each cluster.
 */
struct BalancedClusteringDistModifier {
    virtual void assign_vector(idx_t cluster_id) = 0;
    virtual float get_weight(idx_t cluster_id) const = 0;
    virtual void reset() = 0;
    virtual ~BalancedClusteringDistModifier() {}
};

/**
 * This is the lambda-based balanced k-means clustering implementation of
 * BalancedClusteringDistModifier. It's based on this paper:
 * https://ieeexplore.ieee.org/document/8621917/
 */
struct LambdaBasedDistModifier : BalancedClusteringDistModifier {
    std::array<std::atomic_int64_t, MAX_VECTOR_INDEX_NUM_CLUSTERS> cluster_sizes;
    float lambda;

    explicit LambdaBasedDistModifier(const int num_clusters, const float _lambda) : lambda(_lambda) {
        FAISS_ASSERT(num_clusters <= MAX_VECTOR_INDEX_NUM_CLUSTERS);
        FAISS_ASSERT(_lambda > 0);
        reset();
    }

    void assign_vector(int64_t cluster_id) override {
        ++cluster_sizes[cluster_id];
    }

    float get_weight(int64_t cluster_id) const override {
        return cluster_sizes[cluster_id].load(std::memory_order_relaxed) * lambda;
    }

    void reset() override {
        for (auto & cluster_size : cluster_sizes) {
            cluster_size = 0;
        }
    }
};

struct ClusterSizeCapDistModifier : BalancedClusteringDistModifier {
    std::array<std::atomic_int64_t, MAX_VECTOR_INDEX_NUM_CLUSTERS> cluster_sizes;
    std::array<std::atomic<float>, MAX_VECTOR_INDEX_NUM_CLUSTERS> cluster_weights;
    uint32_t _max_cluster_size;
    int _num_clusters;

    explicit ClusterSizeCapDistModifier(int num_clusters, uint32_t max_cluster_size) : _max_cluster_size(
        max_cluster_size), _num_clusters(num_clusters) {
        FAISS_ASSERT(num_clusters <= MAX_VECTOR_INDEX_NUM_CLUSTERS);
        FAISS_ASSERT(max_cluster_size > 0);
        reset();
    }

    void assign_vector(int64_t cluster_id) override {
        FAISS_ASSERT(cluster_id >= 0 && cluster_id < _num_clusters);
        auto new_size = ++cluster_sizes[cluster_id];
        if (new_size >= _max_cluster_size) {
            // Only update weight if not already set to infinity
            float current_weight = cluster_weights[cluster_id].load();
            if (!std::isinf(current_weight)) {
                cluster_weights[cluster_id].store(std::numeric_limits<float>::infinity());
            }
        }
    }

    float get_weight(int64_t cluster_id) const override {
        return cluster_weights[cluster_id].load();
    }

    void populate_weights(int64_t* cluster_sizes, int num_clusters) {
        for (int i = 0; i < num_clusters; i++) {
            this->cluster_sizes[i] = cluster_sizes[i];
            if (cluster_sizes[i] >= _max_cluster_size) {
                cluster_weights[i].store(std::numeric_limits<float>::infinity());
            }
        }
    }

    void reset() override {
        for (int i = 0; i < cluster_sizes.size(); i++) {
            cluster_sizes[i] = 0;
            cluster_weights[i] = 0.0f;
        }
    }
};

/** Class for the clustering parameters. Can be passed to the
 * constructor of the Clustering object.
 */
struct ClusteringParameters {
    /// number of clustering iterations
    int niter = 25;
    /// redo clustering this many times and keep the clusters with the best
    /// objective
    int nredo = 1;

    bool verbose = false;
    /// whether to normalize centroids after each iteration (useful for inner
    /// product clustering)
    bool spherical = false;
    /// round centroids coordinates to integer after each iteration?
    bool int_centroids = false;
    /// re-train index after each iteration?
    bool update_index = false;

    /// Use the subset of centroids provided as input and do not change them
    /// during iterations
    bool frozen_centroids = false;
    /// If fewer than this number of training vectors per centroid are provided,
    /// writes a warning. Note that fewer than 1 point per centroid raises an
    /// exception.
    int min_points_per_centroid = 39;
    /// to limit size of dataset, otherwise the training set is subsampled
    int max_points_per_centroid = 256;
    /// seed for the random number generator.
    /// negative values lead to seeding an internal rng with
    /// std::high_resolution_clock.
    int seed = 1234;

    /// when the training set is encoded, batch size of the codec decoder
    size_t decode_block_size = 32768;

    /// whether to check for NaNs in an input data
    bool check_input_data_for_NaNs = true;

    /// Whether to use splitmix64-based random number generator for subsampling,
    /// which is faster, but may pick duplicate points.
    bool use_faster_subsampling = false;

    /// This is the distance modifier used for balanced clustering.
    BalancedClusteringDistModifier* dist_modifier = nullptr;
};

struct ClusteringIterationStats {
    float obj;   ///< objective values (sum of distances reported by index)
    double time; ///< seconds for iteration
    double time_search;      ///< seconds for just search
    double imbalance_factor; ///< imbalance factor of iteration
    int nsplit;              ///< number of cluster splits
};

/** K-means clustering based on assignment - centroid update iterations
 *
 * The clustering is based on an Index object that assigns training
 * points to the centroids. Therefore, at each iteration the centroids
 * are added to the index.
 *
 * On output, the centoids table is set to the latest version
 * of the centroids and they are also added to the index. If the
 * centroids table it is not empty on input, it is also used for
 * initialization.
 *
 */
struct Clustering : ClusteringParameters {
    size_t d; ///< dimension of the vectors
    size_t k; ///< nb of centroids

    /** centroids (k * d)
     * if centroids are set on input to train, they will be used as
     * initialization
     */
    std::vector<float> centroids;

    /// stats at every iteration of clustering
    std::vector<ClusteringIterationStats> iteration_stats;

    Clustering(int d, int k);
    Clustering(int d, int k, const ClusteringParameters& cp);

    /** run k-means training
     *
     * @param x          training vectors, size n * d
     * @param index      index used for assignment
     * @param x_weights  weight associated to each vector: NULL or size n
     */
    virtual void train(
            idx_t n,
            const float* x,
            faiss::Index& index,
            const float* x_weights = nullptr);

    /** run with encoded vectors
     *
     * win addition to train()'s parameters takes a codec as parameter
     * to decode the input vectors.
     *
     * @param codec      codec used to decode the vectors (nullptr =
     *                   vectors are in fact floats)
     */
    void train_encoded(
            idx_t nx,
            const uint8_t* x_in,
            const Index* codec,
            Index& index,
            const float* weights = nullptr);

    virtual void compute_centroids(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        const uint8_t* x,
        const Index* codec,
        const int64_t* assign,
        const float* weights,
        float* hassign,
        float* centroids);

    /// Post-process the centroids after each centroid update.
    /// includes optional L2 normalization and nearest integer rounding
    void post_process_centroids();

    virtual ~Clustering() {}
};

/** Exact 1D clustering algorithm
 *
 * Since it does not use an index, it does not overload the train() function
 */
struct Clustering1D : Clustering {
    explicit Clustering1D(int k);

    Clustering1D(int k, const ClusteringParameters& cp);

    void train_exact(idx_t n, const float* x);

    virtual ~Clustering1D() {}
};

struct ProgressiveDimClusteringParameters : ClusteringParameters {
    int progressive_dim_steps; ///< number of incremental steps
    bool apply_pca;            ///< apply PCA on input

    ProgressiveDimClusteringParameters();
};

/** generates an index suitable for clustering when called */
struct ProgressiveDimIndexFactory {
    /// ownership transferred to caller
    virtual Index* operator()(int dim);

    virtual ~ProgressiveDimIndexFactory() {}
};

/** K-means clustering with progressive dimensions used
 *
 * The clustering first happens in dim 1, then with exponentially increasing
 * dimension until d (I steps). This is typically applied after a PCA
 * transformation (optional). Reference:
 *
 * "Improved Residual Vector Quantization for High-dimensional Approximate
 * Nearest Neighbor Search"
 *
 * Shicong Liu, Hongtao Lu, Junru Shao, AAAI'15
 *
 * https://arxiv.org/abs/1509.05195
 */
struct ProgressiveDimClustering : ProgressiveDimClusteringParameters {
    size_t d; ///< dimension of the vectors
    size_t k; ///< nb of centroids

    /** centroids (k * d) */
    std::vector<float> centroids;

    /// stats at every iteration of clustering
    std::vector<ClusteringIterationStats> iteration_stats;

    ProgressiveDimClustering(int d, int k);
    ProgressiveDimClustering(
            int d,
            int k,
            const ProgressiveDimClusteringParameters& cp);

    void train(idx_t n, const float* x, ProgressiveDimIndexFactory& factory);

    virtual ~ProgressiveDimClustering() {}
};

/** simplified interface
 *
 * @param d dimension of the data
 * @param n nb of training vectors
 * @param k nb of output centroids
 * @param x training set (size n * d)
 * @param centroids output centroids (size k * d)
 * @return final quantization error
 */
float kmeans_clustering(
        size_t d,
        size_t n,
        size_t k,
        const float* x,
        float* centroids);

} // namespace faiss

#endif
