#pragma once

#include "distance.h"
#include <random>
#include <cstring>
#include <common.h>

namespace orangedb {
    template<class T>
    using decode_func_t = std::function<float(T, int)>;

    // Perform 1-NN search on the given data in parallel using OpenMP
    class IndexOneNN {
    public:
        explicit IndexOneNN(DelegateDC<float> *dc, int dim, int numEntries, float lambda = 0)
            : dc(dc), dim(dim),
              numEntries(numEntries), lambda(lambda) {
        };

        inline void search(int n, const float *queries, double *distances, int32_t *resultIds);

        inline void search(int n, const float *queries, int32_t *resultIds);

        void knn(int k, const float *queries, double *distances, vector_idx_t *resultIds);

        void knnFiltered(int k, const float *query, double *distance, vector_idx_t *resultIds,
                         const uint8_t *filteredMask);

    private:
        DelegateDC<float> *dc;
        int dim;
        int numEntries;
        float lambda;
    };

    // Goals:
    // 1. Implement a clustering algorithm which is concurrent and takes vector in batches. Reduce Disk I/O & Memory usage.
    // 2. TODO: Works good enough with few iterations. One idea is to use k-means++ (Divide & conquer) on sample set for initialization.
    // 3. TODO: Balance constraints in terms of cluster sizes. Ideally each cluster has similar size. In reality each cluster should not be too big or too small.
    // Non Goals:
    // 1. We only need few clusters. So, we don't have to optimize for large number of clusters.
    // 2. No concept of weights for vectors.
    template<typename T>
    class Clustering {
    public:
        Clustering(int dim, int dataDim, int numCentroids, int nIter, int minCentroidSize, int maxCentroidSize,
                   DelegateDC<T> *dc = nullptr,
                   decode_func_t<T> decodeFunc = [](float a, int d) {return a;}, float lambda = 0);

        void initCentroids(const T *data, int n);

        void train(T *data, int n);

        void assignCentroids(const T *data, int n, int32_t *assign);

        void assignCentroids(const T *data, int n, double *dist, int32_t *assign);

        int getNumCentroids() {
            return numCentroids;
        }

    private:
        void computeCentroids(int n, const T *data, const int *assign, int *hist, float *newCentroids);

        void splitClusters(int *hist, float *newCentroids);

        int sampleData(int n, T *data, T **sampleData);

    public:
        std::vector<float> centroids; // centroids (k * d)

    private:
        int dim; // dimension
        int dataDim; // dimension of the data (if different from centroids)
        int numCentroids; // number of centroids
        int nIter; // number of iterations
        int minCentroidSize; // minimum size of a centroid. This is used to sample the training set.
        int maxCentroidSize; // maximum size of a centroid. This is used to sample the training set.
        float lambda; // regularization parameter
        bool debugMode = false;
        std::unique_ptr<DelegateDC<T>> tempDC = nullptr; // temporary distance computer for training
        DelegateDC<T> *dc; // distance type to use for clustering
        decode_func_t<T> decodeFunc;

        int seed = 1234; // seed for random number generator
    };
}
