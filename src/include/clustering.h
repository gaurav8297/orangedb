#pragma once

#include "distance.h"
#include <random>

namespace orangedb {
    // Perform 1-NN search on the given data in parallel using OpenMP
    class IndexOneNN {
    public:
        struct Node {
            explicit Node(storage_idx_t id, float dist) : id(id), dist(dist) {}

            storage_idx_t id;
            float dist;

            bool operator<(const Node &other) const {
                return dist > other.dist;
            }
        };

        explicit IndexOneNN(DistanceComputer *dc, int dim, int numEntries)
                : dc(dc), dim(dim),
                  numEntries(numEntries) {};

        inline void search(int n, const float *queries, float *distances, int32_t *resultIds);

        void knn(int k, const float *queries, float *distances, int *resultIds);

    private:
        DistanceComputer *dc;
        int dim;
        int numEntries;
    };

    // Goals:
    // 1. Implement a clustering algorithm which is concurrent and takes vector in batches. Reduce Disk I/O & Memory usage.
    // 2. TODO: Works good enough with few iterations. One idea is to use k-means++ (Divide & conquer) on sample set for initialization.
    // 3. TODO: Balance constraints in terms of cluster sizes. Ideally each cluster has similar size. In reality each cluster should not be too big or too small.
    // Non Goals:
    // 1. We only need few clusters. So, we don't have to optimize for large number of clusters.
    // 2. No concept of weights for vectors.
    class Clustering {
    public:
        Clustering(int dim, int numCentroids, int nIter, int minCentroidSize, int maxCentroidSize);

        void initCentroids(int n, const float *data);

        void train(int n, const float *data);

        void assignCentroids(const float *data, int n, int32_t *assign);

        inline int getCentroid(int i, float *centroid) {
            assert(i < numCentroids);
            centroid = new float[dim];
            memcpy(centroid, centroids.data() + i * dim, dim * sizeof(float));
            return dim;
        }

    private:
        inline void computeCentroids(int n, const float *data, const int *assign, int *hist, float *newCentroids);

        inline void splitClusters(int *hist, float *newCentroids);

        inline int sampleData(int n, const float *data, float **sampleData);

        inline void randomPerm(int n, int *perm, int nPerm, int64_t seed);

        inline int randomInt(std::mt19937 &mt, int maxVal) {
            return mt() % maxVal;
        }

    public:
        std::vector<float> centroids; // centroids (k * d)

    private:
        int dim; // dimension
        int numCentroids; // number of centroids
        int nIter; // number of iterations
        int minCentroidSize; // minimum size of a centroid. This is used to sample the training set.
        int maxCentroidSize; // maximum size of a centroid. This is used to sample the training set.

        int seed = 1234; // seed for random number generator
    };
}
