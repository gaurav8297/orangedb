#pragma once

#include <unistd.h>
#include <vector>
#include <clustering.h>
#include <hnsw.h>
#include <faiss/Clustering.h>


namespace orangedb {

    struct ReclusteringIndexConfig {
        // The number of centroids
        int numCentroids = 10;
        // The number of iterations
        int nIter = 25;
        // The minimum size of a centroid
        int minCentroidSize;
        // The maximum size of a centroid
        int maxCentroidSize;
        // Lambda parameter for clustering
        float lambda = 0;
        // The distance threshold for searching in the centroids
        float searchThreshold = 0.4;
        // Distance Method
        DistanceType distanceType = L2;
        // number of centroids to recluster
        int numReclusterCentroids = 10;

        ReclusteringIndexConfig(int numCentroids, int nIter, int minCentroidSize, int maxCentroidSize, float lambda,
                                float searchThreshold, DistanceType distanceType, int numReclusterCentroids)
            : numCentroids(numCentroids), nIter(nIter), minCentroidSize(minCentroidSize),
              maxCentroidSize(maxCentroidSize), lambda(lambda), searchThreshold(searchThreshold),
              distanceType(distanceType), numReclusterCentroids(numReclusterCentroids)
        {}

    };

    class ReclusteringIndex {
    public:
        explicit ReclusteringIndex(int dim, const ReclusteringIndexConfig &config, RandomGenerator* rg);

        void insert(float *data, size_t n);

        void performReclustering();

        void printStats();

        void search(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results, int nProbes);

    private:
        void appendCentroids(const float *centroids, size_t n);

        std::unique_ptr<DistanceComputer> getDistanceComputer(const float *data, int n) const {
            if (config.distanceType == L2) {
                return std::make_unique<L2DistanceComputer>(data, dim, n);
            }
            return std::make_unique<CosineDistanceComputer>(data, dim, n);
        }

    private:
        int dim;
        const ReclusteringIndexConfig &config;
        size_t size;
        RandomGenerator *rg;
        std::vector<float> centroids;
        std::vector<std::vector<float>> clusters;
        std::vector<std::vector<vector_idx_t>> vectorIds;
        std::vector<int> reclusteringCount;
    };
}
