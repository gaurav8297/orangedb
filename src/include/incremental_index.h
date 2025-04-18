#pragma once

#pragma once

#include <unistd.h>
#include <vector>
#include <clustering.h>
#include <hnsw.h>
#include <faiss/Clustering.h>


namespace orangedb {
    struct IncrementalIndexConfig {
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
        // Mega split threshold
        long maxMegaClusterSize = 1000000;

        explicit IncrementalIndexConfig() {}

        explicit IncrementalIndexConfig(int numCentroids, int nIter, int minCentroidSize, int maxCentroidSize,
                        float lambda, float searchThreshold, DistanceType distanceType, long maxMegaClusterSize)
            : numCentroids(numCentroids), nIter(nIter), minCentroidSize(minCentroidSize),
              maxCentroidSize(maxCentroidSize), lambda(lambda), searchThreshold(searchThreshold),
              distanceType(distanceType), maxMegaClusterSize(maxMegaClusterSize) {}

    };

    class IncrementalIndex {
    public:
        explicit IncrementalIndex(int dim, IncrementalIndexConfig config, RandomGenerator* rg);

        explicit IncrementalIndex(const std::string &file_path, RandomGenerator* rg);

        void insert(float *data, size_t n);

        void fixIndex();

        void printStats();

        void flush_to_disk(const std::string &file_path) const;

        void search(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results, int nProbes);

    private:
        void insertFirstTime(float *data, size_t n);

        void appendCentroids(const float *centroids, size_t n);

        void load_from_disk(const std::string &file_path);

        void split();

        void splitMegaCluster(int megaClusterId);

        void storeMegaCluster(int oldMegaClusterId, const float* newMegaCentroid, Clustering* microClustering,
                    const float *data, const vector_idx_t *vectorIds, size_t n);

        void appendMegaCluster(const float* newMegaCentroid, Clustering* microClustering,
            const float *data, const vector_idx_t *vectorIds, size_t n);

        size_t getMegaClusterSize(int megaCentroidId);

        void assignMegaCentroids(const float *data, int n, int32_t *assign);

        void assignMicroCentroids(const float *data, int n, const int32_t *megaAssign, int32_t *microAssign);

        std::unique_ptr<DistanceComputer> getDistanceComputer(const float *data, int n) const {
            if (config.distanceType == L2) {
                return std::make_unique<L2DistanceComputer>(data, dim, n);
            }
            return std::make_unique<CosineDistanceComputer>(data, dim, n);
        }

    private:
        int dim;
        IncrementalIndexConfig config;
        size_t size;
        RandomGenerator *rg;

        // Actual data
        std::vector<float> megaCentroids;
        std::vector<std::vector<vector_idx_t>> megaCentroidAssignment;
        std::vector<float> microCentroids;
        std::vector<std::vector<float>> clusters;
        std::vector<std::vector<vector_idx_t>> vectorIds;
    };
}

