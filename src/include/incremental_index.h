#pragma once

#pragma once

#include <unistd.h>
#include <vector>
#include <clustering.h>
#include <hnsw.h>
#include <faiss/Clustering.h>


namespace orangedb {
    struct IncrementalIndexStats {
        uint64_t numDistanceComp = 0;

        void printStats() {
            printf("Number of distance computations: %lu\n", numDistanceComp);
        }
    };

    struct IncrementalIndexConfig {
        // The number of centroids
        int numCentroids = 10;
        // Average size of the micro centroids
        int avgMicroCentroidSize = 100;
        // The number of iterations
        int nIter = 25;
        // Lambda parameter for clustering
        float lambda = 0;
        // The distance threshold for searching in the centroids
        float searchThreshold = 0.4;
        // Distance Method
        DistanceType distanceType = L2;

        explicit IncrementalIndexConfig() {}

        explicit IncrementalIndexConfig(int numCentroids, int nIter, int avgMicroCentroidSize,
                        float lambda, float searchThreshold, DistanceType distanceType)
            : numCentroids(numCentroids), nIter(nIter), lambda(lambda), searchThreshold(searchThreshold),
              avgMicroCentroidSize(avgMicroCentroidSize), distanceType(distanceType) {}
    };

    class IncrementalIndex {
    public:
        explicit IncrementalIndex(int dim, IncrementalIndexConfig config, RandomGenerator* rg);

        explicit IncrementalIndex(const std::string &file_path, RandomGenerator* rg);

        void insert(float *data, size_t n);

        void printStats();

        void splitMega();

        int splitMicro();

        void flush_to_disk(const std::string &file_path) const;

        void search(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results, int nMegaProbes,
            int nMicroProbes, IncrementalIndexStats& stats);

        double computeSilhouetteMetricOnMicroCentroids();

    private:
        void insertFirstTime(float *data, size_t n);

        void appendCentroids(const float *centroids, size_t n);

        void load_from_disk(const std::string &file_path);

        void splitMegaCluster(int megaClusterId);

        void storeMegaCluster(int oldMegaClusterId, const float* newMegaCentroid, Clustering* microClustering,
                    const float *data, const vector_idx_t *vectorIds, size_t n);

        void appendMegaCluster(const float* newMegaCentroid, Clustering* microClustering,
            const float *data, const vector_idx_t *vectorIds, size_t n);

        void splitMicroCluster(int microClusterId);

        void moveVectorsFromClosestCentroids(int centroidId);

        void findClosestMicroCluster(const float *data, int n, double *dists, int32_t *assign, int skipMicroCentroid);

        size_t getMegaClusterSize(int megaCentroidId);

        void assignMegaCentroids(const float *data, int n, int32_t *assign);

        void assignMicroCentroids(const float *data, int n, const int32_t *megaAssign, int32_t *microAssign);

        inline int getMinCentroidSize(int numVectors, int numCentroids) const {
            return (numVectors / numCentroids) * 0.5;
        }

        inline int getMaxCentroidSize(int numVectors, int numCentroids) const {
            return (numVectors / numCentroids) * 1.2;
        }

        std::unique_ptr<DelegateDC<float>> getDistanceComputer(const float *data, int n) const {
            return createDistanceComputer(data, dim, n, config.distanceType);
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

