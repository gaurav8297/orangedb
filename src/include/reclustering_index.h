#pragma once

#include <unistd.h>
#include <vector>
#include <clustering.h>
#include <hnsw.h>
#include <faiss/Clustering.h>


namespace orangedb {

    struct ReclusteringIndexStats {
        // The number of distance computations
        uint64_t numDistanceCompForSearch = 0;
        uint64_t totalQueries = 0;

        // The number of distance computations for reclustering
        uint64_t numDistanceCompForRecluster = 0;
        uint64_t totalReclusters = 0;

        // Total data written to disk. This will help us measure write amplification.
        uint64_t totalDataWrittenBySystem = 0;
        uint64_t totalDataWrittenByUser = 0;
    };

    struct ReclusteringIndexConfig {
        // The number of iterations
        int nIter = 40;
        // The minimum size of a mega centroid
        int megaCentroidSize = 1000;
        // The minimum size of a mini centroid
        int miniCentroidSize = 500;
        // New miniCentroidSize
        int newMiniCentroidSize = 100;
        // Lambda parameter for clustering
        float lambda = 0;
        // The distance threshold for searching in the centroids
        float searchThreshold = 0.4;
        // Distance Method
        DistanceType distanceType = L2;
        // number of mega centroids to recluster
        int numMegaReclusterCentroids = 10;
        // number of new mini centroids consider for reclustering
        int numNewMiniReclusterCentroids = 100;
        // number of existing mega centroids to consider for reclustering
        int numExistingMegaReclusterCentroids = 5;

        explicit ReclusteringIndexConfig() = default;

        explicit ReclusteringIndexConfig(const int nIter, const int megaCentroidSize, const int miniCentroidSize,
                                         const int newMiniCentroidSize, const float lambda, const float searchThreshold,
                                         const DistanceType distanceType, const int numMegaReclusterCentroids,
                                         const int numNewMiniReclusterCentroids)
            : nIter(nIter), megaCentroidSize(megaCentroidSize), miniCentroidSize(miniCentroidSize),
              newMiniCentroidSize(newMiniCentroidSize), lambda(lambda), searchThreshold(searchThreshold),
              distanceType(distanceType), numMegaReclusterCentroids(numMegaReclusterCentroids),
              numNewMiniReclusterCentroids(numNewMiniReclusterCentroids) {
        }
    };

    class ReclusteringIndex {
    public:
        explicit ReclusteringIndex(int dim, ReclusteringIndexConfig config, RandomGenerator* rg);

        explicit ReclusteringIndex(const std::string &file_path, RandomGenerator* rg);

        void insert(float *data, size_t n);

        void naiveInsert(float *data, size_t n);

        void mergeNewMiniCentroids();

        void reclusterMegaCentroids(int n);

        void recluster(int n, bool fast = false);

        void reclusterFast();

        void reclusterFull(int n);

        void reclusterAllMegaCentroids();

        void storeScoreForMegaClusters();

        void printStats();

        void flush_to_disk(const std::string &file_path) const;

        void search(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results,
                    int nMegaProbes, int nMicroProbes, ReclusteringIndexStats &stats);

    private:
        vector_idx_t getWorstMegaCentroid();

        void reclusterFullMegaCentroids(std::vector<vector_idx_t> megaClusterIds);

        void reclusterFastMegaCentroids(std::vector<vector_idx_t> megaClusterIds);

        void reclusterInternalMegaCentroid(vector_idx_t megaClusterId);

        void mergeNewMiniCentroidsBatch(float *megaCentroid, std::vector<vector_idx_t> newMiniCentroidBatch);

        void mergeNewMiniCentroidsInit();

        void reclusterOnlyMegaCentroids(std::vector<vector_idx_t> megaCentroids);

        void resetInputBuffer();

        void clusterData(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                         std::vector<float>& centroids, std::vector<std::vector<float>>& clusters,
                         std::vector<std::vector<vector_idx_t>> &clusterVectorIds);

        void clusterData(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                 std::vector<float>& centroids, std::vector<std::vector<vector_idx_t>> &clusterVectorIds);

        void calcMeanCentroid(float *data, vector_idx_t *vectorIds, int n, std::vector<float> &centroids,
                          std::vector<std::vector<vector_idx_t> > &clusterVectorIds);

        std::vector<vector_idx_t> appendOrMergeCentroids(std::vector<vector_idx_t> oldMegaCentroids, std::vector<float> &newMegaCentroids,
                                    std::vector<std::vector<vector_idx_t> > &miniClusterIds,
                                    std::vector<float> &newMiniCentroids,
                                    std::vector<std::vector<float> > &newMiniClusters,
                                    std::vector<std::vector<vector_idx_t> > &newMiniClusterVectorIds);

        std::vector<vector_idx_t> appendOrMergeMegaCentroids(std::vector<vector_idx_t> oldMegaCentroidIds, std::vector<float> &newMegaCentroids,
                                       std::vector<std::vector<vector_idx_t> > &newMiniClusterIds);

        inline void updateTotalDataWrittenBySystem(const std::vector<std::vector<vector_idx_t>> &newMiniClusterIds,
                                                   const std::vector<std::vector<vector_idx_t>> &newMiniClusterVectorIds);

        inline void updateTotalDataWrittenByUser(const size_t n);

        void findKClosestMegaCentroids(const float *query, int k, std::vector<vector_idx_t> &ids);

        double calcScoreForMegaCluster(int megaClusterId);

        double calcScoreForMiniCluster(int miniClusterId);

        inline int getNumCentroids(int numVectors, int avgClusterSize) const {
            double ret =  (double)numVectors / avgClusterSize;
            int val = (int)ret;
            if (ret - val > 0.6) {
                val++;
            }
            return val;
        }

        inline int getMinCentroidSize(int numVectors, int numCentroids) const {
            // 50% of the average size
            return (numVectors / numCentroids) * 0.5;
        }

        inline int getMaxCentroidSize(int numVectors, int numCentroids) const {
            // 120% of the average size such all vecs are used during reclustering
            return (numVectors / numCentroids) * 1.2;
        }

        void load_from_disk(const std::string &file_path);

        std::unique_ptr<DistanceComputer> getDistanceComputer(const float *data, int n) const {
            if (config.distanceType == L2) {
                return std::make_unique<L2DistanceComputer>(data, dim, n);
            }
            return std::make_unique<CosineDistanceComputer>(data, dim, n);
        }

    private:
        int dim;
        ReclusteringIndexConfig config;
        size_t size;
        RandomGenerator *rg;

        // Mini and mega centroids
        std::vector<float> megaCentroids;
        std::vector<std::vector<vector_idx_t>> megaMiniCentroidIds;
        std::vector<double> megaClusteringScore;
        std::vector<float> miniCentroids;
        std::vector<std::vector<float>> miniClusters;
        std::vector<std::vector<vector_idx_t>> miniClusterVectorIds;

        // New Mini centroids (Buffering space)
        std::vector<float> newMiniCentroids;
        std::vector<std::vector<float>> newMiniClusters;
        std::vector<std::vector<vector_idx_t>> newMiniClusterVectorIds;

        // Stats
        ReclusteringIndexStats stats;
    };
}
