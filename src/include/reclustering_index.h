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
        // Quantization train percentage
        float quantizationTrainPercentage = 0.1;

        explicit ReclusteringIndexConfig() = default;

        explicit ReclusteringIndexConfig(const int nIter, const int megaCentroidSize, const int miniCentroidSize,
                                         const int newMiniCentroidSize, const float lambda, const float searchThreshold,
                                         const DistanceType distanceType, const int numMegaReclusterCentroids,
                                         const int numNewMiniReclusterCentroids, const float quantizationTrainPercentage = 0.1)
            : nIter(nIter), megaCentroidSize(megaCentroidSize), miniCentroidSize(miniCentroidSize),
              newMiniCentroidSize(newMiniCentroidSize), lambda(lambda), searchThreshold(searchThreshold),
              distanceType(distanceType), numMegaReclusterCentroids(numMegaReclusterCentroids),
              numNewMiniReclusterCentroids(numNewMiniReclusterCentroids), quantizationTrainPercentage(quantizationTrainPercentage) {
        }
    };

    struct SubCells {
        std::vector<float> centroids;  // The centroids of the sub-cells
        std::vector<std::pair<int, int>> start_end_idxes;   // Start and end indices of the vectors in the sub-cell
    };

    class ReclusteringIndex {
    public:
        explicit ReclusteringIndex(int dim, ReclusteringIndexConfig config, RandomGenerator* rg);

        explicit ReclusteringIndex(const std::string &file_path, RandomGenerator* rg);

        void insert(float *data, size_t n);

        void naiveInsert(float *data, size_t n);

        void trainQuant(float *data, size_t n);

        void naiveInsertQuant(float *data, size_t n);

        void mergeNewMiniCentroids();

        void reclusterMegaCentroids(int n);

        void recluster(int n, bool fast = false);

        void reclusterFast(int n = INT_MAX);

        void reclusterFastQuant();

        void reclusterFull(int n);

        void reclusterBasedOnScore(int n);

        void reclusterAllMegaCentroids();

        void reclusterAllMiniCentroidsQuant();

        void storeScoreForMegaClusters(int n = INT_MAX);

        void computeAllSubCells(int avgSubCellSize);

        void quantizeVectors();

        void printStats();

        void flush_to_disk(const std::string &file_path) const;

        void search(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results,
                    int nMegaProbes, int nMiniProbes, ReclusteringIndexStats &stats);

        void searchWithBadClusters(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results,
                    int nMegaProbes, int nMiniProbes, int nMiniProbesForBadClusters, ReclusteringIndexStats &stats, bool skipBadClusters = false);

        void searchMegaCluster(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results,
            int megaClusterId, int nMiniProbes, ReclusteringIndexStats &stats);

        void searchQuantized(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results,
            int nMegaProbes, int nMiniProbes, ReclusteringIndexStats &stats);

        void checkDuplicateMiniClusters();

    private:
        void computeMiniClusterSubcells(int miniClusterId, int avgSubCellSize);

        vector_idx_t getWorstMegaCentroid();

        std::vector<vector_idx_t> reclusterFullMegaCentroids(std::vector<vector_idx_t> megaClusterIds);

        void quantizeVectors(float *data, int n, std::vector<uint8_t> &quantizedVectors);

        void reclusterFastMegaCentroids(std::vector<vector_idx_t> megaClusterIds);

        void reclusterInternalMegaCentroid(vector_idx_t megaClusterId);

        void reclusterInternalMegaCentroidQuant(vector_idx_t megaClusterId);

        void mergeNewMiniCentroidsBatch(float *megaCentroid, std::vector<vector_idx_t> newMiniCentroidBatch);

        void mergeNewMiniCentroidsInit();

        void reclusterOnlyMegaCentroids(std::vector<vector_idx_t> megaCentroids);

        void reclusterOnlyMegaCentroidsQuant(std::vector<vector_idx_t> megaCentroids);

        void resetInputBuffer();

        void clusterData(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                         std::vector<float>& centroids, std::vector<std::vector<float>>& clusters,
                         std::vector<std::vector<vector_idx_t>> &clusterVectorIds);

        void clusterData(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                 std::vector<float>& centroids, std::vector<std::vector<vector_idx_t>> &clusterVectorIds);

        // Quantized clustering methods
        void clusterDataQuant(uint8_t *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                 std::vector<float>& centroids, std::vector<std::vector<uint8_t>>& clusters,
                 std::vector<std::vector<vector_idx_t>> &clusterVectorIds);

        void clusterDataQuant(uint8_t *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                 std::vector<float>& centroids, std::vector<std::vector<vector_idx_t>> &clusterVectorIds);

        void clusterDataWithFaiss(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                  std::vector<float> &centroids, std::vector<std::vector<float> > &clusters,
                                  std::vector<std::vector<vector_idx_t> > &clusterVectorIds);

        void clusterDataWithFaiss(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                  std::vector<float> &centroids,
                                  std::vector<std::vector<vector_idx_t> > &clusterVectorIds);

        // Generic clustering method
        template <typename T>
        void clusterData_(T *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                 std::vector<float>& centroids, std::vector<std::vector<T>>& clusters,
                 std::vector<std::vector<vector_idx_t>> &clusterVectorIds,
                 DelegateDC<T> *dc, int dataDim, decode_func_t<T> decodeFunc);

        template <typename T>
        void clusterData_(T *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                          std::vector<float> &centroids, std::vector<std::vector<vector_idx_t> > &clusterVectorIds,
                          DelegateDC<T> *dc, int dataDim, decode_func_t<T> decodeFunc);

        template <typename T>
        void calcMeanCentroid(T *data, vector_idx_t *vectorIds, int n, int dataDim, std::vector<float> &centroids,
                          std::vector<std::vector<vector_idx_t> > &clusterVectorIds);

        std::vector<vector_idx_t> appendOrMergeCentroids(
            std::vector<vector_idx_t> oldMegaCentroids,
            std::vector<float> &newMegaCentroids,
            std::vector<std::vector<vector_idx_t>> &miniClusterIds,
            std::vector<float> &newMiniCentroids,
            std::vector<std::vector<float>> &newMiniClusters,
            std::vector<std::vector<vector_idx_t>> &newMiniClusterVectorIds);

        // Quantization methods!
        std::vector<vector_idx_t> appendOrMergeCentroidsQuant(
            std::vector<vector_idx_t> oldMegaCentroids,
            std::vector<float> &newMegaCentroids,
            std::vector<std::vector<vector_idx_t> > &miniClusterIds,
            std::vector<uint8_t> &newMiniCentroids,
            std::vector<std::vector<uint8_t> > &newMiniClusters,
            std::vector<std::vector<vector_idx_t> > &newMiniClusterVectorIds);

        std::vector<vector_idx_t> appendOrMergeMegaCentroids(
            std::vector<vector_idx_t> oldMegaCentroidIds,
            std::vector<float> &newMegaCentroids,
            std::vector<std::vector<vector_idx_t>> &newMiniClusterIds);

        inline void updateTotalDataWrittenBySystem(const std::vector<std::vector<vector_idx_t>> &newMiniClusterIds,
                                                   const std::vector<std::vector<vector_idx_t>> &newMiniClusterVectorIds);

        inline void updateTotalDataWrittenByUser(const size_t n);

        void findKClosestMegaCentroids(const float *query, int k, std::vector<vector_idx_t> &ids, ReclusteringIndexStats &stats, bool onlyGoodClusters = false);

        void findKClosestMiniCentroids(const float *query, int k, std::vector<vector_idx_t> megaCentroids, std::vector<vector_idx_t> &ids, ReclusteringIndexStats &stats);

        void findKClosestVectors(const float *query, int k, std::vector<vector_idx_t> miniCentroids,
                                 std::priority_queue<NodeDistCloser> &results, ReclusteringIndexStats &stats);

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

        std::unique_ptr<DelegateDC<float>> getDistanceComputer(const float *data, int n) const {
            return createDistanceComputer(data, dim, n, config.distanceType);
        }

        std::unique_ptr<DelegateDC<uint8_t> > getQuantizedDistanceComputer(
            const uint8_t *data, int n, bool symmetric = false) const {
            return createQuantizedDistanceComputer(data, dim, n, config.distanceType, quantizer.get(), symmetric);
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

        // Minicluster subcells
        std::vector<SubCells> miniClusterSubCells;

        // New Mini centroids (Buffering space)
        std::vector<float> newMiniCentroids;
        std::vector<std::vector<float>> newMiniClusters;
        std::vector<std::vector<vector_idx_t>> newMiniClusterVectorIds;

        // Quantized data
        std::unique_ptr<SQ8Bit> quantizer;
        // std::vector<float> quantizedMegaCentroids;
        std::vector<uint8_t> quantizedMiniCentroids;
        std::vector<std::vector<uint8_t>> quantizedMiniClusters;

        // Stats
        ReclusteringIndexStats stats;
    };
}
