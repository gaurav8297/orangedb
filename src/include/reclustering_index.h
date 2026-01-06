#pragma once

#include <unistd.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <clustering.h>
#include <hnsw.h>
#include <faiss/Clustering.h>


namespace orangedb {

    enum CLUSTERING_MODE {
        HARD_LIMIT,    // 0
        REBALANCE_CENTROIDS, // 1
        REBALANCE_VECTORS, // 2
    };

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
        // limit to max cluster size
        uint64_t hardClusterSizeLimit = 0;
        // Sampling percentage for kmeans clustering
        float kmeansSamplingRatio = 1.0;
        // scoreChangeThreshold
        float scoreChangeThreshold = 0.2;
        // centroidChangeThreshold
        float centroidChangeThreshold = 0.6;
        // Power avg coefficient
        float powerAvgCoefficient = 4.0;
        // Work elements for averaging!
        int workElementsForAveraging = 10;
        // Overlapping score threshold
        float overlappingScoreThreshold = 0.1;

        explicit ReclusteringIndexConfig() = default;

        explicit ReclusteringIndexConfig(const int nIter, const int megaCentroidSize, const int miniCentroidSize,
                                         const int newMiniCentroidSize, const float lambda, const float searchThreshold,
                                         const DistanceType distanceType, const int numMegaReclusterCentroids,
                                         const int numNewMiniReclusterCentroids,
                                         const float quantizationTrainPercentage = 0.1,
                                         const uint64_t hardClusterSizeLimit = 0,
                                         const float kmeansSamplingRatio = 1.0,
                                         const float scoreChangeThreshold = 0.2,
                                         const float centroidChangeThreshold = 0.6,
                                         const float overlappingScoreThreshold = 0.1)
            : nIter(nIter), megaCentroidSize(megaCentroidSize), miniCentroidSize(miniCentroidSize),
              newMiniCentroidSize(newMiniCentroidSize), lambda(lambda), searchThreshold(searchThreshold),
              distanceType(distanceType), numMegaReclusterCentroids(numMegaReclusterCentroids),
              numNewMiniReclusterCentroids(numNewMiniReclusterCentroids),
              quantizationTrainPercentage(quantizationTrainPercentage), hardClusterSizeLimit(hardClusterSizeLimit),
              kmeansSamplingRatio(kmeansSamplingRatio), scoreChangeThreshold(scoreChangeThreshold),
              centroidChangeThreshold(centroidChangeThreshold), overlappingScoreThreshold(overlappingScoreThreshold) {
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

        void simpleInsertWithoutClustering(float *data, size_t n);

        void naiveInsert(float *data, size_t n, bool use_rebalancing = false);

        void trainQuant(float *data, size_t n);

        void naiveInsertQuant(float *data, size_t n);

        void mergeNewMiniCentroids();

        void reclusterMegaCentroids(int n);

        void recluster(int n, bool fast = false);

        void reclusterFast(int n = INT_MAX);

        void getMegaClusterIds(std::vector<vector_idx_t> &megaClusterIds);

        void reclusterFastQuant();

        void reclusterFull(int n);

        void reclusterBasedOnScore(int n);

        void reclusterBasedOnMSEScore();

        void reclusterAllMegaCentroids(int n = INT_MAX);

        void reclusterAllMiniCentroidsQuant();

        void fixBoundaryMiniCentroids(int n = INT_MAX);

        void fixBoundaryMiniCentroidsV2(int n = INT_MAX);

        void storeScoreForMegaClusters(int n = INT_MAX);

        void storeMSEScoreForMegaClusters(int n = INT_MAX);

        void calculateOverlapScoreForL2(int megaCentroidId);

        void calculateOverlapScoreForAngular(int megaCentroidId);

        double calculateRealOverlapScore(vector_idx_t miniCentroidId, std::vector<vector_idx_t> &closestMiniIds);

        double calculateRealOverlapScoreForAngular(vector_idx_t miniCentroidId, std::vector<vector_idx_t> &closestMiniIds);

        void computeOverlapScores();

        void getOverlapScores(const double **_overlapScores, size_t &n) const {
            *_overlapScores = overlapScores.data();
            n = overlapScores.size();
        }

        void getRealOverlapScores(const double **_overlapScores, size_t &n) const {
            *_overlapScores = avgRealOverlapScores.data();
            n = avgRealOverlapScores.size();
        }

        void getMegaCentroids(const float **_centroids, size_t &n) const {
            *_centroids = megaCentroids.data();
            n = megaCentroids.size() / dim;
        }

        void getMiniCentroids(const float **_centroids, size_t &n) const {
            *_centroids = miniCentroids.data();
            n = miniCentroids.size() / dim;
        }

        void getMegaMiniCentroids(std::vector<std::vector<vector_idx_t>> *_megaMiniCentroidIds) const {
            // Copy the mega to mini centroid mapping
            *_megaMiniCentroidIds = megaMiniCentroidIds;
        }
 
        void getMiniClusterVectorIds(std::vector<std::vector<vector_idx_t>> *_miniClusterVectorIds) const {
            *_miniClusterVectorIds = miniClusterVectorIds;
        }
        
        int getDim() const {
            return dim;
        }      

        void saveOldScoreForMegaClusters();

        std::vector<vector_idx_t> getMegaCentroidsToRecluster() const;

        void computeAllSubCells(int avgSubCellSize);

        void quantizeVectors();

        void printChangeClusterStats();

        void printStats();

        void flush_to_disk(const std::string &file_path) const;

        void search(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results,
                    int nMegaProbes, int nMiniProbes, ReclusteringIndexStats &stats);

        void searchWithBadClusters(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results,
                    int nMegaProbes, int nMiniProbes, int nMiniProbesForBadClusters, ReclusteringIndexStats &stats, bool searchEachBadCluster = false);

        void searchMegaCluster(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results,
            int megaClusterId, int nMiniProbes, ReclusteringIndexStats &stats);

        void searchQuantized(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results,
            int nMegaProbes, int nMiniProbes, ReclusteringIndexStats &stats);

        void checkDuplicateMiniClusters();

        void reclusterInternalMegaCentroid(vector_idx_t megaClusterId);

        const float* getVectorData(vector_idx_t vectorId) const;

        // Input query vector and ground truth vectorIds, find the corresponding mini and mega centroids and print
        // the distance between the two centroids before and after reclustering and how the rank of the clusters changed.
        void analyzeQueryClusterChanges(
            const float *query,
            const vector_idx_t *groundTruthVectorIds,
            int nGroundTruth,
            bool onlyStoreChanges = true);

        // Get cluster assignments, vector data, and centroid vectors for given vector IDs
        // Returns: map from vector_idx_t to (miniClusterId, megaClusterId, vector_data, miniCentroid, megaCentroid)
        void getVectorClusterAssignments(
            const float *query,
            const vector_idx_t *vectorIds,
            int n,
            std::unordered_map<vector_idx_t, std::tuple<vector_idx_t, vector_idx_t, vector_idx_t, std::vector<float>,
                std::vector<float> > > &results) const;

        // Find the rank of mega and mini clusters based on distance to query
        // Returns: tuple of (megaRank, miniRankInMega, miniRankOverall) where 0 = closest, 1 = second closest, etc.
        // megaRank: rank among all mega clusters
        // miniRankInMega: rank within the specific mega cluster
        // miniRankOverall: rank among all mini clusters globally
        // Returns (-1, -1, -1) if clusters are not found
        std::tuple<vector_idx_t, vector_idx_t, vector_idx_t> getClusterRanks(
            const float *query, vector_idx_t megaId, vector_idx_t miniId) const;

    private:
        // Helper struct to hold clustering results
        struct ClusteringResult {
            std::vector<int64_t> assignments;
            std::vector<int> histogram;
            std::vector<float> centroids;
            int numClusters;
        };

        // Core clustering logic extracted to avoid duplication
        ClusteringResult performCoreClustering(float *data, int n, int avgClusterSize, 
                                               int nClusters = -1, bool verbose = false);

        void fixBoundaryMiniCentroid(int miniCentroidId, std::unordered_set<vector_idx_t> *alreadyFixed = nullptr);

        void fixBoundaryMiniCentroidV2(int miniCentroidId);

        bool isAtBoundary(vector_idx_t miniClusterId);

        void computeMiniClusterSubcells(int miniClusterId, int avgSubCellSize);

        vector_idx_t getWorstMegaCentroid();

        std::vector<vector_idx_t> reclusterFullMegaCentroids(std::vector<vector_idx_t> megaClusterIds);

        void quantizeVectors(float *data, int n, std::vector<uint8_t> &quantizedVectors);

        void reclusterFastMegaCentroids(std::vector<vector_idx_t> megaClusterIds);

        void reclusterInternalMegaCentroidQuant(vector_idx_t megaClusterId);

        void mergeNewMiniCentroidsBatch(float *megaCentroid, std::vector<vector_idx_t> newMiniCentroidBatch);

        void mergeNewMiniCentroidsInit();

        void reclusterOnlyMegaCentroids(std::vector<vector_idx_t> megaCentroids);

        void reclusterOnlyMegaCentroidsQuant(std::vector<vector_idx_t> megaCentroids);

        void resetInputBuffer();

        void clusterData(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                         std::vector<float>& centroids, std::vector<std::vector<float>>& clusters,
                         std::vector<std::vector<vector_idx_t>> &clusterVectorIds, bool use_rebalancing = false);

        void clusterData(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                 std::vector<float>& centroids, std::vector<std::vector<vector_idx_t>> &clusterVectorIds, 
                 int nClusters = -1, bool use_rebalancing = false);

        // Quantized clustering methods
        void clusterDataQuant(uint8_t *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                 std::vector<float>& centroids, std::vector<std::vector<uint8_t>>& clusters,
                 std::vector<std::vector<vector_idx_t>> &clusterVectorIds);

        void clusterDataQuant(uint8_t *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                 std::vector<float>& centroids, std::vector<std::vector<vector_idx_t>> &clusterVectorIds);

        float findAppropriateLambda(const float *data,
                                    size_t num_rows,
                                    int dim,
                                    int num_clusters,
                                    size_t sample_size = 10000);

        void clusterDataWithFaiss(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                  std::vector<float> &centroids,
                                  std::vector<std::vector<float> > *clusters,
                                  std::vector<std::vector<vector_idx_t> > &clusterVectorIds,
                                  int nClusters = -1);

        void clusterDataWithRebalancing(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                    std::vector<float> &centroids, std::vector<std::vector<float> > *clusters,
                                    std::vector<std::vector<vector_idx_t> > &clusterVectorIds);

        void clusterDataWithCentoidRebalancing(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                    std::vector<float> &centroids, std::vector<std::vector<float> > *clusters,
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
            std::vector<std::vector<vector_idx_t>> &newMiniClusterVectorIds,
            std::vector<vector_idx_t> existingOldMiniClusterIds = std::vector<vector_idx_t>());

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

        void findKClosestMegaCentroids(const float *query, int k, std::vector<vector_idx_t> &ids, std::vector<float> &distances);

        void findKClosestMiniCentroids(const float *query, int k, std::vector<vector_idx_t> megaCentroids, std::vector<vector_idx_t> &ids, ReclusteringIndexStats &stats);

        void findKClosestVectors(const float *query, int k, std::vector<vector_idx_t> miniCentroids,
                                 std::priority_queue<NodeDistCloser> &results, ReclusteringIndexStats &stats);

        double calcScoreForMegaCluster(int megaClusterId);

        double calcScoreForMiniCluster(int miniClusterId, std::unordered_set<vector_idx_t> *closerL1s = nullptr);

        double calcMSEScoreForMegaCluster(int megaClusterId);

        double calcMSEScoreForMiniCluster(int miniClusterId);

        inline double computeAvg(const std::vector<double> &values) const {
            if (values.empty()) {
                return 0.0;
            }
            double sum = 0.0;
            for (double val : values) {
                sum += val;
            }
            return sum / values.size();
        }

        inline double mergeOverlapScores(const std::vector<double> &scores) const {
            if (scores.empty()) {
                return 0.0;
            }
            // double finalScore = 1.0;
            // for (double score : scores) {
            //     finalScore *= (score * 10.0f);
            // }
            // return finalScore;
            double finalScore = std::numeric_limits<double>::max();
            for (double score : scores) {
                finalScore = std::min(finalScore, score);
            }
            return finalScore;
        }

        inline double
        computePowerAvgOnWorstElement(const std::vector<double> &values, bool biggerIsBetter = true,
                                      float powerAvgCoefficient = -1) const {
            if (values.empty()) {
                return 0.0;
            }
            if (powerAvgCoefficient <= 0) {
                // Use provided coefficient
                powerAvgCoefficient = config.powerAvgCoefficient;
            }

            // Take the k worst elements
            int k = std::min(int(values.size()), config.workElementsForAveraging);
            std::vector<double> worstElements;
            worstElements.reserve(k);
            
            // Create indices and sort to find worst elements
            std::vector<size_t> indices(values.size());
            std::iota(indices.begin(), indices.end(), 0);
            
            if (biggerIsBetter) {
                // Worst = smallest values, so sort ascending
                std::sort(indices.begin(), indices.end(),
                    [&values](size_t a, size_t b) { return values[a] < values[b]; });
            } else {
                // Worst = largest values, so sort descending
                std::sort(indices.begin(), indices.end(),
                    [&values](size_t a, size_t b) { return values[a] > values[b]; });
            }
            
            // Extract the k worst elements
            for (int i = 0; i < k; i++) {
                worstElements.push_back(values[indices[i]]);
            }
            
            // Compute power average: (1/n * sum(x_i^p))^(1/p)
            double p = biggerIsBetter ? -1 * powerAvgCoefficient : powerAvgCoefficient;
            
            double sum = 0.0;
            for (double val : worstElements) {
                // Handle negative values: take absolute value and preserve sign
                double sign = (val >= 0.0) ? 1.0 : -1.0;
                double absVal = std::abs(val);
                if (absVal > 0.0) {
                    sum += sign * std::pow(absVal, p);
                }
            }
            
            double avg = sum / worstElements.size();
            if (avg == 0.0) {
                return 0.0;
            }
            
            // Handle negative average for odd powers
            double sign = (avg >= 0.0) ? 1.0 : -1.0;
            double absAvg = std::abs(avg);
            return sign * std::pow(absAvg, 1.0 / p);
        }

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
            // Adjust based on sample size
            auto sampleSize = std::min(std::max(int(numVectors * config.kmeansSamplingRatio), 50000), numVectors);
            return (sampleSize / numCentroids) * 0.4;
        }

        inline int getMaxCentroidSize(int numVectors, int numCentroids) const {
            // 120% of the average size such all vecs are used during reclustering
            auto sampleSize = std::min(std::max(int(numVectors * config.kmeansSamplingRatio), 50000), numVectors);
            return (sampleSize / numCentroids) + 1;
        }

        void load_from_disk(const std::string &file_path);

        std::unique_ptr<DelegateDC<float>> getDistanceComputer(const float *data, int n, DistanceType type = INVALID) const {
            if (type == INVALID) {
                type = config.distanceType;
            }
            return createDistanceComputer(data, dim, n, type);
        }

        std::unique_ptr<DelegateDC<uint8_t> > getQuantizedDistanceComputer(
            const uint8_t *data, int n, bool symmetric = false) const {
            return createQuantizedDistanceComputer(data, dim, n, config.distanceType, quantizer.get(), symmetric);
        }

        void printStatsForTrackId();

    private:
        int dim;
        ReclusteringIndexConfig config;
        size_t size;
        RandomGenerator *rg;

        // Mini and mega centroids
        std::vector<float> megaCentroids;
        std::vector<std::vector<vector_idx_t>> megaMiniCentroidIds;
        std::vector<double> megaClusteringScore;
        std::vector<double> overlapScores;
        std::vector<double> avgRealOverlapScores;
        std::vector<float> miniCentroids;
        std::vector<std::vector<float>> miniClusters;
        std::vector<double> miniClusteringScore;
        std::vector<std::vector<vector_idx_t>> miniClusterVectorIds;

        // Minicluster subcells
        std::vector<SubCells> miniClusterSubCells;

        // old mini centroids
        std::vector<float> oldMegaCentroids;
        std::vector<double> oldMegaClusteringScore;

        // New Mini centroids (Buffering space)
        std::vector<float> newMiniCentroids;
        std::vector<std::vector<float>> newMiniClusters;
        std::vector<std::vector<vector_idx_t>> newMiniClusterVectorIds;

        // Quantized data
        std::unique_ptr<SQ8Bit> quantizer;
        // std::vector<float> quantizedMegaCentroids;

        // Old data
        std::vector<uint8_t> quantizedMiniCentroids;
        std::vector<std::vector<uint8_t>> quantizedMiniClusters;
        std::unordered_map<vector_idx_t, std::tuple<vector_idx_t, vector_idx_t, vector_idx_t, std::vector<float>,
        std::vector<float>>> prevQueryState;

        // Stats
        ReclusteringIndexStats stats;

        // Track micro centroid id
        vector_idx_t nextMiniCentroidId = 6541;
    };
}
