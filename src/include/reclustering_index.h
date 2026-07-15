#pragma once

#include <unistd.h>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <clustering.h>
#include <hnsw.h>
#include <faiss/Clustering.h>
#include <faiss/IndexLSH.h>


namespace orangedb {

    struct WrongAssignmentBucketStats {
        double wrong_pct_avg = 0.0;
        double wrong_pct_std = 0.0;
        double wrong_pct_p90 = 0.0;
        double wrong_pct_p99 = 0.0;
        double gap_avg = 0.0;
        double gap_std = 0.0;
        double gap_p90 = 0.0;
        double gap_p99 = 0.0;
    };

    enum class DynamicMiniProbeStrategy {
        DistanceFactor = 0,
        RadiusOverlap = 1,
        PercentGrowth = 2,
    };

    enum class DynamicMiniProbeRadiusKind {
        Rms = 0,
        P95 = 1,
        P75 = 2,
    };

    enum class DynamicMiniProbeOrderingKind {
        RadiusNormalizedHeuristic = 0,
        CentroidDistance = 1,
        L1Coverage = 2,
        L1CapHeight = 3,
    };

    struct DynamicMiniProbeConfig {
        DynamicMiniProbeStrategy strategy = DynamicMiniProbeStrategy::DistanceFactor;
        DynamicMiniProbeRadiusKind radiusKind = DynamicMiniProbeRadiusKind::P95;
        DynamicMiniProbeOrderingKind orderingKind = DynamicMiniProbeOrderingKind::RadiusNormalizedHeuristic;
        int nMegaProbes = 20;
        int minMiniProbes = 250;
        int maxMiniProbes = 2000;
        double distanceFactor = 1.9;
        double thresholdFactor = 2.0;
        int k = 100;
        int batchSize = 50;
        int growthBatchStep = 10;
        double targetProbabilityMass = 0.95;
        double minBatchProbabilityMass = 0.01;
        int stableQrBatches = 2;
        double qrChangeEps = 0.01;
        bool disableStopping = true;
        double stopHeapMultiplier = 2.0;
        std::vector<double> stopHeapMultipliers;
    };

    struct DynamicMiniProbeBatchLog {
        int batch_index = 0;
        double stop_heap_multiplier = 1.0;
        int selected_l1_count = 0;
        double q_radius_before = 0.0;
        double q_radius_after = 0.0;
        double q_radius_relative_delta = 0.0;
        double batch_probability_mass = 0.0;
        double cumulative_probability_mass = 0.0;
        double stop_heap_radius_before = 0.0;
        double stop_heap_radius_after = 0.0;
        double stop_heap_radius_relative_delta = 0.0;
        double best_raw_centroid_distance = 0.0;
        double remaining_min_raw_centroid_distance = 0.0;
        double centroid_distance_factor_boundary = std::numeric_limits<double>::infinity();
        vector_idx_t stop_heap_worst_id_before = INVALID_VECTOR_ID;
        vector_idx_t stop_heap_worst_id_after = INVALID_VECTOR_ID;
        bool threshold_rule_stop = false;
        bool heap_rule_stop = false;
        bool combined_rule_stop = false;
        bool stopped = false;
        std::vector<vector_idx_t> result_ids;
    };

    struct DynamicMiniProbeStats {
        DynamicMiniProbeStrategy strategy = DynamicMiniProbeStrategy::DistanceFactor;
        DynamicMiniProbeOrderingKind orderingKind = DynamicMiniProbeOrderingKind::RadiusNormalizedHeuristic;
        int candidate_l1_count = 0;
        int selected_l1_count = 0;
        double q_radius = 0.0;
        double cumulative_probability_mass = 0.0;
        uint64_t distance_computations = 0;
        uint64_t angular_cap_disjoint_count = 0;
        uint64_t angular_cap_full_l1_count = 0;
        uint64_t angular_cap_query_inside_l1_count = 0;
        uint64_t angular_cap_partial_count = 0;
        std::string stop_reason;
        std::vector<DynamicMiniProbeBatchLog> batch_logs;
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
        // LSH bucket bits for overlap tracking
        int overlapLshBits = 8;
        // Worst mini centroids used for bucketed overlap score
        int overlapWorstMiniCount = 30;
        // Worst overlap values to average (after bucketing)
        int overlapWorstAvgCount = 30;
        // Threshold for overlap score change to trigger reclustering
        float overlapScoreChangeThreshold = 0.05;
        // Movement-count stopping parameters
        float movementMinTolerancePercent = 5.0;
        float movementMaxTolerancePercent = 80.0;
        int movementToleranceParts = 3;
        int movementMinBadMinis = 5;
        // Use cuVS GPU k-means instead of Faiss CPU k-means
        bool useCuvsKmeans = false;
        // GPU device ID for cuVS k-means
        int cuvsGpuDevice = 0;

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
                                         const float overlappingScoreThreshold = 0.1,
                                         const int overlapLshBits = 8,
                                         const int overlapWorstMiniCount = 20,
                                         const int overlapWorstAvgCount = 30,
                                         const float overlapScoreChangeThreshold = 0.05,
                                         const bool useCuvsKmeans = false,
                                         const int cuvsGpuDevice = 0)
            : nIter(nIter), megaCentroidSize(megaCentroidSize), miniCentroidSize(miniCentroidSize),
              newMiniCentroidSize(newMiniCentroidSize), lambda(lambda), searchThreshold(searchThreshold),
              distanceType(distanceType), numMegaReclusterCentroids(numMegaReclusterCentroids),
              numNewMiniReclusterCentroids(numNewMiniReclusterCentroids),
              quantizationTrainPercentage(quantizationTrainPercentage), hardClusterSizeLimit(hardClusterSizeLimit),
              kmeansSamplingRatio(kmeansSamplingRatio), scoreChangeThreshold(scoreChangeThreshold),
              centroidChangeThreshold(centroidChangeThreshold), overlappingScoreThreshold(overlappingScoreThreshold),
              overlapLshBits(overlapLshBits), overlapWorstMiniCount(overlapWorstMiniCount),
              overlapWorstAvgCount(overlapWorstAvgCount),
              overlapScoreChangeThreshold(overlapScoreChangeThreshold), useCuvsKmeans(useCuvsKmeans),
              cuvsGpuDevice(cuvsGpuDevice) {}
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

        void naiveInsert(float *data, size_t n);

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

        void reclusterBasedOnOverlapHistory();

        void reclusterAllMegaCentroids(int n = INT_MAX);

        void reclusterAllMiniCentroidsQuant();

        void fixBoundaryMiniCentroids(int n = INT_MAX);

        void fixBoundaryMiniCentroidsV2(int n = INT_MAX);

        void storeScoreForMegaClusters(int n = INT_MAX);

        void storeMSEScoreForMegaClusters(int n = INT_MAX);

        void calculateOverlapScoreForL2(int megaCentroidId);

        void calculateOverlapScoreForAngular(int megaCentroidId);

        void calculateOverlapScoreForAngular2(int megaCentroidId);

        double calculateApproxOverlapScoreForAngular(vector_idx_t miniCentroidId, std::vector<vector_idx_t> &closestMiniIds);

        double calculateRealOverlapScore(vector_idx_t miniCentroidId, std::vector<vector_idx_t> &closestMiniIds);

        double calculateRealOverlapScoreForAngular(vector_idx_t megaId, vector_idx_t miniCentroidId, int k = 20);

        void computeOverlapScores();

        // Computes per-mini score as % vectors whose nearest mini centroid is not their own.
        void computeWrongAssignmentScores();

        // Builds bucket-level stats (avg, stdev, P90, P99) for wrong-% and finite boundary gaps
        // over worst minis per mega, keyed by overlap LSH bucket.
        void updateWrongAssignmentBucketAverages();

        // Records the first stable bucket snapshot when consecutive wrong-% avg and stdev absolute
        // change are both <= relativeDeltaFraction. Stable is also refreshed whenever
        // cur.(wrong_pct_avg + wrong_pct_std) < stable.(wrong_pct_avg + wrong_pct_std), even if
        // that convergence condition is not met.
        void updateWrongAssignmentBucketStability(float relativeDeltaFraction = 0.03f);

        // Selects mega centroids for reclustering using stable bucket baselines (see implementation).
        std::vector<vector_idx_t> getMegaCentroidsToReclusterByWrongAssignment(int minTriggerCount = 2);

        // Reclusters selected megas based on wrong-assignment logic.
        void reclusterBasedOnWrongAssignment(float stabilityDeltaFraction = 0.04f, int minTriggerCount = 5);

        // Reclusters selected megas based on saturated vector movement counts.
        void reclusterBasedOnMovementScore();

        // Prints stats over worst-mini wrong-assignment percentages grouped by LSH bucket.
        void printWrongAssignmentStatsForWorstMinis();

        void getApproxOverlapScores(const double **_overlapScores, size_t &n) const {
            *_overlapScores = approxOverlapScores.data();
            n = approxOverlapScores.size();
        }

        void getRealOverlapScores(const double **_overlapScores, size_t &n) const {
            *_overlapScores = realOverlapScores.data();
            n = realOverlapScores.size();
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

        void resetReclusterReadSamples() {
            reclusterReadSamples.clear();
            reclusterReadUnassignedVectors = 0;
            collectReclusterReadSamples = true;
        }

        const std::vector<size_t> &getReclusterReadSamples() const {
            return reclusterReadSamples;
        }

        size_t getReclusterReadUnassignedVectors() const {
            return reclusterReadUnassignedVectors;
        }
        
        int getDim() const {
            return dim;
        }      

        void saveOldScoreForMegaClusters();

        std::vector<vector_idx_t> getMegaCentroidsToRecluster() const;

        std::vector<vector_idx_t> getMegaCentroidsToReclusterByOverlapHistory();

        void computeAllSubCells(int avgSubCellSize);

        void quantizeVectors();

        void printChangeClusterStats();

        void printStats();

        void flush_to_disk(const std::string &file_path) const;

        void search(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results,
                    int nMegaProbes, int nMiniProbes, ReclusteringIndexStats &stats);

        void searchDynamicMiniProbes(const float *query, uint16_t k,
                                     std::priority_queue<NodeDistCloser> &results,
                                     const DynamicMiniProbeConfig &probeConfig,
                                     DynamicMiniProbeStats &dynamicStats,
                                     ReclusteringIndexStats &stats);

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

        void updateOverlapHistory();

    private:
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

        void updateGlobalBucketHistory(const std::unordered_map<uint32_t, double> &bucketScores);

        void ensureOverlapLshIndex();

        uint32_t getOverlapLshBucket(const float *vec);

        std::vector<vector_idx_t> getWorstMiniCentroidsByRealOverlap(
            const std::vector<vector_idx_t> &miniIds, int k) const;

        std::vector<vector_idx_t> getWorstMiniCentroidsByWrongAssignment(
            const std::vector<vector_idx_t> &miniIds, int k) const;

        void calculateOverlapScoreForL2All(int megaCentroidId);

        void calculateWrongAssignmentPercentage(
            vector_idx_t miniCentroidId,
            const std::vector<vector_idx_t> &candidateMiniIds,
            double *wrongly_assigned_percent,
            double *avg_wrongly_assigned_relative_gap = nullptr);

        WrongAssignmentBucketStats computeWrongAssignmentBucketStats(
            const std::vector<double> &wrongPcts, const std::vector<double> &gapsAligned) const;

        void ensureVectorMovementCountsSize(vector_idx_t vectorId);

        void updateMovementCountsForNewMiniAssignments(
            const std::unordered_map<vector_idx_t, vector_idx_t> &oldVectorToMini,
            const std::vector<std::vector<vector_idx_t>> &newMiniClusterVectorIds,
            const std::unordered_map<vector_idx_t, vector_idx_t> &newToOldCentroidIdMap);

        double getMovementWeight(uint8_t count) const;

        double computeMiniMovementScore(vector_idx_t miniId) const;

        std::vector<vector_idx_t> getMegaCentroidsToReclusterByMovementScore() const;

        void resetInputBuffer();

        void clusterData(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                         std::vector<float>& centroids, std::vector<std::vector<float>>& clusters,
                         std::vector<std::vector<vector_idx_t>> &clusterVectorIds);

        void clusterData(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                 std::vector<float>& centroids, std::vector<std::vector<vector_idx_t>> &clusterVectorIds, int nClusters = -1);

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

        void trainKmeansCentroids(const float *data, int n, int numClusters,
                                  std::vector<float> &centroids);

        void assignKmeansLabels(const float *data, int n, const std::vector<float> &centroids,
                                int numClusters, std::vector<int64_t> &assign);

        void kmeansClusterData(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                               std::vector<float> &centroids,
                               std::vector<std::vector<vector_idx_t> > &clusterVectorIds,
                               std::vector<std::vector<float> > *clusters = nullptr,
                               int nClusters = -1);

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

        void findKClosestVectorsForTwoHeaps(const float *query, int k, int heapK,
                                            const std::vector<vector_idx_t> &miniCentroids,
                                            std::priority_queue<NodeDistCloser> &results,
                                            std::priority_queue<NodeDistCloser> &heapResults,
                                            ReclusteringIndexStats &stats);

        void invalidateMiniRadiusStats();

        void ensureMiniRadiusStats();

        const std::vector<double>& getMiniRadiusStats(DynamicMiniProbeRadiusKind radiusKind);

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

        inline double computeAvgOnWorstElement(const std::vector<double> &values, bool biggerIsBetter = true) const {
            if (values.empty()) {
                return 0.0;
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

            // Compute average
            double sum = 0.0;
            for (double val : worstElements) {
                sum += val;
            }
            return sum / worstElements.size();
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

    public:
        ReclusteringIndexConfig config;

    private:
        struct OverlapHistory {
            double prev1 = -10.0;
            double prev2 = -10.0;
            int count = 0;
        };

        int dim;
        size_t size;
        RandomGenerator *rg;

        // Mini and mega centroids
        std::vector<float> megaCentroids;
        std::vector<std::vector<vector_idx_t>> megaMiniCentroidIds;
        std::vector<double> megaClusteringScore;
        std::vector<double> approxOverlapScores;
        std::vector<double> realOverlapScores;
        std::unordered_map<uint32_t, OverlapHistory> globalBucketOverlapHistory;
        std::unordered_map<uint32_t, WrongAssignmentBucketStats> wrongAssignmentBucketCurrentStats;
        std::unordered_map<uint32_t, double> wrongAssignmentBucketPrevPctAvg;
        std::unordered_map<uint32_t, double> wrongAssignmentBucketPrevPctStd;
        std::unordered_map<uint32_t, WrongAssignmentBucketStats> wrongAssignmentBucketStableStats;
        std::vector<float> miniCentroids;
        std::vector<std::vector<float>> miniClusters;
        std::vector<double> miniClusteringScore;
        std::vector<double> wrongAssignmentScores;
        std::vector<double> wrongAssignmentBoundaryAvgRelativeGap;
        std::vector<std::vector<vector_idx_t>> miniClusterVectorIds;
        std::vector<uint8_t> vectorMovementCounts;
        std::vector<size_t> reclusterReadSamples;
        size_t reclusterReadUnassignedVectors = 0;
        bool collectReclusterReadSamples = false;
        std::vector<double> miniRmsRadius;
        std::vector<double> miniP75Radius;
        std::vector<double> miniP95Radius;

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

        std::unique_ptr<faiss::IndexLSH> overlapLshIndex;

        // Stats
        ReclusteringIndexStats stats;

        // Track micro centroid id
        vector_idx_t nextMiniCentroidId = 6541;
    };
}
