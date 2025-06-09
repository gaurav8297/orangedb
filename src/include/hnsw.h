#pragma once

#include <fastQ/common.h>
#include <unistd.h>
#include <omp.h>
#include <vector>
#include <random>
#include <distance.h>
#include <storage.h>
#include <helper_ds.h>
#include <queue>
#include <spdlog/spdlog.h>

using namespace std;

namespace orangedb {
    struct Stats {
        uint64_t totalDistCompDuringSearch = 0;
        uint64_t totalDistCompDuringShrink = 0;
        uint64_t totalDistCompDuringMakeConnection = 0;
        uint64_t totalDistCompDuringDelete = 0;
        uint64_t totalNodesShrinkDuringDelete = 0;
        uint64_t totalShrinkCalls = 0;
        uint64_t totalShrinkMulCalls = 0;
        uint64_t totalShrinkNoUse = 0;
        uint64_t totalGetNbrsCall = 0;
        uint64_t avgNodesExplored = 0;
        uint64_t minNodesExplored = 0;
        uint64_t maxNodesExplored = 0;
        double avgGetNbrsDepth = 0;
        int minDepth = 0;
        int maxDepth = 0;
        uint64_t uselessGetNbrs = 0;
        uint64_t searchIter = 0;
        std::unordered_map<vector_idx_t, uint64_t> shrinkCallsPerNode;


        void logStats() {
            spdlog::info("Total Distance Computations in search: {}", totalDistCompDuringSearch);
            spdlog::info("Total Distance Computations in Shrink: {}", totalDistCompDuringShrink);
            spdlog::info("Total Distance Computations in MakeConnection: {}", totalDistCompDuringMakeConnection);
            spdlog::info("Total Distance Computations in Delete: {}", totalDistCompDuringDelete);
            spdlog::info("Total Nodes Shrink During Delete: {}", totalNodesShrinkDuringDelete);
            spdlog::info("Total Get Nbrs Call: {}", totalGetNbrsCall);
            spdlog::info("Avg Get Nbrs Depth: {}", avgGetNbrsDepth);
            spdlog::info("Useless Get Nbrs: {}", uselessGetNbrs);
            spdlog::info("Total Search Iterations: {}", searchIter);
            spdlog::info("Min Depth: {}", minDepth);
            spdlog::info("Max Depth: {}", maxDepth);
            spdlog::info("Avg Nodes Explored: {}", avgNodesExplored);
            spdlog::info("Min Nodes Explored: {}", minNodesExplored);
            spdlog::info("Max Nodes Explored: {}", maxNodesExplored);
            // Sort shrinkCallsPerNode in descending order and print
//            std::vector<std::pair<vector_idx_t, uint64_t>> sortedShrinkCallsPerNode(shrinkCallsPerNode.begin(),
//                                                                                    shrinkCallsPerNode.end());
//            std::sort(sortedShrinkCallsPerNode.begin(), sortedShrinkCallsPerNode.end(),
//                      [](const std::pair<vector_idx_t, uint64_t> &a, const std::pair<vector_idx_t, uint64_t> &b) {
//                          return a.second > b.second;
//                      });
//
//            // Print top 5% of the nodes
//            if (sortedShrinkCallsPerNode.empty()) {
//                return;
//            }
//            size_t size5Percent = sortedShrinkCallsPerNode.size() * 0.05;
//            size_t start = 0;
//            size_t end = size5Percent;
//            int init = 5;
//            while (end <= sortedShrinkCallsPerNode.size()) {
//                // Sum of all shrink calls
//                uint64_t totalShrinkCalls5Percent = 0;
//                for (size_t i = start; i < end; i++) {
//                    totalShrinkCalls5Percent += sortedShrinkCallsPerNode[i].second;
//                }
//                spdlog::info("Total Shrink Calls {}%-{}% of nodes: {}", init, init + 5, totalShrinkCalls5Percent);
//                spdlog::info("Avg Shrink Calls {}%-{}% of nodes: {}", init, init + 5,
//                             totalShrinkCalls5Percent / size5Percent);
//
//                if (end == sortedShrinkCallsPerNode.size()) {
//                    break;
//                }
//                start += size5Percent;
//                end += size5Percent;
//                end = std::min(end, sortedShrinkCallsPerNode.size());
//                init += 5;
//            }
            spdlog::info("Total Shrink Calls: {}", totalShrinkCalls);
            spdlog::info("Total Shrink Mul Calls: {}", totalShrinkMulCalls);
            spdlog::info("Total Shrink No Use: {}", totalShrinkNoUse);
        }

        void reset() {
            totalDistCompDuringSearch = 0;
            totalDistCompDuringShrink = 0;
            totalDistCompDuringMakeConnection = 0;
            totalDistCompDuringDelete = 0;
            totalNodesShrinkDuringDelete = 0;
            totalShrinkCalls = 0;
            totalShrinkMulCalls = 0;
            totalShrinkNoUse = 0;
            totalGetNbrsCall = 0;
            avgGetNbrsDepth = 0;
            minDepth = 0;
            maxDepth = 0;
            uselessGetNbrs = 0;
            searchIter = 0;
            avgNodesExplored = 0;
            minNodesExplored = 0;
            maxNodesExplored = 0;
            shrinkCallsPerNode.clear();
        }

        void merge(Stats &other) {
#pragma omp critical
            {
                totalDistCompDuringSearch += other.totalDistCompDuringSearch;
                totalDistCompDuringShrink += other.totalDistCompDuringShrink;
                totalDistCompDuringMakeConnection += other.totalDistCompDuringMakeConnection;
                totalDistCompDuringDelete += other.totalDistCompDuringDelete;
                totalNodesShrinkDuringDelete += other.totalNodesShrinkDuringDelete;
                totalShrinkCalls += other.totalShrinkCalls;
                totalShrinkMulCalls += other.totalShrinkMulCalls;
                totalShrinkNoUse += other.totalShrinkNoUse;
                totalGetNbrsCall += other.totalGetNbrsCall;
                avgGetNbrsDepth += other.avgGetNbrsDepth;
                minDepth = std::min(minDepth, other.minDepth);
                maxDepth = std::max(maxDepth, other.maxDepth);
                avgNodesExplored += other.avgNodesExplored;
                minNodesExplored = std::min(minNodesExplored, other.minNodesExplored);
                maxNodesExplored = std::max(maxNodesExplored, other.maxNodesExplored);
                uselessGetNbrs += other.uselessGetNbrs;
                searchIter = std::max(searchIter, other.searchIter);
                for (auto &it: other.shrinkCallsPerNode) {
                    shrinkCallsPerNode[it.first] += it.second;
                }
            }
        }
    };

    struct HNSWConfig {
        // The number of neighbors to keep for each node
        uint16_t M = 16;
        // The number of neighbors to explore during index construction
        uint16_t efConstruction = 200;
        // The number of neighbors to explore during search
        uint64_t efSearch = 50;
        // RNG alpha parameter
        float minAlpha = 0.95;
        // [Experimental] max alpha value
        float maxAlpha = 1.0;
        // Alpha decay
        float alphaDecay = 0.03;

        // Filtered!!
        int filterMinK = 30;
        int maxNeighboursCheck = 64;

        // Compression type (scalar, pair_wise)
        std::string compressionType = "none";

        // Storage path
        std::string storagePath = "/tmp/orangedb";
        bool loadStorage = false;

        // Parallel search
        int nodesToExplore = 20;
        int nodeExpansionPerNode = 10;

        // Number of threads
        int numSearchThreads = 4;

        // Search Parallel Algorithm
        std::string searchParallelAlgorithm;

        HNSWConfig(uint16_t M, uint16_t efConstruction, uint64_t efSearch, float minAlpha,
                   float maxAlpha, float alphaDecay, int filterMinK, int maxNeighboursCheck,
                   std::string compressionType, std::string storagePath, bool loadStorage,
                   int nodesToExplore, int nodeExpansionPerNode, int numSearchThreads,
                   std::string searchParallelAlgorithm)
                : M(M), efConstruction(efConstruction), efSearch(efSearch), minAlpha(minAlpha),
                  maxAlpha(maxAlpha), alphaDecay(alphaDecay), filterMinK(filterMinK),
                  maxNeighboursCheck(maxNeighboursCheck), compressionType(compressionType), storagePath(storagePath),
                  loadStorage(loadStorage), nodesToExplore(nodesToExplore),
                  nodeExpansionPerNode(nodeExpansionPerNode), numSearchThreads(numSearchThreads),
                  searchParallelAlgorithm(searchParallelAlgorithm) {}
    };

    class HNSW {
    public:
        explicit HNSW(HNSWConfig config, RandomGenerator *rg, uint16_t dim);

        void build(const float *data, size_t n);

        inline void flushToDisk() {
            storage->flush_to_disk(config.storagePath);
        }

        void search(
                const float *query,
                uint64_t k,
                uint64_t efSearch,
                orangedb::VisitedTable &visited,
                std::priority_queue<NodeDistCloser> &results,
                Stats &stats);

        void searchWithQuantizer(
                const float *query,
                uint64_t k,
                uint64_t efSearch,
                orangedb::VisitedTable &visited,
                std::priority_queue<NodeDistCloser> &results,
                Stats &stats);

        void searchWithFilter(
                const float *query,
                uint64_t k,
                uint64_t efSearch,
                orangedb::VisitedTable &visited,
                std::priority_queue<NodeDistCloser> &results,
                const uint8_t *filterMask,
                Stats &stats);

        void searchParallel(
                const float *query,
                uint64_t k,
                uint64_t efSearch,
                orangedb::AtomicVisitedTable &visited,
                std::priority_queue<NodeDistCloser> &results,
                Stats &stats,
                PocTaskScheduler *scheduler);

        void deleteNodes(const vector_idx_t *deletedIds, size_t n, int dim, Stats &stats);

        void logStats();

    private:
        void initProbabs(uint16_t M, double levelMult);

        uint8_t randomLevel();

        // Add node to the graph
        void searchNearestOnLevel(
                DelegateDC<float> *dc,
                level_t level,
                vector_idx_t &nearest,
                double &nearestDist,
                Stats &stats);

        void searchNeighbors(
                DelegateDC<float> *dc,
                level_t level,
                std::priority_queue<NodeDistCloser> &results,
                vector_idx_t entrypoint,
                double entrypointDist,
                VisitedTable &visited,
                uint64_t efSearch,
                Stats &stats);

        void searchNeighborsOnLastLevel(
                DelegateDC<float> *dc,
                std::priority_queue<NodeDistCloser> &results,
                vector_idx_t entrypoint,
                double entrypointDist,
                VisitedTable &visited,
                uint64_t efSearch,
                int distCompBatchSize,
                Stats &stats);

        void searchNeighborsOnLastLevel1(
                DelegateDC<float> *dc,
                std::priority_queue<NodeDistCloser> &results,
                vector_idx_t entrypoint,
                double entrypointDist,
                AtomicVisitedTable &visited,
                uint64_t efSearch,
                int distCompBatchSize,
                Stats &stats);

        int findNaiveNeighbours(
                vector_idx_t entrypoint,
                NodeDistCloser *nbrs,
                BitVectorVisitedTable &visited,
                Stats &stats);

        int findNextKNeighbours(
                vector_idx_t entrypoint,
                NodeDistCloser *nbrs,
                BitVectorVisitedTable &visited,
                int minK,
                int maxK,
                int maxNeighboursCheck,
                Stats &stats,
                int& depth);

        int findNextKNeighboursV2(
                DelegateDC<float>* dc,
                vector_idx_t entrypoint,
                NodeDistCloser *nbrs,
                AtomicVisitedTable &visited,
                int minK,
                int maxNeighboursCheck);

        void searchParallelSyncAfterEveryIter(
                DelegateDC<float> *dc,
                std::priority_queue<NodeDistCloser> &results,
                AtomicVisitedTable &visited,
                uint64_t efSearch,
                Stats &stats,
                PocTaskScheduler *scheduler);

        void searchParallelWithParallelQueue(
                DelegateDC<float> *dc,
                std::priority_queue<NodeDistCloser> &results,
                AtomicVisitedTable &visited,
                uint64_t efSearch,
                Stats &stats);

        void searchParallelWithPartitioning(
                DelegateDC<float> *dc,
                std::priority_queue<NodeDistCloser> &results,
                AtomicVisitedTable &visited,
                uint64_t efSearch,
                Stats &stats);

        void searchParallelWithDeltaStepping(
                DelegateDC<float> *dc,
                std::priority_queue<NodeDistCloser> &results,
                AtomicVisitedTable &visited,
                uint64_t efSearch,
                Stats &stats);

        void searchNearestOnLevelWithQuantizer(
                const float *query, fastq::DistanceComputer<float, uint8_t> *dc, orangedb::level_t level,
                orangedb::vector_idx_t &nearest, double &nearestDist,
                orangedb::Stats &stats);

        void searchNeighborsOnLastLevelWithQuantizer(const float *query, fastq::DistanceComputer<float, uint8_t> *dc,
                                                     std::priority_queue<NodeDistCloser> &results,
                                                     orangedb::vector_idx_t entrypoint, double entrypointDist,
                                                     orangedb::VisitedTable &visited, uint64_t efSearch,
                                                     int distCompBatchSize, orangedb::Stats &stats);

        int findNextFilteredKNeighbours(
                DelegateDC<float> *dc,
                vector_idx_t entrypoint,
                std::vector<vector_idx_t> &nbrs,
                const uint8_t *filterMask,
                VisitedTable &visited,
                std::unordered_map<vector_idx_t, int> &nbrsCount,
                int minK,
                int maxK,
                int maxNeighboursCheck,
                bool force,
                Stats &stats);

        void searchNeighborsOnLastLevelWithFilterA(
                DelegateDC<float> *dc,
                std::priority_queue<NodeDistCloser> &results,
                vector_idx_t entrypoint,
                double entrypointDist,
                VisitedTable &visited,
                uint64_t efSearch,
                int distCompBatchSize,
                const uint8_t *filterMask,
                Stats &stats);

        void searchNeighborsOnLastLevelWithFilterB(
                DelegateDC<float> *dc,
                std::priority_queue<NodeDistCloser> &results,
                vector_idx_t entrypoint,
                double entrypointDist,
                VisitedTable &visited,
                uint64_t efSearch,
                int distCompBatchSize,
                const uint8_t *filterMask,
                Stats &stats);

        void shrinkNeighbors(
                DelegateDC<float> *dc,
                vector_idx_t id,
                std::priority_queue<NodeDistCloser> &results,
                int maxSize,
                level_t level,
                int dim,
                Stats &stats);

        void makeConnection(
                DelegateDC<float> *dc,
                vector_idx_t src,
                vector_idx_t dest,
                double distSrcDest,
                level_t level,
                Stats &stats);

        void addNodeOnLevel(
                DelegateDC<float> *dc,
                vector_idx_t id,
                level_t level,
                vector_idx_t entrypoint,
                double entrypointDist,
                VisitedTable &visited,
                std::vector<NodeDistCloser> &neighbors,
                Stats &stats);

        void addNode(
                DelegateDC<float> *dc,
                std::vector<vector_idx_t> node_id,
                level_t level,
                std::vector<omp_lock_t> &locks,
                VisitedTable &visited,
                Stats &stats);

        void deleteNode(
                DelegateDC<float> *dc,
                orangedb::vector_idx_t deletedId,
                std::vector<omp_lock_t> &locks,
                const float *infVector,
                int dim,
                VisitedTable &visited,
                Stats &stats);

        void deleteNodeV2(
                DelegateDC<float> *dc,
                orangedb::vector_idx_t deletedId,
                std::vector<omp_lock_t> &locks,
                const float *infVector,
                VisitedTable &visited,
                Stats &stats);

        void deleteNodeV3(
                DelegateDC<float> *dc,
                orangedb::vector_idx_t deletedId,
                std::vector<omp_lock_t> &locks,
                const float *infVector,
                Stats &stats);

        inline vector_idx_t getActualId(level_t level, vector_idx_t id) const {
            return level != 0 ? storage->actual_ids[level][id] : id;
        }

    public:
        HNSWConfig config;
        Storage *storage;

    private:
        std::vector<double> levelProbabs;
        std::unique_ptr<fastq::Quantizer<uint8_t>> quantizer;
        RandomGenerator *rg;
        Stats stats;
    };
}
