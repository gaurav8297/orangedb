#pragma once

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
        std::unordered_map<vector_idx_t, uint64_t> shrinkCallsPerNode;


        void logStats() {
            spdlog::info("Total Distance Computations in search: {}", totalDistCompDuringSearch);
            spdlog::info("Total Distance Computations in Shrink: {}", totalDistCompDuringShrink);
            spdlog::info("Total Distance Computations in MakeConnection: {}", totalDistCompDuringMakeConnection);
            spdlog::info("Total Distance Computations in Delete: {}", totalDistCompDuringDelete);
            spdlog::info("Total Nodes Shrink During Delete: {}", totalNodesShrinkDuringDelete);
            // Sort shrinkCallsPerNode in descending order and print
            std::vector<std::pair<vector_idx_t, uint64_t>> sortedShrinkCallsPerNode(shrinkCallsPerNode.begin(),
                                                                                     shrinkCallsPerNode.end());
            std::sort(sortedShrinkCallsPerNode.begin(), sortedShrinkCallsPerNode.end(),
                        [](const std::pair<vector_idx_t, uint64_t> &a, const std::pair<vector_idx_t, uint64_t> &b) {
                            return a.second > b.second;
                        });

            // Print top 5% of the nodes
            auto top5Percent = sortedShrinkCallsPerNode.size() * 0.05;
            // Sum of all shrink calls
            uint64_t totalShrinkCalls5Percent = 0;
            for (size_t i = 0; i < top5Percent; i++) {
                totalShrinkCalls5Percent += sortedShrinkCallsPerNode[i].second;
            }
            spdlog::info("Total Shrink Calls in Top 5% of the nodes: {}", totalShrinkCalls5Percent);
            spdlog::info("Total Shrink Calls: {}", totalShrinkCalls);
        }

        void reset() {
            totalDistCompDuringSearch = 0;
            totalDistCompDuringShrink = 0;
            totalDistCompDuringMakeConnection = 0;
            totalDistCompDuringDelete = 0;
            totalNodesShrinkDuringDelete = 0;
            totalShrinkCalls = 0;
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
                for (auto &it : other.shrinkCallsPerNode) {
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
        uint16_t efSearch = 50;
        // RNG alpha parameter
        float alpha = 1.0;

        HNSWConfig(uint16_t M, uint16_t efConstruction, uint16_t efSearch, float alpha)
                : M(M), efConstruction(efConstruction), efSearch(efSearch), alpha(alpha) {}
    };

    class HNSW {
    public:
        explicit HNSW(HNSWConfig config, RandomGenerator *rg, uint16_t dim);

        void build(const float *data, size_t n);

        void search(
                const float *query,
                uint16_t k,
                uint16_t efSearch,
                orangedb::VisitedTable &visited,
                std::priority_queue<NodeDistCloser> &results,
                Stats &stats);

        void deleteNodes(const vector_idx_t *deletedIds, size_t n, int dim, Stats &stats);

        void logStats();

    private:
        void initProbabs(uint16_t M, double levelMult);

        uint8_t randomLevel();

        // Add node to the graph
        void searchNearestOnLevel(
                DistanceComputer *dc,
                level_t level,
                vector_idx_t &nearest,
                double &nearestDist,
                Stats &stats);

        void searchNeighbors(
                DistanceComputer *dc,
                level_t level,
                std::priority_queue<NodeDistCloser> &results,
                vector_idx_t entrypoint,
                double entrypointDist,
                VisitedTable &visited,
                uint16_t efSearch,
                Stats &stats);

        void searchNeighborsOnLastLevel(
                DistanceComputer *dc,
                std::priority_queue<NodeDistCloser> &results,
                vector_idx_t entrypoint,
                double entrypointDist,
                VisitedTable &visited,
                uint16_t efSearch,
                int distCompBatchSize,
                Stats &stats);

        void shrinkNeighbors(
                DistanceComputer *dc,
                vector_idx_t id,
                std::priority_queue<NodeDistCloser> &results,
                int maxSize,
                level_t level,
                int dim,
                Stats &stats);

        void makeConnection(
                DistanceComputer *dc,
                vector_idx_t src,
                vector_idx_t dest,
                double distSrcDest,
                level_t level,
                Stats &stats);

        void addNodeOnLevel(
                DistanceComputer *dc,
                vector_idx_t id,
                level_t level,
                vector_idx_t entrypoint,
                double entrypointDist,
                VisitedTable &visited,
                std::vector<NodeDistCloser> &neighbors,
                Stats &stats);

        void addNode(
                DistanceComputer *dc,
                std::vector<vector_idx_t> node_id,
                level_t level,
                std::vector<omp_lock_t> &locks,
                VisitedTable &visited,
                Stats &stats);

        void deleteNode(
                DistanceComputer* dc,
                orangedb::vector_idx_t deletedId,
                std::vector<omp_lock_t> &locks,
                const float *infVector,
                int dim,
                VisitedTable &visited,
                Stats &stats);

        void deleteNodeV2(
                DistanceComputer* dc,
                orangedb::vector_idx_t deletedId,
                std::vector<omp_lock_t> &locks,
                const float *infVector,
                VisitedTable &visited,
                Stats &stats);

        void deleteNodeV3(
                DistanceComputer* dc,
                orangedb::vector_idx_t deletedId,
                std::vector<omp_lock_t> &locks,
                const float *infVector,
                Stats &stats);

        inline vector_idx_t getActualId(level_t level, vector_idx_t id) {
            return level != 0 ? storage->actual_ids[level][id] : id;
        }

    public:
        HNSWConfig config;

    private:
        vector_idx_t entryPoint;
        level_t maxLevel;
        std::vector<double> levelProbabs;

        Storage *storage;
        RandomGenerator *rg;
        Stats stats;
    };
}
