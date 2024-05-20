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
        uint64_t totalShrinkCalls = 0;


        void logStats() {
            spdlog::info("Total Distance Computations in search: {}", totalDistCompDuringSearch);
            spdlog::info("Total Distance Computations in Shrink: {}", totalDistCompDuringShrink);
            spdlog::info("Total Distance Computations in MakeConnection: {}", totalDistCompDuringMakeConnection);
            spdlog::info("Total Shrink Calls: {}", totalShrinkCalls);
        }

        void reset() {
            totalDistCompDuringSearch = 0;
            totalDistCompDuringShrink = 0;
            totalDistCompDuringMakeConnection = 0;
            totalShrinkCalls = 0;
        }

        void merge(Stats &other) {
#pragma omp critical
            {
                totalDistCompDuringSearch += other.totalDistCompDuringSearch;
                totalDistCompDuringShrink += other.totalDistCompDuringShrink;
                totalDistCompDuringMakeConnection += other.totalDistCompDuringMakeConnection;
                totalShrinkCalls += other.totalShrinkCalls;
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
                std::priority_queue<NodeDistCloser> &results,
                int maxSize,
                level_t level,
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

        inline vector_idx_t getActualId(level_t level, vector_idx_t id) {
            return level != 0 ? storage->actual_ids[level][id] : id;
        }

    private:
        HNSWConfig config;
        vector_idx_t entryPoint;
        level_t maxLevel;
        std::vector<double> levelProbabs;

        Storage *storage;
        RandomGenerator *rg;
        Stats stats;
    };
}
