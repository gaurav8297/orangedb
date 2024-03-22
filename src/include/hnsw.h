#pragma once

#include <unistd.h>
#include <omp.h>
#include <vector>
#include <random>
#include <distance.h>
#include <storage.h>
#include <aux_ds.h>
#include <queue>

using namespace std;

// TODO: Work on optimizations later on
namespace orangedb {
    typedef uint8_t level_t;

    struct Stats {
        explicit Stats() : totalDistComp(0), totalDistCompInShrink(0), totalShrinkCalls1(0), totalShrinkCalls2(0) {}

        atomic_int64_t totalDistComp;
        atomic_int64_t totalDistCompInShrink;
        atomic_int64_t totalShrinkCalls1;
        atomic_int64_t totalShrinkCalls2;
        atomic_int64_t totalShrinkNotReduce;
    };

    class HNSW {
    public:
        struct NodeDistCloser {
            explicit NodeDistCloser(storage_idx_t id, float dist): id(id), dist(dist) {}
            storage_idx_t id;
            float dist;
            bool operator<(const NodeDistCloser& other) const {
                return dist < other.dist;
            }
        };

        struct NodeDistFarther {
            explicit NodeDistFarther(storage_idx_t id, float dist): id(id), dist(dist) {}
            storage_idx_t id;
            float dist;
            bool operator<(const NodeDistFarther& other) const {
                return dist > other.dist;
            }
        };
    public:
        explicit HNSW(uint16_t M, uint16_t ef_construction, uint16_t dim, float explore_factor);
        void build(const float* data, size_t n);
        void search_v1(
                const float* query,
                uint16_t k,
                uint16_t ef_search,
                VisitedTable& visited,
                std::priority_queue<NodeDistCloser>& resultSet);
        void print_stats();
    private:
        void init_probabs(uint16_t M, double levelMult);
        uint8_t random_level();
        // Add node to the graph
        void search_nearest_on_level(
                DistanceComputer *dc,
                level_t level,
                storage_idx_t& nearest,
                float& nearestDist);
        void search_neighbors(
                DistanceComputer *dc,
                level_t level,
                std::priority_queue<NodeDistCloser>& results,
                storage_idx_t entrypoint,
                float entrypointDist,
                VisitedTable& visited,
                uint16_t ef);
        void beam_search_neighbors(
                DistanceComputer *dc,
                level_t level,
                std::priority_queue<NodeDistCloser> &results,
                storage_idx_t entrypoint,
                float entrypointDist,
                VisitedTable &visited,
                uint16_t ef);
        void search_neighbors_optimized(
                DistanceComputer *dc,
                level_t level,
                std::priority_queue<NodeDistCloser>& results,
                storage_idx_t entrypoint,
                float entrypointDist,
                VisitedTable& visited,
                uint16_t ef);
        int shrink_neighbors(DistanceComputer *dc, std::priority_queue<NodeDistCloser>& resultSet, int max_size, uint8_t level);
        void make_connection(DistanceComputer *dc, storage_idx_t src, storage_idx_t dest, float dist_src_dest, level_t level);
        void add_node_on_level(
                DistanceComputer *dc,
                storage_idx_t id,
                level_t level,
                storage_idx_t entrypoint,
                float entrypoint_dist,
                VisitedTable &visited,
                std::vector<pair<storage_idx_t, float>> &neighbors);
        void add_node(
                DistanceComputer *dc,
                std::vector<storage_idx_t> node_id,
                level_t level,
                std::vector<omp_lock_t>& locks,
                VisitedTable &visited);
        inline float rand_float() {
            return mt() / float(mt.max());
        }
        inline storage_idx_t getActualId(uint8_t level, storage_idx_t id) {
            return level != 0 ? storage->actual_ids[level][id] : id;
        }

    private:
        float explore_factor;
        uint16_t ef_construction;
        int64_t entry_point = -1;
        uint8_t max_level = 0;
        std::vector<double> level_probabs;
        Storage* storage;
        std::mt19937 mt;
        Stats stats;
    };
}
