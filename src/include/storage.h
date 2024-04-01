#pragma once

#include <unistd.h>
#include <atomic>
#include <vector>

#define MAX_LEVELS 10

namespace orangedb {
    typedef int32_t storage_idx_t;

    // Separate ID for each level
    struct GraphCSR {
        char padding0[64];
        std::vector<storage_idx_t> neighbors;
        char padding1[64];
    };

    struct LevelCounter {
        explicit LevelCounter(): counter(0) {}
        std::atomic<uint32_t> counter;
        char padding[64];
    };

    // TODO: Make it an interface eventually
    struct Storage {
        explicit Storage(uint16_t dim, uint16_t M, uint16_t max_levels): dim(dim) {
            graphs.resize(max_levels);
            for (uint16_t i = 0; i < max_levels; i++) {
                max_neighbors_per_level.push_back(i == 0 ? M * 2 : M);
            }

            for (uint16_t i = 0; i < max_levels; i++) {
                level_counters[i].counter = 0;
                fast_level_counters[i] = 0;
            }
        }

        inline void resize(uint32_t n, uint8_t level) {
            graphs[level].neighbors.resize(n * max_neighbors_per_level[level], -1);
            if (level > 0) {
                next_level_ids[level].resize(n);
                actual_ids[level].resize(n);
            }
            if (level == 0) {
                num_points = n;
            }
        }

        inline storage_idx_t* get_neighbors(uint8_t level) {
            return graphs[level].neighbors.data();
        }

        inline void get_neighbors_offsets(storage_idx_t id, uint8_t level, size_t& begin, size_t& end) {
            begin = id * max_neighbors_per_level[level];
            end = begin + max_neighbors_per_level[level];
        }

        inline std::vector<storage_idx_t> getNextNodeId(uint8_t max_level) {
            std::vector<storage_idx_t> ids;
            for (uint8_t i = 0; i < max_level; i++) {
                ids.push_back(level_counters[i].counter++);
            }
            return ids;
        }

        // Not thread safe!!
        inline std::vector<storage_idx_t> getFastNextNodeId(uint8_t max_level) {
            std::vector<storage_idx_t> ids;
            for (uint8_t i = 0; i <= max_level; i++) {
                ids.push_back(fast_level_counters[i]++);
            }
            return ids;
        }

        // Right now everything is in memory.
        // TODO: Serialize to disk
        const float* data;
        uint8_t* codes;
        float* vmin;
        float* vdiff;
        const uint16_t dim;
        uint64_t num_points;
        std::vector<uint16_t> max_neighbors_per_level;
        std::vector<GraphCSR> graphs;
        std::vector<storage_idx_t> next_level_ids[MAX_LEVELS];
        std::vector<storage_idx_t> actual_ids[MAX_LEVELS];
        // This is needed because atomics doesn't support copy or more. So we can't dynamically allocate it with resize.
        LevelCounter level_counters[MAX_LEVELS];
        uint32_t fast_level_counters[MAX_LEVELS];
    };
} // namespace orangedb
