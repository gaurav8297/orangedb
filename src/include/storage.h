#pragma once

#include <unistd.h>
#include <atomic>
#include <vector>
#include <fastQ/fastq.h>
#include <fstream>
#include <iostream>
#include <cstring>  // for memcpy
#include <thread>
#include <chrono>

#define MAX_LEVELS 10

namespace orangedb {

    // Separate ID for each level
    struct GraphCSR {
        char padding0[64];
        std::vector<vector_idx_t> neighbors;
        char padding1[64];
    };

    struct LevelCounter {
        explicit LevelCounter() : counter(0) {}

        std::atomic<uint32_t> counter;
        char padding[64];
    };

    struct Storage {
        explicit Storage(uint16_t dim, uint16_t M, uint16_t max_levels, uint16_t code_size)
                : dim(dim), entryPoint(INVALID_VECTOR_ID), maxLevel(0), code_size(code_size) {
            graphs.resize(max_levels);
            for (uint16_t i = 0; i < max_levels; i++) {
                max_neighbors_per_level.push_back(i == 0 ? M * 2 : M);
            }

            for (uint16_t i = 0; i < max_levels; i++) {
                level_counters[i].counter = 0;
                fast_level_counters[i] = 0;
            }
        }

        // Constructor that initializes from a file path (loading from disk)
        explicit Storage(const std::string &file_path, uint16_t max_levels, uint16_t code_size)
                : code_size(code_size) {
            graphs.resize(max_levels);
            load_from_disk(file_path);
        }

        inline void resize(uint32_t n, uint8_t level) {
            graphs[level].neighbors.resize(n * max_neighbors_per_level[level], INVALID_VECTOR_ID);
            if (level > 0) {
                next_level_ids[level].resize(n);
                actual_ids[level].resize(n);
            }
            if (level == 0) {
                numPoints = n;
            }
        }

        inline vector_idx_t *get_neighbors(uint8_t level) {
            return graphs[level].neighbors.data();
        }

        inline size_t get_num_neighbors(uint8_t level) {
            return graphs[level].neighbors.size();
        }

        inline void get_neighbors_offsets(vector_idx_t id, level_t level, size_t &begin, size_t &end) {
            // Add some nanoseconds delay to recreate kuzu behavior
            std::this_thread::sleep_for(std::chrono::nanoseconds(3000));
            begin = id * max_neighbors_per_level[level];
            end = begin + max_neighbors_per_level[level];
        }

        inline std::vector<vector_idx_t> getNextNodeId(level_t max_level) {
            std::vector<vector_idx_t> ids;
            for (uint8_t i = 0; i < max_level; i++) {
                ids.push_back(level_counters[i].counter++);
            }
            return ids;
        }

        inline std::vector<vector_idx_t> getFastNextNodeId(level_t max_level) {
            std::vector<vector_idx_t> ids;
            for (uint8_t i = 0; i <= max_level; i++) {
                ids.push_back(fast_level_counters[i]++);
            }
            return ids;
        }

        // Flush storage to disk as a binary file
        void flush_to_disk(const std::string &file_path) const {
            std::ofstream out(file_path, std::ios::binary);
            if (!out) {
                std::cerr << "Error opening file for writing: " << file_path << std::endl;
                return;
            }

            // Write the basic fields
            out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
            out.write(reinterpret_cast<const char *>(&numPoints), sizeof(numPoints));
            out.write(reinterpret_cast<const char *>(&entryPoint), sizeof(entryPoint));
            out.write(reinterpret_cast<const char *>(&maxLevel), sizeof(maxLevel));

            // Push code to disk
//            out.write(reinterpret_cast<const char *>(codes), numPoints * code_size);

            // Write max_neighbors_per_level
            size_t size = max_neighbors_per_level.size();
            out.write(reinterpret_cast<const char *>(&size), sizeof(size));
            out.write(reinterpret_cast<const char *>(max_neighbors_per_level.data()), size * sizeof(uint16_t));

            // Write graphs (neighbors)
            for (const auto &graph : graphs) {
                size_t neighbors_size = graph.neighbors.size();
                out.write(reinterpret_cast<const char *>(&neighbors_size), sizeof(neighbors_size));
                out.write(reinterpret_cast<const char *>(graph.neighbors.data()), neighbors_size * sizeof(vector_idx_t));
            }

            // Write level_counters (as atomic variables cannot be serialized directly)
            for (const auto &level_counter : level_counters) {
                uint32_t counter_value = level_counter.counter.load();
                out.write(reinterpret_cast<const char *>(&counter_value), sizeof(counter_value));
            }

            // Write fast_level_counters
            out.write(reinterpret_cast<const char *>(fast_level_counters), sizeof(fast_level_counters));

            // Write next_level_ids and actual_ids
            for (int i = 0; i < MAX_LEVELS; ++i) {
                size_t next_size = next_level_ids[i].size();
                size_t actual_size = actual_ids[i].size();
                out.write(reinterpret_cast<const char *>(&next_size), sizeof(next_size));
                out.write(reinterpret_cast<const char *>(next_level_ids[i].data()), next_size * sizeof(vector_idx_t));
                out.write(reinterpret_cast<const char *>(&actual_size), sizeof(actual_size));
                out.write(reinterpret_cast<const char *>(actual_ids[i].data()), actual_size * sizeof(vector_idx_t));
            }

            out.close();
        }

        // Load storage from disk (binary file)
        void load_from_disk(const std::string &file_path) {
            std::ifstream in(file_path, std::ios::binary);
            if (!in) {
                std::cerr << "Error opening file for reading: " << file_path << std::endl;
                return;
            }

            // Read the basic fields
            in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
            in.read(reinterpret_cast<char *>(&numPoints), sizeof(numPoints));
            in.read(reinterpret_cast<char *>(&entryPoint), sizeof(entryPoint));
            in.read(reinterpret_cast<char *>(&maxLevel), sizeof(maxLevel));

            // Read code from disk
//            codes = new uint8_t[numPoints * code_size];
//            in.read(reinterpret_cast<char *>(codes), numPoints * code_size);

            // Read max_neighbors_per_level
            size_t size;
            in.read(reinterpret_cast<char *>(&size), sizeof(size));
            max_neighbors_per_level.resize(size);
            in.read(reinterpret_cast<char *>(max_neighbors_per_level.data()), size * sizeof(uint16_t));

            // Read graphs (neighbors)
            for (auto &graph : graphs) {
                size_t neighbors_size;
                in.read(reinterpret_cast<char *>(&neighbors_size), sizeof(neighbors_size));
                graph.neighbors.resize(neighbors_size);
                in.read(reinterpret_cast<char *>(graph.neighbors.data()), neighbors_size * sizeof(vector_idx_t));
            }

            // Read level_counters (reload atomic variables)
            for (auto &level_counter : level_counters) {
                uint32_t counter_value;
                in.read(reinterpret_cast<char *>(&counter_value), sizeof(counter_value));
                level_counter.counter.store(counter_value);
            }

            // Read fast_level_counters
            in.read(reinterpret_cast<char *>(fast_level_counters), sizeof(fast_level_counters));

            // Read next_level_ids and actual_ids
            for (int i = 0; i < MAX_LEVELS; ++i) {
                size_t next_size, actual_size;
                in.read(reinterpret_cast<char *>(&next_size), sizeof(next_size));
                next_level_ids[i].resize(next_size);
                in.read(reinterpret_cast<char *>(next_level_ids[i].data()), next_size * sizeof(vector_idx_t));

                in.read(reinterpret_cast<char *>(&actual_size), sizeof(actual_size));
                actual_ids[i].resize(actual_size);
                in.read(reinterpret_cast<char *>(actual_ids[i].data()), actual_size * sizeof(vector_idx_t));
            }

            in.close();
        }

        const float *data;
        uint8_t *codes;
        uint16_t code_size;
        uint16_t dim;
        uint64_t numPoints;
        vector_idx_t entryPoint;
        level_t maxLevel;
        std::vector<uint16_t> max_neighbors_per_level;
        std::vector<GraphCSR> graphs;
        std::vector<vector_idx_t> next_level_ids[MAX_LEVELS];
        std::vector<vector_idx_t> actual_ids[MAX_LEVELS];
        LevelCounter level_counters[MAX_LEVELS];
        uint32_t fast_level_counters[MAX_LEVELS];
    };
} // namespace orangedb
