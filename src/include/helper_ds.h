#pragma once

#include <unistd.h>
#include <vector>
#include <common.h>

using namespace std;

namespace orangedb {
    struct NodeDistCloser {
        explicit NodeDistCloser(vector_idx_t id, double dist) : id(id), dist(dist) {}

        vector_idx_t id;
        double dist;

        bool operator<(const NodeDistCloser &other) const {
            return dist < other.dist;
        }
    };

    struct NodeDistFarther {
        explicit NodeDistFarther(vector_idx_t id, double dist) : id(id), dist(dist) {}

        vector_idx_t id;
        double dist;

        bool operator<(const NodeDistFarther &other) const {
            return dist > other.dist;
        }
    };

    class MaxHeap {
    public:
        explicit MaxHeap(int capacity);

        void push(vector_idx_t id, float val);

        vector_idx_t popMin(float *val);

        inline int size() const {
            return logical_size;
        };

        inline vector_idx_t max(float *val) {
            *val = values[1];
            return ids[1];
        }

    private:
        void pushToHeap(vector_idx_t id, float val);

        void popFromHeap();

    private:
        int capacity;
        int physical_size;
        int logical_size;
        std::vector<vector_idx_t> ids;
        std::vector<float> values;
    };

    // TODO: Use bitset instead of vector<uint8_t>
    class VisitedTable {
    public:
        explicit VisitedTable(size_t size) : visited(size, 0), visited_id(1) {};

        inline void set(vector_idx_t id) {
            visited[id] = visited_id;
        }

        inline bool get(vector_idx_t id) {
            return visited[id] == visited_id;
        }

        inline void reset() {
            visited_id++;
            if (visited_id == 250) {
                std::fill(visited.begin(), visited.end(), 0);
                visited_id = 1;
            }
        }

        inline uint8_t *data() {
            return visited.data();
        }

    private:
        std::vector<uint8_t> visited;
        uint8_t visited_id;
    };
} // namespace orangedb
