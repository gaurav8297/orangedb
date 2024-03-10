#pragma once

#include <unistd.h>
#include <vector>
#include "storage.h"

using namespace std;

namespace orangedb {
    class MaxHeap {
    public:
        explicit MaxHeap(int capacity);
        void push(storage_idx_t id, float val);
        storage_idx_t pop_min(float* val);
        inline int size() const {
            return logical_size;
        };
        inline storage_idx_t max(float* val) {
            *val = values[1];
            return ids[1];
        }
    private:
        void push_to_heap(storage_idx_t id, float val);
        void pop_from_heap();
    private:
        int capacity;
        int physical_size;
        int logical_size;
        std::vector<storage_idx_t> ids;
        std::vector<float> values;
    };

    // TODO: Maybe replace it with a bitset or better bloom filters which will take less space and much faster since
    //  it can fit in L3 cache.
    class VisitedTable {
    public:
        // [0, 0, 0, 0, 0... 100M]
        // T1: 5, 9, 1000, 122, 56

        // Visited: [0, 0, 0, 0, 0... 100M]
        // Inserting 5
        // Ngh: 0, 4, 7...
        // Visited: [1, 0, 0, 1, 0... 100M]
        // Reset: []
        explicit VisitedTable(size_t size): visited(size, 0), visited_id(1) {};
        inline void set(storage_idx_t id) {
            visited[id] = visited_id;
        }
        inline bool get(storage_idx_t id) {
            return visited[id] == visited_id;
        }
        inline void reset() {
            visited_id++;
            if (visited_id == 250) {
                std::fill(visited.begin(), visited.end(), 0);
                visited_id = 1;
            }
        }
    private:
        std::vector<uint8_t> visited;
        uint8_t visited_id;
    };
} // namespace orangedb