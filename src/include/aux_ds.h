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
} // namespace orangedb