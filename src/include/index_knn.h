#pragma once

#include "storage.h"
#include "distance.h"
#include <cassert>

namespace orangedb {
    class IndexKnn {
    public:
        IndexKnn(Storage* storage): storage(storage) {};
        void search(int n, const float* queries, int k, float* distances, storage_idx_t* result_ids);
    private:
        Storage* storage;
    };
} // namespace orangedb
