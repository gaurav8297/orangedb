#pragma once

#include <unistd.h>
#include <storage.h>

using namespace std;

namespace orangedb {
    struct DistanceComputer {
        virtual void compute_distance(storage_idx_t id, float& result) = 0;
        virtual void compute_distance(storage_idx_t src, storage_idx_t dest, float& result) = 0;
        virtual ~DistanceComputer() = default;
    };

    struct L2DistanceComputer : public DistanceComputer {
        explicit L2DistanceComputer(const Storage* storage): storage(storage), query(nullptr) {}

        void compute_distance(storage_idx_t id, float& result) override {
            const float* y = storage->data + (id * storage->dim);
            l2_sqr_dist(query, y, storage->dim, result);
        }

        void compute_distance(storage_idx_t src, storage_idx_t dest, float& result) override {
            const float *x = storage->data + (src * storage->dim);
            const float *y = storage->data + (dest * storage->dim);
            l2_sqr_dist(x, y, storage->dim, result);
        }

        void set_query(const float* query) {
            this->query = query;
        }

    private:
        inline void l2_sqr_dist(const float* __restrict x, const float* __restrict y, size_t d, float& result) {
            float res = 0;
            for (size_t i = 0; i < d; i++) {
                float tmp = x[i] - y[i];
                res += tmp * tmp;
            }
            result = res;
        }
        // TODO: add more distance functions with simd support
    private:
        const float* query;
        const Storage* storage;
    };
} // namespace orangedb