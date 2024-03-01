#pragma once

#include <unistd.h>

using namespace std;

namespace orangedb {
    struct Distance {
        inline void l2_sqr_dist(const float* __restrict x, const float* __restrict y, size_t d, float* result) {
            float res = 0;
            for (size_t i = 0; i < d; i++) {
                float tmp = x[i] - y[i];
                res += tmp * tmp;
            }
            *result = res;
        }
        // TODO: add more distance functions with simd support
    };
} // namespace orangedb