#pragma once

#include <vector>
#include "quantizer.h"
#include "utils.h"


namespace orangedb {
    struct ScalarQuantizer : Quantizer {
        explicit ScalarQuantizer(size_t d, float* vmin, float* vdiff) : Quantizer(d, 8) {
            Utils::alloc_aligned((void**)&vmin, d * sizeof(float), 64);
            Utils::alloc_aligned((void**)&vdiff, d * sizeof(float), 64);
            this->vmin = vmin;
            this->vdiff = vdiff;
        };

        void train(size_t n, const float* x);
        void compute_codes(const float* x, uint8_t* codes, size_t n) const;
        void decode(const uint8_t* code, float* x, size_t n) const;
        void print_stats();

        float rangestat_arg = 0;
        float* vmin;
        float* vdiff;
    };
} // namespace orangedb

