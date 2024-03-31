#pragma once

#include <vector>
#include "quantizer.h"


namespace orangedb {
    struct ScalarQuantizer : Quantizer {
        explicit ScalarQuantizer(size_t d) : Quantizer(d, 8) {};

        void train(size_t n, const float* x);
        void compute_codes(const float* x, uint8_t* codes, size_t n) const;
        void decode(const uint8_t* code, float* x, size_t n) const;
        void print_stats();

        float rangestat_arg = 0;
        std::vector<float> trained;
    };
} // namespace orangedb

