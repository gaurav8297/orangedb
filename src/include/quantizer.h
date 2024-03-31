#pragma once

#include <stdint.h>
#include <cstdio>

namespace orangedb {
    struct Quantizer {
        explicit Quantizer(size_t d = 0, size_t code_size = 0)
                : d(d), code_size(code_size) {};
        virtual void train(size_t n, const float* x) = 0;
        virtual void compute_codes(const float* x, uint8_t* codes, size_t n) const = 0;
        virtual void decode(const uint8_t* code, float* x, size_t n) const = 0;
        size_t d;
        size_t code_size;
    };
} // namespace orangedb
