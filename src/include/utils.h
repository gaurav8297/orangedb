#pragma once

#include <vector>

namespace orangedb {
    struct Utils {
        static float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out);
        static int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out);
        static void alloc_aligned(void **ptr, size_t size, size_t align);
    };
} // namespace orangedb