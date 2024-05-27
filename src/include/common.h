#pragma once

#include <sys/stat.h>
#include <iostream>
#include <sys/fcntl.h>
#include <random>
#include "spdlog/fmt/fmt.h"
#include <unordered_map>

#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)

namespace orangedb {
    typedef uint8_t level_t;
    typedef uint64_t vector_idx_t;
    constexpr vector_idx_t INVALID_VECTOR_ID = UINT64_MAX;

    [[noreturn]] inline void failCheckArgument(
            const char *condition_name, const char *file, int linenr, const char *comment) {
        throw std::invalid_argument(fmt::format(
                "Assertion failed in file \"{}\" on line {}: {} with comment: {}", file, linenr, condition_name,
                comment));
    }

#define CHECK_ARGUMENT(condition, comment)                                                            \
    static_cast<bool>(condition) ?                                                                    \
        void(0) :                                                                                     \
        orangedb::failCheckArgument(#condition, __FILE__, __LINE__, static_cast<const char*>(comment))

    static void allocAligned(void **ptr, size_t size, size_t align) {
        *ptr = nullptr;
        if (!IS_ALIGNED(size, align)) {
            printf("size: %lu, align: %lu\n", size, align);
            throw;
        }
#ifdef __APPLE__
        int err = posix_memalign(ptr, align, size);
        if (err) {
            printf("posix_memalign failed with error code %d\n", err);
            throw;
        }
#else
        *ptr = ::aligned_alloc(align, size);
#endif
        if (*ptr == nullptr) {
            printf("aligned_alloc failed\n");
            throw;
        }
    }

    static float *readFvecFile(const char *fName, size_t *d_out, size_t *n_out) {
        FILE *f = fopen(fName, "r");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fName);
            perror("");
            abort();
        }
        int d;
        fread(&d, 1, sizeof(int), f);
        CHECK_ARGUMENT((d > 0 && d < 1000000), "unreasonable dimension");
        fseek(f, 0, SEEK_SET);
        struct stat st{};
        fstat(fileno(f), &st);
        size_t sz = st.st_size;
        CHECK_ARGUMENT(sz % ((d + 1) * 4) == 0, "weird file size");
        size_t n = sz / ((d + 1) * 4);
        *d_out = d;
        *n_out = n;
        auto *x = new float[n * (d + 1)];
        printf("x: %p\n", x);
        size_t nr = fread(x, sizeof(float), n * (d + 1), f);
        CHECK_ARGUMENT(nr == n * (d + 1), "could not read whole file");

        // TODO: Round up the dimensions to the nearest multiple of 8, otherwise the below code will not work
        float *align_x;
        allocAligned(((void **) &align_x), n * d * sizeof(float), 8 * sizeof(float));
        printf("align_x: %p\n", align_x);

        // copy data to aligned memory
        for (size_t i = 0; i < n; i++) {
            memcpy(align_x + i * d, x + 1 + i * (d + 1), d * sizeof(float));
        }

        // free original memory
        delete[] x;
        fclose(f);
        return align_x;
    }

    static int *readIvecFile(const char *fName, size_t *d_out, size_t *n_out) {
        return (int *) readFvecFile(fName, d_out, n_out);
    }

    struct RandomGenerator {
        RandomGenerator(int seed) : mt(seed) {};

        inline float randFloat() {
            return mt() / float(mt.max());
        }

        inline int randInt(int max) {
            return mt() % max;
        }

        inline void randomPerm(int n, int *perm, int nPerm) {
            CHECK_ARGUMENT(nPerm <= n, "Number of permutations should be less than the number of elements");
            std::unordered_map<int, int> m;
            for (int i = 0; i < nPerm - 1; i++) {
                auto i2 = i + randInt(n - i);
                if (m.contains(i2)) {
                    perm[i] = m[i2];
                } else {
                    perm[i] = i2;
                }
                m[i2] = i;
            }

            // last element
            if (m.contains(nPerm - 1)) {
                perm[nPerm - 1] = m[nPerm - 1];
            } else {
                perm[nPerm - 1] = nPerm - 1;
            }
        }

        std::mt19937 mt;
    };
} // namespace orange
