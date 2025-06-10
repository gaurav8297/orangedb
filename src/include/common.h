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

    static float *readBvecFile(const char *fName, size_t *d_out, size_t *n_out, size_t max_rows = SIZE_MAX) {
        // Open the file in binary mode
        FILE *f = fopen(fName, "rb");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fName);
            perror("");
            abort();
        }

        // Read the dimension (first 4 bytes are the dimension in bvecs)
        int d;
        fread(&d, 1, sizeof(int), f);
        CHECK_ARGUMENT((d > 0 && d < 1000000), "unreasonable dimension");

        // Go back to the start of the file
        fseek(f, 0, SEEK_SET);

        // Get the file size to calculate the number of vectors
        struct stat st{};
        fstat(fileno(f), &st);
        size_t sz = st.st_size;
        CHECK_ARGUMENT(sz % (4 + d * sizeof(uint8_t)) == 0, "weird file size");

        // Calculate the total number of vectors and apply the limit
        size_t total_n = sz / (4 + d * sizeof(uint8_t)); // Total number of vectors
        size_t n = (total_n > max_rows) ? max_rows : total_n; // Limit the number of vectors to max_rows
        *d_out = d;
        *n_out = n;

        // Allocate memory for the original uint8_t data (including dimension prefix)
        auto *x = new uint8_t[n * (4 + d)];
        printf("x: %p\n", x);
        size_t nr = fread(x, sizeof(uint8_t), n * (4 + d), f);
        CHECK_ARGUMENT(nr == n * (4 + d), "could not read whole file");

        // Allocate aligned memory for the float data
        float *align_x;
        allocAligned((void **) &align_x, n * d * sizeof(float), 8 * sizeof(float));
        printf("align_x: %p\n", align_x);

        // Convert uint8_t data to float and copy to aligned memory
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                align_x[i * d + j] = static_cast<float>(x[4 + i * (4 + d) + j]); // Skip first 4 bytes (dimension)
            }
        }

        // Free original uint8_t data
        delete[] x;
        fclose(f);
        return align_x;
    }

    static void writeBvecFile(const char *fName, const float *data, size_t d, size_t n) {
        // Open the file in binary write mode
        FILE *f = fopen(fName, "wb");
        if (!f) {
            fprintf(stderr, "could not open %s for writing\n", fName);
            perror("");
            abort();
        }

        // Allocate a temporary buffer for storing uint8_t values
        auto *buffer = new uint8_t[(d + 4) * n]; // 4 bytes for dimension + d bytes for vector values per vector

        // Fill the buffer with dimension + vector data
        for (size_t i = 0; i < n; i++) {
            // Store the dimension (4 bytes)
            int dimension = static_cast<int>(d);
            memcpy(&buffer[i * (d + 4)], &dimension, sizeof(int));

            // Convert float data to uint8_t and store in buffer
            for (size_t j = 0; j < d; j++) {
                // Convert float to uint8_t (clamp to [0, 255] range)
                float value = data[i * d + j];
                buffer[i * (d + 4) + 4 + j] = static_cast<uint8_t>(value < 0.0f ? 0 : (value > 255.0f ? 255 : value));
            }
        }

        // Write the buffer to the file
        size_t bytes_written = fwrite(buffer, sizeof(uint8_t), n * (d + 4), f);
        CHECK_ARGUMENT(bytes_written == n * (d + 4), "could not write whole file");

        // Free the buffer and close the file
        delete[] buffer;
        fclose(f);
    }

    static float *readVecFile(const char *fName, size_t *d_out, size_t *n_out, size_t max_rows = SIZE_MAX) {
        bool is_bvecs = false;
        if (strstr(fName, ".bvecs")) {
            is_bvecs = true;
        }
        if (is_bvecs) {
            return readBvecFile(fName, d_out, n_out, max_rows);
        } else {
            return readFvecFile(fName, d_out, n_out);
        }
    }

    static int *readIvecFile(const char *fName, size_t *d_out, size_t *n_out) {
        return (int *) readFvecFile(fName, d_out, n_out);
    }

    static float *readFbinFile(const char *fName, size_t *d_out, size_t *n_out) {
        FILE *f = fopen(fName, "rb");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fName);
            perror("");
            abort();
        }
        // Read num of vecs
        int n_int;
        fread(&n_int, sizeof(int), 1, f);
        // Read dimension
        int d_int;
        fread(&d_int, sizeof(int), 1, f);
        *d_out = d_int;
        *n_out = n_int;

        auto *x = new float[n_int * d_int];
        size_t bytes_read = fread(x, sizeof(float), n_int * d_int, f);
        CHECK_ARGUMENT(bytes_read == n_int * d_int, "could not read whole file");

        fclose(f);
        return x;
    }

    static void writeFbinFile(const char *fName, const float *data, size_t d, size_t n) {
        FILE *f = fopen(fName, "wb");
        if (!f) {
            fprintf(stderr, "could not open %s for writing\n", fName);
            perror("");
            abort();
        }
        // Write num of vecs
        int n_int = static_cast<int>(n);
        fwrite(&n_int, sizeof(int), 1, f);
        // Write dimension
        int d_int = static_cast<int>(d);
        fwrite(&d_int, sizeof(int), 1, f);
        // Write data
        size_t bytes_written = fwrite(data, sizeof(float), n * d, f);
        CHECK_ARGUMENT(bytes_written == n * d, "could not write whole file");

        // Close the file
        fclose(f);
    }

    struct RandomGenerator {
        RandomGenerator(int seed) : mt(seed) {
        };

        inline float randFloat() {
            return mt() / float(mt.max());
        }

        inline int randInt(int max) {
            return mt() % max;
        }

        inline void randomPerm(uint64_t n, uint64_t *perm, uint64_t nPerm) {
            CHECK_ARGUMENT(nPerm <= n, "Number of permutations should be less than the number of elements");
            std::unordered_map<uint64_t, uint64_t> m;
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


#if _SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("+simd")
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)

    inline static float compute_normalized_factor_neon(const float *vector, int dim) {
        float32x4_t sum_vec = vdupq_n_f32(0.0f); // Initialize sum vector to 0
        int i = 0;
        // Process 4 elements at a time
        for (; i + 4 <= dim; i += 4) {
            float32x4_t vec = vld1q_f32(vector + i); // Load 4 elements
            sum_vec = vfmaq_f32(sum_vec, vec, vec); // Square each element
        }
        // Horizontal addition of sum_vec components
        float sum = vaddvq_f32(sum_vec);

        // Handle the remaining elements (if dim is not divisible by 4)
        for (; i < dim; i++) {
            sum += vector[i] * vector[i];
        }

        return 1.0f / std::sqrt(sum); // Compute the normalization factor
    }

    inline static void normalize_vectors_neon(const float *vector, int dim, float *normalized_vector) {
        float norm = compute_normalized_factor_neon(vector, dim);
        float32x4_t norm_vec = vdupq_n_f32(norm); // Create a vector with the normalization factor
        int i = 0;
        // Process 4 elements at a time
        for (; i + 4 <= dim; i += 4) {
            float32x4_t vec = vld1q_f32(vector + i); // Load 4 elements
            float32x4_t normed_vec = vmulq_f32(vec, norm_vec); // Normalize the vector
            vst1q_f32(normalized_vector + i, normed_vec); // Store the normalized vector
        }

        // Handle the remaining elements (if dim is not divisible by 4)
        for (; i < dim; i++) {
            normalized_vector[i] = vector[i] * norm;
        }
    }

#pragma clang attribute pop
#pragma GCC pop_options
#endif
#endif

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "fma")
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)

inline static float compute_normalized_factor_haswell(const float *vector, int dim) {
    __m256 sum_vec = _mm256_setzero_ps();  // Initialize sum vector to 0
    int i = 0;

    // Process 8 elements at a time using AVX2
    for (; i + 8 <= dim; i += 8) {
        __m256 vec = _mm256_loadu_ps(vector + i);
        // Use FMA for multiply-add operation: sum += vec * vec
        sum_vec = _mm256_fmadd_ps(vec, vec, sum_vec);
    }

    // Reduce with double precision for better accuracy
    double sum = _simsimd_reduce_f32x8_haswell(sum_vec);

    // Handle remaining elements in double precision
    for (; i < dim; i++) {
        sum += vector[i] * vector[i];
    }

    return 1.0 / std::sqrt(sum);
}

inline static void normalize_vectors_haswell(const float *vector, int dim, float *normalized_vector) {
    float norm = compute_normalized_factor_haswell(vector, dim);
    __m256 norm_vec = _mm256_set1_ps(norm);  // Broadcast norm to all elements
    int i = 0;

    // Process 8 elements at a time
    for (; i + 8 <= dim; i += 8) {
        __m256 vec = _mm256_loadu_ps(vector + i);
        __m256 normed_vec = _mm256_mul_ps(vec, norm_vec);
        _mm256_storeu_ps(normalized_vector + i, normed_vec);
    }

    // Handle remaining elements
    for (; i < dim; i++) {
        normalized_vector[i] = vector[i] * norm;
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "bmi2")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,bmi2"))), apply_to = function)
    inline static float compute_normalized_factor_skylake(const float *vector, int dim) {
        __m512 sum_vec = _mm512_setzero_ps();  // Initialize sum vector to 0
        int i = 0;

        for (; i + 16 <= dim; i += 16) {
            __m512 vec = _mm512_loadu_ps(vector + i);     // Load 16 elements
            sum_vec= _mm512_fmadd_ps(vec, vec, sum_vec);     // Square each element
        }

        float sum = _mm512_reduce_add_ps(sum_vec);

        for (; i < dim; i++) {
            sum += vector[i] * vector[i];
        }

        return 1.0f / std::sqrt(sum); // Compute the normalization factor
    }

    inline static void normalize_vectors_skylake(const float *vector, int dim, float *normalized_vector) {
        float norm = compute_normalized_factor_skylake(vector, dim);
        __m512 norm_vec = _mm512_set1_ps(norm); // Broadcast norm to all elements
        int i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 vec = _mm512_loadu_ps(vector + i);
            __m512 normed_vec = _mm512_mul_ps(vec, norm_vec);
            _mm512_storeu_ps(normalized_vector + i, normed_vec);
        }
        for (; i < dim; i++) {
            normalized_vector[i] = vector[i] * norm;
        }
    }
#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE
#endif // SIMSIMD_TARGET_X86


    // Normalize the vectors
    inline static float compute_normalized_factor(const float *vector, int dim) {
#if SIMSIMD_TARGET_NEON
        return compute_normalized_factor_neon(vector, dim);
#elif SIMSIMD_TARGET_SKYLAKE
            return compute_normalized_factor_skylake(vector, dim);
#elif SIMSIMD_TARGET_HASWELL
            return compute_normalized_factor_haswell(vector, dim);
#else
            float norm = 0;
            for (int i = 0; i < dim; i++) {
                norm += vector[i] * vector[i];
            }
            return 1.0f / std::sqrt(norm);
#endif
    }

    inline static void normalize_vector(const float *vector, int dim, float *normalized_vector) {
#if SIMSIMD_TARGET_NEON
        normalize_vectors_neon(vector, dim, normalized_vector);
#elif SIMSIMD_TARGET_SKYLAKE
            normalize_vectors_skylake(vector, dim, normalized_vector);
#elif SIMSIMD_TARGET_HASWELL
            normalize_vectors_haswell(vector, dim, normalized_vector);
#else
            float norm = compute_normalized_factor(vector, dim);
            for (int i = 0; i < dim; i++) {
                normalized_vector[i] = vector[i] * norm;
            }
#endif
    }

    inline static void normalize_vectors(const float *vector, int dim, int n, float *normalized_vector) {
        for (int i = 0; i < n; i++) {
            normalized_vector(vector + i * dim, dim, normalized_vector + i * dim);
        }
    }
} // namespace orange
