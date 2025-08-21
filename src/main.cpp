#include <iostream>
#include "hnsw.h"
#include "spdlog/fmt/fmt.h"

#ifdef __AVX2__
#include <x86intrin.h>
#endif

#include <stdlib.h>    // atoi, getenv
#include <assert.h>    // assert
#include <simsimd/simsimd.h>
#include "include/partitioned_index.h"
#include <fstream>
#include <reclustering_index.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <fastQ/scalar_8bit.h>
#include <fastQ/pair_wise.h>
#include "helper_ds.h"
#include <fastQ/common.h>
#include <nlohmann/json.hpp>

#include "construction.h"
#include "incremental_index.h"
#include "iRG_search.h"
#include "utils.h"
// #include "faiss/IndexACORN.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIVFFlat.h"
#include "fastQ/scalar_test.h"
#include "faiss/IndexPQ.h"

#if 0
#include <liburing.h>
#endif

using namespace orangedb;

#if defined(__GNUC__)
#define PRAGMA_IMPRECISE_LOOP
#define PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("GCC push_options") \
    _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define PRAGMA_IMPRECISE_FUNCTION_END \
    _Pragma("GCC pop_options")
#endif
#include <backward.hpp>


class InputParser {
public:
    InputParser(int &argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            this->tokens.emplace_back(argv[i]);
        }
    }

    const std::string &getCmdOption(const std::string &option) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
            return *itr;
        }
        static const std::string emptyString;
        return emptyString;
    }

private:
    std::vector<std::string> tokens;
};

void exp_omp_lock() {
    omp_set_num_threads(8);
    auto n = 1000000;
    std::vector<omp_lock_t> locks(100000);
    for (int i = 0; i < n; i++) {
        omp_init_lock(&locks[i]);
    }

    std::atomic<int> x = 0;
#pragma omp parallel for
    for (int i = 1; i < n; i++) {
        omp_set_lock(&locks[i]);
        omp_set_lock(&locks[i]);
        x++;
        omp_unset_lock(&locks[i]);
        omp_unset_lock(&locks[i]);
    }


    printf("x = %d\n", x.load());


    for (int i = 0; i < n; i++) {
        omp_destroy_lock(&locks[i]);
    }
}

#ifdef __AVX2__
void l2_sqr_dist(const float* __restrict x, const float* __restrict y, size_t d, float& result) {
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm256_loadu_ps(addr1);                  \
  tmp2 = _mm256_loadu_ps(addr2);                  \
  tmp1 = _mm256_sub_ps(tmp1, tmp2);               \
  tmp1 = _mm256_mul_ps(tmp1, tmp1);               \
  dest = _mm256_add_ps(dest, tmp1);

    __m256 sum;
    __m256 l0, l1;
    __m256 r0, r1;
    size_t qty16 = d >> 4;
    size_t aligned_size = qty16 << 4;
    const float *l = x;
    const float *r = y;

    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm256_loadu_ps(unpack);
    AVX_L2SQR(l, r, sum, l0, r0);

    for (unsigned i = 0; i < aligned_size; i += 32, l += 32, r += 32) {
        AVX_L2SQR(l, r, sum, l0, r0);
        AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
        AVX_L2SQR(l + 16, r + 16, sum, l0, l0);
        AVX_L2SQR(l + 24, r + 24, sum, l1, r1);
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
    for (unsigned i = aligned_size; i < d; ++i, ++l, ++r) {
        float diff = *l - *r;
        result += diff * diff;
    }
}

inline void l1_dist(const float* __restrict x, const float* __restrict y, size_t d, float& result) {
#define AVX_L1(addr1, addr2, dest, tmp1, tmp2, sign_bit) \
  tmp1 = _mm256_loadu_ps(addr1);                  \
  tmp2 = _mm256_loadu_ps(addr2);                  \
  tmp1 = _mm256_sub_ps(tmp1, tmp2);               \
  dest = _mm256_add_ps(dest, tmp1);
//  tmp1 = _mm256_andnot_ps(sign_bit, tmp1);               \

    __m256 sum;
    __m256 l0, l1;
    __m256 r0, r1;
    size_t qty16 = d >> 4;
    size_t aligned_size = qty16 << 4;
    const float *l = x;
    const float *r = y;

    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm256_loadu_ps(unpack);
    __m256 sign_bit = _mm256_set1_ps(-0.0f);

    for (unsigned i = 0; i < aligned_size; i += 16, l += 16, r += 16) {
        AVX_L1(l, r, sum, l0, r0, sign_bit);
        AVX_L1(l + 8, r + 8, sum, l1, r1, sign_bit);
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
}
#else

void l2_sqr_dist(const float *__restrict x, const float *__restrict y, size_t d, float &result) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    result = res;
}

#endif

PRAGMA_IMPRECISE_FUNCTION_BEGIN

inline void fvec_L2sqr_batch_4(
        const float *__restrict x,
        const float *__restrict y0,
        const float *__restrict y1,
        const float *__restrict y2,
        const float *__restrict y3,
        const size_t d,
        float &dis0,
        float &dis1,
        float &dis2,
        float &dis3) {
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
    PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        const float q0 = x[i] - y0[i];
        const float q1 = x[i] - y1[i];
        const float q2 = x[i] - y2[i];
        const float q3 = x[i] - y3[i];
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}

PRAGMA_IMPRECISE_FUNCTION_END


PRAGMA_IMPRECISE_FUNCTION_BEGIN

inline void fvec_L1_batch_4(
        const float *__restrict x,
        const float *__restrict y0,
        const float *__restrict y1,
        const float *__restrict y2,
        const float *__restrict y3,
        const size_t d,
        float &dis0,
        float &dis1,
        float &dis2,
        float &dis3) {
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
    PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        const float q0 = x[i] - y0[i];
        const float q1 = x[i] - y1[i];
        const float q2 = x[i] - y2[i];
        const float q3 = x[i] - y3[i];
        d0 += fabs(q0);
        d1 += fabs(q1);
        d2 += fabs(q2);
        d3 += fabs(q3);
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}

PRAGMA_IMPRECISE_FUNCTION_END

PRAGMA_IMPRECISE_FUNCTION_BEGIN

inline void fvec_L2sqr_batch_8(
        const float *__restrict x,
        const float *__restrict y0,
        const float *__restrict y1,
        const float *__restrict y2,
        const float *__restrict y3,
        const float *__restrict y4,
        const float *__restrict y5,
        const float *__restrict y6,
        const float *__restrict y7,
        const size_t d,
        float &dis0,
        float &dis1,
        float &dis2,
        float &dis3,
        float &dis4,
        float &dis5,
        float &dis6,
        float &dis7) {
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
    float d4 = 0;
    float d5 = 0;
    float d6 = 0;
    float d7 = 0;
    PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        const float q0 = x[i] - y0[i];
        const float q1 = x[i] - y1[i];
        const float q2 = x[i] - y2[i];
        const float q3 = x[i] - y3[i];
        const float q4 = x[i] - y4[i];
        const float q5 = x[i] - y5[i];
        const float q6 = x[i] - y6[i];
        const float q7 = x[i] - y7[i];
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
        d4 += q4 * q4;
        d5 += q5 * q5;
        d6 += q6 * q6;
        d7 += q7 * q7;
    }
    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
    dis4 = d4;
    dis5 = d5;
    dis6 = d6;
    dis7 = d7;
}

PRAGMA_IMPRECISE_FUNCTION_END

#ifdef __AVX2__
inline void fvec_L2sqr_batch_4_vec(
        const float* __restrict x,
        const float* __restrict y0,
        const float* __restrict y1,
        const float* __restrict y2,
        const float* __restrict y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm256_loadu_ps(addr1);                  \
  tmp2 = _mm256_loadu_ps(addr2);                  \
  tmp1 = _mm256_sub_ps(tmp1, tmp2);               \
  tmp1 = _mm256_mul_ps(tmp1, tmp1);               \
  dest = _mm256_add_ps(dest, tmp1);

    __m256 sum0, sum1, sum2, sum3;
    __m256 l0, l1, l2, l3;
    __m256 r0, r1, r2, r3;
    size_t qty16 = d >> 4;
    size_t aligned_size = qty16 << 4;
    const float *l = x;
    const float *m0 = y0;
    const float *m1 = y1;
    const float *m2 = y2;
    const float *m3 = y3;

    float unpack0[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    sum0 = _mm256_loadu_ps(unpack0);
    float unpack1[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    sum1 = _mm256_loadu_ps(unpack1);
    float unpack2[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    sum2 = _mm256_loadu_ps(unpack2);
    float unpack3[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    sum3 = _mm256_loadu_ps(unpack3);

    for (unsigned i = 0; i < aligned_size; i += 16, l += 16, m0 += 16, m1 += 16, m2 += 16, m3 += 16) {
        AVX_L2SQR(l, m0, sum0, l0, r0);
        AVX_L2SQR(l + 8, m0 + 8, sum0, l0, r0);

        AVX_L2SQR(l, m1, sum1, l1, r1);
        AVX_L2SQR(l + 8, m1 + 8, sum1, l1, r1);

        AVX_L2SQR(l, m2, sum2, l2, r2);
        AVX_L2SQR(l + 8, m2 + 8, sum2, l2, r2);

        AVX_L2SQR(l, m3, sum3, l3, r3);
        AVX_L2SQR(l + 8, m3 + 8, sum3, l3, r3);
    }
    _mm256_storeu_ps(unpack0, sum0);
    dis0 = unpack0[0] + unpack0[1] + unpack0[2] + unpack0[3] + unpack0[4] + unpack0[5] + unpack0[6] + unpack0[7];
    _mm256_storeu_ps(unpack1, sum1);
    dis1 = unpack1[0] + unpack1[1] + unpack1[2] + unpack1[3] + unpack1[4] + unpack1[5] + unpack1[6] + unpack1[7];
    _mm256_storeu_ps(unpack2, sum2);
    dis2 = unpack2[0] + unpack2[1] + unpack2[2] + unpack2[3] + unpack2[4] + unpack2[5] + unpack2[6] + unpack2[7];
    _mm256_storeu_ps(unpack3, sum3);
    dis3 = unpack3[0] + unpack3[1] + unpack3[2] + unpack3[3] + unpack3[4] + unpack3[5] + unpack3[6] + unpack3[7];
//    for (unsigned i = aligned_size; i < d; ++i, ++l, ++r) {
//        float diff = *l - *r;
//        result += diff * diff;
//    }
}
#endif


int64_t exp_l1_sqr_dist(const float *baseVecs, size_t baseDimension, size_t baseNumVectors) {
    auto start = std::chrono::high_resolution_clock::now();
    float res = 0;
    const float *query = baseVecs;
    for (size_t i = 1; i < baseNumVectors - 4; i += 4) {
        float res0, res1, res2, res3;
//        fvec_L2sqr_batch_4_vec(
//                query,
//                baseVecs + (i * baseDimension),
//                baseVecs + ((i+1) * baseDimension),
//                baseVecs + ((i+2) * baseDimension),
//                baseVecs + ((i+3) * baseDimension),
//                baseDimension,
//                res0,
//                res1,
//                res2,
//                res3);

        l2_sqr_dist(query, baseVecs + (i * baseDimension), baseDimension, res0);
        l2_sqr_dist(query, baseVecs + ((i + 1) * baseDimension), baseDimension, res1);
        l2_sqr_dist(query, baseVecs + ((i + 2) * baseDimension), baseDimension, res2);
        l2_sqr_dist(query, baseVecs + ((i + 3) * baseDimension), baseDimension, res3);
        res += (res0 + res1 + res2 + res3);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Result: %f\n", res);
    return duration;
}


int64_t exp_l2_sqr_dist(const float *baseVecs, size_t baseDimension, size_t baseNumVectors) {
    auto start = std::chrono::high_resolution_clock::now();
    float res = 0;
    const float *query = baseVecs;
    for (size_t i = 1; i < 200000 - 4; i += 4) {
        float res0, res1, res2, res3;
//        fvec_L2sqr_batch_4_vec(
//                query,
//                baseVecs + (i * baseDimension),
//                baseVecs + ((i+1) * baseDimension),
//                baseVecs + ((i+2) * baseDimension),
//                baseVecs + ((i+3) * baseDimension),
//                baseDimension,
//                res0,
//                res1,
//                res2,
//                res3);

        l2_sqr_dist(query, baseVecs + (i * baseDimension), baseDimension, res0);
        l2_sqr_dist(query, baseVecs + ((i + 1) * baseDimension), baseDimension, res1);
        l2_sqr_dist(query, baseVecs + ((i + 2) * baseDimension), baseDimension, res2);
        l2_sqr_dist(query, baseVecs + ((i + 3) * baseDimension), baseDimension, res3);
        res += (res0 + res1 + res2 + res3);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Result: %f\n", res);
    return duration;
}

int64_t exp_l2_sqr_dist_2(const float *baseVecs, size_t baseDimension, size_t baseNumVectors) {
    auto start = std::chrono::high_resolution_clock::now();
    float res = 0;
    const float *query = baseVecs;
    for (size_t i = 1; i < baseNumVectors - 4; i += 4) {
        float res0, res1, res2, res3;
        fvec_L2sqr_batch_4(
                query,
                baseVecs + (i * baseDimension),
                baseVecs + ((i + 1) * baseDimension),
                baseVecs + ((i + 2) * baseDimension),
                baseVecs + ((i + 3) * baseDimension),
                baseDimension,
                res0,
                res1,
                res2,
                res3);

//        l2_sqr_dist(query, baseVecs + (i * baseDimension), baseDimension, res0);
//        l2_sqr_dist(query, baseVecs + ((i+1) * baseDimension), baseDimension, res1);
//        l2_sqr_dist(query, baseVecs + ((i+2) * baseDimension), baseDimension, res2);
//        l2_sqr_dist(query, baseVecs + ((i+3) * baseDimension), baseDimension, res3);
        res += (res0 + res1 + res2 + res3);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Result: %f\n", res);
    return duration;
}

// Try beam search with SIMD (multiple vectors at the same time)
// - Record the number of vector comparisons

void random_vector_access_exp(
        float *baseVecs,
        size_t baseDimension,
        size_t baseNumVectors,
        size_t nTimes,
        size_t resetQueryAfter) {
    size_t nQueries = (nTimes / resetQueryAfter) + 50;
    printf("Number of queries: %zu\n", nQueries);

    // Get random number between 0 and baseNumVectors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> distribution(0, (baseNumVectors - 1));

//    std::vector<uint64_t> random_vector_ids(nTimes);
//    for (int i = 0; i < nTimes; i++) {
//        random_vector_ids[i] = distribution(gen);
//    }

//    std::vector<size_t> random_query_ids(nQueries);
//    for (int i = 0; i < nQueries; i++) {
//        random_query_ids[i] = distribution(gen);
//    }

    printf("Start benchmark !!!\n");
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel
    {
        float *query = baseVecs + (distribution(gen) * baseDimension);
        float result = 0;
        int j = 0;
#pragma omp for
        for (size_t i = 0; i < nTimes; i += 4) {
            if (j == resetQueryAfter) {
                j = 0;
                query = baseVecs + (distribution(gen) * baseDimension);
            }
            j++;
//            size_t query_idx = i / resetQueryAfter;
//            spdlog::warn("Query idx: {}", query_idx);
//            query = baseVecs + (random_query_ids[query_idx] * baseDimension);
            float res0, res1, res2, res3;
            fvec_L2sqr_batch_4(
                    query,
                    baseVecs + (distribution(gen) * baseDimension),
                    baseVecs + (distribution(gen) * baseDimension),
                    baseVecs + (distribution(gen) * baseDimension),
                    baseVecs + (distribution(gen) * baseDimension),
                    baseDimension,
                    res0,
                    res1,
                    res2,
                    res3);
            result += res0 + res1 + res2 + res3;
        }
        printf("Result: %f\n", result);
    }
//
//    float result = 0;
//    for (size_t i = 0; i < nTimes; i+=8) {
//        int query_idx = i / resetQueryAfter;
//        query = baseVecs + (random_query_ids[query_idx] * baseDimension);
//        float res0, res1, res2, res3;
//        fvec_L2sqr_batch_4(
//                query,
//                baseVecs + (random_vector_ids[i] * baseDimension),
//                baseVecs + (random_vector_ids[i+1] * baseDimension),
//                baseVecs + (random_vector_ids[i+2] * baseDimension),
//                baseVecs + (random_vector_ids[i+3] * baseDimension),
//                baseDimension,
//                res0,
//                res1,
//                res2,
//                res3);
//        result += res0 + res1 + res2 + res3;
//        float res0 = 0, res1 = 0, res2 = 0, res3 = 0, res4 = 0, res5 = 0, res6 = 0, res7 = 0;
//        fvec_L2sqr_batch_8(
//                query,
//                baseVecs + (random_vector_ids[i] * baseDimension),
//                baseVecs + (random_vector_ids[i+1] * baseDimension),
//                baseVecs + (random_vector_ids[i+2] * baseDimension),
//                baseVecs + (random_vector_ids[i+3] * baseDimension),
//                baseVecs + (random_vector_ids[i+4] * baseDimension),
//                baseVecs + (random_vector_ids[i+5] * baseDimension),
//                baseVecs + (random_vector_ids[i+6] * baseDimension),
//                baseVecs + (random_vector_ids[i+7] * baseDimension),
//                baseDimension,
//                res0,
//                res1,
//                res2,
//                res3,
//                res4,
//                res5,
//                res6,
//                res7);
//        result += res0 + res1 + res2 + res3 + res4 + res5 + res6 + res7;
//        fvec_L2sqr_batch_8(
//                query,
//                baseVecs + (random_numbers[i+8] * baseDimension),
//                baseVecs + (random_numbers[i+9] * baseDimension),
//                baseVecs + (random_numbers[i+10] * baseDimension),
//                baseVecs + (random_numbers[i+11] * baseDimension),
//                baseVecs + (random_numbers[i+12] * baseDimension),
//                baseVecs + (random_numbers[i+13] * baseDimension),
//                baseVecs + (random_numbers[i+14] * baseDimension),
//                baseVecs + (random_numbers[i+15] * baseDimension),
//                baseDimension,
//                res9,
//                res10,
//                res11,
//                res12,
//                res4,
//                res13,
//                res14,
//                res15);
//        float res;
//        l2_sqr_dist(query, baseVecs + (random_vector_ids[i] * baseDimension), baseDimension, res);
//        result += res;
//    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//    printf("Result: %f\n", result);
    printf("Duration: %lld ms\n", duration);
    return;
}

void benchmark_random_dist_comp() {
    auto basePath = "/home/g3sehgal/vector_index_exp/gist";
    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readFvecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    printf("Base dimension: %zu, Base num vectors: %zu\n", baseDimension, baseNumVectors);
    omp_set_num_threads(32);

    random_vector_access_exp(baseVecs, baseDimension, baseNumVectors, 3700000000, 3000);
}

void benchmark_simd_distance() {
    auto basePath = "/home/gaurav/vector_index_experiments/vector_index/data/gist_200k";
    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readFvecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    printf("Base dimension: %zu, Base num vectors: %zu\n", baseDimension, baseNumVectors);

    int64_t duration = 0;
    for (int i = 0; i < 100; i++) {
        duration += exp_l1_sqr_dist(baseVecs, baseDimension, baseNumVectors);
    }
    int64_t avg_dur = duration;
    printf("Avg furation: %ld ms\n", avg_dur / 100);

//    duration = 0;
//    for (int i =0; i < 100; i++) {
//        duration += exp_l1_sqr_dist(baseVecs, baseDimension, baseNumVectors);
//    }
//    avg_dur = duration;
//    printf("Avg furation: %ld ms\n", avg_dur / 100);
}


void gen_random_vector(int size, std::vector<float> &random_floats) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < size; ++i) {
        random_floats[i] = dis(gen);
    }
}

void benchmark_n_simd(int64_t n) {
    std::vector<float> vec_1(960), vec_2(960);
    gen_random_vector(960, vec_1);
    gen_random_vector(960, vec_2);

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 5000000)
    for (int i = 0; i < n; i++) {
        float res;
        l2_sqr_dist(vec_1.data(), vec_2.data(), 960, res);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Duration: %ld ms\n", duration);
}

void build_graph(HNSW &hnsw, float *baseVecs, size_t baseNumVectors) {
    auto start = std::chrono::high_resolution_clock::now();
    hnsw.build(baseVecs, baseNumVectors);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Building time: " << duration << " ms" << std::endl;
}

void query_graph_filter(
        HNSW &hnsw,
        const float *queryVecs,
        const uint8_t *filteredMask,
        size_t queryNumVectors,
        size_t queryDimension,
        const vector_idx_t *gtVecs,
        size_t k,
        size_t ef_search,
        size_t baseNumVectors) {
    auto visited = VisitedTable(baseNumVectors);
    auto recall = 0.0;
    Stats stats{};
    long time = 0;
    for (size_t i = 0; i < queryNumVectors; i++) {
        auto localRecall = 0.0;
        std::priority_queue<NodeDistCloser> results;
        std::vector<NodeDistFarther> res;
        auto start = std::chrono::high_resolution_clock::now();
        hnsw.searchWithFilter(queryVecs + (i * queryDimension), k, ef_search, visited, results, filteredMask + (i * baseNumVectors), stats);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        time += duration;
        while (!results.empty()) {
            auto top = results.top();
            res.emplace_back(top.id, top.dist);
            results.pop();
        }
        auto gt = gtVecs + i * k;
        for (auto &result: res) {
            if (std::find(gt, gt + k, result.id) != (gt + k)) {
                recall++;
                localRecall++;
            }
        }
    }
    auto recallPerQuery = recall / queryNumVectors;
    stats.logStats();
    std::cout << "Total Vectors: " << queryNumVectors << std::endl;
    std::cout << "Recall: " << (recallPerQuery / k) * 100 << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Query time: " << time << " ms" << std::endl;
}

void enable_perf() {
    int perf_ctl_fd;
    int perf_ctl_ack_fd;
    char ack[5];

    perf_ctl_fd = atoi(getenv("PERF_CTL_FD"));
    perf_ctl_ack_fd = atoi(getenv("PERF_CTL_ACK_FD"));

    write(perf_ctl_fd, "enable\n", 8);
    read(perf_ctl_ack_fd, ack, 5);
    assert(strcmp(ack, "ack\n") == 0);
}

void disable_perf() {
    int perf_ctl_fd;
    int perf_ctl_ack_fd;
    char ack[5];

    perf_ctl_fd = atoi(getenv("PERF_CTL_FD"));
    perf_ctl_ack_fd = atoi(getenv("PERF_CTL_ACK_FD"));

    write(perf_ctl_fd, "disable\n", 9);
    read(perf_ctl_ack_fd, ack, 5);
    assert(strcmp(ack, "ack\n") == 0);
}

void generateFilterGroundTruth(
        const float* vectors,
        size_t dim,
        size_t numVectors,
        float *queryVecs,
        const uint8_t* filteredMask,
        size_t queryNumVectors,
        int k,
        vector_idx_t *gtVecs) {
    auto dc = createDistanceComputer(vectors, dim, numVectors, COSINE);
#pragma omp parallel
    {
        auto localDc = dc->clone();
        IndexOneNN index(localDc.get(), dim, numVectors);
#pragma omp for schedule(static)
        for (size_t i = 0; i < queryNumVectors; i++) {
            double dists[k];
            index.knnFiltered(k, queryVecs + i * dim, dists, gtVecs + i * k, filteredMask);
        }
    }
}

void writeToFile(const std::string &path, const uint8_t *data, size_t size) {
    std::ofstream outputFile(path, std::ios::binary);
    outputFile.write(reinterpret_cast<const char *>(data), size);
    outputFile.close();
}

void loadFromFile(const std::string &path, uint8_t *data, size_t size) {
    std::ifstream inputFile(path, std::ios::binary);
    inputFile.read(reinterpret_cast<char *>(data), size);
    inputFile.close();
}

void setFilterMaskUsingSelectivity(
        size_t queryNumVectors,
        uint8_t* filteredMask,
        size_t numVectors,
        float selectivity) {
    std::random_device rd;
    std::mt19937 gen(rd());
    printf("Selectivity: %f\n", selectivity);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (size_t i = 0; i < queryNumVectors; i++) {
        for (size_t j = 0; j < numVectors; j++) {
            if (dis(gen) < selectivity) {
                filteredMask[i * numVectors + j] = 1;
            }
        }
    }
}

void populate_mask_and_gt_paths(const std::string &basePath, const std::vector<std::string> &sels,
                         std::vector<std::string> &maskPaths,
                         std::vector<std::string> &gtPath, std::string &queryPath) {
    // Parse the selectivities and efS strings
    for (const auto &sel : sels) {
        auto maskPath = fmt::format("{}/mask_{}.bin", basePath, sel);
        auto gtPathStr = fmt::format("{}/gt_{}.bin", basePath, sel);
        maskPaths.push_back(maskPath);
        gtPath.push_back(gtPathStr);
    }
    // Generate the query path
    queryPath = fmt::format("{}/queries.fvecs", basePath);
}

std::vector<std::string> parseCommaSeparated(const std::string& input) {
    std::vector<std::string> res;
    std::stringstream ss(input);
    std::string temp;

    while (std::getline(ss, temp, ',')) {
        res.push_back(temp);
    }

    return res;
}

void generateFilterGroundTruth(InputParser &input) {
    const std::string &dataPath = input.getCmdOption("-dataPath");
    const std::string &basePath = input.getCmdOption("-basePath");
    const std::vector<std::string> sels = parseCommaSeparated(input.getCmdOption("-sels"));
    auto k = stoi(input.getCmdOption("-k"));
    std::vector<std::string> maskPaths, gtPaths;
    std::string queryPath;
    populate_mask_and_gt_paths(basePath, sels, maskPaths, gtPaths, queryPath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(dataPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryPath.c_str(), &queryDimension, &queryNumVectors);
    printf("Base vectors: %zu, Query vectors: %zu\n", baseNumVectors, queryNumVectors);
    printf("Base dimension: %zu, Query dimension: %zu\n", baseDimension, queryDimension);
    auto *filteredMask = new uint8_t[baseNumVectors];
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    for (size_t i = 0; i < sels.size(); i++) {
        auto maskPath = maskPaths[i];
        auto gtPath = gtPaths[i];

        loadFromFile(maskPath, filteredMask, baseNumVectors);
        // Calculate selectivity from filteredMask
        size_t numFiltered = 0;
        for (int j = 0; j < baseNumVectors; j++) {
            if (filteredMask[j] == 1) {
                numFiltered++;
            }
        }
        float selectivity = (float) numFiltered / baseNumVectors;
        printf("Selectivity: %f\n", selectivity);
        generateFilterGroundTruth(baseVecs, baseDimension, baseNumVectors, queryVecs, filteredMask, queryNumVectors, k, gtVecs);
        printf("Writing gt to file: %s\n", gtPath.c_str());
        writeToFile(gtPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));
    }
}

void generateGroundTruth(
        const float* vectors,
        size_t dim,
        size_t numVectors,
        float *queryVecs,
        size_t queryNumVectors,
        int k,
        vector_idx_t *gtVecs) {
    auto dc = createDistanceComputer(vectors, dim, numVectors, COSINE);
#pragma omp parallel
    {
        auto localDc = dc->clone();
        IndexOneNN index(localDc.get(), dim, numVectors);
#pragma omp for schedule(dynamic, 100)
        for (size_t i = 0; i < queryNumVectors; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            double dists[k];
            index.knn(k, queryVecs + i * dim, dists, gtVecs + i * k);
            auto end = std::chrono::high_resolution_clock::now();
            printf("Query time: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        }
    }
}

void generateGroundTruthParquet(InputParser &input) {
    const std::string &dirPath = input.getCmdOption("-dirPath");
    const std::string &queryPath = input.getCmdOption("-queryPath");
    auto k = stoi(input.getCmdOption("-k"));
    auto numVectors = stoi(input.getCmdOption("-numVectors"));
    const std::string &gtPath = input.getCmdOption("-gtPath");

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readParquetDir(dirPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readFvecFile(queryPath.c_str(), &queryDimension, &queryNumVectors);
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    baseNumVectors = std::min(baseNumVectors, (size_t) numVectors);
    printf("Base num vectors: %zu, Query num vectors: %zu\n", baseNumVectors, queryNumVectors);
    generateGroundTruth(baseVecs, baseDimension, baseNumVectors, queryVecs, queryNumVectors, k, gtVecs);
    // serialize gtVecs to a file
    writeToFile(gtPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));
}

void generateGroundTruth(InputParser &input) {
    const std::string &basePath = input.getCmdOption("-dataPath");
    const std::string &queryPath = input.getCmdOption("-queryPath");
    auto k = stoi(input.getCmdOption("-k"));
    auto numVectors = stoi(input.getCmdOption("-numVectors"));
    const std::string &gtPath = input.getCmdOption("-gtPath");

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readFvecFile(basePath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readFvecFile(queryPath.c_str(), &queryDimension, &queryNumVectors);
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    baseNumVectors = std::min(baseNumVectors, (size_t) numVectors);
    generateGroundTruth(baseVecs, baseDimension, baseNumVectors, queryVecs, queryNumVectors, k, gtVecs);
    // serialize gtVecs to a file
    writeToFile(gtPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));
}

std::vector<int> parseCommaSeparatedIntegers(const std::string& input) {
    std::vector<int> numbers;
    std::stringstream ss(input);
    std::string temp;

    while (std::getline(ss, temp, ',')) {
        numbers.push_back(std::stoi(temp));
    }

    return numbers;
}

void benchmark_filtered_hnsw_queries(InputParser &input) {
    const std::string &basePath = input.getCmdOption("-basePath");
    auto efConstruction = stoi(input.getCmdOption("-efConstruction"));
    auto M = stoi(input.getCmdOption("-M"));
    auto efSearchs = parseCommaSeparatedIntegers(input.getCmdOption("-efSearch"));
    auto thread_count = stoi(input.getCmdOption("-nThreads"));
    auto minAlpha = stof(input.getCmdOption("-minAlpha"));
    auto maxAlpha = stof(input.getCmdOption("-maxAlpha"));
    auto alphaDecay = stof(input.getCmdOption("-alphaDecay"));
    auto k = stoi(input.getCmdOption("-k"));
    auto filterMinK = parseCommaSeparatedIntegers(input.getCmdOption("-filterMinK"));
    auto selectivities = parseCommaSeparatedIntegers(input.getCmdOption("-selectivity"));
    auto maxNeighboursCheck = stoi(input.getCmdOption("-maxNeighboursCheck"));
    bool loadFromStorage = stoi(input.getCmdOption("-loadFromDisk"));

    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);
    auto queryVectorPath = fmt::format("{}/query.fvecs", basePath);
    auto storagePath = fmt::format("{}/storage.bin", basePath);

    CHECK_ARGUMENT(efSearchs.size() == selectivities.size(), "Number of efSearchs and selectivities should be same");
    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");

    HNSWConfig config(M, efConstruction, 100, minAlpha, maxAlpha, alphaDecay, 1, maxNeighboursCheck,
                      "none", storagePath, loadFromStorage, 20, 10, 1, "none");
    omp_set_num_threads(thread_count);
    RandomGenerator rng(1234);

    printf("Base num vectors: %zu\n", baseNumVectors);

    // Print grond truth num vectors
    printf("Query num vectors: %zu\n", queryNumVectors);
    printf("Query dimension: %zu\n", baseDimension);

    HNSW hnsw(config, &rng, baseDimension);
    build_graph(hnsw, baseVecs, baseNumVectors);
    if (!loadFromStorage) {
        hnsw.flushToDisk();
    }
    // hnsw.logStats();

    int i = 0;
    for (auto selectivity : selectivities) {
        auto groundTruthPath = fmt::format("{}/{}_gt.bin", basePath, selectivity);
        auto maskPath = fmt::format("{}/{}_mask.bin", basePath, selectivity);
        auto efSearch = efSearchs[i];
        printf("efSearch: %d, selectivity: %d\n", efSearch, selectivity);
        hnsw.config.filterMinK = filterMinK[i];
        auto *gtVecs = new vector_idx_t[queryNumVectors * k];
        loadFromFile(groundTruthPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));
        auto *filteredMask = new uint8_t[queryNumVectors * baseNumVectors];
        loadFromFile(maskPath, filteredMask, queryNumVectors * baseNumVectors);
        query_graph_filter(hnsw, queryVecs, filteredMask, queryNumVectors, queryDimension, gtVecs, k, efSearch,
                           baseNumVectors);
        printf("Done\n");
        i++;
    }
}

void query_graph(
        HNSW &hnsw,
        const float *queryVecs,
        size_t queryNumVectors,
        size_t queryDimension,
        const vector_idx_t *gtVecs,
        size_t k,
        size_t ef_search,
        size_t baseNumVectors,
        int thread_count,
        int nodeExpansionPerNode) {
    auto start = std::chrono::high_resolution_clock::now();
    auto recall = 0.0;
    auto visited = VisitedTable(baseNumVectors);
    Stats stats{};
    for (size_t i = 0; i < queryNumVectors; i++) {
        auto localRecall = 0.0;
//        PocTaskScheduler taskScheduler(thread_count, &visited, nodeExpansionPerNode, hnsw.storage, ef_search);
        auto startTime = std::chrono::high_resolution_clock::now();
        std::priority_queue<NodeDistCloser> results;
        std::vector<NodeDistFarther> res;
        hnsw.search(queryVecs + (i * queryDimension), k, ef_search, visited, results, stats);
        auto endTime = std::chrono::high_resolution_clock::now();
        while (!results.empty()) {
            auto top = results.top();
            res.emplace_back(top.id, top.dist);
            results.pop();
        }
        auto gt = gtVecs + i * k;
        for (auto &result: res) {
            if (std::find(gt, gt + k, result.id) != (gt + k)) {
                recall++;
                localRecall++;
            }
        }
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        printf("Query time: %lld ms\n", duration);
        printf("Recall: %f\n", localRecall / k);
    }
    auto recallPerQuery = recall / queryNumVectors;
    stats.logStats();
    std::cout << "Total Vectors: " << queryNumVectors << std::endl;
    std::cout << "Recall: " << (recallPerQuery / k) * 100 << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Query time: " << duration << " ms" << std::endl;
}

void benchmark_hnsw_queries(InputParser &input) {
    const std::string &basePath = input.getCmdOption("-basePath");
    auto efConstruction = stoi(input.getCmdOption("-efConstruction"));
    auto M = stoi(input.getCmdOption("-M"));
    auto efSearch = stoi(input.getCmdOption("-efSearch"));
    auto thread_count = stoi(input.getCmdOption("-nThreads"));
    auto minAlpha = stof(input.getCmdOption("-minAlpha"));
    auto maxAlpha = stof(input.getCmdOption("-maxAlpha"));
    auto alphaDecay = stof(input.getCmdOption("-alphaDecay"));
    auto k = stoi(input.getCmdOption("-k"));
    bool loadFromStorage = stoi(input.getCmdOption("-loadFromDisk"));
    std::string compressionType = input.getCmdOption("-compressionType");
    auto nodesToExplore = stoi(input.getCmdOption("-nodesToExplore"));
    auto nodeExpansionPerNode = stoi(input.getCmdOption("-nodeExpansionPerNode"));
    auto searchParallelAlgo = input.getCmdOption("-searchParallelAlgo");

    auto baseVectorPath = fmt::format("{}/base.bvecs", basePath);
    auto queryVectorPath = fmt::format("{}/query.bvecs", basePath);
    auto groundTruthPath = fmt::format("{}/gt.bin", basePath);
    auto storagePath = fmt::format("{}/storage.bin", basePath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    loadFromFile(groundTruthPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));

    // Print grond truth num vectors
    printf("Query num vectors: %zu\n", queryNumVectors);
    printf("k: %zu\n", k);
    printf("base dimension: %zu\n", baseDimension);
    printf("thread count: %d\n", thread_count);


    omp_set_num_threads(thread_count);
    RandomGenerator rng(1234);
    HNSWConfig config(M, efConstruction, efSearch, minAlpha, maxAlpha, alphaDecay, 30, 30, compressionType, storagePath,
                      loadFromStorage, nodesToExplore, nodeExpansionPerNode, thread_count, searchParallelAlgo);
    HNSW hnsw(config, &rng, baseDimension);
    build_graph(hnsw, baseVecs, baseNumVectors);
    if (!loadFromStorage) {
        hnsw.flushToDisk();
    }

    hnsw.logStats();
//    omp_set_num_threads(2);
    query_graph(hnsw, queryVecs, queryNumVectors, queryDimension, gtVecs, k, efSearch, baseNumVectors, thread_count, nodeExpansionPerNode);
}

// Benchmark clustering
void benchmarkClustering(int argc, char **argv) {
    InputParser input(argc, argv);
    const std::string &basePath = input.getCmdOption("-basePath");
    auto nCentroids = stoi(input.getCmdOption("-nCentroids"));
    auto nIter = stoi(input.getCmdOption("-nIter"));
    auto minCentroidSize = stoi(input.getCmdOption("-minCentroidSize"));
    auto maxCentroidSize = stoi(input.getCmdOption("-maxCentroidSize"));
    auto M = stoi(input.getCmdOption("-M"));
    auto K = stoi(input.getCmdOption("-K"));
    auto efConstruction = stoi(input.getCmdOption("-efConstruction"));
    auto efSearch = stoi(input.getCmdOption("-efSearch"));
    auto nThreads = stoi(input.getCmdOption("-nThreads"));
    auto maxSearchCentroids = stoi(input.getCmdOption("-maxSearchCentroids"));
    auto searchThreshold = stof(input.getCmdOption("-searchThreshold"));
    omp_set_num_threads(nThreads);

    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);
    auto queryVectorPath = fmt::format("{}/query.fvecs", basePath);
    auto groundTruthPath = fmt::format("{}/groundtruth.ivecs", basePath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readFvecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readFvecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    size_t gtDimension, gtNumVectors;
    int *gtVecs = readIvecFile(groundTruthPath.c_str(), &gtDimension, &gtNumVectors);

    PartitionedIndexConfig config(M, efConstruction, efSearch, 1.0, nCentroids, nIter, minCentroidSize, maxCentroidSize,
                                  maxSearchCentroids, searchThreshold);
    RandomGenerator rng(1234);
    PartitionedIndex partitionedIndex(baseDimension, config, &rng);

    // Build index
    auto start = std::chrono::high_resolution_clock::now();
    partitionedIndex.build(baseVecs, baseNumVectors);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Building time: " << duration << " ms" << std::endl;

    // search
    auto recall = 0;
    auto avgCentroid = 0;
    Stats stats{};
    VisitedTable visited(baseNumVectors);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < queryNumVectors; i++) {
        std::vector<NodeDistFarther> results;
        avgCentroid += partitionedIndex.search(queryVecs + i * queryDimension, K, visited, results, stats);
        auto gt = gtVecs + i * gtDimension;
        for (auto res: results) {
            if (std::find(gt, gt + gtDimension, res.id) != (gt + gtDimension)) {
                recall++;
            }
        }
    }
    stats.logStats();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Query time: " << duration << " ms" << std::endl;
    std::cout << "Recall: " << recall / queryNumVectors << std::endl;
    std::cout << "Avg Centroid: " << avgCentroid / queryNumVectors << std::endl;
}

void benchmarkPairWise() {
    auto basePath = "/Users/gauravsehgal/work/orangedb/data/openai";
    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);
    auto queryVectorPath = fmt::format("{}/query.fvecs", basePath);
    auto groundTruthPath = fmt::format("{}/groundtruth.ivecs", basePath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readFvecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);

    fastq::scalar_8bit::SQ8Bit sq8(baseDimension);
    sq8.batch_train(baseNumVectors, baseVecs);

    fastq::pair_wise::PairWise2Bit pw2(baseDimension);
    pw2.batch_train(baseNumVectors, baseVecs);

    // encode first vector
    uint8_t *sq8_codes = new uint8_t[sq8.codeSize];
    sq8.encode(baseVecs, sq8_codes, 1);

    uint8_t *pw2_codes = new uint8_t[pw2.codeSize];
    pw2.encode(baseVecs, pw2_codes, 1);

    // decode first vector
    float *sq8_decoded = new float[baseDimension];
    sq8.decode(sq8_codes, sq8_decoded, 1);

    float *pw2_decoded = new float[baseDimension];
    pw2.decode(pw2_codes, pw2_decoded, 1);

    // Print the [original, sq8_decoded, pw2_decoded] vectors
    for (int i = 0; i < baseDimension; i++) {
        if (sq8_decoded[i] != pw2_decoded[i]) {
            printf("[%d, %f, %f, %f] ", i, baseVecs[i], sq8_decoded[i], pw2_decoded[i]);
        }
    }

    printf("\n");
}

void testParallelPriorityQueue() {
    int numThreads = 4;
    int sizeMultiple = 1;
    int initElements = 500;
//    omp_set_num_threads(4);
    ParallelMultiQueue<NodeDistFarther> mq(numThreads, initElements);
    auto start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < 4000; i++) {
//        mq.push(NodeDistFarther(i, i));
//    }

#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < 4000; i++) {
            mq.push(NodeDistFarther(i, i));
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Duration: %lld ms\n", duration);

    start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < initElements; i++) {
//        auto res = mq.popMin();
//        printf("i: %d Result: %f\n", i, res.dist);
//    }

#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < initElements; i++) {
            auto res = mq.popMin();
            printf("i: %d Result: %f\n", i, res.dist);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Duration: %lld ms\n", duration);
}

void read_and_write_bvecs_file(InputParser &input) {
    const std::string &basePath = input.getCmdOption("-basePath");
    const std::string &outBasePath = input.getCmdOption("-outBasePath");
    auto baseVectorPath = fmt::format("{}/base.bvecs", basePath);
    auto outputVectorPath = fmt::format("{}/base.bvecs", outBasePath);
    auto readSize = stoi(input.getCmdOption("-readSize"));
    size_t baseDimension, baseNumVectors;
    float *baseVecs = readBvecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors, readSize);
    printf("Base dimension: %zu, Base num vectors: %zu\n", baseDimension, readSize);
    writeBvecFile(outputVectorPath.c_str(), baseVecs, baseDimension, readSize);
}

void calculate_dists(InputParser &input) {
    const std::string &basePath = input.getCmdOption("-basePath");
    auto baseVectorPath = fmt::format("{}/base.bvecs", basePath);
    size_t baseDimension, baseNumVectors;
    float *baseVecs = readBvecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);

    auto dc = createDistanceComputer(baseVecs, baseDimension, baseNumVectors, COSINE);

    dc->setQuery(baseVecs + (1 * baseDimension));
    double dist;
    dc->computeDistance(18530814, &dist);
    printf("Dist: %f\n", dist);

    dc->setQuery(baseVecs + (18530806 * baseDimension));
    double dist2;
    dc->computeDistance(18530814, &dist2);
    printf("Dist: %f\n", dist2);

    auto q = baseVecs + (18530806 * baseDimension);
    for (int i = 0; i < 10; i++) {
        printf("%f ", q[i]);
    }
}

int tuneEfByStep(std::function<double(int)> getRecall,
                 double targetLow,
                 double targetHigh,
                 int efMin = 100,
                 int efMax = 1000,
                 int step  = 50) {
    // 1) Evaluate at efMin
    double recallMin = getRecall(efMin);
    printf("efMin: %d, recall: %f\n", efMin, recallMin);
    if (recallMin >= targetLow && recallMin <= targetHigh) {
        return efMin;
    }

    // 2) Evaluate at efMax
    double recallMax = getRecall(efMax);
    printf("efMax: %d, recall: %f\n", efMax, recallMax);
    // If even efMax is below your lower bound, just return efMax (best you can do)
    if (recallMax < targetLow) {
        return efMax;
    }
    // Or if efMax falls in range, return it immediately
    if (recallMax >= targetLow && recallMax <= targetHigh) {
        return efMax;
    }

    // 3) Step through [efMin, efMax]
    int ef     = efMin;
    double rec = recallMin;
    int prevEf = -1;
    double prev_rec = rec;
    while (ef != prevEf) {
        prevEf = ef;

        // if too low, step up; if too high, step down
        if (rec < targetLow)        ef = std::min(ef + step, efMax);
        else if (rec > targetHigh)  ef = std::max(ef - step, efMin);
        else                         break;  // in the sweet spot

        rec = getRecall(ef);

        if (prev_rec < targetLow && rec > targetHigh) {
            // we just crossed the lower bound
            printf("ef: %d, recall: %f\n", ef, rec);
            return ef;
        }

        prev_rec = rec;
        printf("ef: %d, recall: %f\n", ef, rec);
    }

    // final check
    if (rec >= targetLow && rec <= targetHigh) {
        return ef;
    }

    // fallback: return efMax because we know recallMax > targetHigh
    // (you could also return 'ef' here if you prefer the last tried value)
    return efMax;
}

void write_json_result(const std::string &basePath, const std::string config, const int totalQueries, const double searchTime,
                        const double distanceComputations, const double nIos, const double recall, const int efSearch,
                        const std::string selectivity) {
    std::string jsonPath = fmt::format("{}/output_{}_{}.json", basePath, selectivity, config);
    nlohmann::json J;
    J["total_queries"] = totalQueries;
    J["avg_execution_time_ms"] = searchTime;
    J["avg_distance_computations"] = distanceComputations;
    J["avg_list_nbrs_calls"] = nIos;
    J["recall_percentage"] = recall * 100;
    J["selectivity"] = stof(selectivity);
    J["efSearch"] = efSearch;

    // Write the JSON object to a file
    std::ofstream ofs(jsonPath);
    ofs << J.dump(4);
    ofs.close();
    std::cout << "Results written to " << jsonPath << std::endl;
}

// void benchmark_acorn(InputParser &input) {
//     const std::string &dataPath = input.getCmdOption("-dataPath");
//     const std::string &basePath = input.getCmdOption("-basePath");
//     const std::vector<std::string> sels = parseCommaSeparated(input.getCmdOption("-sels"));
//     const std::vector<int> efS = parseCommaSeparatedIntegers(input.getCmdOption("-efS"));
//     const int autoEf = stoi(input.getCmdOption("-autoEf"));
//     int k = stoi(input.getCmdOption("-k"));
//     int M = stoi(input.getCmdOption("-M"));
//     int gamma = stoi(input.getCmdOption("-gamma"));
//     int M_beta = stoi(input.getCmdOption("-M_beta"));
//     int nThreads = stoi(input.getCmdOption("-nThreads"));
//     float minRecall = stof(input.getCmdOption("-minRecall"));
//     float maxRecall = stof(input.getCmdOption("-maxRecall"));
//     const int readFromDisk = stoi(input.getCmdOption("-readFromDisk"));
//     const std::string &storagePath = input.getCmdOption("-storagePath");
//     const std::string &resultPath = input.getCmdOption("-resultPath");
//     const int useIp = stoi(input.getCmdOption("-useIp"));
//     std::vector<std::string> maskPaths, gtPath;
//     std::string queryPath;
//     populate_mask_and_gt_paths(basePath, sels, maskPaths, gtPath, queryPath);
//
//     size_t baseDimension, baseNumVectors;
//     float *baseVecs = readVecFile(dataPath.c_str(), &baseDimension, &baseNumVectors);
//     size_t queryDimension, queryNumVectors;
//     float *queryVecs = readVecFile(queryPath.c_str(), &queryDimension, &queryNumVectors);
//     CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
//     printf("Base num vectors: %zu\n", baseNumVectors);
//     printf("Query num vectors: %zu\n", queryNumVectors);
//
//     // First build the index
//     auto *gtVecs = new vector_idx_t[queryNumVectors * k];
//     auto *filteredMask = new uint8_t[baseNumVectors];
//     loadFromFile(maskPaths[0], filteredMask, baseNumVectors);
//     std::vector<int> metadata(baseNumVectors);
//     for (int i = 0; i < baseNumVectors; i++) {
//         metadata[i] = (int) filteredMask[i];
//     }
//     auto index = faiss::IndexACORNFlat(baseDimension, M, gamma, metadata, M_beta, faiss::METRIC_INNER_PRODUCT);
//     faiss::IndexACORNFlat* acorn_index = &index;
//     if (!readFromDisk) {
//         omp_set_num_threads(nThreads);
//         // Print grond truth num vectors
//         printf("Building index\n");
//         auto start = std::chrono::high_resolution_clock::now();
//         acorn_index->add(baseNumVectors, baseVecs);
//         auto end = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//         printf("Building time: %lld ms\n", duration.count());
//         printf("Writing the index on disk!");
//         faiss::write_index(acorn_index, storagePath.c_str());
//     } else {
//         acorn_index = dynamic_cast<faiss::IndexACORNFlat *>(faiss::read_index(storagePath.c_str()));
//         acorn_index->metric_type = faiss::METRIC_INNER_PRODUCT;
//     }
//     omp_set_num_threads(1);
//
//     // Todo: Write the time to build the index
//
//     // Now perform search for each selectivity
//     for (int i = 0; i < sels.size(); i++) {
//         auto& selectivity = sels[i];
//         printf("Selectivity: %s\n", selectivity.c_str());
//         auto efSearch = efS[i];
//         auto& maskPathStr = maskPaths[i];
//         auto& gtPathStr = gtPath[i];
//         printf("gtPath: %s\n", gtPathStr.c_str());
//         printf("maskPath: %s\n", maskPathStr.c_str());
//         loadFromFile(gtPathStr, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));
//         loadFromFile(maskPathStr, filteredMask, baseNumVectors);
//
//         printf("efSearch: %d, selectivity: %s\n", efSearch, sels[i].c_str());
//         if (autoEf) {
//             auto ef = tuneEfByStep([&](int ef) {
//                 acorn_index->acorn.efSearch = ef;
//                 auto labels = new faiss::idx_t[k];
//                 auto distances = new float[k];
//                 auto recall = 0.0;
//                 for (size_t j = 0; j < queryNumVectors; j++) {
//                     acorn_index->search(1, queryVecs + (j * baseDimension), k, distances, labels, reinterpret_cast<char*>(filteredMask));
//                     auto gt = gtVecs + j * k;
//                     for (int m = 0; m < k; m++) {
//                         if (std::find(gt, gt + k, labels[m]) != (gt + k)) {
//                             recall++;
//                         }
//                     }
//                 }
//                 printf("Recall: %f\n", recall);
//                 auto recallPerQuery = recall / queryNumVectors;
//                 return recallPerQuery / k;
//             }, minRecall, maxRecall, 100, 1500, 50);
//             acorn_index->acorn.efSearch = ef;
//         } else {
//             acorn_index->acorn.efSearch = efSearch;
//         }
//
//         // Run the benchmark
//         auto recall = 0.0;
//         auto labels = new faiss::idx_t[k];
//         auto distances = new float[k];
//         long durationPerQuery = 0;
//         for (size_t j = 0; j < queryNumVectors; j++) {
//             auto startTime = std::chrono::high_resolution_clock::now();
//             acorn_index->search(1, queryVecs + (j * baseDimension), k, distances, labels, reinterpret_cast<char*>(filteredMask));
//             auto endTime = std::chrono::high_resolution_clock::now();
//             auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
//             durationPerQuery += duration;
//             auto gt = gtVecs + j * k;
//             for (int m = 0; m < k; m++) {
//                 if (std::find(gt, gt + k, labels[m]) != (gt + k)) {
//                     recall++;
//                 }
//             }
//         }
//         auto config = fmt::format("acorn_{}", gamma);
//         write_json_result(resultPath, config, queryNumVectors, ((double) durationPerQuery / queryNumVectors) * 1e-6,
//                           recall / (queryNumVectors * k), efSearch, selectivity);
//     }
//
//     delete[] filteredMask;
//     delete[] gtVecs;
// }

void benchmark_navix(InputParser &input) {
    const std::string &dataPath = input.getCmdOption("-dataPath");
    const std::string &basePath = input.getCmdOption("-basePath");
    const std::vector<std::string> sels = parseCommaSeparated(input.getCmdOption("-sels"));
    const std::vector<int> efS = parseCommaSeparatedIntegers(input.getCmdOption("-efS"));
    const int autoEf = stoi(input.getCmdOption("-autoEf"));
    int k = stoi(input.getCmdOption("-k"));
    int M = stoi(input.getCmdOption("-M"));
    int efConstruction = stoi(input.getCmdOption("-efConstruction"));
    int nThreads = stoi(input.getCmdOption("-nThreads"));
    float minRecall = stof(input.getCmdOption("-minRecall"));
    float maxRecall = stof(input.getCmdOption("-maxRecall"));
    const int readFromDisk = stoi(input.getCmdOption("-readFromDisk"));
    const std::string &storagePath = input.getCmdOption("-storagePath");
    const std::string &resultPath = input.getCmdOption("-resultPath");
    const int useIp = stoi(input.getCmdOption("-useIp"));
    std::vector<std::string> maskPaths, gtPath;
    std::string queryPath;
    populate_mask_and_gt_paths(basePath, sels, maskPaths, gtPath, queryPath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(dataPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryPath.c_str(), &queryDimension, &queryNumVectors);
    CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
    printf("Base num vectors: %zu\n", baseNumVectors);
    printf("Base dimension: %zu\n", baseDimension);
    printf("Query num vectors: %zu\n", queryNumVectors);

    faiss::MetricType metricType = useIp ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

    // First build the index
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    auto *filteredMask = new uint8_t[baseNumVectors];
    auto index = faiss::IndexHNSWFlat(baseDimension, M, metricType);
    faiss::IndexHNSWFlat* hnsw_index = &index;
    hnsw_index->verbose = true;
    hnsw_index->hnsw.efConstruction = efConstruction;
    if (!readFromDisk) {
        omp_set_num_threads(nThreads);
        // Print grond truth num vectors
        printf("Building index\n");
        auto start = std::chrono::high_resolution_clock::now();
        // hnsw_index->train(baseNumVectors, baseVecs);
        hnsw_index->add(baseNumVectors, baseVecs);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Building time: %lld ms\n", duration.count());
        printf("Writing the index on disk!");
        faiss::write_index(hnsw_index, storagePath.c_str());

        printf("Building time: %lld ms\n", duration.count());
        auto stat_path = fmt::format("{}/navix_{}_build_time.txt", resultPath, M);
        std::ofstream stat_file(stat_path);
        stat_file << "Building time: " << duration.count() << " ms" << std::endl;
        stat_file.close();
    } else {
        delete[] baseVecs;
        baseVecs = nullptr;
        hnsw_index = dynamic_cast<faiss::IndexHNSWFlat *>(faiss::read_index(storagePath.c_str()));
        hnsw_index->hnsw.efConstruction = efConstruction;
        hnsw_index->metric_type = metricType;
    }

    omp_set_num_threads(1);

    // Now perform search for each selectivity
    for (int i = 0; i < sels.size(); i++) {
        auto& selectivity = sels[i];
        printf("Selectivity: %s\n", selectivity.c_str());
        auto efSearch = efS[i];
        auto& maskPathStr = maskPaths[i];
        auto& gtPathStr = gtPath[i];
        printf("gtPath: %s\n", gtPathStr.c_str());
        printf("maskPath: %s\n", maskPathStr.c_str());
        loadFromFile(gtPathStr, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));
        loadFromFile(maskPathStr, filteredMask, baseNumVectors);

        printf("efSearch: %d, selectivity: %s\n", efSearch, sels[i].c_str());
        if (autoEf) {
            auto ef = tuneEfByStep([&](int ef) {
                hnsw_index->hnsw.efSearch = ef;
                faiss::VisitedTable visited(hnsw_index->ntotal);
                faiss::HNSWStats stats;
                auto recall = 0.0;
                for (size_t j = 0; j < queryNumVectors; j++) {
                    auto labels = new faiss::idx_t[k];
                    auto distances = new float[k];
                    // if (selectivity == "100") {
                    //     hnsw_index->single_search(queryVecs + (j * baseDimension), k, distances, labels, visited, stats);
                    // } else {
                    hnsw_index->navix_single_search(queryVecs + (j * baseDimension), k, distances, labels, reinterpret_cast<char*>(filteredMask), visited, stats);
                    // }
                    auto gt = gtVecs + j * k;
                    for (int m = 0; m < k; m++) {
                        if (std::find(gt, gt + k, (vector_idx_t)labels[m]) != (gt + k)) {
                            recall++;
                        }
                    }
                }
                auto recallPerQuery = recall / queryNumVectors;
                return recallPerQuery / k;
            }, minRecall, maxRecall, 100, 1500, 10);
            hnsw_index->hnsw.efSearch = ef;
        } else {
            hnsw_index->hnsw.efSearch = efSearch;
        }

        // Run the benchmark
        auto recall = 0.0;
        auto labels = new faiss::idx_t[k];
        auto distances = new float[k];
        faiss::VisitedTable visited(hnsw_index->ntotal);
        faiss::HNSWStats stats;
        long durationPerQuery = 0;
        for (size_t j = 0; j < queryNumVectors; j++) {
            auto startTime = std::chrono::high_resolution_clock::now();
            // if (selectivity == "100") {
            // hnsw_index->single_search(queryVecs + (j * baseDimension), k, distances, labels, visited, stats);
            // } else {
            hnsw_index->navix_single_search(queryVecs + (j * baseDimension), k, distances, labels, reinterpret_cast<char*>(filteredMask), visited, stats);
            // }
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
            durationPerQuery += duration;
            auto gt = gtVecs + j * k;
            for (int m = 0; m < k; m++) {
                if (std::find(gt, gt + k, labels[m]) != (gt + k)) {
                    recall++;
                }
            }
        }
        auto config = fmt::format("navix_{}", M);
        printf("durationPerQuery: %f ms\n", ((double) durationPerQuery / queryNumVectors) * 1e-6);
        printf("distance: %f\n", ((double) stats.ndis / queryNumVectors));
        printf("graph ios: %f\n", ((double) stats.nhops / queryNumVectors));
        write_json_result(resultPath, config, queryNumVectors, ((double) durationPerQuery / queryNumVectors) * 1e-6,
                            (double) stats.ndis / queryNumVectors, (double) stats.nhops / queryNumVectors,
                          recall / (queryNumVectors * k), efSearch, selectivity);
    }

    delete[] filteredMask;
    delete[] gtVecs;
}

std::pair<int, int> get_range(const char* filteredMask, int n) {
    auto start = 0;
    auto end = 0;
    for (int i = 0; i < n; i++) {
        if (filteredMask[i] == 1) {
            start = i;
            break;
        }
    }

    for (int i = n - 1; i >= 0; i--) {
        if (filteredMask[i] == 1) {
            end = i;
            break;
        }
    }

    // Validate between start and end there's no 0
    for (int i = start; i <= end; i++) {
        if (filteredMask[i] == 0) {
            throw std::runtime_error("Invalid range");
        }
    }

    return std::make_pair(start, end);
}

void benchmark_irangegraph(InputParser &input) {
    const std::string &dataPath = input.getCmdOption("-dataPath");
    const std::string &basePath = input.getCmdOption("-basePath");
    const std::vector<std::string> sels = parseCommaSeparated(input.getCmdOption("-sels"));
    const std::vector<int> efS = parseCommaSeparatedIntegers(input.getCmdOption("-efS"));
    const int autoEf = stoi(input.getCmdOption("-autoEf"));
    int k = stoi(input.getCmdOption("-k"));
    int M = stoi(input.getCmdOption("-M"));
    int efConstruction = stoi(input.getCmdOption("-efConstruction"));
    int nThreads = stoi(input.getCmdOption("-nThreads"));
    float minRecall = stof(input.getCmdOption("-minRecall"));
    float maxRecall = stof(input.getCmdOption("-maxRecall"));
    const int readFromDisk = stoi(input.getCmdOption("-readFromDisk"));
    const std::string &storagePath = input.getCmdOption("-storagePath");
    const std::string &resultPath = input.getCmdOption("-resultPath");
    std::vector<std::string> maskPaths, gtPaths;
    std::string queryPath;
    populate_mask_and_gt_paths(basePath, sels, maskPaths, gtPaths, queryPath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(dataPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryPath.c_str(), &queryDimension, &queryNumVectors);
    CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
    printf("Base num vectors: %zu\n", baseNumVectors);
    printf("Base dimension: %zu\n", baseDimension);
    printf("Query num vectors: %zu\n", queryNumVectors);

    iRangeGraph::DataLoader storage;
    storage.LoadData(baseVecs, baseNumVectors, baseDimension);
    if (!readFromDisk) {
        auto start = std::chrono::high_resolution_clock::now();
        iRangeGraph::iRangeGraph_Build<float> index(&storage, M, efConstruction);
        index.max_threads = nThreads;
        index.buildandsave(storagePath);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // Save the time to build the index in a file
        printf("Building time: %lld ms\n", duration.count());
        auto stat_path = fmt::format("{}/irangegraph_{}_build_time.txt", resultPath, M);
        std::ofstream stat_file(stat_path);
        stat_file << "Building time: " << duration.count() << " ms" << std::endl;
        stat_file.close();
    }

    // First build the index
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    auto *filteredMask = new uint8_t[baseNumVectors];
    storage.LoadQuery(queryVecs, queryNumVectors, baseDimension);
    for (int i = 0; i < sels.size(); i++) {
        auto& selectivity = sels[i];
        printf("Selectivity: %s\n", selectivity.c_str());
        auto efSearch = efS[i];
        auto& maskPathStr = maskPaths[i];
        auto& gtPathStr = gtPaths[i];
        printf("gtPath: %s\n", gtPathStr.c_str());
        printf("maskPath: %s\n", maskPathStr.c_str());
        loadFromFile(gtPathStr, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));
        loadFromFile(maskPathStr, filteredMask, baseNumVectors);

        storage.LoadGroundtruth(gtVecs, k);
        auto query_range = get_range(reinterpret_cast<char *>(filteredMask), baseNumVectors);
        printf("Running range: %d, %d\n", query_range.first, query_range.second);
        storage.LoadQueryRange(query_range.first, query_range.second);
        printf("efSearch: %d, selectivity: %s\n", efSearch, sels[i].c_str());
        iRangeGraph::iRangeGraph_Search<float> searchIndex("", storagePath, &storage, M);
        std::vector<int> finalEfSearches;
        if (autoEf) {
            auto ef = tuneEfByStep([&](int ef) {
                 std::vector<int> efSearches = {ef};
                 auto res = searchIndex.search_new(efSearches, M);
                return res[0].RECALL[0];
            }, minRecall, maxRecall, 100, 1500, 50);
            finalEfSearches.push_back(ef);
        } else {
            finalEfSearches.push_back(efSearch);
        }

        auto res = searchIndex.search_new(finalEfSearches, M);
        auto recall = res[0].RECALL[0];
        auto latency_sec = res[0].latency[0];
        auto ndis = res[0].DCO[0];
        auto nhops = res[0].HOP[0];

        auto config = fmt::format("irangegraph_{}", M);
        printf("durationPerQuery: %f ms\n", latency_sec * 1000);
        printf("distance: %f\n", ndis);
        printf("graph ios: %f\n", nhops);
        write_json_result(resultPath, config, queryNumVectors, latency_sec * 1000,
                          ndis, nhops,
                          recall, efSearch, selectivity);
    }
}

void fvec_to_fbin(InputParser &input) {
    const std::string &vectorPath = input.getCmdOption("-vectorPath");
    const std::string &queryPath = input.getCmdOption("-queryPath");

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(vectorPath.c_str(), &baseDimension, &baseNumVectors);

    writeFbinFile(queryPath.c_str(), baseVecs, baseDimension, baseNumVectors);
}

void benchmark_quantization(InputParser &input) {
    const std::string &basePath = input.getCmdOption("-basePath");

    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);
    auto queryVectorPath = fmt::format("{}/query.fvecs", basePath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);

    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);

    fastq::scalar_test::SQ8Bit sq(baseDimension);
    sq.batch_train(baseVecs, baseNumVectors);

    uint8_t *codes = new uint8_t[sq.codeSize * baseNumVectors];
    sq.encode(baseVecs, codes, baseNumVectors);

    auto dc = createDistanceComputer(baseVecs, baseDimension, baseNumVectors, L2);
    dc->setQuery(queryVecs);
    double dist;
    dc->computeDistance(static_cast<vector_idx_t>(0), &dist);

    auto qdc = sq.get_asym_distance_computer(fastq::scalar_test::L2_SQ);
    double qDist;
    qdc->compute_distance(queryVecs, codes, &qDist);
    printf("Dist: %f, Quantized Dist: %f\n", dist, qDist);

    for (int i = 0; i < 30; i++) {
        printf("%f %d %f\n", baseVecs[i], codes[i], fastq::scalar_test::decode_serial(codes[i], sq.alpha[i], sq.beta[i]));
    }
}

void benchmark_random_pread(InputParser &input) {

}

std::pair<size_t, size_t> get_file_stat(const std::string &filePath) {
    FILE *f = fopen(filePath.c_str(), "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", filePath.c_str());
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
    size_t n = sz / ((d + 1) * 4);
    fclose(f);

    return std::pair(n, d);
}

void get_random_offsets(std::vector<std::pair<uint64_t, uint64_t>> &readInfo, uint64_t dim, uint64_t numVectors) {
    auto now = std::chrono::system_clock::now();
    auto seed = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    RandomGenerator rng(seed);
    std::vector<uint64_t> offsets(readInfo.size());
    printf("Num vectors: %llu\n", numVectors);
    rng.randomPerm(numVectors, offsets.data(), offsets.size());
    // Adjust offsets
    for (int i = 0; i < offsets.size(); i++) {
        printf("Offset: %llu\n", offsets[i]);
        auto offset = offsets[i] * (dim + 1) * 4;
        auto size = (dim + 1) * sizeof(float);
        readInfo[i] = std::make_pair(offset, size);
    }
}

#if 0

struct io_data {
    int read;
    off_t first_offset, offset;
    size_t first_len;
    struct iovec iov;
};

static int setup_context(int fd, unsigned entries, struct io_uring *ring)
{
    int ret;
    // Enable IORING_SETUP_SQPOLL for kernel-side polling of submission queue
    // Enable IORING_SETUP_IOPOLL for kernel-side polling of completions
    struct io_uring_params params = {};
//    params.flags = IORING_SETUP_SQPOLL;
//    params.sq_thread_idle = 2000; // Timeout in milliseconds before sq thread goes idle

    ret = io_uring_queue_init(entries, ring, 0);
    if (ret < 0) {
        fprintf(stderr, "queue_init: %s\n", strerror(-ret));
        return -1;
    }

    // Check if polling was successfully enabled
//    if (!(params.features & IORING_FEAT_SQPOLL)) {
//        fprintf(stderr, "Kernel polling not available\n");
//        return -1;
//    }

    if (io_uring_register_files(ring, &fd, 1) < 0) {
        perror("io_uring_register_files");
        exit(1);
    }

    return 0;
}

static int queue_read(struct io_uring *ring, int fd, off_t size, off_t offset)
{
//    auto start = std::chrono::high_resolution_clock::now();
    struct io_uring_sqe *sqe;
    struct io_data *data;

    data = static_cast<io_data *>(malloc(size + sizeof(*data)));
    if (!data)
        return 1;
//    auto end = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
//    printf("Duration malloc: %lld ns\n", duration);

//    start = std::chrono::high_resolution_clock::now();
    sqe = io_uring_get_sqe(ring);
    if (!sqe) {
        free(data);
        return 1;
    }

//    end = std::chrono::high_resolution_clock::now();
//    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
//    printf("Duration get_sqe: %lld ns\n", duration);

//    start = std::chrono::high_resolution_clock::now();

    data->read = 1;
    data->offset = data->first_offset = offset;

    data->iov.iov_base = data + 1;
    data->iov.iov_len = size;
    data->first_len = size;

    // Use fixed file descriptor for better performance
    io_uring_prep_readv(sqe, fd, &data->iov, 1, offset);
    // Set IOPOLL flag for this request
//    sqe->flags |= IOSQE_FIXED_FILE | IOSQE_IO_LINK;
    io_uring_sqe_set_data(sqe, data);

//    end = std::chrono::high_resolution_clock::now();
//    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
//    printf("Duration prep_readv: %lld ns\n", duration);
    return 0;
}

static int open_file(const char *file, bool useODirect = true)
{
#ifdef __linux__
    auto flags = useODirect ? O_DIRECT | O_RDONLY : O_RDONLY;
    int fd = open(file, flags);
    if (fd < 0) {
        perror("open");
        return -1;
    }
    return fd;
#elif defined(__APPLE__)
    // macOS-specific: Open file and set F_NOCACHE
    int fd = open(file, O_RDONLY);
    if (fd == -1) {
        perror("macOS open failed");
        return 1;
    }
    if (useODirect && fcntl(fd, F_NOCACHE, 1) == -1) {
        perror("macOS fcntl F_NOCACHE failed");
        close(fd);
        return 1;
    }
    std::cout << "Opened file with F_NOCACHE on macOS.\n";
    return fd;
#endif
}

void benchmark_io_uring(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    auto numRandomReads = stoi(input.getCmdOption("-numRandomReads"));
    bool useODirect = stoi(input.getCmdOption("-useODirect"));
    printf("O_DIRECT: %d\n", useODirect);

    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);

    auto stat = get_file_stat(baseVectorPath);
    std::vector<std::pair<uint64_t, uint64_t>> readInfo(numRandomReads);
    get_random_offsets(readInfo, stat.second, stat.first);

    // Open with O_DIRECT for potentially better performance with polling
    int fd = open_file(baseVectorPath.c_str(), useODirect);
    if (fd < 0) {
        perror("open failed");
        abort();
    }

    struct io_uring ring;
    setup_context(fd, numRandomReads, &ring);

    // Batch submission metrics
    const int BATCH_SIZE = 64;
    auto start = std::chrono::high_resolution_clock::now();
    struct io_uring_cqe *cqe;
    int pending = 0;

    // Queue reads in batches
    for (int i = 0; i < numRandomReads; i++) {
        auto offset = readInfo[i].first;
        auto size = readInfo[i].second;
        if (queue_read(&ring, fd, size, offset))
            break;

        pending++;

        // Submit in batches for better performance
        if (pending == BATCH_SIZE || i == numRandomReads - 1) {
            auto ret = io_uring_submit(&ring);
            if (ret < 0) {
                fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
                abort();
            }
            pending = 0;
        }
    }

//    auto end = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
//    printf("Duration for queuing and submitting reads: %lld ns\n", duration);

//    start = std::chrono::high_resolution_clock::now();
    std::vector<double> dists(numRandomReads);

    // Process completions
    for (int i = 0; i < numRandomReads; i++) {
        struct io_data *data;
        // Use IORING_ENTER_GETEVENTS to actively poll for completions
        auto ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0) {
            fprintf(stderr, "io_uring_wait_cqe: %s\n", strerror(-ret));
            abort();
        }

        data = static_cast<io_data *>(io_uring_cqe_get_data(cqe));
        if (cqe->res < 0) {
            fprintf(stderr, "Read failed: %s\n", strerror(-cqe->res));
            abort();
        }

        assert(data->read == 1);
        // Compute distance
        simsimd_cos_f32(queryVecs, reinterpret_cast<float *>(data->iov.iov_base) + 1, queryDimension, &dists[i]);

        // Free the allocated memory
        free(data);
        io_uring_cqe_seen(&ring, cqe);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Duration for processing completions: %lld ns\n", duration);

    // Cleanup
    auto dist_sum = 0.0;
    for (int i = 0; i < numRandomReads; i++) {
        dist_sum += dists[i];
    }
    printf("Average distance: %f\n", dist_sum / numRandomReads);

    io_uring_queue_exit(&ring);
    close(fd);
}

void benchmark_pread(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    auto numRandomReads = stoi(input.getCmdOption("-numRandomReads"));
    bool useODirect = stoi(input.getCmdOption("-useODirect"));
    printf("O_DIRECT: %d\n", useODirect);

    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);

    auto stat = get_file_stat(baseVectorPath);
    std::vector<std::pair<uint64_t, uint64_t>> readInfo(numRandomReads);
    get_random_offsets(readInfo, stat.second, stat.first);

    int fd = open_file(baseVectorPath.c_str(), useODirect);
    if (fd < 0) {
        perror("open failed");
        abort();
    }

    auto start = std::chrono::high_resolution_clock::now();

    std:vector<double> dists(numRandomReads);
    for (int i = 0; i < numRandomReads; i++) {
        auto offset = readInfo[i].first;
        auto size = readInfo[i].second;
        float *baseVecs = reinterpret_cast<float *>(malloc(size));
        if (baseVecs == nullptr) {
            perror("malloc failed");
            abort();
        }
        auto ret = pread(fd, baseVecs, size, offset);
        if (ret < 0) {
            perror("pread failed");
            abort();
        }
        simsimd_cos_f32(queryVecs, baseVecs + 1, queryDimension, &dists[i]);
        free(baseVecs);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Time: %lld ms\n", duration);

    // Cleanup
    auto dist_sum = 0.0;
    for (int i = 0; i < numRandomReads; i++) {
        dist_sum += dists[i];
    }
    printf("Average distance: %f\n", dist_sum / numRandomReads);
    close(fd);
}
#endif

void test_clustering_data(InputParser &input) {
    // TODO: Replace with FAISS IVF FLAT
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    const std::string &groundTruthPath = input.getCmdOption("-groundTruthPath");
    const int numVectors = stoi(input.getCmdOption("-numVectors"));
    const int clusterSize = stoi(input.getCmdOption("-clusterSize"));
    const int nIter = stoi(input.getCmdOption("-nIter"));
    const float lambda = stof(input.getCmdOption("-lambda"));
    const int k = stoi(input.getCmdOption("-k"));
    const int nProbes = stoi(input.getCmdOption("-nProbes"));
    const int numThreads = stoi(input.getCmdOption("-numThreads"));

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);

    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);

    CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    loadFromFile(groundTruthPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));

    omp_set_num_threads(numThreads);
    baseNumVectors = std::min(baseNumVectors, (size_t) numVectors);
    printf("baseNumVectors: %d, baseDimension: %d\n", baseNumVectors, baseDimension);
    int numCentroids = numVectors / clusterSize;
    int minCentroidSize = (numVectors / numCentroids) * 0.5;
    int maxCentroidSize = (numVectors / numCentroids) * 1.2;
    auto dc = createDistanceComputer(baseVecs, baseDimension, baseNumVectors, L2);
    auto clustering = Clustering<float>(baseDimension, baseDimension, numCentroids, nIter, minCentroidSize,
                                        maxCentroidSize, dc.get(), [](const float a, int j) { return a; }, lambda);

    // Init centroids and train!!
    printf("Init centroids\n");
    clustering.initCentroids(baseVecs, baseNumVectors);
    printf("Train\n");
    clustering.train(baseVecs, baseNumVectors);

    auto labels = new int32_t[baseNumVectors];
    clustering.assignCentroids(baseVecs, baseNumVectors, labels);

    // Print the distribution interms of avg, min, max cluster size
    std::vector<size_t> clusterSizeHist(numCentroids, 0);
    for (int i = 0; i < baseNumVectors; i++) {
        clusterSizeHist[labels[i]]++;
    }
    auto minSize = std::numeric_limits<size_t>::max();
    size_t maxSize = 0;
    size_t avgSize = 0;
    for (const auto &clusterSize: clusterSizeHist) {
        minSize = std::min(minSize, clusterSize);
        maxSize = std::max(maxSize, clusterSize);
        avgSize += clusterSize;
    }
    printf("Min size of clusters: %zu\n", minSize);
    printf("Max size of clusters: %zu\n", maxSize);
    printf("Avg size of clusters: %zu\n", avgSize / numCentroids);

    // Run search by first finding nProbes centroids and then searching in those
    double totalDC = 0.0;
    double recall = 0.0;
    auto centroidDc = createDistanceComputer(clustering.centroids.data(), baseDimension, clustering.getNumCentroids(), L2);
    for (size_t i = 0; i < queryNumVectors; i++) {
        // Find the nearest nProbes centroids for the current query
        centroidDc->setQuery(queryVecs + i * queryDimension);
        std::priority_queue<NodeDistCloser> closestCentroids;
        for (int j = 0; j < numCentroids; j++) {
            double dist;
            centroidDc->computeDistance(j, &dist);
            totalDC++;
            if (closestCentroids.size() < nProbes || dist < closestCentroids.top().dist) {
                closestCentroids.emplace(j, dist);
                if (closestCentroids.size() > nProbes) {
                    closestCentroids.pop();
                }
            }
        }

        // Search within base vectors belonging to the selected centroids
        std::priority_queue<NodeDistCloser> results;
        while (!closestCentroids.empty()) {
            auto closestCentroidId = closestCentroids.top();
            closestCentroids.pop();
            // Iterate over all base vectors and check if assigned to this centroid
            for (size_t v = 0; v < baseNumVectors; v++) {
                if (labels[v] == closestCentroidId.id) {
                    double dist;
                    centroidDc->computeSymDistance(queryVecs + i * queryDimension, baseVecs + v * baseDimension, &dist);
                    totalDC++;
                    if (results.size() < k || dist < results.top().dist) {
                        results.emplace(v, dist);
                        if (results.size() > k) {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Calculate recall
        auto gt = gtVecs + i * k;
        while (!results.empty()) {
            auto res = results.top();
            results.pop();
            if (std::find(gt, gt + k, res.id) != (gt + k)) {
                recall++;
            }
        }
    }

    // Print avg distance computation and recall
    printf("Avg Distance Computation: %f\n", totalDC / queryNumVectors);
    printf("Recall: %f\n", recall / queryNumVectors);
    delete[] labels;
    delete[] baseVecs;
    delete[] queryVecs;
}

void benchmark_faiss_clustering(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    const std::string &groundTruthPath = input.getCmdOption("-groundTruthPath");
    const int numVectors = stoi(input.getCmdOption("-numVectors"));
    const int clusterSize = stoi(input.getCmdOption("-clusterSize"));
    const int nIter = stoi(input.getCmdOption("-nIter"));
    const int nThreads = stoi(input.getCmdOption("-nThreads"));
    const int k = stoi(input.getCmdOption("-k"));
    const int numQueries = stoi(input.getCmdOption("-numQueries"));
    const int sampleSize = stoi(input.getCmdOption("-sampleSize"));
    const int nProbes = stoi(input.getCmdOption("-nProbes"));
    const int readFromDisk = stoi(input.getCmdOption("-readFromDisk"));
    const std::string &storagePath = input.getCmdOption("-storagePath");

    // Read dataset
    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);

    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    baseNumVectors = std::min(baseNumVectors, (size_t) numVectors);
    queryNumVectors = std::min(queryNumVectors, (size_t) numQueries);
    auto sampleSizeAdjusted = std::min((size_t)sampleSize, baseNumVectors);
    CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    loadFromFile(groundTruthPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));
    auto quantizer = faiss::IndexFlatIP(baseDimension);
    auto numCentroids = baseNumVectors / clusterSize;
    faiss::IndexIVFFlat idx(&quantizer, baseDimension, numCentroids, faiss::METRIC_INNER_PRODUCT);
    faiss::IndexIVFFlat* index = &idx;
    index->cp.niter = nIter;
    index->cp.max_points_per_centroid = (sampleSizeAdjusted / numCentroids);
    index->cp.min_points_per_centroid = (sampleSizeAdjusted / numCentroids) * 0.5;
    printf("max_points_per_centroid: %d, min_points_per_centroid: %d\n",
           index->cp.max_points_per_centroid, index->cp.min_points_per_centroid);
    index->cp.verbose = true;
    if (!readFromDisk) {
        omp_set_num_threads(nThreads);
        // Print grond truth num vectors
        printf("Building index\n");
        auto start = std::chrono::high_resolution_clock::now();
        index->train(baseNumVectors, baseVecs);
        index->add(baseNumVectors, baseVecs);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Building time: %lld ms\n", duration.count());
        printf("Writing the index on disk!");
        faiss::write_index(index, storagePath.c_str());
    } else {
        index = dynamic_cast<faiss::IndexIVFFlat *>(faiss::read_index(storagePath.c_str()));
    }
    omp_set_num_threads(1);
    index->nprobe = nProbes;
    auto recall = 0.0;
    auto labels = new faiss::idx_t[k];
    auto distances = new float[k];
    auto startTime = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < queryNumVectors; i++) {
        index->search(1, queryVecs + (i * baseDimension), k, distances, labels);
        auto gt = gtVecs + i * k;
        for (int j = 0; j < k; j++) {
            printf("Query %zu, label: %lld, gt: %llu, dist: %f\n", i, labels[j], gt[j], distances[j]);
            if (std::find(gt, gt + k, labels[j]) != (gt + k)) {
                recall++;
            }
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration_search = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    auto recallPerQuery = recall / queryNumVectors;
    std::cout << "Total Vectors: " << queryNumVectors << std::endl;
    std::cout << "Num of centroids: " << numCentroids << std::endl;
    std::cout << "Recall: " << (recallPerQuery / k) * 100 << std::endl;
    std::cout << "Avg Distances comps: " << faiss::indexIVF_stats.ndis / queryNumVectors << std::endl;
    std::cout << "Query time: " << duration_search << " ms" << std::endl;
}

double get_recall(ReclusteringIndex &index, float *queryVecs, size_t queryDimension, size_t queryNumVectors, int k,
                  vector_idx_t *gtVecs, int nMegaProbes, int nMiniProbes) {
    // search
    double recall = 0;
    ReclusteringIndexStats stats;
    for (int i = 0; i < queryNumVectors; i++) {
        std::priority_queue<NodeDistCloser> results;
        index.search(queryVecs + i * queryDimension, k, results, nMegaProbes, nMiniProbes, stats);
        auto gt = gtVecs + i * k;
        while (!results.empty()) {
            auto res = results.top();
            results.pop();
            if (std::find(gt, gt + k, res.id) != (gt + k)) {
                recall++;
            }
        }
    }
    printf("Avg Distance Computation: %llu\n", stats.numDistanceCompForSearch / queryNumVectors);
    return recall / queryNumVectors;
}

double get_recall_with_bad_clusters(ReclusteringIndex &index, float *queryVecs, size_t queryDimension, size_t queryNumVectors, int k,
                  vector_idx_t *gtVecs, int nMegaProbes, int nMiniProbes, int nMiniProbesForBadCluster, bool skipBadClusters) {
    // search
    double recall = 0;
    ReclusteringIndexStats stats;
    for (int i = 0; i < queryNumVectors; i++) {
        std::priority_queue<NodeDistCloser> results;
        index.searchWithBadClusters(queryVecs + i * queryDimension, k, results, nMegaProbes, nMiniProbes, nMiniProbesForBadCluster, stats, skipBadClusters);
        auto gt = gtVecs + i * k;
        while (!results.empty()) {
            auto res = results.top();
            results.pop();
            if (std::find(gt, gt + k, res.id) != (gt + k)) {
                recall++;
            }
        }
    }
    printf("Avg Distance Computation: %llu\n", stats.numDistanceCompForSearch / queryNumVectors);
    return recall / queryNumVectors;
}

double get_quantized_recall(ReclusteringIndex &index, float *queryVecs, size_t queryDimension, size_t queryNumVectors, int k,
                  vector_idx_t *gtVecs, int nMegaProbes, int nMiniProbes) {
    // search
    double recall = 0;
    ReclusteringIndexStats stats;
    for (int i = 0; i < queryNumVectors; i++) {
        std::priority_queue<NodeDistCloser> results;
        index.searchQuantized(queryVecs + i * queryDimension, k, results, nMegaProbes, nMiniProbes, stats);
        auto gt = gtVecs + i * k;
        while (!results.empty()) {
            auto res = results.top();
            results.pop();
            if (std::find(gt, gt + k, res.id) != (gt + k)) {
                recall++;
            }
        }
    }
    printf("Avg Distance Computation: %llu\n", stats.numDistanceCompForSearch / queryNumVectors);
    return recall / queryNumVectors;
}

void benchmark_reclustering_index(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    const std::string &groundTruthPath = input.getCmdOption("-groundTruthPath");
    const int numInserts = stoi(input.getCmdOption("-numInserts"));
    const int numVectors = stoi(input.getCmdOption("-numVectors"));
    const int k = stoi(input.getCmdOption("-k"));
    const int numIters = stoi(input.getCmdOption("-numIters"));
    const int megaCentroidSize = stoi(input.getCmdOption("-megaCentroidSize"));
    const int miniCentroidSize = stoi(input.getCmdOption("-miniCentroidSize"));
    const int newMiniCentroidSize = stoi(input.getCmdOption("-newMiniCentroidSize"));
    const float lambda = stof(input.getCmdOption("-lambda"));
    const int numMegaReclusterCentroids = stoi(input.getCmdOption("-numMegaReclusterCentroids"));
    const int numNewMiniReclusterCentroids = stoi(input.getCmdOption("-numNewMiniReclusterCentroids"));
    const int nMegaProbes = stoi(input.getCmdOption("-nMegaProbes"));
    const int nMiniProbes = stoi(input.getCmdOption("-nMiniProbes"));
    const int readFromDisk = stoi(input.getCmdOption("-readFromDisk"));
    const std::string &storagePath = input.getCmdOption("-storagePath");

    // Read dataset
    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);

    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    baseNumVectors = std::min(baseNumVectors, (size_t) numVectors);

    ReclusteringIndexConfig config(numIters, megaCentroidSize, miniCentroidSize, newMiniCentroidSize, lambda, 0.4, L2,
                                   numMegaReclusterCentroids, numNewMiniReclusterCentroids);
    CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    loadFromFile(groundTruthPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));

    RandomGenerator rng(1234);
    ReclusteringIndex index(baseDimension, config, &rng);

    if (readFromDisk) {
        index = ReclusteringIndex(storagePath, &rng);
    } else {
        printf("Building index\n");
        auto chunkSize = baseNumVectors / numInserts;
        printf("Chunk size: %d\n", chunkSize);
        for (long i = 0; i < numInserts; i++) {
            auto start = i * chunkSize;
            auto end = (i + 1) * chunkSize;
            if (i == (numInserts - 1)) {
                end = baseNumVectors;
            }
            printf("processing chunk: %d, start: %lu, end: %lu\n", i, start, end);
            index.insert(baseVecs + start * baseDimension, end - start);

            printf("performing merging of mega centroids\n");
            index.mergeNewMiniCentroids();
        }

        printf("Writing index to disk\n");
        index.flush_to_disk(storagePath);
    }

    index.storeScoreForMegaClusters();
    index.printStats();

    auto recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbes, nMiniProbes);
    printf("Recall: %f\n", recall);
    index.reclusterAllMegaCentroids();
    recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbes, nMiniProbes);
    printf("Recall: %f\n", recall);

    index.storeScoreForMegaClusters();
    index.printStats();
}

void benchmark_fast_reclustering(InputParser &input) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    const std::string &groundTruthPath = input.getCmdOption("-groundTruthPath");
    const bool isParquet = stoi(input.getCmdOption("-isParquet"));
    int numInserts = stoi(input.getCmdOption("-numInserts"));
    const int numVectors = stoi(input.getCmdOption("-numVectors"));
    const int k = stoi(input.getCmdOption("-k"));
    const int numIters = stoi(input.getCmdOption("-numIters"));
    const int megaCentroidSize = stoi(input.getCmdOption("-megaCentroidSize"));
    const int miniCentroidSize = stoi(input.getCmdOption("-miniCentroidSize"));
    const float lambda = stof(input.getCmdOption("-lambda"));
    const int numMegaReclusterCentroids = stoi(input.getCmdOption("-numMegaReclusterCentroids"));
    const int reclusterOnScore = stoi(input.getCmdOption("-reclusterOnScore"));
    const int nMegaProbes = stoi(input.getCmdOption("-nMegaProbes"));
    const int nMiniProbes = stoi(input.getCmdOption("-nMiniProbes"));
    const int iterations = stoi(input.getCmdOption("-iterations"));
    // const bool fast = stoi(input.getCmdOption("-fast"));
    const int numQueries = stoi(input.getCmdOption("-numQueries"));
    const int readFromDisk = stoi(input.getCmdOption("-readFromDisk"));
    const std::string &storagePath = input.getCmdOption("-storagePath");
    const int numThreads = stoi(input.getCmdOption("-numThreads"));
    const bool useIP = stoi(input.getCmdOption("-useIP"));
    const float quantTrainPercentage = stof(input.getCmdOption("-quantTrainPercentage"));
    const bool quantBuild = stoi(input.getCmdOption("-quantBuild"));
    const int avgSubCellSize = stoi(input.getCmdOption("-avgSubCellSize"));
    const int nMiniProbesForBadCluster = stoi(input.getCmdOption("-nMiniProbesForBadCluster"));
    const int nMegaRecluster = stoi(input.getCmdOption("-nMegaRecluster"));
    omp_set_num_threads(numThreads);

    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    queryNumVectors = std::min(queryNumVectors, (size_t) numQueries);

    DistanceType distanceType = useIP ? COSINE : L2;
    ReclusteringIndexConfig config(numIters, megaCentroidSize, miniCentroidSize, 0, lambda, 0.4, distanceType,
                                   0, 0, quantTrainPercentage);
    // CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    loadFromFile(groundTruthPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));

    RandomGenerator rng(1234);
    ReclusteringIndex index(queryDimension, config, &rng);

    if (readFromDisk) {
        index = ReclusteringIndex(storagePath, &rng);
    } else {
        // Read dataset
        size_t baseDimension, baseNumVectors;
        float *baseVecs;
        std::vector<std::string> filePaths;
        if (isParquet) {
            list_parquet_dir(baseVectorPath.c_str(), filePaths);
            if (filePaths.empty()) {
                fprintf(stderr, "No parquet files found in the directory: %s\n", baseVectorPath.c_str());
                exit(1);
            }
            auto status = readParquetFileStats(filePaths.at(0).c_str(), &baseDimension, &baseNumVectors);
            if (!status.ok()) {
                fprintf(stderr, "Failed to read parquet file stats: %s\n", status.ToString().c_str());
                exit(1);
            }
        } else {
            baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
        }
        baseNumVectors = std::min(baseNumVectors, (size_t) numVectors);
        assert(baseDimension == queryDimension);
        if (isParquet) {
            auto numFiles = filePaths.size();
            numInserts = std::min(numInserts, (int)numFiles);
            auto chunkSize = numFiles / numInserts;
            printf("Reading parquet files: %zu\n", numFiles);
            auto totalVectors = 0;
            for (int i = 0; i < numInserts; i++) {
                auto start = i * chunkSize;
                auto end = (i + 1) * chunkSize;
                if (i == (numInserts - 1)) {
                    end = numFiles;
                }
                std:vector<std::string> paths;
                printf("Processing parquet files: %d, start: %d, end: %d\n", i, start, end);
                for (int j = start; j < end; j++) {
                    paths.push_back(filePaths[j]);
                    printf("Working on parquet file: %s\n", filePaths[j].c_str());
                }
                auto data = readParquetFiles(paths, &baseDimension, &baseNumVectors);
                index.naiveInsert(data, baseNumVectors);
                totalVectors += baseNumVectors;
                delete[] data;
            }
            printf("Total vectors inserted: %d\n", totalVectors);
        } else {
            if (quantBuild) {
                index.trainQuant(baseVecs, baseNumVectors);
            }
            printf("Building index\n");
            auto chunkSize = baseNumVectors / numInserts;
            printf("Chunk size: %lu\n", chunkSize);
            for (long i = 0; i < numInserts; i++) {
                auto start = i * chunkSize;
                auto end = (i + 1) * chunkSize;
                if (i == (numInserts - 1)) {
                    end = baseNumVectors;
                }
                printf("processing chunk: %d, start: %lu, end: %lu\n", i, start, end);
                if (quantBuild) {
                    index.naiveInsertQuant(baseVecs + start * baseDimension, end - start);
                } else {
                    index.naiveInsert(baseVecs + start * baseDimension, end - start);
                }
            }
        }
        printf("Writing index to disk\n");
        index.flush_to_disk(storagePath);
    }
    // index.quantizeVectors();
    auto recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbes,
                                 nMiniProbes);
    // auto recallWithoutBadClusters = get_recall_with_bad_clusters(index, queryVecs, queryDimension,
    //                                         queryNumVectors, k, gtVecs, nMegaProbes, nMiniProbes,
    //                                         nMiniProbesForBadCluster, true);
    //
    // auto recallWithBadCluster = get_recall_with_bad_clusters(index, queryVecs, queryDimension,
    //                                         queryNumVectors, k, gtVecs, nMegaProbes, nMiniProbes,
    //                                         nMiniProbesForBadCluster, false);

    // index.computeAllSubCells(avgSubCellSize);
    // auto quantizedRecall = get_quantized_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs,
    //                                             nMegaProbes, nMiniProbes);
    // printf("Recall: %f, Recall without bad clusters: %f, Recall with bad clusters: %f\n", recall, recallWithoutBadClusters, recallWithBadCluster);
    index.printStats();

    // index.storeScoreForMegaClusters();
    // index.flush_to_disk(storagePath);
    for (int iter = 0; iter < iterations; iter++) {
        printf("Iteration: %d\n", iter);
        // index.reclusterAllMiniCentroidsQuant();
        index.reclusterAllMegaCentroids();
        // recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbes,
        //                  nMiniProbes);
        // index.printStats();
        // quantizedRecall = get_quantized_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs,
                                             // nMegaProbes, nMiniProbes);
        printf("After reclustering only mega centroids, Recall: %f\n", recall);
        if (numMegaReclusterCentroids == 1) {
            index.reclusterFast(nMegaProbes);
        } else {
            if (reclusterOnScore) {
                index.reclusterBasedOnScore(numMegaReclusterCentroids);
            } else {
                index.reclusterFull(numMegaReclusterCentroids);
            }
        }
        // index.quantizeVectors();
        recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbes,
                                 nMiniProbes);
        // quantizedRecall = get_quantized_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs,
        //                              nMegaProbes, nMiniProbes);
        printf("After micro reclustering, recall: %f\n", recall);
        index.storeScoreForMegaClusters();
        index.printStats();
        printf("Done iteration: %d\n", iter);
    }
    if (iterations > 0) {
        printf("Flushing to disk\n");
        index.flush_to_disk(storagePath);
    }
}

double get_recall(IncrementalIndex &index, float *queryVecs, size_t queryDimension, size_t queryNumVectors, int k,
                  vector_idx_t *gtVecs, int nMegaProbes, int nMicroProbes) {
    IncrementalIndexStats stats;
    // search
    double recall = 0;
    for (int i = 0; i < queryNumVectors; i++) {
        std::priority_queue<NodeDistCloser> results;
        index.search(queryVecs + i * queryDimension, k, results, nMegaProbes, nMicroProbes, stats);
        auto gt = gtVecs + i * k;
        while (!results.empty()) {
            auto res = results.top();
            results.pop();
            if (std::find(gt, gt + k, res.id) != (gt + k)) {
                recall++;
            }
        }
    }
    printf("Avg Distance Computation: %llu\n", stats.numDistanceComp / queryNumVectors);
    return recall / queryNumVectors;
}

void benchmark_splitting(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    const std::string &groundTruthPath = input.getCmdOption("-groundTruthPath");
    const int numInserts = stoi(input.getCmdOption("-numInserts"));
    const int numVectors = stoi(input.getCmdOption("-numVectors"));
    const int k = stoi(input.getCmdOption("-k"));
    const int numIters = stoi(input.getCmdOption("-numIters"));
    const int avgCentroidSize = stoi(input.getCmdOption("-avgCentroidSize"));
    const int nMegaProbes = stoi(input.getCmdOption("-nMegaProbes"));
    const int nMicroProbes = stoi(input.getCmdOption("-nMicroProbes"));
    const float lambda = stof(input.getCmdOption("-lambda"));
    const int readFromDisk = stoi(input.getCmdOption("-readFromDisk"));
    const std::string &storagePath = input.getCmdOption("-storagePath");

    // Read dataset
    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    baseNumVectors = std::min(baseNumVectors, (size_t) numVectors);
    auto chunkSize = baseNumVectors / numInserts;
    auto numCentroids = chunkSize / avgCentroidSize;
    printf("Chunk size: %lu, Num centroids: %lu\n", chunkSize, numCentroids);
    IncrementalIndexConfig config(numCentroids, numIters, avgCentroidSize, lambda, 0.4, L2);

    CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    loadFromFile(groundTruthPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));

    RandomGenerator rng(1234);
    IncrementalIndex index(baseDimension, config, &rng);

    std::vector<double> scores;
    int numSplits = 0;
    if (readFromDisk) {
        index = IncrementalIndex(storagePath, &rng);
    } else {
        printf("Building index\n");
        for (long i = 0; i < numInserts; i++) {
            auto start = i * chunkSize;
            auto end = (i + 1) * chunkSize;
            if (i == (numInserts - 1)) {
                end = baseNumVectors;
            }
            printf("processing chunk: %ld, start: %lu, end: %lu\n", i, start, end);
            index.insert(baseVecs + start * baseDimension, end - start);
            scores.push_back(index.computeSilhouetteMetricOnMicroCentroids());
            numSplits += index.splitMicro();
            scores.push_back(index.computeSilhouetteMetricOnMicroCentroids());
        }

        printf("Writing index to disk\n");
        index.flush_to_disk(storagePath);
    }
    index.printStats();
    printf("Num splits: %d\n", numSplits);
    int iter = 0;
    for (int i = 0; i < scores.size(); i+=2) {
        printf("Silhouette score after %d insert and split: %f->%f\n", iter, scores[i], scores[i+1]);
        iter += 1;
    }

    auto initRecall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbes,
        nMicroProbes);
    // std::vector<double> recalls;
    // for (int i = 0; i < numReclusters; i++) {
    //     auto recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbes,
    //     nMicroProbes);
    //     recalls.push_back(recall);
    // }
    // index.printStats();
    // auto final_recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbes,
    //     nMicroProbes);
    printf("Recall: %f\n", initRecall);
    // for (int i = 0; i < numReclusters; i++) {
    //     printf("Recall after reclustering %d: %f\n", i, recalls[i]);
    // }
    // printf("Final Recall: %f\n", final_recall);
}

void generate_quantized_vectors() {
    const int dims = 34;
    const int numVectors = 10000;

    // Generate random vectors with 50 dimensions
    std::vector<float> random_vecs(numVectors * dims);
    RandomGenerator rng(1234);
    for (int i = 0; i < numVectors * dims; i++) {
        random_vecs[i] = rng.randFloat();
    }

    // Normalize vectors
    std::vector<float> normalize_vecs(numVectors * dims);
    normalize_vectors(random_vecs.data(), dims, numVectors, normalize_vecs.data());

    SQ8Bit quantizer(dims);
    quantizer.batch_train(numVectors, normalize_vecs.data());

    // print vmin and vmax
    printf("vmin and vmax for each dimension:\n");
    for (int i = 0; i < dims; i++) {
        printf("%f, ", quantizer.vmin[i]);
    }
    printf("\n");
    for (int i = 0; i < dims; i++) {
        printf("%f, ", quantizer.vdiff[i]);
    }
    printf("\n");

    quantizer.finalize_train();

    std::vector<uint8_t> quantized_vectors(numVectors * quantizer.codeSize);
    quantizer.encode(normalize_vecs.data(), quantized_vectors.data(), numVectors);

    // print first encoded vector, and random normalized vector
    printf("First normalized vector: ");
    for (int i = 0; i < dims; i++) {
        printf("%f, ", normalize_vecs[i]);
    }
    printf("\n");
    printf("First encoded vector: ");
    for (int i = 0; i < quantizer.codeSize; i++) {
        printf("%d, ", quantized_vectors[i]);
    }
    printf("\n");
    printf("random normalized vector: ");
    for (int i = dims * 5485; i < dims * 5486; i++) {
        printf("%f, ", normalize_vecs[i]);
    }
    printf("\n");
    printf("random quantized vector: ");
    for (int i = quantizer.codeSize * 5485; i < quantizer.codeSize * 5486; i++) {
        printf("%d, ", quantized_vectors[i]);
    }
}

void benchmark_quantized_dc(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    const int n = stoi(input.getCmdOption("-n"));
    const int M = stoi(input.getCmdOption("-M"));
    const int nBits = stoi(input.getCmdOption("-nBits"));

    // Read dataset
    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);

    faiss::IndexPQ indexPQ(baseDimension, M, nBits, faiss::MetricType::METRIC_L2);

    printf("Training index\n");
    indexPQ.train(baseNumVectors, baseVecs);

    printf("Adding base vectors\n");
    indexPQ.add(baseNumVectors, baseVecs);

    printf("Computing symmetric distances\n");
    indexPQ.pq.compute_sdc_table();
    auto dc = indexPQ.get_FlatCodesDistanceComputer();
    dc->set_query(queryVecs);
    auto start = std::chrono::high_resolution_clock::now();
    double dist = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < baseNumVectors; j++) {
            dist += dc->symmetric_dis(0, j);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    printf("Symmetric Distance: %f\n", dist);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // Number of distance computations per sec
    printf("Symmetric Distance computation time: %lld ms\n", duration.count());
    printf("Symmetric Distance computation per sec: %f\n", (n * baseNumVectors) / (duration.count() / 1000.0));

    printf("Computing asymmetric distances\n");
    start = std::chrono::high_resolution_clock::now();
    dist = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < baseNumVectors; j++) {
            dist += (*dc)(j);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    printf("Asymmetric Distance: %f\n", dist);
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // Number of distance computations per sec
    printf("Asymmetric Distance computation time: %lld ms\n", duration.count());
    printf("Asymmetric Distance computation per sec: %f\n", (n * baseNumVectors) / (duration.count() / 1000.0));

    // Run normal distance computation
    printf("Computing non quantized distances\n");
    start = std::chrono::high_resolution_clock::now();
    dist = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < baseNumVectors; j++) {
            dist += faiss::fvec_L2sqr(queryVecs, baseVecs + j * baseDimension, baseDimension);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    printf("Actual Distance: %f\n", dist);
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // Number of distance computations per sec
    printf("Total Distance Computations: %lu\n", n * baseNumVectors);
    printf("Actual Distance computation time: %lld ms\n", duration.count());
    printf("Actual Distance computation per sec: %f\n", (n * baseNumVectors) / (duration.count() / 1000.0));
}

void read_parquet_file(InputParser &input) {
    const std::string &dirPath = input.getCmdOption("-dirPath");
    size_t numVectors, dim;
    // Read parquet file
    auto data = readParquetDir(dirPath.c_str(), &dim, &numVectors);
    // Print status
    delete data;
}

void check_omp_threads(InputParser &input) {
    const int numThreads = stoi(input.getCmdOption("-numThreads"));
    omp_set_num_threads(numThreads);
    printf("Number of OpenMP threads set to: %d\n", omp_get_num_threads());
    // Print the thread id
    auto thread_id = std::this_thread::get_id();
    printf("Current thread id: %lu\n", std::hash<std::thread::id>()(thread_id));
    auto num = 100000000;
#pragma omp parallel
    {
        auto main_thread_id = std::hash<std::thread::id>()(thread_id);
#pragma omp for
        for (auto i = 0; i < num; i++) {
            auto new_id = std::hash<std::thread::id>()(std::this_thread::get_id());
            if (new_id != main_thread_id) {
                printf("Thread id changed from %lu to %lu\n", main_thread_id, new_id);
                main_thread_id = new_id;
            }
        }
    }
}

int main(int argc, char **argv) {
    backward::SignalHandling sh;
//    benchmarkPairWise();
    InputParser input(argc, argv);
//    benchmark_quantization(input);
//    calculate_dists(input);
    const std::string &run = input.getCmdOption("-run");
    if (run == "benchmark") {
        benchmark_hnsw_queries(input);
    } else if (run == "generateGT") {
        generateGroundTruth(input);
    } else if (run == "generateFilterGT") {
        generateFilterGroundTruth(input);
    } else if (run == "benchmarkFiltered") {
        benchmark_filtered_hnsw_queries(input);
    } else if (run == "benchmarkAcorn") {
        // benchmark_acorn(input);
    } else if (run == "generateGTParquet") {
        generateGroundTruthParquet(input);
    }
#if 0
    else if (run == "benchmarkIoUring") {
        benchmark_io_uring(input);
    } else if (run == "benchmarkPread") {
        benchmark_pread(input);
    }
#endif
    else if (run == "benchmarkClustering") {
        test_clustering_data(input);
    }
    else if (run == "benchmarkReclusteringIndex") {
        benchmark_reclustering_index(input);
    }
    else if (run == "benchmarkSplitting") {
        benchmark_splitting(input);
    }
    else if (run == "benchmarkQuantized") {
        benchmark_quantized_dc(input);
    }
    else if (run == "benchmarkFastReclustering") {
        benchmark_fast_reclustering(input);
    }
    else if (run == "benchmarkNavix") {
        benchmark_navix(input);
    }
    else if (run == "benchmarkIRangeGraph") {
        benchmark_irangegraph(input);
    }
    else if (run == "benchmarkFaissClustering") {
        benchmark_faiss_clustering(input);
    }
    else if (run == "generateQuantizedData") {
        generate_quantized_vectors();
    }
    else if (run == "readParquetFile") {
        read_parquet_file(input);
    }
    else if (run == "checkOmpThreads") {
        check_omp_threads(input);
    }
//    testParallelPriorityQueue();
//    benchmark_simd_distance();
//    benchmark_n_simd(5087067004);
//    benchmark_random_dist_comp();
//    benchmark_scalar_quantizer();
//    benchmark_quantizer();
//    benchmark_explore_data();
//    benchmarkClustering(argc, argv);
//    benchmarkSimSimd();
    return 0;
}
