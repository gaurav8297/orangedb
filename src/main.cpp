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
#include <fastQ/scalar_8bit.h>
#include <fastQ/pair_wise.h>
#include "helper_ds.h"
#include <fastQ/common.h>

using namespace orangedb;

#if defined(__GNUC__)
#define PRAGMA_IMPRECISE_LOOP
#define PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("GCC push_options") \
    _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define PRAGMA_IMPRECISE_FUNCTION_END \
    _Pragma("GCC pop_options")
#endif

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

void build_graph(HNSW &hnsw, const float *baseVecs, size_t baseNumVectors) {
    auto start = std::chrono::high_resolution_clock::now();
    auto qq1 = baseVecs + (18530806 * 128);
    for (int i = 0; i < 10; i++) {
        printf("qq1: %f\n", qq1[i]);
    }
    printf("pointer address: %p\n", baseVecs);
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
    auto start = std::chrono::high_resolution_clock::now();
    auto visited = VisitedTable(baseNumVectors);
    auto recall = 0.0;
    Stats stats{};
    printf("baseNumVectors: %zu\n", baseNumVectors);
    for (size_t i = 0; i < queryNumVectors; i++) {
        auto localRecall = 0.0;
        std::priority_queue<NodeDistCloser> results;
        std::vector<NodeDistFarther> res;
        hnsw.searchWithFilter(queryVecs + (i * queryDimension), k, ef_search, visited, results, filteredMask + (i * baseNumVectors), stats);
        while (!results.empty()) {
            auto top = results.top();
            res.emplace_back(top.id, top.dist);
            printf("ID: %d, Dist: %f\n", top.id, top.dist);
            results.pop();
        }
        auto gt = gtVecs + i * k;
        for (auto &result: res) {
            if (std::find(gt, gt + k, result.id) != (gt + k)) {
                recall++;
                localRecall++;
            }
        }
        printf("Recall: %f\n", localRecall);
        break;
//        printf("Recall: %f\n", localRecall);
    }
    auto recallPerQuery = recall / queryNumVectors;
    stats.logStats();
    std::cout << "Total Vectors: " << queryNumVectors << std::endl;
    std::cout << "Recall: " << (recallPerQuery / k) * 100 << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Query time: " << duration << " ms" << std::endl;
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
    L2DistanceComputer dc(vectors, dim, numVectors);
#pragma omp parallel
    {
        auto localDc = dc.clone();
        IndexOneNN index(localDc.get(), dim, numVectors);
#pragma omp for schedule(static)
        for (size_t i = 0; i < queryNumVectors; i++) {
            double dists[k];
            index.knnFiltered(k, queryVecs + i * dim, dists, gtVecs + i * k, filteredMask + i * numVectors);
            for (int j = 0; j < k; j++) {
                printf("GT: %llu, Dist: %f\n", gtVecs[i * k + j], dists[j]);
            }
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

void generateFilterGroundTruth(InputParser &input) {
    const std::string &basePath = input.getCmdOption("-basePath");
    auto k = stoi(input.getCmdOption("-k"));
    const std::string &gtPath = input.getCmdOption("-gtPath");
    auto selectivity = stof(input.getCmdOption("-selectivity"));
    const std::string &filteredMaskPath = input.getCmdOption("-filteredMaskPath");
    auto baseVectorPath = fmt::format("{}/base.bvecs", basePath);
    auto queryVectorPath = fmt::format("{}/query.bvecs", basePath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readBvecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readBvecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];

    // print first 2 vectors
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < baseDimension; j++) {
            printf("%f ", baseVecs[i * baseDimension + j]);
        }
        printf("\n");
    }

    // print first 2 query vectors
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < queryDimension; j++) {
            printf("%f ", queryVecs[i * queryDimension + j]);
        }
        printf("\n");
    }

    printf("Base vectors: %zu, Query vectors: %zu\n", baseNumVectors, queryNumVectors);
    printf("Base dimension: %zu, Query dimension: %zu\n", baseDimension, queryDimension);

    auto *filteredMask = new uint8_t[queryNumVectors * baseNumVectors];
    memset(filteredMask, 0, queryNumVectors * baseNumVectors);
    setFilterMaskUsingSelectivity(queryNumVectors, filteredMask, baseNumVectors, selectivity);
    generateFilterGroundTruth(baseVecs, baseDimension, baseNumVectors, queryVecs, filteredMask, queryNumVectors, k, gtVecs);
    // serialize gtVecs to a file
//    writeToFile(gtPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));
    // serialize filteredMask to a file
//    writeToFile(filteredMaskPath, filteredMask, queryNumVectors * baseNumVectors);
}

void generateGroundTruth(
        const float* vectors,
        size_t dim,
        size_t numVectors,
        float *queryVecs,
        size_t queryNumVectors,
        int k,
        vector_idx_t *gtVecs) {
    CosineDistanceComputer dc(vectors, dim, numVectors);
#pragma omp parallel
    {
        auto localDc = dc.clone();
        IndexOneNN index(localDc.get(), dim, numVectors);
#pragma omp for schedule(dynamic, 100)
        for (size_t i = 0; i < queryNumVectors; i++) {
            double dists[k];
            index.knn(k, queryVecs + i * dim, dists, gtVecs + i * k);
        }
    }
}

void generateGroundTruth(InputParser &input) {
    const std::string &basePath = input.getCmdOption("-basePath");
    auto k = stoi(input.getCmdOption("-k"));
    const std::string &gtPath = input.getCmdOption("-gtPath");
    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);
    auto queryVectorPath = fmt::format("{}/query.fvecs", basePath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readFvecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readFvecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];

    generateGroundTruth(baseVecs, baseDimension, baseNumVectors, queryVecs, queryNumVectors, k, gtVecs);
    // serialize gtVecs to a file
    writeToFile(gtPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));
}

void benchmark_filtered_hnsw_queries(InputParser &input) {
    const std::string &basePath = input.getCmdOption("-basePath");
    auto efConstruction = stoi(input.getCmdOption("-efConstruction"));
    auto M = stoi(input.getCmdOption("-M"));
    auto efSearch = stoi(input.getCmdOption("-efSearch"));
    auto thread_count = stoi(input.getCmdOption("-nThreads"));
    auto minAlpha = stof(input.getCmdOption("-minAlpha"));
    auto maxAlpha = stof(input.getCmdOption("-maxAlpha"));
    auto alphaDecay = stof(input.getCmdOption("-alphaDecay"));
    auto k = stoi(input.getCmdOption("-k"));
    auto filterMinK = stoi(input.getCmdOption("-filterMinK"));
    auto selectivity = stoi(input.getCmdOption("-selectivity"));
    auto maxNeighboursCheck = stoi(input.getCmdOption("-maxNeighboursCheck"));
    bool loadFromStorage = stoi(input.getCmdOption("-loadFromDisk"));

    auto baseVectorPath = fmt::format("{}/base.bvecs", basePath);
    auto queryVectorPath = fmt::format("{}/query.bvecs", basePath);
    auto groundTruthPath = fmt::format("{}/{}_gt.bin", basePath, selectivity);
    auto maskPath = fmt::format("{}/{}_mask.bin", basePath, selectivity);
    auto storagePath = fmt::format("{}/storage.bin", basePath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readBvecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readBvecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    loadFromFile(groundTruthPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));
    auto *filteredMask = new uint8_t[queryNumVectors * baseNumVectors];
    loadFromFile(maskPath, filteredMask, queryNumVectors * baseNumVectors);
    printf("Base num vectors: %zu\n", baseNumVectors);
    auto q = baseVecs + (18530806 * baseDimension);
    for (int i = 0; i < 10; i++) {
        printf("q vec %f ", q[i]);
    }

    // Print grond truth num vectors
    printf("Query num vectors: %zu\n", queryNumVectors);
    printf("Query dimension: %zu\n", baseDimension);

    omp_set_num_threads(thread_count);
    RandomGenerator rng(1234);
    HNSWConfig config(M, efConstruction, efSearch, minAlpha, maxAlpha, alphaDecay, filterMinK, maxNeighboursCheck,
                      "none", storagePath, loadFromStorage, 20, 10, 1, "none");
    HNSW hnsw(config, &rng, baseDimension);
    build_graph(hnsw, baseVecs, baseNumVectors);
    if (!loadFromStorage) {
        hnsw.flushToDisk();
    }
//    generateFilterGroundTruth(baseVecs, baseDimension, baseNumVectors, queryVecs, filteredMask, 1, k, gtVecs);
//    hnsw.logStats();
    query_graph_filter(hnsw, queryVecs, filteredMask, queryNumVectors, queryDimension, gtVecs, k, efSearch, baseNumVectors);
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
    Stats stats{};
    for (size_t i = 0; i < queryNumVectors; i++) {
        auto localRecall = 0.0;
        auto visited = AtomicVisitedTable(baseNumVectors);
        PocTaskScheduler taskScheduler(thread_count, &visited, nodeExpansionPerNode, hnsw.storage, ef_search);
        auto startTime = std::chrono::high_resolution_clock::now();
        std::priority_queue<NodeDistCloser> results;
        std::vector<NodeDistFarther> res;
        hnsw.searchParallel(queryVecs + (i * queryDimension), k, ef_search, visited, results, stats, &taskScheduler);
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
        break;
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

    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);
    auto queryVectorPath = fmt::format("{}/query.fvecs", basePath);
    auto groundTruthPath = fmt::format("{}/gt.bin", basePath);
    auto storagePath = fmt::format("{}/storage.bin", basePath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readFvecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readFvecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
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

    L2DistanceComputer dc(baseVecs, baseDimension, baseNumVectors);

    dc.setQuery(baseVecs + (1 * baseDimension));
    double dist;
    dc.computeDistance(18530814, &dist);
    printf("Dist: %f\n", dist);

    dc.setQuery(baseVecs + (18530806 * baseDimension));
    double dist2;
    dc.computeDistance(18530814, &dist2);
    printf("Dist: %f\n", dist2);

    auto q = baseVecs + (18530806 * baseDimension);
    for (int i = 0; i < 10; i++) {
        printf("%f ", q[i]);
    }
}

int main(int argc, char **argv) {
//    benchmarkPairWise();
    InputParser input(argc, argv);
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
