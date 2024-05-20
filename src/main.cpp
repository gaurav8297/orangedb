#include <iostream>
#include "hnsw.h"
#include "scalar_quantizer.h"
#include "spdlog/fmt/fmt.h"

#ifdef __AVX2__
#include <x86intrin.h>
#endif

#include <stdlib.h>    // atoi, getenv
#include <assert.h>    // assert
#include <simsimd/simsimd.h>
#include "include/partitioned_index.h"

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
    hnsw.build(baseVecs, baseNumVectors);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Building time: " << duration << " ms" << std::endl;
}

void query_graph(
        HNSW &hnsw,
        const float *queryVecs,
        size_t queryNumVectors,
        size_t queryDimension,
        const int *gtVecs,
        size_t k,
        size_t ef_search,
        size_t baseNumVectors) {
    auto start = std::chrono::high_resolution_clock::now();
    auto visited = VisitedTable(baseNumVectors);
    auto recall = 0.0;
    Stats stats{};
    for (size_t i = 0; i < queryNumVectors; i++) {
        std::priority_queue<NodeDistCloser> results;
        std::vector<NodeDistFarther> res;
        hnsw.search(queryVecs + (i * queryDimension), k, ef_search, visited, results, stats);
        int s = 0;
        while (!results.empty() && s < k) {
            auto top = results.top();
            res.emplace_back(top.id, top.dist);
            results.pop();
            s++;
        }
        auto gt = gtVecs + i * 100;
        for (auto &result: res) {
            if (std::find(gt, gt + 100, result.id) != (gt + 100)) {
                recall++;
            }
        }
    }
    stats.logStats();
    std::cout << "Recall: " << recall / queryNumVectors << std::endl;
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

void benchmark_hnsw_queries(int argc, char **argv) {
    InputParser input(argc, argv);
    const std::string &basePath = input.getCmdOption("-basePath");
    auto efConstruction = stoi(input.getCmdOption("-efConstruction"));
    auto M = stoi(input.getCmdOption("-M"));
    auto efSearch = stoi(input.getCmdOption("-efSearch"));
    auto thread_count = stoi(input.getCmdOption("-nThreads"));
    auto num_vectors = stoi(input.getCmdOption("-numVectors"));
    auto alpha = stof(input.getCmdOption("-alpha"));

    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);
    auto queryVectorPath = fmt::format("{}/query.fvecs", basePath);
    auto groundTruthPath = fmt::format("{}/groundtruth.ivecs", basePath);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readFvecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readFvecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    size_t gtDimension, gtNumVectors;
    int *gtVecs = readIvecFile(groundTruthPath.c_str(), &gtDimension, &gtNumVectors);

    omp_set_num_threads(thread_count);
    RandomGenerator rng(1234);
    HNSWConfig config(M, efConstruction, efSearch, alpha);
    HNSW hnsw(config, &rng, baseDimension);
    build_graph(hnsw, baseVecs, num_vectors);
    hnsw.logStats();
    // ./orangedb_main -basePath /home/g3sehgal/vector_index_exp/gist -efConstruction 128 -M 64 -efSearch 150 -nThreads 32 -exploreFactor 1
    query_graph(hnsw, queryVecs, queryNumVectors, queryDimension, gtVecs, 100, efSearch, baseNumVectors);
}

//void benchmark_scalar_quantizer() {
//    auto basePath = "/home/gaurav/vector_index_experiments/vector_index/data/gist_50k";
//    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);
//
//    size_t baseDimension, baseNumVectors;
//    float *baseVecs = Utils::fvecs_read(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
//    uint8_t *codes;
//    Utils::alloc_aligned((void **) &codes, baseNumVectors * baseDimension, 64);
//    printf("Base dimension: %zu, Base num vectors: %zu\n", baseDimension, baseNumVectors);
//    float *vmin;
//    float *vdiff;
//    ScalarQuantizer sq(baseDimension, vmin, vdiff);
//    sq.train(baseNumVectors, baseVecs);
//    sq.compute_codes(baseVecs, codes, baseNumVectors);
////    float *decoded;
////    Utils::alloc_aligned((void**)&decoded, baseNumVectors * baseDimension * sizeof(float), 64);
////    sq.decode(codes, decoded, baseNumVectors);
//    Storage storage(baseDimension, 64, 2);
//    storage.data = baseVecs;
//    storage.codes = codes;
//    storage.vmin = sq.vmin;
//    storage.vdiff = sq.vdiff;
//
//    // Print 0 dimension for first 10 vectors
////    auto error = 0.0;
////    for (int i = 0; i < baseNumVectors; i++) {
////        printf("Actual 0th location %f\n", baseVecs[i * 960]);
////        printf("Decoded 0th location %f\n", decoded[i * 960]);
////        error += baseVecs[i * 960] - decoded[i * 960];
////    }
////    printf("Error: %f\n", (error / baseNumVectors));
//
//    L2DistanceComputer normaldc(storage.data, baseDimension, baseNumVectors);
//    SQDistanceComputer sqdc(storage.codes, baseDimension, baseNumVectors, storage.vmin, storage.vdiff);
//    normaldc.setQuery(baseVecs);
//    sqdc.setQuery(baseVecs);
//
//    auto error = 0.0;
//    for (int i = 1; i < baseNumVectors; i++) {
//        float actual, fake;
//        normaldc.computeDistance(4, i, actual);
//        sqdc.computeDistance(4, i, fake);
//        printf("Actual: %f, Fake: %f\n", actual, fake);
//        error += fake - actual;
//    }
//    printf("Error: %f\n", (error / baseNumVectors));
//
//    // TODO - Calculate distance between two vectors. First convert to float and then calc distances.
//    // TODO - Try other way where normalize the equation and precompute some values to reduce computation/
////    sq.print_stats();
//}

void benchmark_quantizer() {
    auto val = 0.5645566777f;
    uint8_t quantized = int(val * 255.0f);
    float dequantized_v1 = quantized / 255.0f;
    printf("Quantized: %d, Dequantized: %f\n", quantized, dequantized_v1);
    float dequantized_v2 = (quantized + 0.5f) / 255.0f;
    printf("Dequantized: %f\n", dequantized_v2);
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
    for (int i = 0; i < queryNumVectors; i++) {
        std::vector<NodeDistFarther> results;
        Stats stats{};
        partitionedIndex.search(queryVecs + i * queryDimension, K, results, stats);
        auto gt = gtVecs + i * gtDimension;
        for (auto res: results) {
            if (std::find(gt, gt + gtDimension, res.id) != (gt + gtDimension)) {
                recall++;
            }
        }
    }
    std::cout << "Recall: " << recall / queryNumVectors << std::endl;
}

int main(int argc, char **argv) {
    benchmark_hnsw_queries(argc, argv);
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
