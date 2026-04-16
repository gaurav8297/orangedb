#include <iostream>
#include "hnsw.h"
#include "spdlog/fmt/fmt.h"

#ifdef __AVX2__
#include <x86intrin.h>
#endif

#include <stdlib.h>    // atoi, getenv
#include <assert.h>    // assert
#include <climits>
#include <cmath>       // isnan, isinf
#include <memory>
#include <random>      // mt19937, uniform_int_distribution
#include <unordered_set>
#include <simsimd/simsimd.h>
#include "include/partitioned_index.h"
#include <fstream>
#include <reclustering_index.h>
#include <faiss/index_io.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/VectorTransform.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/impl/RaBitQuantizer.h>
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
#include "faiss/IndexScalarQuantizer.h"
#include "fastQ/scalar_test.h"
#include "faiss/IndexPQ.h"
#include "umappp/umappp.hpp"
#include "knncolle/knncolle.hpp"
#include <faiss/impl/VisitedTable.h>
// #include <cblas.h>

#ifdef CUVS_ENABLED
#include "cuvs_kmeans.h"
#endif

#if 0
#include <liburing.h>
#endif

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {
int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

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

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

enum CLUSTER_HIRARCHY {
    C_L1,    // 0
    C_L2, // 1
};

enum UMAP_VISUALIZATION_MODE {
    NO_UMAP,    // 0
    LIVE_UMAP, // 1
    OFFLINE_UMAP, // 2
};

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

static size_t get_current_rss_bytes() {
#if defined(__APPLE__)
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
        KERN_SUCCESS) {
        return 0;
    }
    return static_cast<size_t>(info.resident_size);
#elif defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    size_t total_pages = 0;
    size_t resident_pages = 0;
    statm >> total_pages >> resident_pages;
    return resident_pages * static_cast<size_t>(sysconf(_SC_PAGESIZE));
#else
    return 0;
#endif
}

static double bytes_to_mb(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static void print_memory_usage(const char *label) {
    printf("%s RSS: %.2f MB\n", label, bytes_to_mb(get_current_rss_bytes()));
}

static std::unique_ptr<faiss::IndexIVF> create_faiss_ivf_index(
        size_t dimension,
        size_t num_centroids,
        faiss::MetricType metric,
        bool use_scalar_quantizer) {
    auto *quantizer = new faiss::IndexFlat(dimension, metric);
    std::unique_ptr<faiss::IndexIVF> index;
    if (use_scalar_quantizer) {
        index = std::make_unique<faiss::IndexIVFScalarQuantizer>(
                quantizer, dimension, num_centroids, faiss::ScalarQuantizer::QT_8bit, metric);
    } else {
        index = std::make_unique<faiss::IndexIVFFlat>(quantizer, dimension, num_centroids, metric);
    }
    index->own_fields = true;
    return index;
}

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

void writeNestedVectorToFile(const std::string &path, const std::vector<std::vector<vector_idx_t>> &data) {
    std::ofstream outputFile(path, std::ios::binary);
    
    // Write number of outer vectors
    uint64_t numOuter = data.size();
    outputFile.write(reinterpret_cast<const char *>(&numOuter), sizeof(uint64_t));
    
    // Write each inner vector
    for (const auto &inner : data) {
        // Write size of inner vector
        uint64_t innerSize = inner.size();
        outputFile.write(reinterpret_cast<const char *>(&innerSize), sizeof(uint64_t));
        
        // Write inner vector data
        if (innerSize > 0) {
            outputFile.write(reinterpret_cast<const char *>(inner.data()), innerSize * sizeof(vector_idx_t));
        }
    }
    
    outputFile.close();
}

void loadNestedVectorFromFile(const std::string &path, std::vector<std::vector<vector_idx_t>> &data) {
    std::ifstream inputFile(path, std::ios::binary);
    
    // Read number of outer vectors
    uint64_t numOuter;
    inputFile.read(reinterpret_cast<char *>(&numOuter), sizeof(uint64_t));
    
    data.resize(numOuter);
    
    // Read each inner vector
    for (uint64_t i = 0; i < numOuter; i++) {
        // Read size of inner vector
        uint64_t innerSize;
        inputFile.read(reinterpret_cast<char *>(&innerSize), sizeof(uint64_t));
        
        // Read inner vector data
        data[i].resize(innerSize);
        if (innerSize > 0) {
            inputFile.read(reinterpret_cast<char *>(data[i].data()), innerSize * sizeof(vector_idx_t));
        }
    }
    
    inputFile.close();
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
    omp_set_num_threads(32);
    auto dc = createDistanceComputer(vectors, dim, numVectors, L2);
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
    auto numQueries = stoi(input.getCmdOption("-numQueries"));

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(basePath.c_str(), &baseDimension, &baseNumVectors, numVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryPath.c_str(), &queryDimension, &queryNumVectors, numQueries);
    queryNumVectors = std::min(queryNumVectors, (size_t) numQueries);
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    baseNumVectors = std::min(baseNumVectors, (size_t) numVectors);
    printf("Base num vectors: %zu, Query num vectors: %zu\n", baseNumVectors, queryNumVectors);
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
                    // hnsw_index->navix_single_search(queryVecs + (j * baseDimension), k, distances, labels, reinterpret_cast<char*>(filteredMask), visited, stats);
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
            // hnsw_index->navix_single_search(queryVecs + (j * baseDimension), k, distances, labels, reinterpret_cast<char*>(filteredMask), visited, stats);
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
    const float lambda = stof(input.getCmdOption("-lambda"));
    const int nThreads = stoi(input.getCmdOption("-nThreads"));
    const int k = stoi(input.getCmdOption("-k"));
    const int numQueries = stoi(input.getCmdOption("-numQueries"));
    const int sampleSize = stoi(input.getCmdOption("-sampleSize"));
    const int nProbes = stoi(input.getCmdOption("-nProbes"));
    const int readFromDisk = stoi(input.getCmdOption("-readFromDisk"));
    const std::string &storagePath = input.getCmdOption("-storagePath");
    const int isParquet = stoi(input.getCmdOption("-isParquet"));
    int nFiles = isParquet ? stoi(input.getCmdOption("-nFiles")) : 0;
    const int useIP = stoi(input.getCmdOption("-useIP"));
    const float factor = stof(input.getCmdOption("-factor"));

    size_t baseDimension, totalBaseNumVectors;
    std::vector<std::string> filePaths;

    float *baseVecs = nullptr;
    
    // Get file information first
    if (isParquet) {
        list_parquet_dir(baseVectorPath.c_str(), filePaths);
        if (filePaths.empty()) {
            fprintf(stderr, "No parquet files found in the directory: %s\n", baseVectorPath.c_str());
            exit(1);
        }
        nFiles = std::min(nFiles, (int) filePaths.size());
        if (nFiles != filePaths.size()) {
            std::vector<std::string> temp(nFiles);
            for (int i = 0; i < nFiles; i++) {
                temp[i] = filePaths[i];
            }
            filePaths = temp;
        }
        auto status = readParquetFileStats(filePaths.at(0).c_str(), &baseDimension, &totalBaseNumVectors);
        if (!status.ok()) {
            fprintf(stderr, "Failed to read parquet file stats: %s\n", status.ToString().c_str());
            exit(1);
        }
        // Calculate total vectors across all files
        totalBaseNumVectors = 0;
        for (const auto& path : filePaths) {
            size_t fileVectors;
            auto status = readParquetFileStats(path.c_str(), &baseDimension, &fileVectors);
            if (status.ok()) {
                totalBaseNumVectors += fileVectors;
            }
        }
    } else {
        baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &totalBaseNumVectors, numVectors);
    }
    totalBaseNumVectors = std::min(totalBaseNumVectors, (size_t) numVectors);

    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    queryNumVectors = std::min(queryNumVectors, (size_t) numQueries);
    auto sampleSizeAdjusted = std::min((size_t)sampleSize, totalBaseNumVectors);
    CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    loadFromFile(groundTruthPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));

    auto metric = useIP ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
    auto quantizer = faiss::IndexFlat(baseDimension, metric);
    auto numCentroids = totalBaseNumVectors / clusterSize;
    faiss::IndexIVFFlat idx(&quantizer, baseDimension, numCentroids, metric);
    faiss::IndexIVFFlat* index = &idx;
    index->cp.niter = nIter;
    index->cp.max_points_per_centroid = (sampleSizeAdjusted / numCentroids) + 1;
    index->cp.min_points_per_centroid = (sampleSizeAdjusted / numCentroids) * 0.5;
    printf("max_points_per_centroid: %d, min_points_per_centroid: %d\n",
           index->cp.max_points_per_centroid, index->cp.min_points_per_centroid);
    index->cp.verbose = true;
    // index->cp.lambda = lambda;

    if (!readFromDisk) {
        omp_set_num_threads(nThreads);
        
        // Phase 1: Train index with sample data
        printf("Phase 1: Training index with sample of %zu vectors\n", sampleSizeAdjusted);
        float *sampleVecs = nullptr;
        
        if (isParquet) {
            // Read first few parquet files for training sample
            std::vector<std::string> sampleFilePaths;
            size_t sampledVectors = 0;
            for (const auto& path : filePaths) {
                sampleFilePaths.push_back(path);
                size_t fileVectors;
                auto status = readParquetFileStats(path.c_str(), &baseDimension, &fileVectors);
                if (status.ok()) {
                    sampledVectors += fileVectors;
                    if (sampledVectors >= sampleSizeAdjusted) break;
                }
            }
            size_t actualSampleDim, actualSampleVectors;
            sampleVecs = readParquetFiles(sampleFilePaths, &actualSampleDim, &actualSampleVectors);
            actualSampleVectors = std::min(actualSampleVectors, sampleSizeAdjusted);
            
            auto trainStart = std::chrono::high_resolution_clock::now();
            index->train(actualSampleVectors, sampleVecs);
            auto trainEnd = std::chrono::high_resolution_clock::now();
            auto trainDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);
            printf("Training time: %lld ms\n", trainDuration.count());
            
            delete[] sampleVecs; // Free sample data
        } else {
            auto trainStart = std::chrono::high_resolution_clock::now();
            index->train(totalBaseNumVectors, baseVecs);
            auto trainEnd = std::chrono::high_resolution_clock::now();
            auto trainDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);
            printf("Training time: %lld ms\n", trainDuration.count());
        }
        
        // // Save trained index
        // std::string trainedIndexPath = storagePath + "_trained";
        // faiss::write_index(index, trainedIndexPath.c_str());
        // printf("Saved trained index to: %s\n", trainedIndexPath.c_str());
        
        // Phase 2: Add all data in batches
        printf("Phase 2: Adding all data in batches\n");
        size_t totalAdded = 0;
        auto addStart = std::chrono::high_resolution_clock::now();
        
        if (isParquet) {
            // Process parquet files in batches of 10 files
            const size_t filesPerBatch = 10;
            size_t numBatches = (filePaths.size() + filesPerBatch - 1) / filesPerBatch;
            
            for (size_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
                std::vector<std::string> batchPaths;
                size_t startIdx = batchIdx * filesPerBatch;
                size_t endIdx = std::min(startIdx + filesPerBatch, filePaths.size());
                
                // Collect files for this batch
                for (size_t i = startIdx; i < endIdx; i++) {
                    batchPaths.push_back(filePaths[i]);
                }
                
                size_t batchDim, batchVectors;
                float *batchData = readParquetFiles(batchPaths, &batchDim, &batchVectors);
                
                printf("Adding batch %zu/%zu (%zu files: %zu-%zu) with %zu vectors\n", 
                       batchIdx + 1, numBatches, batchPaths.size(), startIdx, endIdx - 1, batchVectors);
                index->add(batchVectors, batchData);
                totalAdded += batchVectors;
                
                delete[] batchData; // Free batch data immediately
                
                if (totalAdded >= totalBaseNumVectors) break;
            }
        } else {
            // For non-parquet files, add in chunks
            index->add(totalBaseNumVectors, baseVecs);
        }
        
        auto addEnd = std::chrono::high_resolution_clock::now();
        auto addDuration = std::chrono::duration_cast<std::chrono::milliseconds>(addEnd - addStart);
        printf("Adding time: %lld ms\n", addDuration.count());
        printf("Total vectors added: %zu\n", totalAdded);
        
        printf("Writing final index to disk: %s\n", storagePath.c_str());
        faiss::write_index(index, storagePath.c_str());
    } else {
        index = dynamic_cast<faiss::IndexIVFFlat *>(faiss::read_index(storagePath.c_str()));
    }
    printf("Calculating min / max / avg cluster sizes from assignments\n");
    omp_set_num_threads(nThreads);
    auto indexFlat = static_cast<faiss::IndexFlat *>(index->quantizer);
    std::vector<int64_t> assignment(totalBaseNumVectors);
    if (!isParquet) {
        indexFlat->assign(totalBaseNumVectors, baseVecs, assignment.data());
        std::vector<int> histogram(numCentroids, 0);
        for (size_t i = 0; i < totalBaseNumVectors; ++i) {
            if (assignment[i] >= 0 && assignment[i] < numCentroids) {
                histogram[assignment[i]]++;
            }
        }

        int min_cluster_size = *std::min_element(histogram.begin(), histogram.end());
        int max_cluster_size = *std::max_element(histogram.begin(), histogram.end());
        double avg_cluster_size = static_cast<double>(totalBaseNumVectors) / numCentroids;

        double sum_squared_diff = 0.0;
        for (int count : histogram) {
            double diff = count - avg_cluster_size;
            sum_squared_diff += diff * diff;
        }
        double std_dev = std::sqrt(sum_squared_diff / numCentroids);

        printf("Assignment histogram statistics:\n");
        printf("  Min cluster size: %d\n", min_cluster_size);
        printf("  Max cluster size: %d\n", max_cluster_size);
        printf("  Average cluster size: %.2f\n", avg_cluster_size);
        printf("  Standard deviation: %.2f\n", std_dev);
    }

    // omp_set_num_threads(1);
    index->nprobe = nProbes;
    // float* centroids = reinterpret_cast<float *>(indexFlat->codes.data());
    auto recall = 0.0;
    auto labels = new faiss::idx_t[k];
    auto distances = new float[k];
    auto startTime = std::chrono::high_resolution_clock::now();
    printf("baseDimension: %lu\n", baseDimension);
    printf("queryNumVectors: %zu\n", queryNumVectors);
    std::vector<float> centroidDists(numCentroids);
    std::vector<faiss::idx_t> indices(numCentroids);

    // Calculate silhouette score for each centroid using OMP
    if (!isParquet) {
        printf("Calculating silhouette scores for %zu centroids...\n", numCentroids);
        std::vector<double> silhouetteScores(numCentroids, 0.0);

        // Get centroids from the flat index
        float* centroids = indexFlat->get_xb();

        auto silhouetteStart = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (size_t centroidId = 0; centroidId < numCentroids; centroidId++) {
            // Collect points belonging to this centroid
            std::vector<size_t> clusterPoints;
            for (size_t i = 0; i < totalBaseNumVectors; i++) {
                if (assignment[i] == static_cast<int64_t>(centroidId)) {
                    clusterPoints.push_back(i);
                }
            }

            if (clusterPoints.empty()) {
                silhouetteScores[centroidId] = 0.0;
                continue;
            }

            double totalSilhouette = 0.0;
            const float* curCentroid = centroids + centroidId * baseDimension;

            // Calculate silhouette for each point in this cluster
            for (size_t pointIdx : clusterPoints) {
                const float* curPoint = baseVecs + pointIdx * baseDimension;

                // 1) a = distance to own centroid
                double a = 0.0;
                if (useIP) {
                    a = -faiss::fvec_inner_product(curPoint, curCentroid, baseDimension);
                } else {
                    a = faiss::fvec_L2sqr(curPoint, curCentroid, baseDimension);
                }

                // 2) b = min distance to any other centroid
                double b = std::numeric_limits<double>::infinity();
                for (size_t otherCentroidId = 0; otherCentroidId < numCentroids; otherCentroidId++) {
                    if (otherCentroidId == centroidId) continue;

                    const float* otherCentroid = centroids + otherCentroidId * baseDimension;
                    double dist;
                    if (useIP) {
                        dist = -faiss::fvec_inner_product(curPoint, otherCentroid, baseDimension);
                    } else {
                        dist = faiss::fvec_L2sqr(curPoint, otherCentroid, baseDimension);
                    }
                    b = std::min(b, dist);
                }

                // 3) silhouette for this point
                double m = std::max(a, b);
                if (m < 0) {
                    m = std::max(-a, -b);
                }
                double s = (m != 0.0) ? (b - a) / m : 0.0;

                totalSilhouette += s;
            }

            silhouetteScores[centroidId] = totalSilhouette / clusterPoints.size();
        }

        auto silhouetteEnd = std::chrono::high_resolution_clock::now();
        auto silhouetteDuration = std::chrono::duration_cast<std::chrono::milliseconds>(silhouetteEnd - silhouetteStart);

        // Print statistics
        double avgSilhouette = 0.0;
        double minSilhouette = std::numeric_limits<double>::max();
        double maxSilhouette = std::numeric_limits<double>::lowest();

        for (size_t i = 0; i < numCentroids; i++) {
            avgSilhouette += silhouetteScores[i];
            minSilhouette = std::min(minSilhouette, silhouetteScores[i]);
            maxSilhouette = std::max(maxSilhouette, silhouetteScores[i]);
        }
        avgSilhouette /= numCentroids;

        printf("Silhouette score statistics:\n");
        printf("  Average: %.4f\n", avgSilhouette);
        printf("  Min: %.4f\n", minSilhouette);
        printf("  Max: %.4f\n", maxSilhouette);
        printf("  Calculation time: %lld ms\n", silhouetteDuration.count());
    }

    for (size_t i = 0; i < queryNumVectors; i++) {
        index->search(1, queryVecs + (i * baseDimension), k, distances, labels);
        indexFlat->search(1, queryVecs + (i * baseDimension), numCentroids, centroidDists.data(), indices.data());

        if (useIP) {
            for (size_t c = 0; c < numCentroids; c++) {
                centroidDists[c] = -centroidDists[c];
            }
        }

        // Sort the centroidDists
        std::sort(centroidDists.begin(), centroidDists.end());
        auto closest = centroidDists[0];
        auto closest_factor = closest + std::abs(closest) * (factor - 1);
        auto furthest = centroidDists[0];
        int m = 0;
        for (int c = 0; c < numCentroids; c++) {
            if (centroidDists[c] > closest_factor) {
                // printf("Closest centroid index for query %zu: %lld\n", i, indices[c]);
                break;
            }
            furthest = centroidDists[c];
            m++;
        }
        printf(
            "Closest centroid distance %zu: %f and Number of centroids within %fx closest distance: %d with furthest distance %f\n",
            i, closest, factor, m, furthest);
        auto gt = gtVecs + i * k;
        auto localRecall = 0;
        for (int j = 0; j < k; j++) {
            if (std::find(gt, gt + k, labels[j]) != (gt + k)) {
                recall++;
                localRecall++;
            }
        }
        printf("Query %zu: Recall: %f%%\n", i, (localRecall / (double)k) * 100);
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

void benchmark_faiss_clustering_on_bvec(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    const std::string &groundTruthPath = input.getCmdOption("-groundTruthPath");
    const int numVectors = stoi(input.getCmdOption("-numVectors"));
    const int clusterSize = stoi(input.getCmdOption("-clusterSize"));
    const int nIter = stoi(input.getCmdOption("-nIter"));
    const int nThreads = stoi(input.getCmdOption("-nThreads"));
    const int k = stoi(input.getCmdOption("-k"));
    const int numQueries = stoi(input.getCmdOption("-numQueries"));
    const size_t sampleSize = input.getCmdOption("-sampleSize").empty()
                                      ? 0
                                      : stoull(input.getCmdOption("-sampleSize"));
    const double samplePercent = !input.getCmdOption("-samplePercent").empty()
                                         ? stod(input.getCmdOption("-samplePercent"))
                                         : (!input.getCmdOption("-samplePercentage").empty()
                                                    ? stod(input.getCmdOption("-samplePercentage"))
                                                    : 0.0);
    const int nProbes = stoi(input.getCmdOption("-nProbes"));
    const int readFromDisk = stoi(input.getCmdOption("-readFromDisk"));
    const std::string &storagePath = input.getCmdOption("-storagePath");
    const bool useIP = input.getCmdOption("-useIP").empty() ? false : stoi(input.getCmdOption("-useIP"));
    const bool useScalarQuantizer = input.getCmdOption("-useScalarQuantizer").empty()
                                            ? false
                                            : stoi(input.getCmdOption("-useScalarQuantizer"));
    const size_t addBatchSize = input.getCmdOption("-addBatchSize").empty()
                                        ? static_cast<size_t>(250000)
                                        : stoull(input.getCmdOption("-addBatchSize"));

    CHECK_ARGUMENT(baseVectorPath.find(".bvec") != std::string::npos, "base vector path must be a .bvecs file");
    CHECK_ARGUMENT(!storagePath.empty(), "storage path is required");
    CHECK_ARGUMENT(sampleSize > 0 || samplePercent > 0.0, "sampleSize or samplePercent is required");

    size_t baseDimension, totalBaseNumVectors;
    readBvecFileStats(baseVectorPath.c_str(), &baseDimension, &totalBaseNumVectors);
    totalBaseNumVectors = std::min(totalBaseNumVectors, static_cast<size_t>(numVectors));

    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors, numQueries);
    queryNumVectors = std::min(queryNumVectors, static_cast<size_t>(numQueries));
    CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");

    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    loadFromFile(groundTruthPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));

    auto metric = useIP ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
    const size_t numCentroids = std::max<size_t>(1, totalBaseNumVectors / clusterSize);
    const size_t sampleSizeAdjusted = resolveBvecSampleCount(totalBaseNumVectors, sampleSize, samplePercent);
    const size_t avgPointsPerCentroid = std::max<size_t>(1, sampleSizeAdjusted / numCentroids);

    std::unique_ptr<faiss::IndexIVF> ownedIndex;
    std::unique_ptr<faiss::Index> loadedIndex;
    faiss::IndexIVF *index = nullptr;

    if (!readFromDisk) {
        ownedIndex = create_faiss_ivf_index(baseDimension, numCentroids, metric, useScalarQuantizer);
        index = ownedIndex.get();
        index->cp.niter = nIter;
        index->cp.max_points_per_centroid = avgPointsPerCentroid + 1;
        index->cp.min_points_per_centroid = std::max<size_t>(1, avgPointsPerCentroid / 2);
        index->cp.verbose = true;

        printf("Training sample size: %zu\n", sampleSizeAdjusted);
        if (samplePercent > 0.0) {
            printf("Training sample percent: %.4f\n", samplePercent);
        }
        printf("Add batch size: %zu\n", addBatchSize);
        printf("Index type: %s\n", useScalarQuantizer ? "IVFScalarQuantizer(QT_8bit)" : "IVFFlat");
        printf("Code size per vector: %zu bytes\n", index->code_size);
        print_memory_usage("Initial");

        omp_set_num_threads(nThreads);

        size_t sampleDim = 0;
        size_t sampledVectors = 0;
        float *sampleVecs = readBvecTrainingSample(
                baseVectorPath.c_str(),
                sampleSize,
                &sampleDim,
                &sampledVectors,
                samplePercent,
                totalBaseNumVectors);
        CHECK_ARGUMENT(sampleDim == baseDimension, "sample dimension mismatch");

        auto trainStart = std::chrono::high_resolution_clock::now();
        index->train(sampledVectors, sampleVecs);
        auto trainEnd = std::chrono::high_resolution_clock::now();
        printf(
                "Training time: %lld ms\n",
                std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart).count());
        std::free(sampleVecs);
        print_memory_usage("After training");

        size_t totalAdded = 0;
        size_t batchId = 0;
        auto addStart = std::chrono::high_resolution_clock::now();
        while (totalAdded < totalBaseNumVectors) {
            size_t batchDim = 0;
            size_t batchVectors = 0;
            const size_t rowsToRead = std::min(addBatchSize, totalBaseNumVectors - totalAdded);
            float *batchVecs = readBvecFileChunk(
                    baseVectorPath.c_str(), totalAdded, rowsToRead, &batchDim, &batchVectors);
            CHECK_ARGUMENT(batchDim == baseDimension, "batch dimension mismatch");
            index->add(batchVectors, batchVecs);
            totalAdded += batchVectors;
            batchId++;
            printf("Added batch %zu with %zu vectors. Total added: %zu\n", batchId, batchVectors, totalAdded);
            if (batchId == 1 || batchId % 10 == 0 || totalAdded == totalBaseNumVectors) {
                print_memory_usage("After add batch");
            }
            std::free(batchVecs);
        }
        auto addEnd = std::chrono::high_resolution_clock::now();
        printf(
                "Adding time: %lld ms\n",
                std::chrono::duration_cast<std::chrono::milliseconds>(addEnd - addStart).count());

        faiss::write_index(index, storagePath.c_str());
        printf("Stored index at: %s\n", storagePath.c_str());
        if (std::filesystem::exists(storagePath)) {
            printf("Index file size: %.2f MB\n", bytes_to_mb(std::filesystem::file_size(storagePath)));
        }
        print_memory_usage("After storing index");
    } else {
        loadedIndex.reset(faiss::read_index(storagePath.c_str()));
        index = dynamic_cast<faiss::IndexIVF *>(loadedIndex.get());
        CHECK_ARGUMENT(index != nullptr, "stored index is not an IVF index");
        printf("Loaded index from: %s\n", storagePath.c_str());
        print_memory_usage("After loading index");
    }

    index->nprobe = nProbes;
    faiss::indexIVF_stats.reset();

    std::vector<faiss::idx_t> labels(k);
    std::vector<float> distances(k);
    double recall = 0.0;

    auto searchStart = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < queryNumVectors; i++) {
        index->search(1, queryVecs + i * baseDimension, k, distances.data(), labels.data());
        auto *gt = gtVecs + i * k;
        int localRecall = 0;
        for (int j = 0; j < k; j++) {
            if (std::find(gt, gt + k, labels[j]) != (gt + k)) {
                recall++;
                localRecall++;
            }
        }
        printf("Query %zu: Recall: %.2f%%\n", i, (localRecall / static_cast<double>(k)) * 100.0);
    }
    auto searchEnd = std::chrono::high_resolution_clock::now();
    auto searchDuration = std::chrono::duration_cast<std::chrono::milliseconds>(searchEnd - searchStart).count();

    std::cout << "Total Vectors: " << queryNumVectors << std::endl;
    std::cout << "Num of centroids: " << numCentroids << std::endl;
    std::cout << "Recall: " << ((recall / queryNumVectors) / k) * 100 << std::endl;
    std::cout << "Avg Distances comps: " << faiss::indexIVF_stats.ndis / queryNumVectors << std::endl;
    std::cout << "Query time: " << searchDuration << " ms" << std::endl;
    print_memory_usage("After queries");
}

void debug_fbin_ivf_query(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    std::string groundTruthPath = input.getCmdOption("-groundTruthPath");
    const size_t numVectors = input.getCmdOption("-numVectors").empty()
                                      ? SIZE_MAX
                                      : stoull(input.getCmdOption("-numVectors"));
    const size_t queryIndex = stoull(input.getCmdOption("-queryIndex"));
    const int k = stoi(input.getCmdOption("-k"));
    const int numThreads = input.getCmdOption("-numThreads").empty() ? 1 : stoi(input.getCmdOption("-numThreads"));
    const int nIter = input.getCmdOption("-nIter").empty() ? 10 : stoi(input.getCmdOption("-nIter"));
    const int nProbes = stoi(input.getCmdOption("-nProbes"));
    const double factor = input.getCmdOption("-factor").empty() ? 1.0 : stod(input.getCmdOption("-factor"));
    const bool useIP = input.getCmdOption("-useIP").empty() ? false : stoi(input.getCmdOption("-useIP"));
    const bool useScalarQuantizer = input.getCmdOption("-useScalarQuantizer").empty()
                                            ? false
                                            : stoi(input.getCmdOption("-useScalarQuantizer"));
    size_t numCentroids = !input.getCmdOption("-numCentroids").empty()
                                        ? stoull(input.getCmdOption("-numCentroids"))
                                        : 0;
    const double samplePercent = input.getCmdOption("-samplePercent").empty()
                                         ? 0.2
                                         : stod(input.getCmdOption("-samplePercent"));

    CHECK_ARGUMENT(!baseVectorPath.empty(), "base vector path is required");
    CHECK_ARGUMENT(!queryVectorPath.empty(), "query vector path is required");

    size_t baseDimension = 0;
    size_t totalBaseNumVectors = 0;
    float *baseVecs = readFbinFile(baseVectorPath.c_str(), &baseDimension, &totalBaseNumVectors);
    totalBaseNumVectors = std::min(totalBaseNumVectors, numVectors);
    CHECK_ARGUMENT(totalBaseNumVectors > 0, "no base vectors available");
    if (numCentroids == 0) {
        numCentroids = std::max<size_t>(1, totalBaseNumVectors / stoi(input.getCmdOption("-clusterSize")));
    }
    printf("Base vectors: %zu, dimension: %zu numCentroids: %zu\n", totalBaseNumVectors, baseDimension, numCentroids);
    CHECK_ARGUMENT(numCentroids <= totalBaseNumVectors, "numCentroids must be <= number of base vectors");

    size_t queryDimension = 0;
    size_t queryNumVectors = 0;
    float *queryVecs = readFvecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
    CHECK_ARGUMENT(queryIndex < queryNumVectors, "queryIndex out of range");

    if (groundTruthPath.empty()) {
        groundTruthPath = fmt::format("{}.gt_k{}.bin", queryVectorPath, k);
    }

    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    auto metric = useIP ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
    const size_t sampledTrainVectors = std::max<size_t>(
            1, static_cast<size_t>(std::ceil(totalBaseNumVectors * samplePercent)));
    omp_set_num_threads(numThreads);
    faiss::IndexFlat exactIndex(baseDimension, metric);
    exactIndex.add(totalBaseNumVectors, baseVecs);

    if (!std::filesystem::exists(groundTruthPath)) {
        printf("Ground truth not found. Computing exact GT and writing to: %s\n", groundTruthPath.c_str());
        std::vector<faiss::idx_t> exactLabels(queryNumVectors * k);
        std::vector<float> exactDistances(queryNumVectors * k);
        exactIndex.search(queryNumVectors, queryVecs, k, exactDistances.data(), exactLabels.data());

        for (size_t i = 0; i < queryNumVectors * static_cast<size_t>(k); i++) {
            gtVecs[i] = static_cast<vector_idx_t>(exactLabels[i]);
        }

        const auto gtParent = std::filesystem::path(groundTruthPath).parent_path();
        if (!gtParent.empty()) {
            std::filesystem::create_directories(gtParent);
        }
        writeToFile(groundTruthPath, reinterpret_cast<const uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));
    } else {
        loadFromFile(groundTruthPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));
    }

    auto index = create_faiss_ivf_index(baseDimension, numCentroids, metric, useScalarQuantizer);
    index->cp.niter = nIter;
    index->cp.max_points_per_centroid = std::max<size_t>(1, sampledTrainVectors / numCentroids);
    index->cp.min_points_per_centroid = std::max<size_t>(1, index->cp.max_points_per_centroid / 2);
    index->cp.verbose = true;

    auto buildStart = std::chrono::high_resolution_clock::now();
    index->train(totalBaseNumVectors, baseVecs);
    index->add(totalBaseNumVectors, baseVecs);
    auto buildEnd = std::chrono::high_resolution_clock::now();

    index->nprobe = nProbes;
    faiss::indexIVF_stats.reset();

    auto *queryVec = queryVecs + queryIndex * queryDimension;
    std::vector<faiss::idx_t> labels(k);
    std::vector<float> distances(k);
    index->search(1, queryVec, k, distances.data(), labels.data());
    std::vector<faiss::idx_t> exactVectorIds(totalBaseNumVectors);
    std::vector<float> exactVectorDists(totalBaseNumVectors);
    exactIndex.search(1, queryVec, totalBaseNumVectors, exactVectorDists.data(), exactVectorIds.data());
    if (useIP) {
        for (float &dist : exactVectorDists) {
            dist = -dist;
        }
    }

    auto *gt = gtVecs + queryIndex * k;
    int hitCount = 0;
    for (int j = 0; j < k; j++) {
        if (std::find(gt, gt + k, labels[j]) != (gt + k)) {
            hitCount++;
        }
    }

    std::vector<float> centroidDists(numCentroids);
    std::vector<faiss::idx_t> centroidIds(numCentroids);
    index->quantizer->search(1, queryVec, numCentroids, centroidDists.data(), centroidIds.data());
    if (useIP) {
        for (size_t i = 0; i < numCentroids; i++) {
            centroidDists[i] = -centroidDists[i];
        }
    }
    std::sort(centroidDists.begin(), centroidDists.end());

    const double minCentroidDist = centroidDists.front();
    const double threshold = minCentroidDist + std::abs(minCentroidDist) * (factor - 1.0);
    int centroidsWithinFactor = 0;
    double furthestAccepted = minCentroidDist;
    for (double dist : centroidDists) {
        if (dist > threshold) {
            break;
        }
        furthestAccepted = dist;
        centroidsWithinFactor++;
    }

    const double minVectorDist = exactVectorDists.front();
    const double maxVectorDist = exactVectorDists.back();
    const double vectorThreshold = minVectorDist + std::abs(minVectorDist) * (factor - 1.0);
    size_t vectorsWithinFactor = 0;
    double furthestAcceptedVectorDist = minVectorDist;
    for (double dist : exactVectorDists) {
        if (dist > vectorThreshold) {
            break;
        }
        furthestAcceptedVectorDist = dist;
        vectorsWithinFactor++;
    }

    printf("Debug IVF query\n");
    printf("Index type: %s\n", useScalarQuantizer ? "IVFScalarQuantizer(QT_8bit)" : "IVFFlat");
    printf("Base vectors: %zu, dimension: %zu, centroids: %zu\n", totalBaseNumVectors, baseDimension, numCentroids);
    printf("OpenMP threads: %d\n", numThreads);
    printf("Faiss training cap: %zu vectors (20%%)\n", sampledTrainVectors);
    printf("Build time: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(buildEnd - buildStart).count());
    printf("Query index: %zu, nprobe: %d, factor: %.4f\n", queryIndex, nProbes, factor);
    printf("Recall@%d: %.2f%% (%d/%d)\n", k, (hitCount / static_cast<double>(k)) * 100.0, hitCount, k);
    printf("Average IVF distance computations: %zu\n", faiss::indexIVF_stats.ndis);
    printf("Min centroid distance: %.6f\n", minCentroidDist);
    printf("Centroids within factor %.4f: %d\n", factor, centroidsWithinFactor);
    printf("Furthest accepted centroid distance: %.6f\n", furthestAccepted);
    printf("Min vector distance from query: %.6f\n", minVectorDist);
    printf("Max vector distance from query: %.6f\n", maxVectorDist);
    printf("Vectors within factor %.4f: %zu\n", factor, vectorsWithinFactor);
    printf("Furthest accepted vector distance: %.6f\n", furthestAcceptedVectorDist);

    const int printK = std::min(k, 10);
    printf("Top-%d result ids: ", printK);
    for (int j = 0; j < printK; j++) {
        printf("%lld ", static_cast<long long>(labels[j]));
    }
    printf("\n");
    printf("Top-%d gt ids: ", printK);
    for (int j = 0; j < printK; j++) {
        printf("%llu ", static_cast<unsigned long long>(gt[j]));
    }
    printf("\n");
}

double get_recall(ReclusteringIndex &index, float *queryVecs, size_t queryDimension, size_t queryNumVectors, int k,
                  vector_idx_t *gtVecs, int nMegaProbes, int nMiniProbes, std::vector<double> &queryRecalls) {
    queryRecalls.resize(queryNumVectors);
    // search
    double recall = 0;
    ReclusteringIndexStats stats;
    double max_recall = 0;
    double min_recall = std::numeric_limits<double>::max();
    double num_recall_below_75 = 0;
    for (int i = 0; i < queryNumVectors; i++) {
        std::priority_queue<NodeDistCloser> results;
        index.search(queryVecs + i * queryDimension, k, results, nMegaProbes, nMiniProbes, stats);
        auto gt = gtVecs + i * k;
        double localRecall = 0;
        while (!results.empty()) {
            auto res = results.top();
            results.pop();
            if (std::find(gt, gt + k, res.id) != (gt + k)) {
                recall++;
                localRecall++;
            }
        }
        localRecall = (localRecall / k) * 100;
        max_recall = std::max(max_recall, localRecall);
        min_recall = std::min(min_recall, localRecall);
        if (localRecall < 75.0) {
            num_recall_below_75++;
        }
        queryRecalls[i] = localRecall;
        printf("Query %d: Recall: %f%%\n", i, localRecall);
    }
    printf("Avg Distance Computation: %llu\n", stats.numDistanceCompForSearch / queryNumVectors);
    printf("Max Recall: %f, Min Recall: %f, Num Recall below 75%%: %f\n", max_recall, min_recall, num_recall_below_75);
    return recall / queryNumVectors;
}

double get_recall_with_bad_clusters(ReclusteringIndex &index, float *queryVecs, size_t queryDimension,
                                    size_t queryNumVectors, int k,
                                    vector_idx_t *gtVecs, int nMegaProbes, int nMiniProbes,
                                    int nMiniProbesForBadCluster, bool searchEachBadCluster) {
    // search
    double recall = 0;
    ReclusteringIndexStats stats;
    for (int i = 0; i < queryNumVectors; i++) {
        std::priority_queue<NodeDistCloser> results;
        index.searchWithBadClusters(queryVecs + i * queryDimension, k, results, nMegaProbes, nMiniProbes, nMiniProbesForBadCluster, stats, searchEachBadCluster);
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

    std::vector<double> queryRecalls;
    auto recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbes, nMiniProbes, queryRecalls);
    printf("Recall: %f\n", recall);
    index.reclusterAllMegaCentroids();
    recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbes, nMiniProbes, queryRecalls);
    printf("Recall: %f\n", recall);

    index.storeScoreForMegaClusters();
    index.printStats();
}

void read_and_write_chunk(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const int numVectors = stoi(input.getCmdOption("-numVectors"));
    int numInserts = stoi(input.getCmdOption("-numInserts"));
    int idx = stoi(input.getCmdOption("-idx"));
    size_t baseDimension, baseNumVectors;
    float* baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors, numVectors);

    auto chunkSize = baseNumVectors / numInserts;
    printf("Chunk size: %lu\n", chunkSize);
    for (long i = 0; i < numInserts; i++) {
        auto start = i * chunkSize;
        auto end = (i + 1) * chunkSize;
        if (i == (numInserts - 1)) {
            end = baseNumVectors;
        }
        if (i != idx) {
            continue;
        }
        printf("processing chunk: %ld, start: %lu, end: %lu\n", i, start, end);
        writeFvecFile("out.fvecs", baseVecs + start * baseDimension, baseDimension, end - start);
    }
}

void run_umap_2D_without_clustering(InputParser &input) {
    
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const int numVectors = stoi(input.getCmdOption("-numVectors"));
    const std::string &outputPath = input.getCmdOption("-outputPath");
    size_t baseDimension, baseNumVectors;
    float* baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors, numVectors);
 
    using namespace orangedb;
  
    // K-NN
    std::vector<float> embedding(numVectors * 2);   
    auto metric = std::make_shared<knncolle::EuclideanDistance<float, float>>();
    knncolle::VptreeBuilder<int, float, float> builder(metric);
    
    // UMAP
    umappp::Options opt;
    opt.num_neighbors = 15; 
    opt.min_dist = 0.1;
    opt.num_epochs = 500;
    auto status = umappp::initialize(
        (int)baseDimension, numVectors, baseVecs, builder, 2, embedding.data(), opt
    );
    status.run(embedding.data());    
    
    // Binary output
    FILE* fp = fopen(outputPath.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file %s for writing\n", outputPath.c_str());
        return;
    }
    
    // header
    fwrite(&numVectors, sizeof(int), 1, fp);
    
    // vectors
    for (int i = 0; i < numVectors; ++i) {
        float umap_1 = embedding[i*2];
        float umap_2 = embedding[i*2+1];
        fwrite(&umap_1, sizeof(float), 1, fp);
        fwrite(&umap_2, sizeof(float), 1, fp);
        fwrite(&i, sizeof(int), 1, fp);
    }    
    fclose(fp);
    printf("UMAP projection of the dataset (without clustering) is written to %s\n", outputPath.c_str());
    printf("Binary format: num_vectors (int), then for each vector: UMAP_1 (float), UMAP_2 (float), row_id (int)\n");
}

void run_umap_3D_without_clustering(InputParser &input) {
    
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const int numVectors = stoi(input.getCmdOption("-numVectors"));
    const std::string &outputPath = input.getCmdOption("-outputPath");
    size_t baseDimension, baseNumVectors;
    float* baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors, numVectors);
 
    using namespace orangedb;
  
    // K-NN
    std::vector<float> embedding(numVectors * 3);   
    auto metric = std::make_shared<knncolle::EuclideanDistance<float, float>>();
    knncolle::VptreeBuilder<int, float, float> builder(metric);
    
    // UMAP
    umappp::Options opt;
    opt.num_neighbors = 15; 
    opt.min_dist = 0.1;
    opt.num_epochs = 500;
    auto status = umappp::initialize(
        (int)baseDimension, numVectors, baseVecs, builder, 3, embedding.data(), opt
    );
    status.run(embedding.data());    
    
    // Binary output
    FILE* fp = fopen(outputPath.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file %s for writing\n", outputPath.c_str());
        return;
    }
    
    // header
    fwrite(&numVectors, sizeof(int), 1, fp);
    
    // vectors
    for (int i = 0; i < numVectors; ++i) {
        float umap_1 = embedding[i*3];
        float umap_2 = embedding[i*3+1];
        float umap_3 = embedding[i*3+2];
        fwrite(&umap_1, sizeof(float), 1, fp);
        fwrite(&umap_2, sizeof(float), 1, fp);
        fwrite(&umap_3, sizeof(float), 1, fp);
        fwrite(&i, sizeof(int), 1, fp);
    }    
    fclose(fp);
    printf("UMAP projection of the dataset (without clustering) is written to %s\n", outputPath.c_str());
    printf("Binary format: num_vectors (int), then for each vector: UMAP_1 (float), UMAP_2 (float), UMAP_3 (float), row_id (int)\n");
}

void run_umap_2D_with_cluster_data(
    const orangedb::ReclusteringIndex& index,
    float* baseVecs, 
    int numVectors, 
    size_t baseDimension,
    const std::string& outputPath,
    int hirarchyLevel,
    int sampleSize = 100000  // Default sample size to reduce UMAP cost
) {
    using namespace orangedb;
    
    // Early validation
    if (baseDimension == 0) {
        printf("Error: baseDimension is 0, skipping UMAP 2D\n");
        return;
    }
    if (baseVecs == nullptr) {
        printf("Error: baseVecs is null, skipping UMAP 2D\n");
        return;
    }
    if (numVectors <= 0) {
        printf("Error: numVectors is %d, skipping UMAP 2D\n", numVectors);
        return;
    }
    
    std::unordered_map<vector_idx_t, int> vectorToCluster;  
    const float* centroids = nullptr;
    size_t numCentroids = 0;
    
    // Cluster Hirarchy 
    if (hirarchyLevel == C_L2) {
        std::vector<std::vector<vector_idx_t>> L1_ClusterVectorIds;
        std::vector<std::vector<vector_idx_t>> L2_CentroidIds;
        index.getMiniClusterVectorIds(&L1_ClusterVectorIds);
        index.getMegaMiniCentroids(&L2_CentroidIds);
        index.getMegaCentroids(&centroids,numCentroids);
        
        for (size_t L2_ClusterId = 0; L2_ClusterId < L2_CentroidIds.size(); ++L2_ClusterId) {
            for (auto L1_ClusterId : L2_CentroidIds[L2_ClusterId]) {
                if (L1_ClusterId < L1_ClusterVectorIds.size()) {
                    for (auto vectorId : L1_ClusterVectorIds[L1_ClusterId]) {
                        vectorToCluster[vectorId] = (int)L2_ClusterId;
                    }
                }
            }
        }
    }
    else if (hirarchyLevel == C_L1) {
        std::vector<std::vector<vector_idx_t>> L1_ClusterVectorIds;
        index.getMiniClusterVectorIds(&L1_ClusterVectorIds);
        index.getMiniCentroids(&centroids,numCentroids);
        
        for (size_t L1_clusterId = 0; L1_clusterId < L1_ClusterVectorIds.size(); ++L1_clusterId) {
            for (auto vectorId : L1_ClusterVectorIds[L1_clusterId]) {
                vectorToCluster[vectorId] = (int)L1_clusterId;
            }
        }
    }
    else {
        printf("Invalid hirarchy level: %d\n", hirarchyLevel);
        return;
    }
    printf("Total vectors assigned to clusters: %zu\n", vectorToCluster.size());
    printf("Number of centroids: %zu\n", numCentroids);
    
    // Subsample vectors to reduce UMAP computation cost
    std::vector<vector_idx_t> sampledVectorIds;
    if (sampleSize > 0 && sampleSize < numVectors) {
        // Group vectors by cluster for stratified sampling
        std::unordered_map<int, std::vector<vector_idx_t>> clusterToVectors;
        for (const auto& [vecId, clusterId] : vectorToCluster) {
            clusterToVectors[clusterId].push_back(vecId);
        }
        
        // Calculate samples per cluster (proportional to cluster size)
        int totalAssigned = (int)vectorToCluster.size();
        RandomGenerator rng(42);  // Fixed seed for reproducibility
        
        for (auto& [clusterId, vecIds] : clusterToVectors) {
            // Proportional sampling: each cluster gets samples proportional to its size
            int clusterSampleSize = std::max(1, (int)(((double)vecIds.size() / totalAssigned) * sampleSize));
            clusterSampleSize = std::min(clusterSampleSize, (int)vecIds.size());
            
            // Use randomPerm to get random indices
            std::vector<uint64_t> perm(clusterSampleSize);
            rng.randomPerm(vecIds.size(), perm.data(), clusterSampleSize);
            for (int i = 0; i < clusterSampleSize; ++i) {
                sampledVectorIds.push_back(vecIds[perm[i]]);
            }
        }
        printf("Subsampled %zu vectors from %d total (requested sample size: %d)\n", 
               sampledVectorIds.size(), numVectors, sampleSize);
    } else {
        // No subsampling, use all vectors
        for (const auto& [vecId, clusterId] : vectorToCluster) {
            sampledVectorIds.push_back(vecId);
        }
        printf("Using all %zu vectors (no subsampling, sampleSize=%d, numVectors=%d)\n", 
               sampledVectorIds.size(), sampleSize, numVectors);
    }
    
    int numSampledVectors = (int)sampledVectorIds.size();
    int totalVectors = numSampledVectors + (int)numCentroids;
    
    if (numSampledVectors == 0) {
        printf("Warning: No vectors to process for UMAP 2D\n");
        return;
    }
    
    std::vector<float> allVectors(totalVectors * baseDimension);
    
    // Copy sampled vectors
    for (int i = 0; i < numSampledVectors; ++i) {
        vector_idx_t origId = sampledVectorIds[i];
        std::memcpy(allVectors.data() + i * baseDimension, 
                    baseVecs + origId * baseDimension, 
                    baseDimension * sizeof(float));
    }
    
    // Copy centroids
    if (centroids && numCentroids > 0) {
        std::memcpy(allVectors.data() + numSampledVectors * baseDimension, 
                    centroids, 
                    numCentroids * baseDimension * sizeof(float));
    }
    
    // Check for NaN/Inf values in the data
    int nanCount = 0, infCount = 0;
    for (size_t i = 0; i < allVectors.size(); ++i) {
        if (std::isnan(allVectors[i])) nanCount++;
        if (std::isinf(allVectors[i])) infCount++;
    }
    if (nanCount > 0 || infCount > 0) {
        printf("Warning: Data contains %d NaN and %d Inf values, skipping UMAP 2D\n", nanCount, infCount);
        return;
    }
    
    std::vector<float> embedding(totalVectors * 2);
    
    // K-NN
    auto metric = std::make_shared<knncolle::EuclideanDistance<float, float>>();
    knncolle::VptreeBuilder<int, float, float> builder(metric);
    
    // UMAP
    umappp::Options opt;
    opt.num_neighbors = 15; 
    opt.min_dist = 0.1;
    opt.num_epochs = 500;
    
    // Ensure we have enough vectors for UMAP (need at least num_neighbors + 1)
    if (totalVectors <= opt.num_neighbors) {
        printf("Skipping UMAP 2D: not enough vectors (%d) for num_neighbors (%d)\n", totalVectors, opt.num_neighbors);
        return;
    }
    
    printf("Running UMAP 2D dimensionality reduction on %d sampled vectors + %zu centroids (dim=%zu)...\n", 
           numSampledVectors, numCentroids, baseDimension);
    auto umap_start = std::chrono::high_resolution_clock::now();
    auto status = umappp::initialize(
        (int)baseDimension, totalVectors, allVectors.data(), builder, 2, embedding.data(), opt
    );
    status.run(embedding.data());
    auto umap_end = std::chrono::high_resolution_clock::now();
    auto umap_duration = std::chrono::duration_cast<std::chrono::milliseconds>(umap_end - umap_start).count();
    printf("UMAP 2D took %ld ms (%.2f seconds)\n", umap_duration, umap_duration / 1000.0);
    
    // Binary output
    FILE* fp = fopen(outputPath.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file %s for writing\n", outputPath.c_str());
        return;
    }
    
    // header
    fwrite(&totalVectors, sizeof(int), 1, fp);
    
    // sampled vectors
    for (int i = 0; i < numSampledVectors; ++i) {
        vector_idx_t origId = sampledVectorIds[i];
        int clusterId = -1; // Default if not found
        auto it = vectorToCluster.find(origId);
        if (it != vectorToCluster.end()) {
            clusterId = it->second;
        }
        float umap_1 = embedding[i*2];
        float umap_2 = embedding[i*2+1];
        int isCentroid = 0;
        fwrite(&umap_1, sizeof(float), 1, fp);
        fwrite(&umap_2, sizeof(float), 1, fp);
        fwrite(&clusterId, sizeof(int), 1, fp);
        fwrite(&isCentroid, sizeof(int), 1, fp);
    }
    
    // centroids
    for (size_t i = 0; i < numCentroids; ++i) {
        int idx = numSampledVectors + i;
        float umap_1 = embedding[idx*2];
        float umap_2 = embedding[idx*2+1];
        int clusterId = (int)i;
        int isCentroid = 1;
        fwrite(&umap_1, sizeof(float), 1, fp);
        fwrite(&umap_2, sizeof(float), 1, fp);
        fwrite(&clusterId, sizeof(int), 1, fp);
        fwrite(&isCentroid, sizeof(int), 1, fp);
    }
    
    fclose(fp);
    printf("UMAP visualization with clusters written to %s\n", outputPath.c_str());
    printf("Binary format: num_records (int), then for each record: UMAP_1 (float), UMAP_2 (float), Cluster_ID (int), Is_Centroid (int, 0=vector, 1=centroid)\n");
}

void save_clustering_data(
    const orangedb::ReclusteringIndex& index,
    float* baseVecs, 
    int numVectors, 
    size_t baseDimension,
    const std::string& outputPath,
    int hirarchyLevel
) {
    using namespace orangedb;
    
    std::unordered_map<vector_idx_t, int> vectorToCluster;  
    
    // Cluster Hirarchy 
    if (hirarchyLevel == C_L2) {
        std::vector<std::vector<vector_idx_t>> L1_ClusterVectorIds;
        std::vector<std::vector<vector_idx_t>> L2_CentroidIds;
        index.getMiniClusterVectorIds(&L1_ClusterVectorIds);
        index.getMegaMiniCentroids(&L2_CentroidIds);
        
        for (size_t L2_ClusterId = 0; L2_ClusterId < L2_CentroidIds.size(); ++L2_ClusterId) {
            for (auto L1_ClusterId : L2_CentroidIds[L2_ClusterId]) {
                if (L1_ClusterId < L1_ClusterVectorIds.size()) {
                    for (auto vectorId : L1_ClusterVectorIds[L1_ClusterId]) {
                        vectorToCluster[vectorId] = (int)L2_ClusterId;
                    }
                }
            }
        }
    }
    else if (hirarchyLevel == C_L1) {
        std::vector<std::vector<vector_idx_t>> L1_ClusterVectorIds;
        index.getMiniClusterVectorIds(&L1_ClusterVectorIds);
        
        for (size_t L1_clusterId = 0; L1_clusterId < L1_ClusterVectorIds.size(); ++L1_clusterId) {
            for (auto vectorId : L1_ClusterVectorIds[L1_clusterId]) {
                vectorToCluster[vectorId] = (int)L1_clusterId;
            }
        }
    }
    else {
        printf("Invalid hirarchy level: %d\n", hirarchyLevel);
        return;
    }
    printf("Total vectors assigned to clusters: %zu\n", vectorToCluster.size());       

    // Save clustering data to bin
    FILE* fp = fopen(outputPath.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file %s for writing\n", outputPath.c_str());
        return;
    }
    
    // header
    int numRecords = (int)vectorToCluster.size();
    fwrite(&numRecords, sizeof(int), 1, fp);
    
    // ROW_ID, Cluster_ID
    for (auto it = vectorToCluster.begin(); it != vectorToCluster.end(); ++it) {
        int rowId = it->first;
        int clusterId = it->second;
        fwrite(&rowId, sizeof(int), 1, fp);
        fwrite(&clusterId, sizeof(int), 1, fp);
    }

    fclose(fp);
    printf("Clustering data written to %s\n", outputPath.c_str());
    printf("Binary format: num_records (int), then for each record: ROW_ID (int), Cluster_ID (int)\n");
}

void run_umap_3D_with_cluster_data(
    const ReclusteringIndex& index,
    const std::string& outputPath,
    int hirarchyLevel,
    int sampleSize = 100000,  // Default sample size to reduce UMAP cost
    int numThreads = 32
) {
    using namespace orangedb;
    
    int dim = index.getDim();
    if (dim <= 0) {
        printf("Error: dimension is %d, skipping UMAP 3D\n", dim);
        return;
    }
    
    std::unordered_map<vector_idx_t, int> vectorToCluster;      
    const float* centroids = nullptr;
    size_t numCentroids = 0;
    
    if (hirarchyLevel == C_L2) {
        std::vector<std::vector<vector_idx_t>> L1_ClusterVectorIds;
        std::vector<std::vector<vector_idx_t>> L2_CentroidIds;
        index.getMiniClusterVectorIds(&L1_ClusterVectorIds);
        index.getMegaMiniCentroids(&L2_CentroidIds);
        index.getMegaCentroids(&centroids,numCentroids);
        
        for (size_t L2_ClusterId = 0; L2_ClusterId < L2_CentroidIds.size(); ++L2_ClusterId) {
            for (auto L1_ClusterId : L2_CentroidIds[L2_ClusterId]) {
                if (L1_ClusterId < L1_ClusterVectorIds.size()) {
                    for (auto vectorId : L1_ClusterVectorIds[L1_ClusterId]) {
                        vectorToCluster[vectorId] = (int)L2_ClusterId;
                    }
                }
            }
        }
    }
    else if (hirarchyLevel == C_L1) {
        std::vector<std::vector<vector_idx_t>> L1_ClusterVectorIds;
        index.getMiniClusterVectorIds(&L1_ClusterVectorIds);
        index.getMiniCentroids(&centroids,numCentroids);
        
        for (size_t L1_clusterId = 0; L1_clusterId < L1_ClusterVectorIds.size(); ++L1_clusterId) {
            for (auto vectorId : L1_ClusterVectorIds[L1_clusterId]) {
                vectorToCluster[vectorId] = (int)L1_clusterId;
            }
        }
    }
    else {
        printf("Invalid hirarchy level: %d\n", hirarchyLevel);
        return;
    }
    
    int numVectors = (int)vectorToCluster.size();
    printf("Total vectors assigned to clusters: %d\n", numVectors);
    printf("Number of centroids: %zu\n", numCentroids);
    
    // Subsample vectors to reduce UMAP computation cost
    std::vector<vector_idx_t> sampledVectorIds;
    if (sampleSize > 0 && sampleSize < numVectors) {
        // Group vectors by cluster for stratified sampling
        std::unordered_map<int, std::vector<vector_idx_t>> clusterToVectors;
        for (const auto& [vecId, clusterId] : vectorToCluster) {
            clusterToVectors[clusterId].push_back(vecId);
        }
        
        // Calculate samples per cluster (proportional to cluster size)
        int totalAssigned = numVectors;
        RandomGenerator rng(42);  // Fixed seed for reproducibility
        
        for (auto& [clusterId, vecIds] : clusterToVectors) {
            // Proportional sampling: each cluster gets samples proportional to its size
            int clusterSampleSize = std::max(1, (int)(((double)vecIds.size() / totalAssigned) * sampleSize));
            clusterSampleSize = std::min(clusterSampleSize, (int)vecIds.size());
            
            // Use randomPerm to get random indices
            std::vector<uint64_t> perm(clusterSampleSize);
            rng.randomPerm(vecIds.size(), perm.data(), clusterSampleSize);
            for (int i = 0; i < clusterSampleSize; ++i) {
                sampledVectorIds.push_back(vecIds[perm[i]]);
            }
        }
        printf("Subsampled %zu vectors from %d total (requested sample size: %d)\n", 
               sampledVectorIds.size(), numVectors, sampleSize);
    } else {
        // No subsampling, use all vectors
        for (const auto& [vecId, clusterId] : vectorToCluster) {
            sampledVectorIds.push_back(vecId);
        }
        printf("Using all %zu vectors (no subsampling, sampleSize=%d, numVectors=%d)\n", 
               sampledVectorIds.size(), sampleSize, numVectors);
    }
    
    int numSampledVectors = (int)sampledVectorIds.size();
    int totalVectors = numSampledVectors + (int)numCentroids;
    
    if (numSampledVectors == 0) {
        printf("Warning: No vectors to process for UMAP 3D\n");
        return;
    }
    
    std::vector<float> allVectors(totalVectors * dim);
    
    // Copy sampled vectors from index
    for (int i = 0; i < numSampledVectors; ++i) {
        vector_idx_t origId = sampledVectorIds[i];
        const float* vecData = index.getVectorData(origId);
        if (vecData == nullptr) {
            printf("Error: getVectorData returned null for vectorId %d, skipping UMAP 3D\n", origId);
            return;
        }
        std::memcpy(allVectors.data() + i * dim, vecData, dim * sizeof(float));
    }
    
    // Copy centroids
    if (centroids && numCentroids > 0) {
        std::memcpy(allVectors.data() + numSampledVectors * dim, 
                    centroids, 
                    numCentroids * dim * sizeof(float));
    }
    std::vector<float> embedding(totalVectors * 3);
    
    // K-NN
    auto metric = std::make_shared<knncolle::EuclideanDistance<float, float>>();
    knncolle::VptreeBuilder<int, float, float> builder(metric);
    
    // UMAP
    umappp::Options opt;
    opt.num_neighbors = 15; 
    opt.min_dist = 0.1;
    opt.num_epochs = 500;
    opt.num_threads = numThreads;
    
    // Ensure we have enough vectors for UMAP (need at least num_neighbors + 1)
    if (totalVectors <= opt.num_neighbors) {
        printf("Skipping UMAP 3D: not enough vectors (%d) for num_neighbors (%d)\n", totalVectors, opt.num_neighbors);
        return;
    }
    
    printf("Running UMAP 3D dimensionality reduction on %d sampled vectors + %zu centroids...\n", numSampledVectors, numCentroids);
    auto umap_start = std::chrono::high_resolution_clock::now();
    auto status = umappp::initialize(
        dim, totalVectors, allVectors.data(), builder, 3, embedding.data(), opt
    );
    status.run(embedding.data());
    auto umap_end = std::chrono::high_resolution_clock::now();
    auto umap_duration = std::chrono::duration_cast<std::chrono::milliseconds>(umap_end - umap_start).count();
    printf("UMAP 3D took %ld ms (%.2f seconds)\n", umap_duration, umap_duration / 1000.0);
    
    // Binary output
    FILE* fp = fopen(outputPath.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file %s for writing\n", outputPath.c_str());
        return;
    }
    
    // header
    fwrite(&totalVectors, sizeof(int), 1, fp);
    
    // sampled vectors
    for (int i = 0; i < numSampledVectors; ++i) {
        vector_idx_t origId = sampledVectorIds[i];
        int clusterId = -1; // Default if not found
        auto it = vectorToCluster.find(origId);
        if (it != vectorToCluster.end()) {
            clusterId = it->second;
        }
        float umap_1 = embedding[i*3];
        float umap_2 = embedding[i*3+1];
        float umap_3 = embedding[i*3+2];
        int isCentroid = 0;
        fwrite(&umap_1, sizeof(float), 1, fp);
        fwrite(&umap_2, sizeof(float), 1, fp);
        fwrite(&umap_3, sizeof(float), 1, fp);
        fwrite(&clusterId, sizeof(int), 1, fp);
        fwrite(&isCentroid, sizeof(int), 1, fp);
    }
    
    // centroids
    for (size_t i = 0; i < numCentroids; ++i) {
        int idx = numSampledVectors + i;
        float umap_1 = embedding[idx*3];
        float umap_2 = embedding[idx*3+1];
        float umap_3 = embedding[idx*3+2];
        int clusterId = (int)i;
        int isCentroid = 1;
        fwrite(&umap_1, sizeof(float), 1, fp);
        fwrite(&umap_2, sizeof(float), 1, fp);
        fwrite(&umap_3, sizeof(float), 1, fp);
        fwrite(&clusterId, sizeof(int), 1, fp);
        fwrite(&isCentroid, sizeof(int), 1, fp);
    }
    
    fclose(fp);
    printf("UMAP 3D visualization with clusters written to %s\n", outputPath.c_str());
}

void write_debug_data(ReclusteringIndex* index, int iter,  std::vector<double> queryRecalls, const std::string& baseDir = "scores/") {
    auto dim = index->getDim();
    // Write mega and mini centroids
    auto mega_centroids_file_path = baseDir + "mega_centroids_iter_" + std::to_string(iter) + ".bin";
    const float* megaCentroids;
    size_t numMegaCentroids;
    index->getMegaCentroids(&megaCentroids, numMegaCentroids);
    writeToFile(mega_centroids_file_path, reinterpret_cast<const uint8_t *>(megaCentroids),
                numMegaCentroids * dim * sizeof(float));

    auto mini_centroids_file_path = baseDir + "mini_centroids_iter_" + std::to_string(iter) + ".bin";
    const float* miniCentroids;
    size_t numMiniCentroids;
    index->getMiniCentroids(&miniCentroids, numMiniCentroids);
    writeToFile(mini_centroids_file_path, reinterpret_cast<const uint8_t *>(miniCentroids),
                numMiniCentroids * dim * sizeof(float));

    // Write mega-mini centroid ids
    std::vector<std::vector<vector_idx_t>> megaMiniCentroidIds;
    index->getMegaMiniCentroids(&megaMiniCentroidIds);
    auto mega_mini_centroids_file_path = baseDir + "mega_mini_centroids_iter_" + std::to_string(iter) + ".bin";
    writeNestedVectorToFile(mega_mini_centroids_file_path, megaMiniCentroidIds);

    // Write overlapping scores
    auto approx_overlapping_file_path = baseDir + "approx_overlap_scores_iter_" + std::to_string(iter) + ".bin";
    auto real_overlapping_file_path = baseDir + "real_overlap_scores_iter_" + std::to_string(iter) + ".bin";
    const double* overlapScores;
    size_t numScores;
    index->getApproxOverlapScores(&overlapScores, numScores);
    writeToFile(approx_overlapping_file_path, reinterpret_cast<const uint8_t *>(overlapScores), numScores * sizeof(double));
    index->getRealOverlapScores(&overlapScores, numScores);
    writeToFile(real_overlapping_file_path, reinterpret_cast<const uint8_t *>(overlapScores), numScores * sizeof(double));

    if (!queryRecalls.empty()) {
        // Write recall
        auto recall_file_path = baseDir + "recall_iter_" + std::to_string(iter) + ".bin";
        writeToFile(recall_file_path, reinterpret_cast<const uint8_t *>(queryRecalls.data()),
                    queryRecalls.size() * sizeof(double));
    }
}

void benchmark_fast_reclustering(InputParser &input) {
    auto getStringOption = [&](const std::string &option, const std::string &defaultValue = "") -> std::string {
        const auto &value = input.getCmdOption(option);
        return value.empty() ? defaultValue : value;
    };
    auto getIntOption = [&](const std::string &option, int defaultValue) -> int {
        const auto &value = input.getCmdOption(option);
        return value.empty() ? defaultValue : stoi(value);
    };
    auto getFloatOption = [&](const std::string &option, float defaultValue) -> float {
        const auto &value = input.getCmdOption(option);
        return value.empty() ? defaultValue : stof(value);
    };
    auto getBoolOption = [&](const std::string &option, bool defaultValue) -> bool {
        const auto &value = input.getCmdOption(option);
        return value.empty() ? defaultValue : (stoi(value) != 0);
    };
    auto getIntListOption = [&](const std::string &option, const std::string &defaultValue) -> std::vector<int> {
        const auto &value = input.getCmdOption(option);
        return parseCommaSeparatedIntegers(value.empty() ? defaultValue : value);
    };

    const std::string baseVectorPath = getStringOption("-baseVectorPath");
    const std::string queryVectorPath = getStringOption("-queryVectorPath");
    const std::string groundTruthPath = getStringOption("-groundTruthPath");
    if (baseVectorPath.empty() || queryVectorPath.empty() || groundTruthPath.empty()) {
        fprintf(stderr, "benchmarkFastReclustering requires -baseVectorPath, -queryVectorPath, and -groundTruthPath\n");
        exit(1);
    }

    const bool isParquet = getBoolOption("-isParquet", false);
    int numInserts = getIntOption("-numInserts", 100);
    const int numVectors = getIntOption("-numVectors", 1000000);
    const int k = getIntOption("-k", 100);
    const int numIters = getIntOption("-numIters", 10);
    const int megaCentroidSize = getIntOption("-megaCentroidSize", 1000);
    const int miniCentroidSize = getIntOption("-miniCentroidSize", 1000);
    const float lambda = getFloatOption("-lambda", 0.0f);
    const int numMegaReclusterCentroids = getIntOption("-numMegaReclusterCentroids", 1);
    const int reclusterOnScore = getIntOption("-reclusterOnScore", 0);
    auto nMegaProbes = getIntListOption("-nMegaProbes", "20");
    auto nMiniProbes = getIntListOption("-nMiniProbes", "250");
    const int iterations = getIntOption("-iterations", 7);
    // const bool fast = getBoolOption("-fast", false);
    const int numQueries = getIntOption("-numQueries", 10);
    const int readFromDisk = getIntOption("-readFromDisk", 0);
    const std::string storagePath = getStringOption("-storagePath", "orangedb_recluster.bin");
    const int numThreads = getIntOption("-numThreads", std::max(1, omp_get_max_threads()));
    const bool useIP = getBoolOption("-useIP", true);
    const float quantTrainPercentage = getFloatOption("-quantTrainPercentage", 0.1f);
    const bool quantBuild = getBoolOption("-quantBuild", false);
    // const int avgSubCellSize = getIntOption("-avgSubCellSize", 1000);
    auto nMiniProbesForBadCluster = getIntListOption("-nMiniProbesForBadCluster", "50");
    const int nMegaRecluster = getIntOption("-nMegaRecluster", 1000000000);
    int nFiles = getIntOption("-nFiles", 10);
    int hardClusterSizeLimit = getIntOption("-hardClusterSizeLimit", 0);
    float kmeansSamplingRatio = getFloatOption("-kmeansSamplingRatio", 0.2f);
    int numFixBoundaries = getIntOption("-numFixBoundaries", 10);
    float scoreChangeThreshold = getFloatOption("-scoreChangeThreshold", 0.25f);
    float centroidChangeThreshold = getFloatOption("-centroidChangeThreshold", 0.7f);
    const bool useMSEToRecluster = getBoolOption("-useMSEToRecluster", false);
    const int umap_mode = getIntOption("-umap_mode", 0);
    const float overlapScoreChangeThreshold = getFloatOption("-overlapScoreChangeThreshold", 0.2f);
    const int LshNbits = getIntOption("-LshNbits", 8);
    const bool useCuvsKmeans = getBoolOption("-useCuvsKmeans", false);
    const int cuvsGpuDevice = getIntOption("-cuvsGpuDevice", 0);
    omp_set_num_threads(numThreads);

    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors, numQueries);
    queryNumVectors = std::min(queryNumVectors, (size_t) numQueries);

    DistanceType distanceType = useIP ? IP : L2;
    ReclusteringIndexConfig config(numIters, megaCentroidSize, miniCentroidSize, 0, lambda, 0.4, distanceType,
                                   0, 0, quantTrainPercentage, hardClusterSizeLimit, kmeansSamplingRatio,
                                   scoreChangeThreshold, centroidChangeThreshold, 0.1, LshNbits, 20, 30,
                                   overlapScoreChangeThreshold, useCuvsKmeans, cuvsGpuDevice);
    // CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    loadFromFile(groundTruthPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));

    RandomGenerator rng(1234);
    ReclusteringIndex index(queryDimension, config, &rng);

    size_t baseDimension = queryDimension;
    size_t baseNumVectors = 0;
    float *baseVecs = nullptr;

    if (readFromDisk) {
        index = ReclusteringIndex(storagePath, &rng);
        index.config.overlapScoreChangeThreshold = overlapScoreChangeThreshold;
    } else {
        // Read dataset
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
            baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors, numVectors);
        }
        baseNumVectors = std::min(baseNumVectors, (size_t) numVectors);
        assert(baseDimension == queryDimension);
        if (isParquet) {
            auto numFiles = std::min(nFiles, (int)filePaths.size());
            // Calculate how many batch inserts per file
            int insertsPerFile = std::max(1, numInserts / numFiles);
            int totalBatches = numFiles * insertsPerFile;
            printf("Reading %d parquet files with %d batch inserts per file (total %d batches)\n", 
                   numFiles, insertsPerFile, totalBatches);
            
            size_t totalVectors = 0;
            int batchCount = 0;
            for (int fileIdx = 0; fileIdx < numFiles; fileIdx++) {
                printf("Processing parquet file %d/%d: %s\n", fileIdx + 1, numFiles, filePaths[fileIdx].c_str());
                
                // Read single file
                std::vector<std::string> paths = {filePaths[fileIdx]};
                size_t fileNumVectors;
                auto data = readParquetFiles(paths, &baseDimension, &fileNumVectors);
                
                // Split file data into insertsPerFile batches
                size_t vectorsPerBatch = fileNumVectors / insertsPerFile;
                for (int batchIdx = 0; batchIdx < insertsPerFile; batchIdx++) {
                    size_t batchStart = batchIdx * vectorsPerBatch;
                    size_t batchEnd = (batchIdx == insertsPerFile - 1) ? fileNumVectors : (batchIdx + 1) * vectorsPerBatch;
                    size_t batchSize = batchEnd - batchStart;
                    
                    printf("  Batch %d/%d: inserting vectors [%zu, %zu) (%zu vectors)\n", 
                           batchIdx + 1, insertsPerFile, batchStart, batchEnd, batchSize);
                    
                    // Insert batch (offset into data array)
                    index.naiveInsert(data + batchStart * baseDimension, batchSize);
                    totalVectors += batchSize;
                    batchCount++;

                    // if (fileIdx == numFiles - 1 && batchIdx == insertsPerFile - 2) {
                    //     // One file before last, adjust insertsPerFile to match total numInserts. Run reclustering
                    //     index.storeMSEScoreForMegaClusters();
                    //     index.computeOverlapScores();
                    //     // Run 6 iteration of reclustering
                    //     for (int iter = 0; iter < iterations; iter++) {
                    //         printf("Reclustering Iteration: %d\n", iter);
                    //         index.updateOverlapHistory();
                    //         index.reclusterAllMegaCentroids(nMegaRecluster);
                    //         index.storeMSEScoreForMegaClusters();
                    //         index.computeOverlapScores();
                    //         index.reclusterBasedOnOverlapHistory();
                    //         index.printStats();
                    //         printf("Reclustering Iteration %d completed\n", iter);
                    //     }
                    // }
                }
                delete[] data;
            }
            printf("Total vectors inserted: %zu in %d batches\n", totalVectors, batchCount);
        } else {
            if (quantBuild) {
                index.trainQuant(baseVecs, baseNumVectors);
            }
            printf("Building index with realtime reclustering\n");
            auto chunkSize = baseNumVectors / numInserts;
            printf("Chunk size: %lu\n", chunkSize);
            auto startReclusterPoint = (numInserts / 2) - 1;
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

                // Recluster after 50 inserts, then every 2 inserts thereafter
                bool should_recluster = false;
                if (i == startReclusterPoint) {
                    // After 50 inserts (0-indexed, so i == 49)
                    printf("=== Completed %ld inserts - Running reclustering ===\n", i + 1);
                    should_recluster = true;
                } else if (i > startReclusterPoint && (i - startReclusterPoint) % 2 == 0) {
                    // Every 2 inserts after the 50th (i.e., at 51, 53, 55, ...)
                    printf("=== Completed %ld inserts - Running reclustering ===\n", i + 1);
                    should_recluster = true;
                }

                if (useMSEToRecluster && should_recluster) {
                    for (int iter = 0; iter < iterations; iter++) {
                        printf("Reclustering Iteration: %d\n", iter);
                        index.storeMSEScoreForMegaClusters();
                        index.saveOldScoreForMegaClusters();
                        index.reclusterAllMegaCentroids(nMegaRecluster);
                        index.storeMSEScoreForMegaClusters();
                        index.reclusterBasedOnMSEScore();
                        printf("Reclustering Iteration %d completed\n", iter);
                    }
                    printf("=== Reclustering completed ===\n");
                }
            }
        }
        printf("Writing index to disk\n");
        index.flush_to_disk(storagePath);
    }
    // index.quantizeVectors();

    // auto recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbes,
    //                              nMiniProbes);

    // auto recallWithoutBadClusters = get_recall_with_bad_clusters(index, queryVecs, queryDimension,
    //                                         queryNumVectors, k, gtVecs, nMegaProbes, nMiniProbes,
    //                                         nMiniProbesForBadCluster, true);
    // index.storeScoreForMegaClusters();
    // auto recallWithBadCluster = get_recall_with_bad_clusters(index, queryVecs, queryDimension,
    //                                          queryNumVectors, k, gtVecs, nMegaProbes, nMiniProbes,
    //                                          nMiniProbesForBadCluster, false);

    // index.computeAllSubCells(avgSubCellSize);
    // auto quantizedRecall = get_quantized_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs,
    //                                             nMegaProbes, nMiniProbes);
    // printf("Recall: %f, Recall without bad clusters: %f, Recall with bad clusters: %f\n", recall, recallWithoutBadClusters, recallWithBadCluster);
    // index.reclusterAllMegaCentroids(nMegaRecluster);
    // index.flush_to_disk(storagePath);
    // index.storeMSEScoreForMegaClusters();
    // index.computeOverlapScores();
    index.printStats();

    std::vector<std::vector<double>> prevRecallValues;
    for (auto nMegaProbe : nMegaProbes) {
        for (auto nMiniProbe : nMiniProbes) {
            std::vector<double> recallValues;
            auto recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbe,
                                    nMiniProbe, recallValues);
            printf("nMegaProbes: %d, nMiniProbes: %d, Recall: %f, Recall with bad clusters: %f\n", nMegaProbe, nMiniProbe, recall, 0.0f);
            prevRecallValues.push_back(std::move(recallValues));
        }
    }

    // Calculate and write recall after writing overlap scores
    // Write per-query recall for the first probe combination
    // std::vector<double> queryRecalls;
    // if (!nMegaProbes.empty() && !nMiniProbes.empty()) {
    //     auto recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbes[0],
    //                              nMiniProbes[0], queryRecalls);
    // }
    // write_debug_data(&index, 0, queryRecalls);

    if (useMSEToRecluster) {
        // Generate UMAP visualization with cluster assignments (before early return)
        if (baseVecs != nullptr && baseNumVectors > 0) {
            if(umap_mode==LIVE_UMAP) {
                printf("\n=== Generating UMAP Visualization ===\n");
                // run_umap_2D_with_cluster_data(index, baseVecs, (int)baseNumVectors, baseDimension, "umap_l2_clusters_2D.bin", C_L2);
                run_umap_3D_with_cluster_data(index, "umap_l2_clusters_3D.bin", C_L2, 100000, numThreads);
                // run_umap_2D_with_cluster_data(index, baseVecs, (int)baseNumVectors, baseDimension, "umap_l1_clusters_2D.bin", C_L1);
                run_umap_3D_with_cluster_data(index, "umap_l1_clusters_3D.bin", C_L1, 100000, numThreads);
            } else if(umap_mode==OFFLINE_UMAP) {
                printf("\n=== saving clustering data ===\n");
                save_clustering_data(index, baseVecs, (int)baseNumVectors, baseDimension, "clustering_data_l2.bin", C_L2);
                save_clustering_data(index, baseVecs, (int)baseNumVectors, baseDimension, "clustering_data_l1.bin", C_L1);
            } 
        } else {
            printf("Skipping UMAP visualization\n");
        }
        return;
    }
    printf("Starting reclustering iterations\n");
    // auto track_query_id = 0;
    // index.flush_to_disk(storagePath);
    // index.storeMSEScoreForMegaClusters();
    for (int iter = 0; iter < iterations; iter++) {
        printf("Started Iteration: %d\n", iter);
        // index.updateOverlapHistory();
        index.reclusterAllMegaCentroids(nMegaRecluster);
        // index.storeMSEScoreForMegaClusters();
        // index.computeOverlapScores();
        // index.printStats();
        index.printWrongAssignmentStatsForWorstMinis();

        // Calculate and write recall after writing overlap scores
        // Write per-query recall for the first probe combination
        // std::vector<double> queryRecalls;
        // if (!nMegaProbes.empty() && !nMiniProbes.empty()) {
        //     auto recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbes[0],
        //                              nMiniProbes[0], queryRecalls);
        // }
        // write_debug_data(&index, iter + 1, queryRecalls);

        // Generate UMAP visualization with cluster assignments (before early return)
        if(umap_mode==LIVE_UMAP) {
            printf("\n=== Generating UMAP Visualization ===\n");
            // run_umap_2D_with_cluster_data(index, baseVecs, (int) baseNumVectors, baseDimension,
                                          // "umap_l2_clusters_2D_iter_" + std::to_string(iter + 1) + ".bin", C_L2);
            run_umap_3D_with_cluster_data(index,
                                          "umap_l2_clusters_3D_iter_" + std::to_string(iter + 1) + ".bin", C_L2,
                                          100000, numThreads);
            // run_umap_2D_with_cluster_data(index, baseVecs, (int) baseNumVectors, baseDimension,
                                          // "umap_l1_clusters_2D_iter_" + std::to_string(iter + 1) + ".bin", C_L1);
            run_umap_3D_with_cluster_data(index,
                                          "umap_l1_clusters_3D_iter_" + std::to_string(iter + 1) + ".bin", C_L1,
                                          100000, numThreads);
        } else if(umap_mode==OFFLINE_UMAP) {
            printf("\n=== saving clustering data ===\n");
            save_clustering_data(index, baseVecs, (int)baseNumVectors, baseDimension, "clustering_data_l2.bin", C_L2);
            save_clustering_data(index, baseVecs, (int)baseNumVectors, baseDimension, "clustering_data_l1.bin", C_L1);
        }

        // quantizedRecall = get_quantized_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs,
                                             // nMegaProbes, nMiniProbes);
        if (numMegaReclusterCentroids == 1) {
            index.reclusterFast();
            // std::vector<vector_idx_t> megaClusterIds;
            // index.getMegaClusterIds(megaClusterIds);
            // for (auto megaClusterId : megaClusterIds) {
            //     index.reclusterInternalMegaCentroid(megaClusterId);
            //
            //     // bool bigChangeInRecall = false;
            //     // for (int i = 0; i < nMegaProbes.size(); i++) {
            //     //     auto nMegaProbe = nMegaProbes[i];
            //     //     for (int j = 0; j < nMiniProbes.size(); j++) {
            //     //         auto nMiniProbe = nMiniProbes[j];
            //     //         std::vector<double> queryRecalls;
            //     //         auto recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs,
            //     //                                  nMegaProbe,
            //     //                                  nMiniProbe, queryRecalls);
            //     //         auto &prevRecall = prevRecallValues[i * nMiniProbes.size() + j];
            //     //         for (size_t m = 0; m < queryRecalls.size(); m++) {
            //     //             if (queryRecalls[m] < prevRecall[m] - 5) {
            //     //                 if (m == track_query_id) {
            //     //                     bigChangeInRecall = true;
            //     //                 }
            //     //                 printf(
            //     //                     "Warning: Recall decreased for nMegaProbes: %d, nMiniProbes: %d, Query %zu, Previous Recall: %f, Current Recall: %f\n",
            //     //                     nMegaProbe, nMiniProbe, m, prevRecall[m], queryRecalls[m]);
            //     //             }
            //     //         }
            //     //     }
            //     // }
            //
            //     // if (bigChangeInRecall) {
            //     //     index.analyzeQueryClusterChanges(queryVecs + track_query_id * queryDimension,
            //     //                                      gtVecs + k * track_query_id, k,
            //     //                                      false);
            //     // }
            // }
        } else {
            if (reclusterOnScore) {
                index.reclusterBasedOnScore(numMegaReclusterCentroids);
            } else {
                index.reclusterFull(numMegaReclusterCentroids);
            }
        }
        // index.quantizeVectors();
        // index.fixBoundaryMiniCentroidsV2();
        // index.storeScoreForMegaClusters();
        prevRecallValues.clear();
        for (auto nMegaProbe : nMegaProbes) {
            for (auto nMiniProbe : nMiniProbes) {
                std::vector<double> queryRecalls;
                auto recall = get_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs, nMegaProbe,
                                        nMiniProbe, queryRecalls);
                // auto recallWithBadClusters = get_recall_with_bad_clusters(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs,
                //                                       nMegaProbe,
                //                                       nMiniProbe, 5, false);
                prevRecallValues.push_back(std::move(queryRecalls));
                printf("nMegaProbes: %d, nMiniProbes: %d, Recall: %f, Recall with bad clusters: %f\n", nMegaProbe,
                       nMiniProbe, recall, 0.0f);
            }
        }
        // quantizedRecall = get_quantized_recall(index, queryVecs, queryDimension, queryNumVectors, k, gtVecs,
        //                              nMegaProbes, nMiniProbes);
        // index.storeMSEScoreForMegaClusters();
        // index.storeScoreForMegaClusters();
        // index.printStats();

    }
    // index.storeMSEScoreForMegaClusters();
    // index.computeOverlapScores();
    index.printStats();
    if (iterations > 0) {
        // index.storeScoreForMegaClusters();
        // index.printStats();
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

void benchmark_faiss_flat(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    const std::string &groundTruthPath = input.getCmdOption("-groundTruthPath");
    const std::string parquetColumnName = input.getCmdOption("-parquetColumnName").empty()
                                                  ? "emb"
                                                  : input.getCmdOption("-parquetColumnName");
    const int numVectors = stoi(input.getCmdOption("-numVectors"));
    const int nThreads = stoi(input.getCmdOption("-nThreads"));
    const int k = stoi(input.getCmdOption("-k"));
    const int numQueries = stoi(input.getCmdOption("-numQueries"));
    const int isParquet = stoi(input.getCmdOption("-isParquet"));
    int nFiles = stoi(input.getCmdOption("-nFiles"));
    const bool useIP = stoi(input.getCmdOption("-useIP"));

    size_t baseDimension, totalBaseNumVectors;
    std::vector<std::string> filePaths;

    // Load base vectors
    if (isParquet) {
        list_parquet_dir(baseVectorPath.c_str(), filePaths);
        if (filePaths.empty()) {
            fprintf(stderr, "No parquet files found in the directory: %s\n", baseVectorPath.c_str());
            exit(1);
        }
    }

    auto metric = useIP ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

    // Load query vectors
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors, numQueries);
    queryNumVectors = std::min(queryNumVectors, (size_t) numQueries);

    // Create Flat IP index for exact search
    faiss::IndexFlat index(queryDimension, metric);
    omp_set_num_threads(nThreads);
    float *allVecs;
    // Load base vectors into index
    if (isParquet) {
        // Allocate memory for all vectors at once
        nFiles = std::min(nFiles, (int) filePaths.size());
        std::vector<std::string> newFilePaths(nFiles);
        for (int i = 0; i < nFiles; i++) {
            newFilePaths[i] = filePaths[i];
        }
        totalBaseNumVectors = 0;
        float* fileData = readParquetFiles(newFilePaths, &baseDimension, &totalBaseNumVectors, parquetColumnName);
        CHECK_ARGUMENT(baseDimension == queryDimension, "Base and query dimensions are not same");
        printf("Total number of vectors: %zu\n", totalBaseNumVectors);
        // Directly assign to IndexFlatIP codes without copying
        index.ntotal = totalBaseNumVectors;
        index.codes = faiss::MaybeOwnedVector<uint8_t>::create_view(reinterpret_cast<uint8_t *>(fileData),
                                                                    totalBaseNumVectors * baseDimension * sizeof(float), nullptr);
    } else {
        allVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &totalBaseNumVectors, numVectors);
        index.ntotal = std::min(totalBaseNumVectors, (size_t) numVectors);
        index.codes = faiss::MaybeOwnedVector<uint8_t>::create_view(reinterpret_cast<uint8_t *>(allVecs),
                                                                    totalBaseNumVectors * baseDimension * sizeof(float),
                                                                    nullptr);
    }
    printf("Generating ground truth using Flat IP index with %zu vectors\n", totalBaseNumVectors);
    // Generate ground truth
    auto *gtVecs = new vector_idx_t[queryNumVectors * k];
    auto *labels = new faiss::idx_t[k];
    auto *distances = new float[k];

    printf("Generating ground truth for %zu queries with k=%d\n", queryNumVectors, k);
    auto startTime = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < queryNumVectors; i++) {
        printf("Processing query %zu/%zu\n", i, queryNumVectors);
        index.search(1, queryVecs + (i * queryDimension), k, distances, labels);
        // Copy exact search results to ground truth
        for (int j = 0; j < k; j++) {
            gtVecs[i * k + j] = labels[j];
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Save ground truth to file
    printf("Writing ground truth to: %s\n", groundTruthPath.c_str());
    writeToFile(groundTruthPath, reinterpret_cast<uint8_t *>(gtVecs), queryNumVectors * k * sizeof(vector_idx_t));

    printf("Ground truth generation completed!\n");
    printf("Time taken: %lld ms\n", duration.count());
    printf("Queries processed: %zu\n", queryNumVectors);
    printf("k value: %d\n", k);

    // Cleanup
    delete[] labels;
    delete[] distances;
    delete[] gtVecs;
    delete[] queryVecs;
    delete[] allVecs;
}

void benchmark_balanced_clustering(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const int clusterSize = stoi(input.getCmdOption("-clusterSize"));
    const int nIter = stoi(input.getCmdOption("-nIter"));
    const int nThreads = stoi(input.getCmdOption("-nThreads"));
    const float lambda = stof(input.getCmdOption("-lambda"));
    const int sampleSize = stoi(input.getCmdOption("-sampleSize"));
    const bool useIP = stoi(input.getCmdOption("-useIP"));

    auto metric = useIP ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    auto numCentroids = baseNumVectors / clusterSize;
    faiss::Clustering clustering(baseDimension, clusterSize);

    omp_set_num_threads(nThreads);
    auto sampleSizeAdjusted = std::min((size_t)sampleSize, baseNumVectors);
    clustering.niter = nIter;
    clustering.max_points_per_centroid = (sampleSizeAdjusted / numCentroids);
    clustering.min_points_per_centroid = (sampleSizeAdjusted / numCentroids) * 0.3;
    printf("max_points_per_centroid: %d, min_points_per_centroid: %d\n",
           clustering.max_points_per_centroid, clustering.min_points_per_centroid);
    clustering.verbose = true;
    // clustering.lambda = lambda;

    faiss::IndexFlat index(baseDimension, metric);
    clustering.train(baseNumVectors, baseVecs, index);

    std::vector<int64_t> assignment(baseNumVectors);
    index.assign(baseNumVectors, baseVecs, assignment.data());

    std::vector<int> histogram(numCentroids, 0);
    for (size_t i = 0; i < baseNumVectors; ++i) {
        if (assignment[i] >= 0 && assignment[i] < numCentroids) {
            histogram[assignment[i]]++;
        }
    }
    
    int min_cluster_size = *std::min_element(histogram.begin(), histogram.end());
    int max_cluster_size = *std::max_element(histogram.begin(), histogram.end());
    double avg_cluster_size = static_cast<double>(baseNumVectors) / numCentroids;
    
    double sum_squared_diff = 0.0;
    for (int count : histogram) {
        double diff = count - avg_cluster_size;
        sum_squared_diff += diff * diff;
    }
    double std_dev = std::sqrt(sum_squared_diff / numCentroids);
    
    printf("Assignment histogram statistics:\n");
    printf("  Min cluster size: %d\n", min_cluster_size);
    printf("  Max cluster size: %d\n", max_cluster_size);
    printf("  Average cluster size: %.2f\n", avg_cluster_size);
    printf("  Standard deviation: %.2f\n", std_dev);
}

void test_something(InputParser &input) {
    RandomGenerator rg(1234);
    auto total_selected = 0;
    auto tota_rows = 0;
    for (int i = 0; i < 100; i++) {
        if (rg.randFloat() < 0.1) {
            total_selected += 1;
        }
        tota_rows++;
    }
    printf("total_selected: %d, total_rows: %d, Average selected: %f\n", total_selected, tota_rows,
           (double) total_selected / tota_rows);
}

void test_bug(InputParser &input) {
    const std::string &dataPath = input.getCmdOption("-dataPath");
    const std::string &centroidsPath = input.getCmdOption("-centroidsPath");
    const std::string &clusterSizePath = input.getCmdOption("-clusterSizePath");
    const int nThreads = stoi(input.getCmdOption("-nThreads"));
    omp_set_num_threads(nThreads);
    size_t numCentroids = 5710;
    size_t dimension = 128;
    size_t numVectors = 3078;
    auto hardLimit = 3800;
    // Allocate centroids
    std::vector<float> data(numVectors * dimension);
    std::vector<float> centroids(numCentroids * dimension);
    std::vector<int64_t> clusterSizes(numCentroids, 0);
    loadFromFile(dataPath,
             reinterpret_cast<uint8_t *>(data.data()),
             numVectors * dimension * sizeof(float));
    loadFromFile(centroidsPath,
                 reinterpret_cast<uint8_t *>(centroids.data()),
                 numCentroids * dimension * sizeof(float));
    loadFromFile(clusterSizePath,
                 reinterpret_cast<uint8_t *>(clusterSizes.data()),
                 numCentroids * sizeof(int64_t));

    for (size_t i = 0; i < 100; i++) {
        printf("Running iteration %zu\n", i);
        auto index = faiss::IndexFlatL2(dimension);
        index.add(numCentroids, centroids.data());
        std::vector<int64_t> assign(numVectors);
        std::vector<float> distances(numVectors);
        faiss::ClusterSizeCapDistModifier hardLimitDistModifier(numCentroids, hardLimit);
        hardLimitDistModifier.populate_weights(clusterSizes.data(), numCentroids);
        faiss::SearchParameters params;
        params.dist_modifier = &hardLimitDistModifier;
        index.search(numVectors, data.data(), 1, distances.data(), assign.data(), &params);
    }
}

void test_another_bug(InputParser &input) {
    omp_set_num_threads(8);
    int dimension = 100;
    int numVectors = 65536;
    int avg_cluster_size = 128;
    RandomGenerator rg(1234);
    std::vector<float> data(numVectors * dimension);
    for (int i = 0; i < numVectors; i++) {
        data[i * dimension + 0] = 0;
        data[i * dimension + 1] = (i % 512) * 1000;
        for (int d = 2; d < dimension; d++) {
            data[i * dimension + d] = rg.randFloat();
        }
    }

    faiss::ClusteringParameters cl;
    cl.niter = 20;
    cl.nredo = 1;
    cl.min_points_per_centroid = avg_cluster_size * 0.5;
    cl.max_points_per_centroid = avg_cluster_size * 1.7;
    std::unique_ptr<faiss::BalancedClusteringDistModifier> distModifier;
    cl.verbose = true;
    auto numCentroids = numVectors / avg_cluster_size + 1;
    faiss::Clustering clustering(dimension, numCentroids, cl);
    auto index = faiss::IndexFlatL2(dimension);
    clustering.train(numVectors, data.data(), index);

    std::vector<int64_t> assign(numVectors);
    index.assign(numVectors, data.data(), assign.data());

    // Print histogram
    std::vector<int> histogram(numCentroids, 0);
    for (int i = 0; i < numVectors; i++) {
        histogram[assign[i]]++;
    }
    // for (int i = 0; i < numCentroids; i++) {
    //     printf("Cluster %d: Size %d\n", i, histogram[i]);
    // }
    // if any centroid is greater than 128 then print the vector assigned to it
    // for (int i = 0; i < numCentroids; i++) {
    //     if (histogram[i] > avg_cluster_size) {
    //         printf("Centroid %d has size %d, Vectors assigned:\n", i, histogram[i]);
    //         for (int j = 0; j < numVectors; j++) {
    //             if (assign[j] == i) {
    //                 printf("Vector %d: ", j);
    //                 for (int d = 0; d < dimension; d++) {
    //                     printf("%f, ", data[j * dimension + d]);
    //                 }
    //                 printf("\n");
    //             }
    //         }
    //     }
    // }
}

void test_final_bug(InputParser &input) {
    int dimension = 3;
    int avg_cluster_size = 10;
    int numL2s = 10;
    int numL1s = 100;
    int numVectors = numL2s * numL1s;
    RandomGenerator rg(1234);
    std::vector<float> data(numVectors * dimension);
    for (int i = 0; i < numVectors; i++) {
        data[i * dimension + 0] = 100000 * (i % numL2s);
        data[i * dimension + 1] = 100 * (i % numL1s);
        data[i * dimension + 2] = rg.randFloat();
    }

    ReclusteringIndexConfig config;
    config.megaCentroidSize = avg_cluster_size;
    config.miniCentroidSize = avg_cluster_size;
    config.nIter = 20;
    ReclusteringIndex index(dimension, config, &rg);
    index.simpleInsertWithoutClustering(data.data(), numVectors);
    index.storeScoreForMegaClusters();
    index.printStats();
    for (int iter = 0; iter < 5; iter++) {
        printf("Iteration %d\n", iter);
        index.reclusterAllMegaCentroids();
        index.reclusterFast();
    }
    index.storeScoreForMegaClusters();
    index.printStats();
    // ReclusteringIndex index(dimension, 10, 10, 10, 0.5, L2);
}

void test_final_bug_2(InputParser &input) {
    // omp_set_num_threads(8);
    // size_t baseDimension, baseNumVectors;
    // auto filePath = "/Users/gaurav.sehgal/work/orangedb/data/ifmwcvluoe.parquet";
    // std::vector<std::string> file_paths;
    // file_paths.push_back(filePath);
    // auto data = readParquetFiles(file_paths, &baseDimension, &baseNumVectors);
    // printf("Read %zu vectors of dimension %zu from %s\n", baseNumVectors, baseDimension, filePath);
    // RandomGenerator rg(1234);
    // ReclusteringIndexConfig config;
    // config.megaCentroidSize = 1000;
    // config.miniCentroidSize = 1000;
    // config.kmeansSamplingRatio = 0.2;
    // config.nIter = 20;
    // ReclusteringIndex index(baseDimension, config, &rg);
    // index.naiveInsert(data, baseNumVectors);
    // // index.storeScoreForMegaClusters();
    // // index.printStats();
    // for (int iter = 0; iter < 1; iter++) {
    //     printf("Iteration %d\n", iter);
    //     index.reclusterAllMegaCentroids();
    //     index.reclusterFast();
    // }
    // // index.storeScoreForMegaClusters();
    // // index.printStats();
    //
    // int numQueries = 10;
    // std::vector<float> queryVecs(baseDimension * 10);
    // for (size_t i = 0; i < baseDimension * numQueries; i++) {
    //     queryVecs[i] = 30 * rg.randFloat();
    // }
    // int k = 10;
    //
    // faiss::IndexFlatL2 flat_ind(baseDimension);
    // faiss::IndexIVFFlat ivf_index(&flat_ind, baseDimension, 1048, faiss::METRIC_L2);
    // ivf_index.cp.max_points_per_centroid = 200;
    // ivf_index.nprobe = 700;
    // ivf_index.cp.verbose = true;
    // ivf_index.cp.niter = 20;
    // ivf_index.train(baseNumVectors, data);
    // ivf_index.add(baseNumVectors, data);
    // printf("Added data to faiss index\n");
    // std::vector<faiss::idx_t> act_gt_labels(numQueries * k);
    // std::vector<float> act_gt_distances(numQueries * k);
    // ivf_index.search(numQueries, queryVecs.data(), k, act_gt_distances.data(), act_gt_labels.data());
    //
    // faiss::IndexFlatL2 flat_index(baseDimension);
    // flat_index.add(baseNumVectors, data);
    // std::vector<faiss::idx_t> gt_labels(numQueries * k);
    // std::vector<float> gt_distances(numQueries * k);
    // flat_index.search(numQueries, queryVecs.data(), k, gt_distances.data(), gt_labels.data());
    //
    // // Check recall
    // for (int i = 0; i < numQueries; i++) {
    //     double localRecall = 0;
    //     auto gt = gt_labels.data() + i * k;
    //     auto act_gt = act_gt_labels.data() + i * k;
    //     for (int j = 0; j < k; j++) {
    //         if (std::find(gt, gt + k, act_gt[j]) != (gt + k)) {
    //             localRecall++;
    //         }
    //     }
    //     printf("Query %d: Faiss IVF Recall: %f\n", i, (localRecall / k) * 100.0);
    // }
    //
    // for (int i = 0; i < numQueries; i++) {
    //     std::priority_queue<NodeDistCloser> results;
    //     ReclusteringIndexStats stats;
    //     index.search(queryVecs.data() + i * baseDimension, k, results, 10, 500, stats);
    //     auto gt = gt_labels.data() + i * k;
    //     double localRecall = 0;
    //     while (!results.empty()) {
    //         auto res = results.top();
    //         results.pop();
    //         if (std::find(gt, gt + k, res.id) != (gt + k)) {
    //             localRecall++;
    //         }
    //     }
    //     printf("Query %d: Recall: %f\n", i, (localRecall / k) * 100.0);
    // }

    std::vector<float> queryVec = {1.0, 2.0, 8.0, 31.0, 19.0, 3.0, 0.0, 0.0, 1.0, 47.0, 86.0, 27.0, 7.0, 0.0, 5.0, 2.0, 2.0, 42.0, 75.0, 10.0, 7.0, 6.0, 7.0, 1.0, 7.0, 8.0, 5.0, 2.0, 4.0, 3.0, 1.0, 1.0, 0.0, 0.0, 62.0, 120.0, 31.0, 0.0, 0.0, 0.0, 4.0, 41.0, 120.0, 120.0, 37.0, 8.0, 7.0, 4.0, 117.0, 120.0, 120.0, 11.0, 0.0, 1.0, 4.0, 15.0, 77.0, 31.0, 3.0, 1.0, 1.0, 3.0, 1.0, 2.0, 9.0, 3.0, 103.0, 86.0, 3.0, 10.0, 77.0, 73.0, 5.0, 5.0, 22.0, 70.0, 20.0, 92.0, 120.0, 38.0, 120.0, 30.0, 9.0, 3.0, 1.0, 9.0, 60.0, 120.0, 107.0, 12.0, 1.0, 3.0, 4.0, 0.0, 0.0, 11.0, 46.0, 15.0, 5.0, 0.0, 3.0, 24.0, 41.0, 74.0, 0.0, 0.0, 9.0, 2.0, 6.0, 53.0, 98.0, 20.0, 16.0, 2.0, 16.0, 2.0, 0.0, 2.0, 59.0, 72.0, 20.0, 8.0, 27.0, 5.0, 0.0, 0.0, 0.0, 10.0};
    std::vector<float> vec1 = {0.0, 11.0, 32.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 80.0, 39.0, 32.0, 16.0, 3.0, 0.0, 17.0, 35.0, 40.0, 11.0, 17.0, 25.0, 46.0, 43.0, 26.0, 11.0, 0.0, 0.0, 3.0, 3.0, 7.0, 44.0, 0.0, 5.0, 76.0, 125.0, 28.0, 1.0, 0.0, 0.0, 6.0, 22.0, 126.0, 126.0, 48.0, 13.0, 6.0, 5.0, 126.0, 126.0, 126.0, 18.0, 2.0, 2.0, 6.0, 25.0, 80.0, 44.0, 0.0, 0.0, 0.0, 1.0, 1.0, 6.0, 16.0, 17.0, 81.0, 64.0, 21.0, 17.0, 17.0, 12.0, 16.0, 6.0, 20.0, 55.0, 52.0, 57.0, 122.0, 58.0, 126.0, 45.0, 9.0, 7.0, 3.0, 10.0, 69.0, 121.0, 116.0, 31.0, 6.0, 1.0, 1.0, 0.0, 0.0, 10.0, 34.0, 11.0, 0.0, 1.0, 1.0, 14.0, 37.0, 26.0, 15.0, 5.0, 5.0, 8.0, 4.0, 20.0, 89.0, 36.0, 56.0, 28.0, 15.0, 8.0, 3.0, 9.0, 54.0, 70.0, 44.0, 37.0, 14.0, 0.0, 0.0, 0.0, 1.0, 37.0};
    std::vector<float> vec2 = {0.0, 0.0, 12.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 47.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0, 31.0, 58.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 9.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 129.0, 110.0, 0.0, 0.0, 1.0, 0.0, 5.0, 21.0, 129.0, 129.0, 14.0, 14.0, 7.0, 3.0, 129.0, 129.0, 123.0, 43.0, 2.0, 3.0, 4.0, 19.0, 35.0, 31.0, 11.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 87.0, 68.0, 0.0, 2.0, 100.0, 94.0, 6.0, 4.0, 29.0, 63.0, 20.0, 111.0, 129.0, 51.0, 129.0, 28.0, 7.0, 9.0, 3.0, 25.0, 114.0, 129.0, 41.0, 7.0, 9.0, 3.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 68.0, 60.0, 0.0, 0.0, 3.0, 4.0, 2.0, 20.0, 65.0, 11.0, 1.0, 1.0, 14.0, 2.0, 0.0, 5.0, 40.0, 26.0, 1.0, 1.0, 6.0, 4.0, 1.0, 0.0, 0.0, 2.0};

    auto dist = std::sqrt(faiss::fvec_L2sqr(queryVec.data(), vec1.data(), queryVec.size()));
    auto dist2 = std::sqrt(faiss::fvec_L2sqr(queryVec.data(), vec2.data(), queryVec.size()));
    printf("Distance to vec1: %f\n", dist);
    printf("Distance to vec2: %f\n", dist2);
}

void test_quantization_issue(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const int numVectors = stoi(input.getCmdOption("-numVectors"));
    const int trainOffset = stoi(input.getCmdOption("-trainOffset"));
    const int numTrainVectors = stoi(input.getCmdOption("-numTrainVectors"));
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    const int sampleSize = stoi(input.getCmdOption("-sampleSize"));
    const int queryIndex = stoi(input.getCmdOption("-queryIndex"));

    // Load base vectors
    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors, numVectors);
    // Load query vectors
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    baseNumVectors = std::min(baseNumVectors, (size_t) numVectors);
    faiss::ScalarQuantizer sq(baseDimension, faiss::ScalarQuantizer::QT_8bit);
    std::vector<uint8_t> codes(baseNumVectors * sq.code_size);
    printf("Training scalar quantizer on %zu vectors of dimension %zu\n", baseNumVectors, baseDimension);
    sq.train(numTrainVectors, baseVecs + trainOffset * baseDimension);
    printf("Computing codes for base vectors\n");
    sq.compute_codes(baseVecs, codes.data(), baseNumVectors);

    // Randomly sample 10k points and compare distances
    RandomGenerator rg(1234);
    std::vector<vector_idx_t> sampleIndices(sampleSize);
    rg.randomPerm(baseNumVectors, sampleIndices.data(), sampleSize);

    // TODO: Compute distances using quantized vectors from query vector 3 onwards
    std::vector<float> actualDistances(baseNumVectors);
    std::vector<float> codesDistances(baseNumVectors);
    std::vector<float> itsOwnDistances(baseNumVectors);
    auto queryVec = queryVecs + queryIndex * queryDimension;
    auto dc = sq.get_distance_computer();
    double distance_diff = 0;
    double avg_its_own_diff = 0;
    double avg_distance_from_query = 0;
    for (size_t i = 0; i < sampleSize; i++) {
        actualDistances[i] = faiss::fvec_L2sqr(queryVec,
                                               baseVecs + sampleIndices[i] * baseDimension,
                                               baseDimension);

        dc->set_query(queryVec);
        codesDistances[i] = dc->distance_to_code(codes.data() + sampleIndices[i] * sq.code_size);
        distance_diff += std::abs(actualDistances[i] - codesDistances[i]);

        dc->set_query(baseVecs + sampleIndices[i] * baseDimension);
        itsOwnDistances[i] = dc->distance_to_code(codes.data() + sampleIndices[i] * sq.code_size);
        avg_its_own_diff += itsOwnDistances[i];
        avg_distance_from_query += actualDistances[i];
    }
    printf("Average distance difference from query over %d (avg %f) samples: %f\n", sampleSize,
           avg_distance_from_query / sampleSize, distance_diff / sampleSize);
    printf("Average distance difference from it's vector over %d samples: %f\n", sampleSize,
           avg_its_own_diff / sampleSize);
    // Write the actualDistances and codesDistances to file for comparison
    writeToFile("./actual_distances.bin",
                reinterpret_cast<uint8_t *>(actualDistances.data()),
                actualDistances.size() * sizeof(float));
    writeToFile("./codes_distances.bin",
                reinterpret_cast<uint8_t *>(codesDistances.data()),
                codesDistances.size() * sizeof(float));
}

static void init_topk_heap(
        size_t k,
        bool useIP,
        std::vector<float> &distances,
        std::vector<int64_t> &labels) {
    distances.resize(k);
    labels.resize(k);
    if (useIP) {
        faiss::minheap_heapify(k, distances.data(), labels.data());
    } else {
        faiss::maxheap_heapify(k, distances.data(), labels.data());
    }
}

static void add_to_topk_heap(
        size_t k,
        bool useIP,
        std::vector<float> &distances,
        std::vector<int64_t> &labels,
        float value,
        int64_t id) {
    if (useIP) {
        if (value > distances[0]) {
            faiss::minheap_replace_top(k, distances.data(), labels.data(), value, id);
        }
    } else {
        if (value < distances[0]) {
            faiss::maxheap_replace_top(k, distances.data(), labels.data(), value, id);
        }
    }
}

static double compute_topk_match_percentage(
        size_t exactK,
        size_t quantizedK,
        bool useIP,
        std::vector<float> &exactDistances,
        std::vector<int64_t> &exactLabels,
        std::vector<float> &quantizedDistances,
        std::vector<int64_t> &quantizedLabels) {
    if (useIP) {
        faiss::minheap_reorder(exactK, exactDistances.data(), exactLabels.data());
        faiss::minheap_reorder(quantizedK, quantizedDistances.data(), quantizedLabels.data());
    } else {
        faiss::maxheap_reorder(exactK, exactDistances.data(), exactLabels.data());
        faiss::maxheap_reorder(quantizedK, quantizedDistances.data(), quantizedLabels.data());
    }

    std::unordered_set<int64_t> exactSet(exactLabels.begin(), exactLabels.end());
    size_t matches = 0;
    for (int64_t id : quantizedLabels) {
        if (exactSet.count(id) != 0) {
            matches++;
        }
    }

    return 100.0 * static_cast<double>(matches) / static_cast<double>(exactK);
}

struct TopKHeapState {
    std::vector<float> distances;
    std::vector<int64_t> labels;
};

struct DistanceMseRunConfig {
    size_t nbits = 0;
    double factor = 1.0;
    size_t quantizedK = 0;
    std::vector<double> perQuerySquaredError;
    std::vector<TopKHeapState> quantizedTopK;
};

struct NbitsRunGroup {
    size_t nbits = 0;
    std::vector<size_t> runIndices;
};

struct RaBitClusterCodes {
    std::vector<size_t> indices;
    std::vector<uint8_t> codes;
    faiss::RaBitQuantizer quantizer;
    const float *centroid = nullptr;

    RaBitClusterCodes(size_t dim, faiss::MetricType metricType, size_t bits)
            : quantizer(dim, metricType, bits) {}
};

struct RaBitChunkConfig {
    size_t nbits = 0;
    std::vector<RaBitClusterCodes> clusters;
};

struct ScalarClusterCodes {
    std::vector<uint8_t> codes;
    faiss::ScalarQuantizer quantizer;
    size_t nbits = 0;

    ScalarClusterCodes(size_t dim, size_t bits)
            : quantizer(dim, bits == 4 ? faiss::ScalarQuantizer::QT_4bit : faiss::ScalarQuantizer::QT_8bit),
              nbits(bits) {}
};

static std::vector<double> parseCommaSeparatedDoubles(const std::string &input) {
    std::vector<double> numbers;
    std::stringstream ss(input);
    std::string temp;

    while (std::getline(ss, temp, ',')) {
        numbers.push_back(std::stod(temp));
    }

    return numbers;
}

static void init_topk_heaps_for_queries(
        size_t numQueries,
        size_t k,
        bool useIP,
        std::vector<TopKHeapState> &heaps) {
    heaps.resize(numQueries);
    for (size_t q = 0; q < numQueries; q++) {
        init_topk_heap(k, useIP, heaps[q].distances, heaps[q].labels);
    }
}

static size_t get_requested_vector_limit(const std::string &numVectorsArg) {
    return numVectorsArg.empty() ? SIZE_MAX : stoull(numVectorsArg);
}

static void collect_parquet_input_files(
        const std::string &baseVectorPath,
        int requestedFiles,
        size_t maxVectors,
        const std::string &parquetColumnName,
        std::vector<std::string> &selectedPaths,
        size_t *baseDimension,
        size_t *baseNumVectors) {
    std::vector<std::string> allPaths;
    list_parquet_dir(baseVectorPath.c_str(), allPaths);
    CHECK_ARGUMENT(!allPaths.empty(), "no parquet files found in baseVectorPath");

    const int numFiles = std::min(requestedFiles, static_cast<int>(allPaths.size()));
    CHECK_ARGUMENT(numFiles > 0, "nFiles must be positive");

    selectedPaths.clear();
    *baseNumVectors = 0;
    bool hasDimension = false;
    for (int fileIdx = 0; fileIdx < numFiles; fileIdx++) {
        size_t fileDimension = 0;
        size_t fileVectors = 0;
        printf("Checking parquet file %s for dimension and vector count\n", allPaths[fileIdx].c_str());
        auto status = readParquetFileStats(
                allPaths[fileIdx].c_str(),
                &fileDimension,
                &fileVectors,
                parquetColumnName);
        CHECK_ARGUMENT(status.ok(), status.ToString().c_str());
        if (!hasDimension) {
            *baseDimension = fileDimension;
            hasDimension = true;
        } else {
            CHECK_ARGUMENT(*baseDimension == fileDimension, "parquet file dimensions do not match");
        }
        if (*baseNumVectors >= maxVectors) {
            break;
        }
        selectedPaths.push_back(allPaths[fileIdx]);
        *baseNumVectors += std::min(fileVectors, maxVectors - *baseNumVectors);
    }

    CHECK_ARGUMENT(!selectedPaths.empty(), "no parquet files selected");
}

static float *read_parquet_sample_files(
        const std::vector<std::string> &filePaths,
        size_t numSampleFiles,
        size_t *dimension,
        size_t *numVectors,
        const std::string &parquetColumnName) {
    std::vector<std::string> samplePaths;
    const size_t sampleFiles = std::min(numSampleFiles, filePaths.size());
    for (size_t i = 0; i < sampleFiles; i++) {
        samplePaths.push_back(filePaths[i]);
    }
    return readParquetFiles(samplePaths, dimension, numVectors, parquetColumnName);
}

static size_t resolve_train_sample_files(
        const std::string &trainSampleFilesArg,
        size_t defaultSampleFiles,
        size_t maxAvailableFiles) {
    const size_t requestedSampleFiles = trainSampleFilesArg.empty()
                                        ? defaultSampleFiles
                                        : stoull(trainSampleFilesArg);
    CHECK_ARGUMENT(requestedSampleFiles > 0, "trainSampleFiles must be positive");
    return std::min(requestedSampleFiles, maxAvailableFiles);
}

static size_t resolve_train_sample_size(
        const std::string &trainSampleSizeArg,
        size_t defaultSampleSize,
        size_t maxAvailableVectors) {
    const size_t requestedSampleSize = trainSampleSizeArg.empty()
                                       ? defaultSampleSize
                                       : stoull(trainSampleSizeArg);
    CHECK_ARGUMENT(requestedSampleSize > 0, "trainSampleSize must be positive");
    return std::min(requestedSampleSize, maxAvailableVectors);
}

static size_t get_clustering_sample_size(size_t totalVectors, size_t numCentroids) {
    size_t sampleVectors = numCentroids * 2000;
    sampleVectors = std::max(sampleVectors, numCentroids);
    return std::min(totalVectors, sampleVectors);
}

static size_t get_scalar_training_sample_size(size_t totalVectors) {
    return std::min(totalVectors, static_cast<size_t>(200000));
}

static std::vector<float> gather_indexed_vectors(
        const float *vectors,
        size_t dimension,
        const std::vector<size_t> &indices) {
    std::vector<float> gathered(indices.size() * dimension);
    for (size_t i = 0; i < indices.size(); i++) {
        memcpy(
                gathered.data() + i * dimension,
                vectors + indices[i] * dimension,
                dimension * sizeof(float));
    }
    return gathered;
}

static std::vector<float> train_kmeans_centroids(
        const float *trainVecs,
        size_t numTrainVectors,
        size_t dimension,
        size_t numCentroids,
        int kmeansNiter,
        bool useIP) {
    CHECK_ARGUMENT(numCentroids > 0, "numCentroids must be positive");
    CHECK_ARGUMENT(numTrainVectors >= numCentroids, "not enough sample vectors for requested numCentroids");

    faiss::ClusteringParameters cp;
    cp.niter = kmeansNiter;
    cp.max_points_per_centroid = INT_MAX;
    cp.verbose = true;
    if (useIP) {
        cp.spherical = true;
    }

    faiss::MetricType metric = useIP ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
    faiss::Clustering clustering(static_cast<int>(dimension), numCentroids, cp);
    faiss::IndexFlat assigner(static_cast<faiss::idx_t>(dimension), metric);
    clustering.train(numTrainVectors, trainVecs, assigner);
    return clustering.centroids;
}

static std::vector<std::vector<size_t>> assign_vectors_to_centroids(
        const float *vectors,
        size_t numVectors,
        size_t dimension,
        faiss::IndexFlat &centroidIndex,
        size_t numCentroids) {
    std::vector<std::vector<size_t>> assignments(numCentroids);
    std::vector<faiss::idx_t> labels(numVectors);
    std::vector<float> distances(numVectors);
    centroidIndex.search(numVectors, vectors, 1, distances.data(), labels.data());
    for (size_t i = 0; i < numVectors; i++) {
        CHECK_ARGUMENT(labels[i] >= 0 && labels[i] < static_cast<faiss::idx_t>(numCentroids),
                       "invalid centroid assignment");
        assignments[labels[i]].push_back(i);
    }
    return assignments;
}

static std::vector<DistanceMseRunConfig> create_distance_mse_run_configs(
        const std::vector<int> &nbitsValues,
        const std::vector<double> &factorValues,
        size_t baseNumVectors,
        size_t numQueries,
        size_t k,
        bool useIP,
        std::vector<NbitsRunGroup> &runGroups) {
    std::vector<DistanceMseRunConfig> runs;
    runGroups.clear();
    for (size_t i = 0; i < nbitsValues.size(); i++) {
        NbitsRunGroup group;
        group.nbits = static_cast<size_t>(nbitsValues[i]);
        for (size_t j = 0; j < factorValues.size(); j++) {
            CHECK_ARGUMENT(factorValues[j] > 0, "factor must be positive");
            DistanceMseRunConfig run;
            run.nbits = static_cast<size_t>(nbitsValues[i]);
            run.factor = factorValues[j];
            run.quantizedK = std::min(
                    baseNumVectors,
                    std::max(k, static_cast<size_t>(std::ceil(run.factor * static_cast<double>(k)))));
            run.perQuerySquaredError.assign(numQueries, 0.0);
            init_topk_heaps_for_queries(numQueries, run.quantizedK, useIP, run.quantizedTopK);
            group.runIndices.push_back(runs.size());
            runs.push_back(std::move(run));
        }
        runGroups.push_back(std::move(group));
    }
    return runs;
}

static void print_distance_mse_results(
        const char *label,
        size_t baseNumVectors,
        size_t numQueries,
        size_t k,
        bool useIP,
        const std::vector<TopKHeapState> &exactTopK,
        const std::vector<DistanceMseRunConfig> &runs) {
    for (const auto &run : runs) {
        double totalSquaredError = 0.0;
        double totalTopKMatch = 0.0;
        printf("%s config: nbits=%zu factor=%.3f quantized_k=%zu\n",
               label,
               run.nbits,
               run.factor,
               run.quantizedK);
        for (size_t q = 0; q < numQueries; q++) {
            std::vector<float> exactDistances = exactTopK[q].distances;
            std::vector<int64_t> exactLabels = exactTopK[q].labels;
            std::vector<float> quantizedDistances = run.quantizedTopK[q].distances;
            std::vector<int64_t> quantizedLabels = run.quantizedTopK[q].labels;
            const double topKMatch = compute_topk_match_percentage(
                    k,
                    run.quantizedK,
                    useIP,
                    exactDistances,
                    exactLabels,
                    quantizedDistances,
                    quantizedLabels);
            const double mse = run.perQuerySquaredError[q] / static_cast<double>(baseNumVectors);
            totalSquaredError += run.perQuerySquaredError[q];
            totalTopKMatch += topKMatch;
            printf("Query %zu MSE: %.10f, top-%zu in quantized top-%zu: %.2f%%\n",
                   q,
                   mse,
                   k,
                   run.quantizedK,
                   topKMatch);
        }
        const double overallMse = totalSquaredError / static_cast<double>(numQueries * baseNumVectors);
        printf("Overall %s MSE (nbits=%zu factor=%.3f): %.10f\n",
               label,
               run.nbits,
               run.factor,
               overallMse);
        printf("Average top-%zu in quantized top-%zu (nbits=%zu factor=%.3f): %.2f%%\n",
               k,
               run.quantizedK,
               run.nbits,
               run.factor,
               totalTopKMatch / static_cast<double>(numQueries));
    }
}

static void build_rabitq_chunk_configs(
        const float *rotatedChunkVecs,
        size_t chunkVectors,
        size_t dimension,
        faiss::MetricType metric,
        const std::vector<int> &nbitsValues,
        bool useCentroids,
        const std::vector<float> &centroids,
        faiss::IndexFlat *centroidIndex,
        std::vector<RaBitChunkConfig> &chunkConfigs) {
    chunkConfigs.clear();
    std::vector<std::vector<size_t>> centroidAssignments;
    if (useCentroids) {
        centroidAssignments = assign_vectors_to_centroids(
                rotatedChunkVecs,
                chunkVectors,
                dimension,
                *centroidIndex,
                centroids.size() / dimension);
    }

    for (size_t nbitsIdx = 0; nbitsIdx < nbitsValues.size(); nbitsIdx++) {
        RaBitChunkConfig config;
        config.nbits = static_cast<size_t>(nbitsValues[nbitsIdx]);
        if (!useCentroids) {
            config.clusters.emplace_back(dimension, metric, config.nbits);
            auto &cluster = config.clusters.back();
            cluster.indices.resize(chunkVectors);
            for (size_t i = 0; i < chunkVectors; i++) {
                cluster.indices[i] = i;
            }
            cluster.quantizer.train(chunkVectors, rotatedChunkVecs);
            cluster.codes.resize(chunkVectors * cluster.quantizer.code_size);
            cluster.quantizer.compute_codes(rotatedChunkVecs, cluster.codes.data(), chunkVectors);
        } else {
            const size_t numCentroids = centroids.size() / dimension;
            config.clusters.reserve(numCentroids);
            for (size_t centroidId = 0; centroidId < numCentroids; centroidId++) {
                config.clusters.emplace_back(dimension, metric, config.nbits);
                auto &cluster = config.clusters.back();
                cluster.centroid = centroids.data() + centroidId * dimension;
                cluster.indices = centroidAssignments[centroidId];
                if (cluster.indices.empty()) {
                    continue;
                }
                std::vector<float> clusterVecs = gather_indexed_vectors(
                        rotatedChunkVecs,
                        dimension,
                        cluster.indices);
                cluster.quantizer.train(cluster.indices.size(), clusterVecs.data());
                cluster.codes.resize(cluster.indices.size() * cluster.quantizer.code_size);
                cluster.quantizer.compute_codes_core(
                        clusterVecs.data(),
                        cluster.codes.data(),
                        cluster.indices.size(),
                        cluster.centroid);
            }
        }
        chunkConfigs.push_back(std::move(config));
    }
}

static void process_rabitq_chunk(
        const float *chunkVecs,
        size_t chunkVectors,
        size_t baseOffset,
        size_t dimension,
        const float *queryVecs,
        const float *rotatedQueryVecs,
        size_t numQueries,
        size_t k,
        bool useIP,
        faiss::MetricType metric,
        faiss::RandomRotationMatrix &rotationMatrix,
        const std::vector<int> &nbitsValues,
        const std::vector<NbitsRunGroup> &runGroups,
        bool useCentroids,
        const std::vector<float> &centroids,
        faiss::IndexFlat *centroidIndex,
        std::vector<TopKHeapState> &exactTopK,
        std::vector<DistanceMseRunConfig> &runs) {
    std::vector<float> rotatedChunkVecs(chunkVectors * dimension);
    rotationMatrix.apply_noalloc(chunkVectors, chunkVecs, rotatedChunkVecs.data());

    std::vector<RaBitChunkConfig> chunkConfigs;
    build_rabitq_chunk_configs(
            rotatedChunkVecs.data(),
            chunkVectors,
            dimension,
            metric,
            nbitsValues,
            useCentroids,
            centroids,
            centroidIndex,
            chunkConfigs);

#pragma omp parallel for
    for (int64_t q = 0; q < static_cast<int64_t>(numQueries); q++) {
        const float *queryVec = queryVecs + q * dimension;
        const float *rotatedQueryVec = rotatedQueryVecs + q * dimension;
        std::vector<float> exactChunkDistances(chunkVectors);

        for (size_t i = 0; i < chunkVectors; i++) {
            const float *baseVec = chunkVecs + i * dimension;
            const float exactDistance = useIP
                                        ? faiss::fvec_inner_product(queryVec, baseVec, dimension)
                                        : faiss::fvec_L2sqr(queryVec, baseVec, dimension);
            exactChunkDistances[i] = exactDistance;
            add_to_topk_heap(
                    k,
                    useIP,
                    exactTopK[q].distances,
                    exactTopK[q].labels,
                    exactDistance,
                    static_cast<int64_t>(baseOffset + i));
        }

        for (size_t configIdx = 0; configIdx < chunkConfigs.size(); configIdx++) {
            auto &config = chunkConfigs[configIdx];
            for (auto &cluster : config.clusters) {
                if (cluster.indices.empty()) {
                    continue;
                }
                std::unique_ptr<faiss::FlatCodesDistanceComputer> dc(
                        cluster.quantizer.get_distance_computer(0, cluster.centroid, false));
                dc->set_query(rotatedQueryVec);
                for (size_t i = 0; i < cluster.indices.size(); i++) {
                    const size_t localIdx = cluster.indices[i];
                    const float exactDistance = exactChunkDistances[localIdx];
                    const float quantizedDistance =
                            dc->distance_to_code(cluster.codes.data() + i * cluster.quantizer.code_size);
                    const double diff = static_cast<double>(exactDistance) - static_cast<double>(quantizedDistance);
                    for (size_t runIdx : runGroups[configIdx].runIndices) {
                        runs[runIdx].perQuerySquaredError[q] += diff * diff;
                        add_to_topk_heap(
                                runs[runIdx].quantizedK,
                                useIP,
                                runs[runIdx].quantizedTopK[q].distances,
                                runs[runIdx].quantizedTopK[q].labels,
                                quantizedDistance,
                                static_cast<int64_t>(baseOffset + localIdx));
                    }
                }
            }
        }
    }
}

static std::vector<ScalarClusterCodes> train_scalar_quantizers(
        const float *trainVecs,
        size_t trainVectors,
        size_t dimension,
        const std::vector<int> &nbitsValues) {
    std::vector<ScalarClusterCodes> quantizers;
    quantizers.reserve(nbitsValues.size());
    for (int nbits : nbitsValues) {
        quantizers.emplace_back(dimension, static_cast<size_t>(nbits));
        auto &sq = quantizers.back();
        sq.quantizer.train(trainVectors, trainVecs);
    }
    return quantizers;
}

static void process_scalar_chunk(
        const float *chunkVecs,
        size_t chunkVectors,
        size_t baseOffset,
        size_t dimension,
        const float *queryVecs,
        size_t numQueries,
        size_t k,
        bool useIP,
        faiss::MetricType metric,
        const std::vector<ScalarClusterCodes> &trainedQuantizers,
        const std::vector<NbitsRunGroup> &runGroups,
        std::vector<TopKHeapState> &exactTopK,
        std::vector<DistanceMseRunConfig> &runs) {
    std::vector<ScalarClusterCodes> chunkConfigs;
    chunkConfigs.reserve(trainedQuantizers.size());
    for (const auto &trainedSq : trainedQuantizers) {
        chunkConfigs.emplace_back(dimension, trainedSq.nbits);
        auto &chunkSq = chunkConfigs.back();
        chunkSq.quantizer = trainedSq.quantizer;
        chunkSq.codes.resize(chunkVectors * chunkSq.quantizer.code_size);
        chunkSq.quantizer.compute_codes(chunkVecs, chunkSq.codes.data(), chunkVectors);
    }

#pragma omp parallel for
    for (int64_t q = 0; q < static_cast<int64_t>(numQueries); q++) {
        const float *queryVec = queryVecs + q * dimension;
        std::vector<float> exactChunkDistances(chunkVectors);

        for (size_t i = 0; i < chunkVectors; i++) {
            const float *baseVec = chunkVecs + i * dimension;
            const float exactDistance = useIP
                                        ? faiss::fvec_inner_product(queryVec, baseVec, dimension)
                                        : faiss::fvec_L2sqr(queryVec, baseVec, dimension);
            exactChunkDistances[i] = exactDistance;
            add_to_topk_heap(
                    k,
                    useIP,
                    exactTopK[q].distances,
                    exactTopK[q].labels,
                    exactDistance,
                    static_cast<int64_t>(baseOffset + i));
        }

        for (size_t configIdx = 0; configIdx < chunkConfigs.size(); configIdx++) {
            auto &config = chunkConfigs[configIdx];
            std::unique_ptr<faiss::ScalarQuantizer::SQDistanceComputer> dc(
                    config.quantizer.get_distance_computer(metric));
            dc->set_query(queryVec);
            for (size_t i = 0; i < chunkVectors; i++) {
                const float exactDistance = exactChunkDistances[i];
                const float quantizedDistance =
                        dc->distance_to_code(config.codes.data() + i * config.quantizer.code_size);
                const double diff = static_cast<double>(exactDistance) - static_cast<double>(quantizedDistance);
                for (size_t runIdx : runGroups[configIdx].runIndices) {
                    runs[runIdx].perQuerySquaredError[q] += diff * diff;
                    add_to_topk_heap(
                            runs[runIdx].quantizedK,
                            useIP,
                            runs[runIdx].quantizedTopK[q].distances,
                            runs[runIdx].quantizedTopK[q].labels,
                            quantizedDistance,
                            static_cast<int64_t>(baseOffset + i));
                }
            }
        }
    }
}

void compute_quantized_ip_mse(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    const std::string &baseQuantizedVectorPath = input.getCmdOption("-baseQuantizedVectorPath");
    const std::string &queryQuantizedVectorPath = input.getCmdOption("-queryQuantizedVectorPath");
    const std::string &numQueriesArg = input.getCmdOption("-numQueries");
    const std::string &kArg = input.getCmdOption("-k");
    const std::string &factorArg = input.getCmdOption("-factor");
    const std::string &useIPArg = input.getCmdOption("-useIP");

    CHECK_ARGUMENT(!baseVectorPath.empty(), "base vector path is required");
    CHECK_ARGUMENT(!queryVectorPath.empty(), "query vector path is required");
    CHECK_ARGUMENT(!baseQuantizedVectorPath.empty(), "base quantized vector path is required");
    CHECK_ARGUMENT(!queryQuantizedVectorPath.empty(), "query quantized vector path is required");

    const size_t numQueries = numQueriesArg.empty() ? 10 : stoull(numQueriesArg);
    const size_t k = kArg.empty() ? 10 : stoull(kArg);
    const double factor = factorArg.empty() ? 1.0 : stod(factorArg);
    const bool useIP = useIPArg.empty() ? true : (stoi(useIPArg) != 0);
    CHECK_ARGUMENT(numQueries > 0, "numQueries must be positive");
    CHECK_ARGUMENT(k > 0, "k must be positive");
    CHECK_ARGUMENT(factor > 0, "factor must be positive");
    size_t baseNumVectors, baseDimension;
    float *baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors);
    size_t queryNumVectors, queryDimension;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    size_t baseQuantizedNumVectors, baseQuantizedDimension;
    float *baseQuantizedVecs = readVecFile(baseQuantizedVectorPath.c_str(), &baseQuantizedDimension,
                                           &baseQuantizedNumVectors);
    size_t queryQuantizedNumVectors, queryQuantizedDimension;
    float *queryQuantizedVecs = readVecFile(queryQuantizedVectorPath.c_str(), &queryQuantizedDimension,
                                            &queryQuantizedNumVectors);

    CHECK_ARGUMENT(baseNumVectors == baseQuantizedNumVectors, "base vector counts do not match");
    CHECK_ARGUMENT(queryNumVectors == queryQuantizedNumVectors, "query vector counts do not match");
    CHECK_ARGUMENT(baseDimension == queryDimension, "base and query dimensions do not match");
    CHECK_ARGUMENT(baseQuantizedDimension == queryQuantizedDimension,
                   "quantized base and query dimensions do not match");
    CHECK_ARGUMENT(baseDimension == baseQuantizedDimension,
                   "original and quantized vector dimensions do not match");
    CHECK_ARGUMENT(queryNumVectors >= numQueries, "not enough queries in query vectors");
    CHECK_ARGUMENT(k <= baseNumVectors, "k must be <= number of base vectors");
    const size_t quantizedK = std::min(
            baseNumVectors,
            std::max(k, static_cast<size_t>(std::ceil(factor * static_cast<double>(k)))));
    std::vector<double> perQuerySquaredError(numQueries, 0.0);
    std::vector<double> perQueryTopKMatch(numQueries, 0.0);

#pragma omp parallel for
    for (int64_t q = 0; q < static_cast<int64_t>(numQueries); q++) {
        const float *queryVec = queryVecs + q * queryDimension;
        const float *queryQuantizedVec = queryQuantizedVecs + q * queryQuantizedDimension;
        double querySquaredError = 0.0;
        std::vector<float> exactTopKDistances;
        std::vector<int64_t> exactTopKLabels;
        std::vector<float> quantizedTopKDistances;
        std::vector<int64_t> quantizedTopKLabels;
        init_topk_heap(k, useIP, exactTopKDistances, exactTopKLabels);
        init_topk_heap(quantizedK, useIP, quantizedTopKDistances, quantizedTopKLabels);

        for (size_t i = 0; i < baseNumVectors; i++) {
            const float *baseVec = baseVecs + i * baseDimension;
            const float *baseQuantizedVec = baseQuantizedVecs + i * baseQuantizedDimension;
            double distance = useIP
                              ? faiss::fvec_inner_product(queryVec, baseVec, baseDimension)
                              : faiss::fvec_L2sqr(queryVec, baseVec, baseDimension);
            double quantizedDistance = useIP
                                       ? faiss::fvec_inner_product(queryQuantizedVec, baseQuantizedVec,
                                                                   baseQuantizedDimension)
                                       : faiss::fvec_L2sqr(queryQuantizedVec, baseQuantizedVec,
                                                           baseQuantizedDimension);
            double diff = distance - quantizedDistance;
            querySquaredError += diff * diff;
            add_to_topk_heap(k, useIP, exactTopKDistances, exactTopKLabels, distance, i);
            add_to_topk_heap(
                    quantizedK,
                    useIP,
                    quantizedTopKDistances,
                    quantizedTopKLabels,
                    quantizedDistance,
                    i);
        }
        perQuerySquaredError[q] = querySquaredError;
        perQueryTopKMatch[q] = compute_topk_match_percentage(
                k,
                quantizedK,
                useIP,
                exactTopKDistances,
                exactTopKLabels,
                quantizedTopKDistances,
                quantizedTopKLabels);
    }

    free(baseVecs);
    free(baseQuantizedVecs);
    free(queryVecs);
    free(queryQuantizedVecs);

    double totalSquaredError = 0.0;
    double totalTopKMatch = 0.0;
    printf("Computed %s MSE for %zu queries across %zu base vectors (k=%zu, factor=%.3f, quantized_k=%zu)\n",
           useIP ? "IP" : "L2", numQueries, baseNumVectors, k, factor, quantizedK);
    for (size_t q = 0; q < numQueries; q++) {
        double mse = perQuerySquaredError[q] / static_cast<double>(baseNumVectors);
        totalSquaredError += perQuerySquaredError[q];
        totalTopKMatch += perQueryTopKMatch[q];
        printf("Query %zu MSE: %.10f, top-%zu in quantized top-%zu: %.2f%%\n",
               q, mse, k, quantizedK, perQueryTopKMatch[q]);
    }

    double overallMse = totalSquaredError / static_cast<double>(numQueries * baseNumVectors);
    printf("Overall MSE: %.10f\n", overallMse);
    printf("Average top-%zu in quantized top-%zu: %.2f%%\n",
           k, quantizedK, totalTopKMatch / static_cast<double>(numQueries));
}

void compute_rabitq_rotated_distance_mse(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    const std::string &nbitsArg = input.getCmdOption("-nbits");
    const std::string &useCentroidsArg = input.getCmdOption("-useCentroids");
    const std::string &numQueriesArg = input.getCmdOption("-numQueries");
    const std::string &kArg = input.getCmdOption("-k");
    const std::string &factorArg = input.getCmdOption("-factor");
    const std::string &useIPArg = input.getCmdOption("-useIP");
    const std::string &rotationSeedArg = input.getCmdOption("-rotationSeed");
    const std::string &kmeansNiterArg = input.getCmdOption("-kmeansNiter");
    const std::string &numCentroidsArg = input.getCmdOption("-numCentroids");
    const std::string &isParquetArg = input.getCmdOption("-isParquet");
    const std::string &nFilesArg = input.getCmdOption("-nFiles");
    const std::string &numVectorsArg = input.getCmdOption("-numVectors");
    const std::string &trainSampleSizeArg = input.getCmdOption("-trainSampleSize");
    const std::string &trainSampleFilesArg = input.getCmdOption("-trainSampleFiles");
    const std::string parquetColumnName = input.getCmdOption("-parquetColumnName").empty()
                                                  ? "emb"
                                                  : input.getCmdOption("-parquetColumnName");

    CHECK_ARGUMENT(!baseVectorPath.empty(), "base vector path is required");
    CHECK_ARGUMENT(!queryVectorPath.empty(), "query vector path is required");
    CHECK_ARGUMENT(!nbitsArg.empty(), "nbits is required");
    CHECK_ARGUMENT(!useCentroidsArg.empty(), "useCentroids is required");

    const std::vector<int> nbitsValues = parseCommaSeparatedIntegers(nbitsArg);
    CHECK_ARGUMENT(!nbitsValues.empty(), "nbits must not be empty");
    for (int nbits : nbitsValues) {
        CHECK_ARGUMENT(nbits >= 1 && nbits <= 9, "nbits must be between 1 and 9");
    }
    const bool useCentroids = stoi(useCentroidsArg) != 0;
    const bool useIP = !useIPArg.empty() && stoi(useIPArg) != 0;
    const std::vector<double> factorValues = factorArg.empty()
                                             ? std::vector<double>{1.0}
                                             : parseCommaSeparatedDoubles(factorArg);
    const int rotationSeed = rotationSeedArg.empty() ? 123 : stoi(rotationSeedArg);
    const int kmeansNiter = kmeansNiterArg.empty() ? 25 : stoi(kmeansNiterArg);
    const size_t numCentroids = numCentroidsArg.empty() ? 0 : stoull(numCentroidsArg);
    const bool isParquet = !isParquetArg.empty() && stoi(isParquetArg) != 0;
    const int requestedFiles = nFilesArg.empty() ? INT_MAX : stoi(nFilesArg);
    const size_t requestedBaseVectors = get_requested_vector_limit(numVectorsArg);
    const faiss::MetricType metric = useIP ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
    if (useCentroids) {
        CHECK_ARGUMENT(numCentroids > 0, "numCentroids is required when useCentroids=1");
    }

    size_t queryNumVectors, queryDimension;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);

    size_t baseNumVectors = 0;
    size_t baseDimension = 0;
    float *baseVecs = nullptr;
    std::vector<std::string> parquetFilePaths;
    if (isParquet) {
        collect_parquet_input_files(
                baseVectorPath,
                requestedFiles,
                requestedBaseVectors,
                parquetColumnName,
                parquetFilePaths,
                &baseDimension,
                &baseNumVectors);
    } else {
        baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors, requestedBaseVectors);
    }

    CHECK_ARGUMENT(baseDimension == queryDimension, "base and query dimensions do not match");
    CHECK_ARGUMENT(baseDimension > 0, "vector dimension must be positive");
    CHECK_ARGUMENT(baseDimension <= static_cast<size_t>(INT_MAX), "vector dimension is too large");

    const size_t requestedQueries = numQueriesArg.empty() ? queryNumVectors : stoull(numQueriesArg);
    const size_t k = kArg.empty() ? 10 : stoull(kArg);
    CHECK_ARGUMENT(requestedQueries > 0, "numQueries must be positive");
    CHECK_ARGUMENT(k > 0, "k must be positive");
    CHECK_ARGUMENT(requestedQueries <= queryNumVectors, "not enough query vectors");
    const size_t numQueries = requestedQueries;
    CHECK_ARGUMENT(k <= baseNumVectors, "k must be <= number of base vectors");

    printf("Loaded %zu base vectors and %zu query vectors of dimension %zu\n",
           baseNumVectors, numQueries, baseDimension);
    printf("RaBitQ config: metric=%s useCentroids=%d rotationSeed=%d isParquet=%d\n",
           useIP ? "ip" : "l2", useCentroids ? 1 : 0, rotationSeed, isParquet ? 1 : 0);

    faiss::RandomRotationMatrix rrot(static_cast<int>(baseDimension), static_cast<int>(baseDimension));
    rrot.init(rotationSeed);

    std::vector<float> rotatedQueryVecs(numQueries * baseDimension);
    printf("Applying random rotation to query vectors\n");
    rrot.apply_noalloc(numQueries, queryVecs, rotatedQueryVecs.data());

    std::vector<float> centroids;
    std::unique_ptr<faiss::IndexFlat> centroidIndex;
    if (useCentroids) {
        if (isParquet) {
            size_t sampleDimension = 0;
            size_t sampleVectors = 0;
            const size_t sampleFiles = resolve_train_sample_files(
                    trainSampleFilesArg,
                    3,
                    parquetFilePaths.size());
            printf("Training centroids from %zu parquet files\n", sampleFiles);
            float *sampleVecs = read_parquet_sample_files(
                    parquetFilePaths,
                    sampleFiles,
                    &sampleDimension,
                    &sampleVectors,
                    parquetColumnName);
            CHECK_ARGUMENT(sampleDimension == baseDimension, "sample dimension mismatch");
            const size_t trainVectors = resolve_train_sample_size(
                    trainSampleSizeArg,
                    sampleVectors,
                    sampleVectors);
            std::vector<float> rotatedSampleVecs(trainVectors * baseDimension);
            rrot.apply_noalloc(trainVectors, sampleVecs, rotatedSampleVecs.data());
            centroids = train_kmeans_centroids(
                    rotatedSampleVecs.data(),
                    trainVectors,
                    baseDimension,
                    numCentroids,
                    kmeansNiter,
                    useIP);
            free(sampleVecs);
        } else {
            const size_t trainVectors = resolve_train_sample_size(
                    trainSampleSizeArg,
                    get_clustering_sample_size(baseNumVectors, numCentroids),
                    baseNumVectors);
            printf("Training centroids from %zu sampled vectors\n", trainVectors);
            std::vector<float> rotatedSampleVecs(trainVectors * baseDimension);
            rrot.apply_noalloc(trainVectors, baseVecs, rotatedSampleVecs.data());
            centroids = train_kmeans_centroids(
                    rotatedSampleVecs.data(),
                    trainVectors,
                    baseDimension,
                    numCentroids,
                    kmeansNiter,
                    useIP);
        }
        centroidIndex = std::make_unique<faiss::IndexFlat>(static_cast<faiss::idx_t>(baseDimension), metric);
        centroidIndex->add(numCentroids, centroids.data());
    }

    std::vector<TopKHeapState> exactTopK;
    init_topk_heaps_for_queries(numQueries, k, useIP, exactTopK);
    std::vector<NbitsRunGroup> runGroups;
    std::vector<DistanceMseRunConfig> runs = create_distance_mse_run_configs(
            nbitsValues,
            factorValues,
            baseNumVectors,
            numQueries,
            k,
            useIP,
            runGroups);
    printf("Computing MSE over %s distances\n", useIP ? "inner product" : "L2");
    if (isParquet) {
        size_t processedVectors = 0;
        for (size_t fileIdx = 0; fileIdx < parquetFilePaths.size() && processedVectors < baseNumVectors; fileIdx++) {
            size_t fileDimension = 0;
            size_t fileVectors = 0;
            std::vector<std::string> singleFilePath = {parquetFilePaths[fileIdx]};
            float *fileVecs = readParquetFiles(
                    singleFilePath,
                    &fileDimension,
                    &fileVectors,
                    parquetColumnName);
            CHECK_ARGUMENT(fileDimension == baseDimension, "parquet file dimension mismatch");
            const size_t vectorsToProcess = std::min(fileVectors, baseNumVectors - processedVectors);
            printf("Processing parquet file %zu/%zu with %zu vectors\n",
                   fileIdx + 1,
                   parquetFilePaths.size(),
                   vectorsToProcess);
            process_rabitq_chunk(
                    fileVecs,
                    vectorsToProcess,
                    processedVectors,
                    baseDimension,
                    queryVecs,
                    rotatedQueryVecs.data(),
                    numQueries,
                    k,
                    useIP,
                    metric,
                    rrot,
                    nbitsValues,
                    runGroups,
                    useCentroids,
                    centroids,
                    centroidIndex.get(),
                    exactTopK,
                    runs);
            processedVectors += vectorsToProcess;
            free(fileVecs);
        }
    } else {
        process_rabitq_chunk(
                baseVecs,
                baseNumVectors,
                0,
                baseDimension,
                queryVecs,
                rotatedQueryVecs.data(),
                numQueries,
                k,
                useIP,
                metric,
                rrot,
                nbitsValues,
                runGroups,
                useCentroids,
                centroids,
                centroidIndex.get(),
                exactTopK,
                runs);
    }

    print_distance_mse_results(
            "RaBitQ rotated distance",
            baseNumVectors,
            numQueries,
            k,
            useIP,
            exactTopK,
            runs);

    free(baseVecs);
    free(queryVecs);
}

void compute_scalar_quantizer_distance_mse(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    const std::string &nbitsArg = input.getCmdOption("-nbits");
    const std::string &numQueriesArg = input.getCmdOption("-numQueries");
    const std::string &kArg = input.getCmdOption("-k");
    const std::string &factorArg = input.getCmdOption("-factor");
    const std::string &useIPArg = input.getCmdOption("-useIP");
    const std::string &isParquetArg = input.getCmdOption("-isParquet");
    const std::string &nFilesArg = input.getCmdOption("-nFiles");
    const std::string &numVectorsArg = input.getCmdOption("-numVectors");
    const std::string &trainSampleSizeArg = input.getCmdOption("-trainSampleSize");
    const std::string &trainSampleFilesArg = input.getCmdOption("-trainSampleFiles");
    const std::string parquetColumnName = input.getCmdOption("-parquetColumnName").empty()
                                                  ? "emb"
                                                  : input.getCmdOption("-parquetColumnName");

    CHECK_ARGUMENT(!baseVectorPath.empty(), "base vector path is required");
    CHECK_ARGUMENT(!queryVectorPath.empty(), "query vector path is required");
    CHECK_ARGUMENT(!nbitsArg.empty(), "nbits is required");

    const std::vector<int> nbitsValues = parseCommaSeparatedIntegers(nbitsArg);
    CHECK_ARGUMENT(!nbitsValues.empty(), "nbits must not be empty");
    for (int nbits : nbitsValues) {
        CHECK_ARGUMENT(nbits == 4 || nbits == 8, "nbits must be 4 or 8");
    }
    const bool useIP = !useIPArg.empty() && stoi(useIPArg) != 0;
    const std::vector<double> factorValues = factorArg.empty()
                                             ? std::vector<double>{1.0}
                                             : parseCommaSeparatedDoubles(factorArg);
    const bool isParquet = !isParquetArg.empty() && stoi(isParquetArg) != 0;
    const int requestedFiles = nFilesArg.empty() ? INT_MAX : stoi(nFilesArg);
    const size_t requestedBaseVectors = get_requested_vector_limit(numVectorsArg);
    const faiss::MetricType metric = useIP ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

    size_t queryNumVectors, queryDimension;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);

    size_t baseNumVectors = 0;
    size_t baseDimension = 0;
    float *baseVecs = nullptr;
    std::vector<std::string> parquetFilePaths;
    if (isParquet) {
        collect_parquet_input_files(
                baseVectorPath,
                requestedFiles,
                requestedBaseVectors,
                parquetColumnName,
                parquetFilePaths,
                &baseDimension,
                &baseNumVectors);
    } else {
        baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors, requestedBaseVectors);
    }

    CHECK_ARGUMENT(baseDimension == queryDimension, "base and query dimensions do not match");

    const size_t requestedQueries = numQueriesArg.empty() ? queryNumVectors : stoull(numQueriesArg);
    const size_t k = kArg.empty() ? 10 : stoull(kArg);
    CHECK_ARGUMENT(requestedQueries > 0, "numQueries must be positive");
    CHECK_ARGUMENT(k > 0, "k must be positive");
    CHECK_ARGUMENT(requestedQueries <= queryNumVectors, "not enough query vectors");
    const size_t numQueries = requestedQueries;
    CHECK_ARGUMENT(k <= baseNumVectors, "k must be <= number of base vectors");

    printf("Loaded %zu base vectors and %zu query vectors of dimension %zu\n",
           baseNumVectors, numQueries, baseDimension);
    printf("Scalar quantizer config: metric=%s isParquet=%d\n",
           useIP ? "ip" : "l2", isParquet ? 1 : 0);

    std::vector<ScalarClusterCodes> trainedQuantizers;
    if (isParquet) {
        size_t sampleDimension = 0;
        size_t sampleVectors = 0;
        const size_t sampleFiles = resolve_train_sample_files(
                trainSampleFilesArg,
                3,
                parquetFilePaths.size());
        printf("Training scalar quantizers from %zu parquet files\n", sampleFiles);
        float *sampleVecs = read_parquet_sample_files(
                parquetFilePaths,
                sampleFiles,
                &sampleDimension,
                &sampleVectors,
                parquetColumnName);
        CHECK_ARGUMENT(sampleDimension == baseDimension, "sample dimension mismatch");
        const size_t trainVectors = resolve_train_sample_size(
                trainSampleSizeArg,
                get_scalar_training_sample_size(baseNumVectors),
                sampleVectors);
        trainedQuantizers = train_scalar_quantizers(
                sampleVecs,
                trainVectors,
                baseDimension,
                nbitsValues);
        free(sampleVecs);
    } else {
        const size_t trainVectors = resolve_train_sample_size(
                trainSampleSizeArg,
                get_scalar_training_sample_size(baseNumVectors),
                baseNumVectors);
        printf("Training scalar quantizers from %zu sampled vectors\n", trainVectors);
        trainedQuantizers = train_scalar_quantizers(
                baseVecs,
                baseNumVectors,
                baseDimension,
                nbitsValues);
    }

    std::vector<TopKHeapState> exactTopK;
    init_topk_heaps_for_queries(numQueries, k, useIP, exactTopK);
    std::vector<NbitsRunGroup> runGroups;
    std::vector<DistanceMseRunConfig> runs = create_distance_mse_run_configs(
            nbitsValues,
            factorValues,
            baseNumVectors,
            numQueries,
            k,
            useIP,
            runGroups);

    if (isParquet) {
        size_t processedVectors = 0;
        for (size_t fileIdx = 0; fileIdx < parquetFilePaths.size() && processedVectors < baseNumVectors; fileIdx++) {
            size_t fileDimension = 0;
            size_t fileVectors = 0;
            std::vector<std::string> singleFilePath = {parquetFilePaths[fileIdx]};
            float *fileVecs = readParquetFiles(
                    singleFilePath,
                    &fileDimension,
                    &fileVectors,
                    parquetColumnName);
            CHECK_ARGUMENT(fileDimension == baseDimension, "parquet file dimension mismatch");
            const size_t vectorsToProcess = std::min(fileVectors, baseNumVectors - processedVectors);
            printf("Processing parquet file %zu/%zu with %zu vectors\n",
                   fileIdx + 1,
                   parquetFilePaths.size(),
                   vectorsToProcess);
            process_scalar_chunk(
                    fileVecs,
                    vectorsToProcess,
                    processedVectors,
                    baseDimension,
                    queryVecs,
                    numQueries,
                    k,
                    useIP,
                    metric,
                    trainedQuantizers,
                    runGroups,
                    exactTopK,
                    runs);
            processedVectors += vectorsToProcess;
            free(fileVecs);
        }
    } else {
        process_scalar_chunk(
                baseVecs,
                baseNumVectors,
                0,
                baseDimension,
                queryVecs,
                numQueries,
                k,
                useIP,
                metric,
                trainedQuantizers,
                runGroups,
                exactTopK,
                runs);
    }

    print_distance_mse_results(
            "scalar quantizer distance",
            baseNumVectors,
            numQueries,
            k,
            useIP,
            exactTopK,
            runs);

    free(baseVecs);
    free(queryVecs);
}

/**
 * Test scalar quantization quality using parquet files.
 *
 * Reads a train parquet file to train a faiss::ScalarQuantizer, then reads
 * multiple data parquet files and query vectors. Samples random points from
 * the data and compares actual L2 distances vs quantized-code distances from
 * query vectors, mirroring what test_quantization_issue does but with parquet
 * data sources.
 *
 * Args (via InputParser):
 *   -trainParquetPath : directory containing train parquet file(s)
 *   -dataParquetPath  : directory containing data parquet files
 *   -queryVectorPath  : path to query vectors (.fvecs / .bvecs)
 *   -nFiles           : max number of data parquet files to read
 *   -sampleSize       : number of random points to sample for comparison
 *   -queryIndex       : which query vector to use
 */
void test_quantization_parquet(InputParser &input) {
    const std::string &trainParquetPath = input.getCmdOption("-trainParquetPath");
    const std::string &dataParquetPath = input.getCmdOption("-dataParquetPath");
    const std::string &queryVectorPath = input.getCmdOption("-queryVectorPath");
    const int nFiles = stoi(input.getCmdOption("-nFiles"));
    const int sampleSize = stoi(input.getCmdOption("-sampleSize"));
    const int queryIndex = stoi(input.getCmdOption("-queryIndex"));

    // ---- Step 1: Read training parquet file(s) and train the scalar quantizer ----
    size_t trainDim, trainNumVectors;
    float *trainVecs = readParquetDir(trainParquetPath.c_str(), &trainDim, &trainNumVectors);
    printf("Loaded %zu training vectors of dimension %zu\n", trainNumVectors, trainDim);

    faiss::ScalarQuantizer sq(trainDim, faiss::ScalarQuantizer::QT_8bit);
    printf("Training scalar quantizer on %zu vectors of dimension %zu\n", trainNumVectors, trainDim);
    sq.train(trainNumVectors, trainVecs);
    delete[] trainVecs;

    // ---- Step 2: Read data parquet files ----
    std::vector<std::string> filePaths;
    list_parquet_dir(dataParquetPath.c_str(), filePaths);
    if (filePaths.empty()) {
        fprintf(stderr, "No parquet files found in: %s\n", dataParquetPath.c_str());
        exit(1);
    }
    int numFiles = std::min(nFiles, (int)filePaths.size());
    std::vector<std::string> selectedPaths(filePaths.begin(), filePaths.begin() + numFiles);

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readParquetFiles(selectedPaths, &baseDimension, &baseNumVectors);
    printf("Loaded %zu base vectors of dimension %zu from %d parquet files\n",
           baseNumVectors, baseDimension, numFiles);
    assert(baseDimension == trainDim);

    // ---- Step 3: Compute SQ codes for all base vectors ----
    std::vector<uint8_t> codes(baseNumVectors * sq.code_size);
    printf("Computing codes for %zu base vectors\n", baseNumVectors);
    sq.compute_codes(baseVecs, codes.data(), baseNumVectors);

    // ---- Step 4: Load query vectors ----
    size_t queryDimension, queryNumVectors;
    float *queryVecs = readVecFile(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    assert(queryDimension == baseDimension);
    assert((size_t)queryIndex < queryNumVectors);
    auto queryVec = queryVecs + queryIndex * queryDimension;

    // ---- Step 5: Sample random points and compare distances ----
    RandomGenerator rg(1234);
    std::vector<uint64_t> sampleIndices(sampleSize);
    rg.randomPerm(baseNumVectors, sampleIndices.data(), sampleSize);

    std::vector<float> actualDistances(sampleSize);
    std::vector<float> codesDistances(sampleSize);
    std::vector<float> itsOwnDistances(sampleSize);
    auto dc = sq.get_distance_computer();

    double distance_diff = 0;
    double avg_its_own_diff = 0;
    double avg_distance_from_query = 0;
    for (int i = 0; i < sampleSize; i++) {
        auto idx = sampleIndices[i];
        // Actual L2 distance from query to base vector
        actualDistances[i] = faiss::fvec_L2sqr(queryVec,
                                                baseVecs + idx * baseDimension,
                                                baseDimension);

        // Quantized distance from query to code
        dc->set_query(queryVec);
        codesDistances[i] = dc->distance_to_code(codes.data() + idx * sq.code_size);
        distance_diff += std::abs(actualDistances[i] - codesDistances[i]);

        // Self-distance: base vector vs its own code (measures quantization error)
        dc->set_query(baseVecs + idx * baseDimension);
        itsOwnDistances[i] = dc->distance_to_code(codes.data() + idx * sq.code_size);
        avg_its_own_diff += itsOwnDistances[i];
        avg_distance_from_query += actualDistances[i];
    }

    printf("Average distance difference from query over %d (avg dist %f) samples: %f\n",
           sampleSize, avg_distance_from_query / sampleSize, distance_diff / sampleSize);
    printf("Average self-quantization error over %d samples: %f\n",
           sampleSize, avg_its_own_diff / sampleSize);

    // Write results to files for external analysis
    writeToFile("./actual_distances_parquet.bin",
                reinterpret_cast<uint8_t *>(actualDistances.data()),
                actualDistances.size() * sizeof(float));
    writeToFile("./codes_distances_parquet.bin",
                reinterpret_cast<uint8_t *>(codesDistances.data()),
                codesDistances.size() * sizeof(float));
    writeToFile("./self_distances_parquet.bin",
                reinterpret_cast<uint8_t *>(itsOwnDistances.data()),
                itsOwnDistances.size() * sizeof(float));

    free(baseVecs);
    delete[] queryVecs;
}

void print_quantization_parquet_data(InputParser &input) {
    const std::string &trainParquetPath = input.getCmdOption("-trainParquetPath");
    const std::string &dataPath = input.getCmdOption("-dataPath");
    const std::string &quantizedDataPath = input.getCmdOption("-quantizedDataPath");
    const int trainSize = stoi(input.getCmdOption("-trainSize"));

    // ---- Step 1: Read training parquet file(s) and train the scalar quantizer ----
    size_t trainDim, trainNumVectors;
    float *trainVecs = readParquetDir(trainParquetPath.c_str(), &trainDim, &trainNumVectors);
    printf("Loaded %zu training vectors of dimension %zu\n", trainNumVectors, trainDim);

    size_t actualTrainSize = std::min((size_t)trainSize, trainNumVectors);
    faiss::ScalarQuantizer sq(trainDim, faiss::ScalarQuantizer::QT_8bit);
    printf("Training scalar quantizer on %zu vectors of dimension %zu\n", actualTrainSize, trainDim);
    sq.train(actualTrainSize, trainVecs);
    delete[] trainVecs;

    // Print trained SQ parameters: min, diff (vmax-vmin), max per dimension
    // sq.trained layout for QT_8bit: [vmin_0..vmin_{d-1}, vdiff_0..vdiff_{d-1}]
    const float *vmin = sq.trained.data();
    const float *vdiff = sq.trained.data() + trainDim;
    printf("min:  [");
    for (size_t i = 0; i < trainDim; i++) {
        printf("%s%.6f", i > 0 ? ", " : "", vmin[i]);
    }
    printf("]\n");
    printf("diff: [");
    for (size_t i = 0; i < trainDim; i++) {
        printf("%s%.6f", i > 0 ? ", " : "", vdiff[i]);
    }
    printf("]\n");
    printf("max:  [");
    for (size_t i = 0; i < trainDim; i++) {
        printf("%s%.6f", i > 0 ? ", " : "", vmin[i] + vdiff[i]);
    }
    printf("]\n");

    // ---- Step 2: Read data fvec file ----
    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(dataPath.c_str(), &baseDimension, &baseNumVectors);
    printf("Loaded %zu base vectors of dimension %zu from %s\n",
           baseNumVectors, baseDimension, dataPath.c_str());
    assert(baseDimension == trainDim);

    // ---- Step 3: Compute SQ codes for all base vectors ----
    std::vector<uint8_t> ourCodes(baseNumVectors * sq.code_size);
    printf("Computing codes for %zu base vectors\n", baseNumVectors);
    sq.compute_codes(baseVecs, ourCodes.data(), baseNumVectors);

    // ---- Step 4: Load pre-quantized data ----
    size_t refDim, refNumVectors;
    float *refVecs = readVecFile(quantizedDataPath.c_str(), &refDim, &refNumVectors);
    printf("Loaded %zu reference quantized vectors of dimension %zu from %s\n",
           refNumVectors, refDim, quantizedDataPath.c_str());
    assert(refDim == baseDimension);

    // ---- Step 5: Compare and print side by side ----
    size_t compareCount = std::min(baseNumVectors, refNumVectors);

    size_t totalAllMismatches = 0;
    size_t totalInterestingDims = 0;
    size_t totalInterestingMismatches = 0;
    size_t vectorsWithMismatch = 0;

    for (size_t vecIdx = 0; vecIdx < compareCount; vecIdx++) {
        bool hasMismatchInInterestingDim = false;
        int vecMismatches = 0;
        for (size_t d = 0; d < baseDimension; d++) {
            uint8_t ours = ourCodes[vecIdx * sq.code_size + d];
            uint8_t ref = (uint8_t)refVecs[vecIdx * refDim + d];
            if (ours != ref) {
                vecMismatches++;
                float val = baseVecs[vecIdx * baseDimension + d];
                if (val > 1.0f || val < -1.0f) {
                    hasMismatchInInterestingDim = true;
                }
            }
        }
        totalAllMismatches += vecMismatches;
        if (vecMismatches > 0) vectorsWithMismatch++;

        if (hasMismatchInInterestingDim) {
            printf("\n=== Vector %zu (%d total mismatches) ===\n", vecIdx, vecMismatches);
            printf("%-6s %-14s %-10s %-10s %-5s\n", "Dim", "ActualVal", "OurCode", "RefCode", "Match");
            for (size_t d = 0; d < baseDimension; d++) {
                float val = baseVecs[vecIdx * baseDimension + d];
                if (val > 1.0f || val < -1.0f) {
                    uint8_t ours = ourCodes[vecIdx * sq.code_size + d];
                    uint8_t ref = (uint8_t)refVecs[vecIdx * refDim + d];
                    totalInterestingDims++;
                    if (ours != ref) totalInterestingMismatches++;
                    printf("%-6zu %-14.6f %-10u %-10u %s\n",
                           d, val, ours, ref,
                           ours == ref ? "YES" : "NO");
                }
            }
        }
    }

    printf("\n=== Summary ===\n");
    printf("Compared %zu vectors of dimension %zu\n", compareCount, baseDimension);
    printf("Vectors with at least one mismatch: %zu / %zu\n", vectorsWithMismatch, compareCount);
    printf("Total dimension mismatches: %zu / %zu\n", totalAllMismatches, compareCount * baseDimension);
    printf("Interesting dims (|val| > 1) checked: %zu, mismatches: %zu\n",
           totalInterestingDims, totalInterestingMismatches);

    free(baseVecs);
    free(refVecs);
}

/**
 * Benchmark comparing kmeans performance across different configurations:
 * - Dimensions: 1024 vs 128
 * - Scalar quantization: with vs without
 * - Distance metrics: L2 vs IP (Inner Product)
 *
 * Uses Faiss Clustering for all kmeans operations.
 */
void benchmark_kmeans_dimensions_quantization(InputParser &input) {
    const int numVectors = stoi(input.getCmdOption("-numVectors"));
    const int numClusters = stoi(input.getCmdOption("-numClusters"));
    const int nIter = stoi(input.getCmdOption("-nIter"));
    const int nThreads = stoi(input.getCmdOption("-nThreads"));
    const int seed = stoi(input.getCmdOption("-seed"));

    omp_set_num_threads(nThreads);

    // Dimensions to test
    std::vector<int> dimensions = {128, 1024};
    // Metrics to test
    std::vector<std::pair<faiss::MetricType, std::string>> metrics = {
        {faiss::METRIC_L2, "L2"},
        {faiss::METRIC_INNER_PRODUCT, "IP"}
    };
    // Quantization modes
    std::vector<std::pair<bool, std::string>> quantizeModes = {
        {false, "raw"},
        {true, "sq8"}
    };

    printf("=======================================================\n");
    printf("Kmeans Benchmark: Dimensions x Quantization x Metrics\n");
    printf("=======================================================\n");
    printf("NumVectors: %d, NumClusters: %d, nIter: %d, nThreads: %d\n\n", numVectors, numClusters, nIter, nThreads);

    for (int dim : dimensions) {
        // Generate random data for this dimension
        printf("Generating %d random vectors of dimension %d...\n", numVectors, dim);
        RandomGenerator rg(seed);
        std::vector<float> data(numVectors * dim);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = rg.randFloat();
        }

        // Normalize for IP metric (to make results meaningful)
        std::vector<float> normalizedData(numVectors * dim);
        for (int i = 0; i < numVectors; i++) {
            float norm = 0;
            for (int j = 0; j < dim; j++) {
                norm += data[i * dim + j] * data[i * dim + j];
            }
            norm = std::sqrt(norm);
            for (int j = 0; j < dim; j++) {
                normalizedData[i * dim + j] = data[i * dim + j] / norm;
            }
        }

        for (auto& [metric, metricName] : metrics) {
            for (auto& [useQuantized, quantizeName] : quantizeModes) {
                printf("\n-------------------------------------------------------\n");
                printf("Config: dim=%d, metric=%s, data=%s\n", dim, metricName.c_str(), quantizeName.c_str());
                printf("-------------------------------------------------------\n");

                // Select appropriate input data (normalized for IP)
                float* inputData = (metric == faiss::METRIC_INNER_PRODUCT)
                                   ? normalizedData.data() : data.data();

                // Setup clustering parameters
                faiss::ClusteringParameters cp;
                cp.niter = nIter;
                cp.verbose = true;
                cp.seed = seed;
                cp.max_points_per_centroid = INT_MAX;  // Disable sampling
                if (metric == faiss::METRIC_INNER_PRODUCT) {
                    cp.spherical = true;
                }

                // Create clustering object
                faiss::Clustering clustering(dim, numClusters, cp);

                // Create index for distance computation
                faiss::IndexFlat index(dim, metric);

                // Run kmeans and measure time
                auto start = std::chrono::high_resolution_clock::now();

                if (useQuantized) {
                    // Create IndexScalarQuantizer as codec for train_encoded
                    faiss::IndexScalarQuantizer sqIndex(dim, faiss::ScalarQuantizer::QT_8bit, metric);
                    printf("Training scalar quantizer codec...\n");
                    sqIndex.train(numVectors, inputData);

                    // Encode data using the codec
                    std::vector<uint8_t> codes(numVectors * sqIndex.code_size);
                    sqIndex.sa_encode(numVectors, inputData, codes.data());

                    // Run kmeans directly on encoded data using train_encoded
                    printf("Running kmeans with train_encoded on scalar quantized data...\n");
                    clustering.train_encoded(numVectors, codes.data(), &sqIndex, index);
                } else {
                    clustering.train(numVectors, inputData, index);
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

                printf("\nResults:\n");
                printf("  Training time: %lld ms\n", duration.count());
                if (!clustering.iteration_stats.empty()) {
                    printf("  Final objective: %f\n", clustering.iteration_stats.back().obj);
                    printf("  Iterations completed: %zu\n", clustering.iteration_stats.size());
                }

                // Compute cluster sizes for analysis using the centroids now in the index
                std::vector<int64_t> assignments(numVectors);
                index.assign(numVectors, inputData, assignments.data());

                std::vector<int> histogram(numClusters, 0);
                for (int i = 0; i < numVectors; i++) {
                    if (assignments[i] >= 0 && assignments[i] < numClusters) {
                        histogram[assignments[i]]++;
                    }
                }

                int minClusterSize = *std::min_element(histogram.begin(), histogram.end());
                int maxClusterSize = *std::max_element(histogram.begin(), histogram.end());
                double avgClusterSize = static_cast<double>(numVectors) / numClusters;

                double sumSquaredDiff = 0.0;
                for (int count : histogram) {
                    double diff = count - avgClusterSize;
                    sumSquaredDiff += diff * diff;
                }
                double stdDev = std::sqrt(sumSquaredDiff / numClusters);

                printf("  Cluster size stats:\n");
                printf("    Min: %d, Max: %d, Avg: %.2f, StdDev: %.2f\n",
                       minClusterSize, maxClusterSize, avgClusterSize, stdDev);
            }
        }
        printf("\n");
    }

    printf("=======================================================\n");
    printf("Benchmark Complete\n");
    printf("=======================================================\n");
}

/**
 * Benchmark Faiss ScalarQuantizer 8-bit distance computation
 */
void benchmark_faiss_sq8_distance(InputParser &input) {
    const size_t dim = input.getCmdOption("-dim").empty() ? 1024 : stoi(input.getCmdOption("-dim"));
    const size_t numBase = input.getCmdOption("-numBase").empty() ? 100000 : stoi(input.getCmdOption("-numBase"));
    const size_t numQuery = input.getCmdOption("-numQuery").empty() ? 100 : stoi(input.getCmdOption("-numQuery"));
    const int numIterations = input.getCmdOption("-n").empty() ? 10 : stoi(input.getCmdOption("-n"));

    printf("=======================================================\n");
    printf("Faiss ScalarQuantizer 8-bit Distance Benchmark\n");
    printf("=======================================================\n");
    printf("Base vectors: %zu x %zu\n", numBase, dim);
    printf("Query vectors: %zu x %zu\n", numQuery, dim);
    printf("Iterations: %d\n", numIterations);
    printf("=======================================================\n\n");

    // Generate random vectors using Faiss
    std::vector<float> baseVecs(numBase * dim);
    std::vector<float> queryVecs(numQuery * dim);

    printf("Generating random vectors...\n");
    faiss::float_rand(baseVecs.data(), baseVecs.size(), 1234);
    faiss::float_rand(queryVecs.data(), queryVecs.size(), 5678);

    // Create and train the 8-bit scalar quantizer
    faiss::ScalarQuantizer sq(dim, faiss::ScalarQuantizer::QT_8bit);

    printf("Training ScalarQuantizer...\n");
    auto trainStart = std::chrono::high_resolution_clock::now();
    sq.train(numBase, baseVecs.data());
    auto trainEnd = std::chrono::high_resolution_clock::now();
    auto trainDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);
    printf("Training time: %lld ms\n", trainDuration.count());
    printf("Code size per vector: %zu bytes\n", sq.code_size);

    // Encode base vectors
    std::vector<uint8_t> codes(sq.code_size * numBase);
    printf("Encoding base vectors...\n");
    auto encodeStart = std::chrono::high_resolution_clock::now();
    sq.compute_codes(baseVecs.data(), codes.data(), numBase);
    auto encodeEnd = std::chrono::high_resolution_clock::now();
    auto encodeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(encodeEnd - encodeStart);
    printf("Encoding time: %lld ms\n\n", encodeDuration.count());

    // // Benchmark non-quantized L2 distance (baseline)
    // printf("--- Baseline: Non-quantized L2 distance ---\n");
    // {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     double totalDist = 0;
    //     for (int iter = 0; iter < numIterations; iter++) {
    //         for (size_t q = 0; q < numQuery; q++) {
    //             for (size_t b = 0; b < numBase; b++) {
    //                 totalDist += faiss::fvec_L2sqr(
    //                     queryVecs.data() + q * dim,
    //                     baseVecs.data() + b * dim,
    //                     dim
    //                 );
    //             }
    //         }
    //     }
    //     auto end = std::chrono::high_resolution_clock::now();
    //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //
    //     size_t totalComputations = (size_t)numIterations * numQuery * numBase;
    //     double throughput = totalComputations / (duration.count() / 1000.0);
    //
    //     printf("Total distance sum: %.6e\n", totalDist);
    //     printf("Time: %lld ms\n", duration.count());
    //     printf("Throughput: %.2f M distances/sec\n\n", throughput / 1e6);
    // }
    //
    // // Benchmark SimSIMD non-quantized L2 distance
    // printf("--- SimSIMD: Non-quantized L2 distance ---\n");
    // {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     double totalDist = 0;
    //     simsimd_distance_t dist;
    //     for (int iter = 0; iter < numIterations; iter++) {
    //         for (size_t q = 0; q < numQuery; q++) {
    //             for (size_t b = 0; b < numBase; b++) {
    //                 simsimd_l2sq_f32(
    //                     queryVecs.data() + q * dim,
    //                     baseVecs.data() + b * dim,
    //                     dim,
    //                     &dist
    //                 );
    //                 totalDist += dist;
    //             }
    //         }
    //     }
    //     auto end = std::chrono::high_resolution_clock::now();
    //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //
    //     size_t totalComputations = (size_t)numIterations * numQuery * numBase;
    //     double throughput = totalComputations / (duration.count() / 1000.0);
    //
    //     printf("Total distance sum: %.6e\n", totalDist);
    //     printf("Time: %lld ms\n", duration.count());
    //     printf("Throughput: %.2f M distances/sec\n\n", throughput / 1e6);
    // }
    //
    // // Benchmark SQ8 L2 distance
    // printf("--- SQ8 L2 distance ---\n");
    // {
    //     std::unique_ptr<faiss::ScalarQuantizer::SQDistanceComputer> dc(
    //         sq.get_distance_computer(faiss::METRIC_L2)
    //     );
    //     dc->codes = codes.data();
    //     dc->code_size = sq.code_size;
    //
    //     auto start = std::chrono::high_resolution_clock::now();
    //     double totalDist = 0;
    //     for (int iter = 0; iter < numIterations; iter++) {
    //         for (size_t q = 0; q < numQuery; q++) {
    //             dc->set_query(queryVecs.data() + q * dim);
    //             for (size_t b = 0; b < numBase; b++) {
    //                 totalDist += dc->distance_to_code(codes.data() + b * sq.code_size);
    //             }
    //         }
    //     }
    //     auto end = std::chrono::high_resolution_clock::now();
    //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //
    //     size_t totalComputations = (size_t)numIterations * numQuery * numBase;
    //     double throughput = totalComputations / (duration.count() / 1000.0);
    //
    //     printf("Total distance sum: %.6e\n", totalDist);
    //     printf("Time: %lld ms\n", duration.count());
    //     printf("Throughput: %.2f M distances/sec\n\n", throughput / 1e6);
    // }

    // // Benchmark non-quantized Inner Product distance (baseline)
    // printf("--- Baseline: Non-quantized Inner Product ---\n");
    // {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     double totalDist = 0;
    //     for (int iter = 0; iter < numIterations; iter++) {
    //         for (size_t q = 0; q < numQuery; q++) {
    //             for (size_t b = 0; b < numBase; b++) {
    //                 totalDist += faiss::fvec_inner_product(
    //                     queryVecs.data() + q * dim,
    //                     baseVecs.data() + b * dim,
    //                     dim
    //                 );
    //             }
    //         }
    //     }
    //     auto end = std::chrono::high_resolution_clock::now();
    //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //
    //     size_t totalComputations = (size_t)numIterations * numQuery * numBase;
    //     double throughput = totalComputations / (duration.count() / 1000.0);
    //
    //     printf("Total distance sum: %.6e\n", totalDist);
    //     printf("Time: %lld ms\n", duration.count());
    //     printf("Throughput: %.2f M distances/sec\n\n", throughput / 1e6);
    // }

    // Benchmark SimSIMD non-quantized Inner Product
    printf("--- SimSIMD: Non-quantized Inner Product ---\n");
    {
        auto start = std::chrono::high_resolution_clock::now();
        double totalDist = 0;
        simsimd_distance_t dist;
        for (int iter = 0; iter < numIterations; iter++) {
            for (size_t q = 0; q < numQuery; q++) {
                for (size_t b = 0; b < numBase; b++) {
                    simsimd_dot_f32(
                        queryVecs.data() + q * dim,
                        baseVecs.data() + b * dim,
                        dim,
                        &dist
                    );
                    totalDist += dist;
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        size_t totalComputations = (size_t)numIterations * numQuery * numBase;
        double throughput = totalComputations / (duration.count() / 1000.0);

        printf("Total distance sum: %.6e\n", totalDist);
        printf("Time: %lld ms\n", duration.count());
        printf("Throughput: %.2f M distances/sec\n\n", throughput / 1e6);
    }

    // Benchmark Blocked SimSIMD Inner Product (demonstrating sgemm_-like blocking)
    // This shows that blocking alone gives SOME benefit, but nowhere near sgemm_
    printf("--- Blocked SimSIMD: Inner Product (sgemm_-like blocking) ---\n");
    {
        struct BatchConfig {
            size_t bs_x;  // query batch size
            size_t bs_y;  // database batch size
        };
        std::vector<BatchConfig> configs = {
            {64, 64},
            {512, 512},
            {1, 1024},
        };

        for (const auto& cfg : configs) {
            size_t bs_x = cfg.bs_x;
            size_t bs_y = cfg.bs_y;

            auto start = std::chrono::high_resolution_clock::now();
            double totalDist = 0;
            simsimd_distance_t dist;

            for (int iter = 0; iter < numIterations; iter++) {
                // Outer loops: iterate over blocks (like sgemm_)
                for (size_t i0 = 0; i0 < numQuery; i0 += bs_x) {
                    size_t i1 = std::min(i0 + bs_x, numQuery);

                    for (size_t j0 = 0; j0 < numBase; j0 += bs_y) {
                        size_t j1 = std::min(j0 + bs_y, numBase);

                        // Inner loops: process all pairs within the block
                        // This is where sgemm_ would do a cache-optimized matmul
                        // We can only do vector-by-vector here
                        for (size_t q = i0; q < i1; q++) {
                            const float* query_ptr = queryVecs.data() + q * dim;
                            for (size_t b = j0; b < j1; b++) {
                                simsimd_dot_f32(
                                    query_ptr,
                                    baseVecs.data() + b * dim,
                                    dim,
                                    &dist
                                );
                                totalDist += dist;
                            }
                        }
                    }
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            size_t totalComputations = (size_t)numIterations * numQuery * numBase;
            double throughput = totalComputations / (duration.count() / 1000.0);

            printf("  bs_x=%5zu, bs_y=%5zu | Time: %6lld ms | %.2f M dist/sec\n",
                   bs_x, bs_y, duration.count(), throughput / 1e6);
        }
        printf("\n");
        printf("  NOTE: Blocking helps cache locality, but individual dot products\n");
        printf("        still suffer from poor arithmetic intensity (2D loads/FLOP).\n");
        printf("        sgemm_ uses micro-kernels that keep data in registers.\n\n");
    }

    // Benchmark Manual Micro-Kernel Style Inner Product
    // This demonstrates the KEY insight: compute multiple outputs per base vector load
    printf("--- Manual Micro-Kernel Style: Multiple queries per base load ---\n");
    {
        // The key insight of sgemm_'s micro-kernel:
        // For each base vector loaded, compute dot products against MULTIPLE queries
        // This amortizes the memory load cost across multiple outputs

        constexpr size_t QUERY_UNROLL = 4;  // Process 4 queries per base vector load

        auto start = std::chrono::high_resolution_clock::now();
        double totalDist = 0;
        simsimd_distance_t dist[QUERY_UNROLL];

        for (int iter = 0; iter < numIterations; iter++) {
            // Process queries in groups of QUERY_UNROLL
            for (size_t q = 0; q < numQuery; q += QUERY_UNROLL) {
                size_t q_end = std::min(q + QUERY_UNROLL, numQuery);
                size_t actual_queries = q_end - q;

                // For each base vector (loaded ONCE)
                for (size_t b = 0; b < numBase; b++) {
                    const float* base_ptr = baseVecs.data() + b * dim;

                    // Compute dot product against ALL queries in the group
                    // This means: 1 base load -> QUERY_UNROLL outputs
                    // Much better arithmetic intensity!
                    for (size_t qi = 0; qi < actual_queries; qi++) {
                        simsimd_dot_f32(
                            queryVecs.data() + (q + qi) * dim,
                            base_ptr,
                            dim,
                            &dist[qi]
                        );
                        totalDist += dist[qi];
                    }
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        size_t totalComputations = (size_t)numIterations * numQuery * numBase;
        double throughput = totalComputations / (duration.count() / 1000.0);

        printf("  QUERY_UNROLL=%zu | Time: %6lld ms | %.2f M dist/sec\n",
               QUERY_UNROLL, duration.count(), throughput / 1e6);
        printf("\n");
        printf("  This is STILL slow because:\n");
        printf("  1. simsimd_dot_f32 loads base_ptr anew each call (no register reuse)\n");
        printf("  2. sgemm_ keeps base data in SIMD registers across multiple FMAs\n");
        printf("  3. sgemm_ unrolls BOTH query and base dimensions in registers\n\n");
    }

    // Benchmark BLAS sgemm_ Inner Product with various batch sizes
    printf("--- BLAS sgemm_: Inner Product (varying batch sizes) ---\n");
    {
        struct BatchConfig {
            size_t bs_x;  // query batch size
            size_t bs_y;  // database batch size
        };
        std::vector<BatchConfig> configs = {
            {1, 1},
            {1, 64},
            {64, 64},
            {256, 256},
            {512, 512},
            {1024, 1024},
            {4096, 1024},   // Faiss default
            {4096, 4096},
            {8192, 8192},
        };

        for (const auto& cfg : configs) {
            size_t bs_x = cfg.bs_x;
            size_t bs_y = cfg.bs_y;
            std::vector<float> ip_block(bs_x * bs_y);

            auto start = std::chrono::high_resolution_clock::now();
            double totalDist = 0;

            for (int iter = 0; iter < numIterations; iter++) {
                for (size_t i0 = 0; i0 < numQuery; i0 += bs_x) {
                    size_t i1 = std::min(i0 + bs_x, numQuery);

                    for (size_t j0 = 0; j0 < numBase; j0 += bs_y) {
                        size_t j1 = std::min(j0 + bs_y, numBase);

                        float one = 1.0f, zero = 0.0f;
                        FINTEGER nyi = j1 - j0;
                        FINTEGER nxi = i1 - i0;
                        FINTEGER di = dim;

                        sgemm_("T", "N",
                               &nyi, &nxi, &di,
                               &one,
                               baseVecs.data() + j0 * dim, &di,
                               queryVecs.data() + i0 * dim, &di,
                               &zero,
                               ip_block.data(), &nyi);

                        for (size_t i = 0; i < (size_t)(nxi * nyi); i++) {
                            totalDist += ip_block[i];
                        }
                    }
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            size_t totalComputations = (size_t)numIterations * numQuery * numBase;
            double throughput = totalComputations / (duration.count() / 1000.0);

            printf("  bs_x=%5zu, bs_y=%5zu | Time: %6lld ms | %.2f M dist/sec\n",
                   bs_x, bs_y, duration.count(), throughput / 1e6);
        }
        printf("\n");
    }

    // Benchmark SQ8 Inner Product distance
    printf("--- SQ8 Inner Product ---\n");
    {
        std::unique_ptr<faiss::ScalarQuantizer::SQDistanceComputer> dc(
            sq.get_distance_computer(faiss::METRIC_INNER_PRODUCT)
        );
        dc->codes = codes.data();
        dc->code_size = sq.code_size;

        auto start = std::chrono::high_resolution_clock::now();
        double totalDist = 0;
        for (int iter = 0; iter < numIterations; iter++) {
            for (size_t q = 0; q < numQuery; q++) {
                dc->set_query(queryVecs.data() + q * dim);
                for (size_t b = 0; b < numBase; b++) {
                    totalDist += dc->distance_to_code(codes.data() + b * sq.code_size);
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        size_t totalComputations = (size_t)numIterations * numQuery * numBase;
        double throughput = totalComputations / (duration.count() / 1000.0);

        printf("Total distance sum: %.6e\n", totalDist);
        printf("Time: %lld ms\n", duration.count());
        printf("Throughput: %.2f M distances/sec\n\n", throughput / 1e6);
    }

    // Benchmark Blocked SQ8 Inner Product (demonstrating sgemm_-like blocking)
    printf("--- Blocked SQ8: Inner Product (sgemm_-like blocking) ---\n");
    {
        struct BatchConfig {
            size_t bs_x;  // query batch size
            size_t bs_y;  // database batch size
        };
        std::vector<BatchConfig> configs = {
            {64, 64},
            {512, 512},
            {1, 1024},
        };

        std::unique_ptr<faiss::ScalarQuantizer::SQDistanceComputer> dc(
            sq.get_distance_computer(faiss::METRIC_INNER_PRODUCT)
        );
        dc->codes = codes.data();
        dc->code_size = sq.code_size;

        for (const auto& cfg : configs) {
            size_t bs_x = cfg.bs_x;
            size_t bs_y = cfg.bs_y;

            auto start = std::chrono::high_resolution_clock::now();
            double totalDist = 0;

            for (int iter = 0; iter < numIterations; iter++) {
                // Outer loops: iterate over blocks (like sgemm_)
                for (size_t i0 = 0; i0 < numQuery; i0 += bs_x) {
                    size_t i1 = std::min(i0 + bs_x, numQuery);

                    for (size_t j0 = 0; j0 < numBase; j0 += bs_y) {
                        size_t j1 = std::min(j0 + bs_y, numBase);

                        // Inner loops: process all pairs within the block
                        for (size_t q = i0; q < i1; q++) {
                            dc->set_query(queryVecs.data() + q * dim);
                            for (size_t b = j0; b < j1; b++) {
                                totalDist += dc->distance_to_code(codes.data() + b * sq.code_size);
                            }
                        }
                    }
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            size_t totalComputations = (size_t)numIterations * numQuery * numBase;
            double throughput = totalComputations / (duration.count() / 1000.0);

            printf("  bs_x=%5zu, bs_y=%5zu | Time: %6lld ms | %.2f M dist/sec\n",
                   bs_x, bs_y, duration.count(), throughput / 1e6);
        }
        printf("\n");
    }

    printf("\n=======================================================\n");
    printf("Benchmark Complete\n");
    printf("=======================================================\n");
}

/**
 * Test function to validate OpenBLAS thread safety with knn_inner_product.
 * Calls knn_inner_product in parallel from multiple threads to check for
 * segmentation faults that may arise from OpenBLAS thread safety issues.
 */
void test_knn_inner_product_parallel(InputParser& input) {
    printf("=======================================================\n");
    printf("Testing knn_inner_product in parallel for OpenBLAS thread safety\n");
    printf("(with different data sizes, data, and queries per thread)\n");
    printf("=======================================================\n");
    // openblas_set_num_threads(1);

    const size_t dim = 128;
    const size_t k = 10;
    const int numIterations = 50;
    const int nThreads = input.getCmdOption("-nThreads").empty() ? 10 : stoi(input.getCmdOption("-nThreads"));
    omp_set_num_threads(nThreads);
    const int numThreads = omp_get_max_threads();

    // Different sizes for each thread
    const size_t minNumBase = 50000;
    const size_t maxNumBase = 300000;
    const size_t minNumQueries = 50;
    const size_t maxNumQueries = 1000;

    printf("Config: dim=%zu, k=%zu\n", dim, k);
    printf("Base vectors per thread: %zu - %zu\n", minNumBase, maxNumBase);
    printf("Queries per thread: %zu - %zu\n", minNumQueries, maxNumQueries);
    printf("Iterations: %d, Threads: %d\n\n", numIterations, numThreads);

    // Generate different data for each thread with different sizes
    std::vector<size_t> threadNumBase(numThreads);
    std::vector<size_t> threadNumQueries(numThreads);
    std::vector<std::vector<float>> threadBaseVecs(numThreads);
    std::vector<std::vector<float>> threadQueryVecs(numThreads);
    std::vector<std::vector<float>> threadDistances(numThreads);
    std::vector<std::vector<int64_t>> threadIndices(numThreads);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> baseDist(minNumBase, maxNumBase);
    std::uniform_int_distribution<size_t> queryDist(minNumQueries, maxNumQueries);

    for (int t = 0; t < numThreads; t++) {
        threadNumBase[t] = baseDist(rng);
        threadNumQueries[t] = queryDist(rng);

        threadBaseVecs[t].resize(threadNumBase[t] * dim);
        threadQueryVecs[t].resize(threadNumQueries[t] * dim);

        // Use different seeds for each thread's data
        faiss::float_rand(threadBaseVecs[t].data(), threadBaseVecs[t].size(), 42 + t * 1000);
        faiss::float_rand(threadQueryVecs[t].data(), threadQueryVecs[t].size(), 123 + t * 1000);

        threadDistances[t].resize(threadNumQueries[t] * k);
        threadIndices[t].resize(threadNumQueries[t] * k);

        printf("Thread %d: numBase=%zu, numQueries=%zu\n", t, threadNumBase[t], threadNumQueries[t]);
    }

    printf("\nGenerated random vectors for all threads\n");

    std::atomic<int> completedIterations(0);
    std::atomic<bool> hasCrash(false);

    printf("Starting parallel knn_inner_product calls...\n");
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float* distances = threadDistances[tid].data();
        int64_t* indices = threadIndices[tid].data();
        const float* baseVecs = threadBaseVecs[tid].data();
        const float* queryVecs = threadQueryVecs[tid].data();
        size_t numBase = threadNumBase[tid];
        size_t numQueries = threadNumQueries[tid];

        for (int iter = 0; iter < numIterations && !hasCrash.load(); iter++) {
            try {
                faiss::knn_inner_product(
                    queryVecs,
                    baseVecs,
                    dim,
                    numQueries,
                    numBase,
                    k,
                    distances,
                    indices,
                    nullptr,
                    nullptr
                );

                int completed = completedIterations.fetch_add(1) + 1;
                if (completed % (numThreads * 10) == 0) {
#pragma omp critical
                    {
                        printf("  Completed %d / %d total calls\n",
                               completed, numIterations * numThreads);
                    }
                }
            } catch (const std::exception& e) {
#pragma omp critical
                {
                    printf("Exception in thread %d: %s\n", tid, e.what());
                    hasCrash.store(true);
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("\n=======================================================\n");
    if (hasCrash.load()) {
        printf("TEST FAILED: Crash or exception detected!\n");
    } else {
        printf("TEST PASSED: No segfault detected\n");
        printf("Total calls: %d\n", completedIterations.load());
        printf("Time: %lld ms\n", duration.count());

        // Verify results are reasonable (not NaN/inf)
        bool resultsValid = true;
        for (int t = 0; t < numThreads && resultsValid; t++) {
            for (size_t i = 0; i < threadNumQueries[t] * k; i++) {
                if (std::isnan(threadDistances[t][i]) || std::isinf(threadDistances[t][i])) {
                    printf("Invalid distance found in thread %d at index %zu\n", t, i);
                    resultsValid = false;
                    break;
                }
            }
        }
        if (resultsValid) {
            printf("Result validation: PASSED (no NaN/inf values)\n");
        }
    }
    printf("=======================================================\n");
}

#ifdef CUVS_ENABLED
void benchmark_cuvs_balanced_kmeans_wrapper(InputParser &input) {
    const std::string &baseVectorPath = input.getCmdOption("-baseVectorPath");
    const int numVectors = stoi(input.getCmdOption("-numVectors"));
    const int clusterSize = stoi(input.getCmdOption("-clusterSize"));
    const int nIter = stoi(input.getCmdOption("-nIter"));
    const bool useIP = stoi(input.getCmdOption("-useIP"));

    size_t baseDimension, baseNumVectors;
    float *baseVecs = readVecFile(baseVectorPath.c_str(), &baseDimension, &baseNumVectors, numVectors);
    auto numClusters = baseNumVectors / clusterSize;

    printf("Loaded %zu vectors of dimension %zu, targeting %zu clusters\n",
           baseNumVectors, baseDimension, numClusters);

    std::vector<float> centroids(numClusters * baseDimension);
    std::vector<uint32_t> labels(baseNumVectors);

    cuvs_kmeans_fit(baseVecs, baseNumVectors, baseDimension, numClusters, nIter, useIP,
                    centroids.data());

    cuvs_kmeans_predict(baseVecs, baseNumVectors, baseDimension, centroids.data(),
                        numClusters, nIter, useIP, labels.data());

    delete[] baseVecs;
}
#endif

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    backward::SignalHandling sh;
    InputParser input(argc, argv);
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
    else if (run == "benchmarkFaissClusteringOnBvec") {
        benchmark_faiss_clustering_on_bvec(input);
    }
    else if (run == "debugFbinIvfQuery") {
        debug_fbin_ivf_query(input);
    }
    else if (run == "benchmarkFaissFlat") {
        benchmark_faiss_flat(input);
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
    else if (run == "benchmarkBalancedClustering") {
        benchmark_balanced_clustering(input);
    }
    else if (run == "testSomething") {
        test_something(input);
    }
    else if (run == "read_and_write_chunk") {
        read_and_write_chunk(input);
    }
    else if (run == "testBug") {
        test_final_bug_2(input);
    }
    else if (run == "testQuantizationIssue") {
        test_quantization_issue(input);
    }
    else if (run == "testQuantizationParquet") {
        test_quantization_parquet(input);
    }
    else if (run == "run_umap_2D_without_clustering") {
        run_umap_2D_without_clustering(input);
    }
    else if (run == "run_umap_3D_without_clustering") {
        run_umap_3D_without_clustering(input);
    }
    else if (run == "benchmarkFaissSQ8Distance") {
        benchmark_faiss_sq8_distance(input);
    }
    else if (run == "benchmarkKmeansDimensionsQuantization") {
        benchmark_kmeans_dimensions_quantization(input);
    }
    else if (run == "testKnnInnerProductParallel") {
        test_knn_inner_product_parallel(input);
    }
    else if (run == "printQuantizationParquetData") {
        print_quantization_parquet_data(input);
    }
    else if (run == "computeQuantizedIpMse") {
        compute_quantized_ip_mse(input);
    }
    else if (run == "computeRaBitQRotatedDistanceMse") {
        compute_rabitq_rotated_distance_mse(input);
    }
    else if (run == "computeScalarQuantizerDistanceMse") {
        compute_scalar_quantizer_distance_mse(input);
    }
#ifdef CUVS_ENABLED
    else if (run == "benchmarkCuvsBalancedKmeans") {
        benchmark_cuvs_balanced_kmeans_wrapper(input);
    }
#endif
    return 0;
}
