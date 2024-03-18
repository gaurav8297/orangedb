#include <iostream>
#include "hnsw.h"
#include "utils.h"
#include "spdlog/fmt/fmt.h"
#include <chrono>
#ifdef __AVX2__
#include <x86intrin.h>
#endif
#include <stdlib.h>    // atoi, getenv
#include <assert.h>    // assert

using namespace orangedb;

#if defined(__GNUC__)
#define PRAGMA_IMPRECISE_LOOP
#define PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("GCC push_options") \
    _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define PRAGMA_IMPRECISE_FUNCTION_END \
    _Pragma("GCC pop_options")
#endif

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
inline void l2_sqr_dist(const float* __restrict x, const float* __restrict y, size_t d, float& result) {
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

    for (unsigned i = 0; i < aligned_size; i += 480, l += 480, r += 480) {
        AVX_L2SQR(l, r, sum, l0, r0);
        AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
        AVX_L2SQR(l + 16, r + 16, sum, l0, r0);
        AVX_L2SQR(l + 24, r + 24, sum, l1, r1);

        AVX_L2SQR(l + 32, r + 32, sum, l0, r0);
        AVX_L2SQR(l + 40, r + 40, sum, l1, r1);
        AVX_L2SQR(l + 48, r + 48, sum, l0, r0);
        AVX_L2SQR(l + 56, r + 56, sum, l1, r1);

        AVX_L2SQR(l + 64, r + 64, sum, l0, r0);
        AVX_L2SQR(l + 72, r + 72, sum, l1, r1);
        AVX_L2SQR(l + 80, r + 80, sum, l0, r0);
        AVX_L2SQR(l + 88, r + 88, sum, l1, r1);

        AVX_L2SQR(l + 96, r + 96, sum, l0, r0);
        AVX_L2SQR(l + 104, r + 104, sum, l1, r1);
        AVX_L2SQR(l + 112, r + 112, sum, l0, r0);
        AVX_L2SQR(l + 120, r + 120, sum, l1, r1);

        AVX_L2SQR(l + 128, r + 128, sum, l0, r0);
        AVX_L2SQR(l + 136, r + 136, sum, l1, r1);
        AVX_L2SQR(l + 144, r + 144, sum, l0, r0);
        AVX_L2SQR(l + 152, r + 152, sum, l1, r1);

        AVX_L2SQR(l + 160, r + 160, sum, l0, r0);
        AVX_L2SQR(l + 168, r + 168, sum, l1, r1);
        AVX_L2SQR(l + 176, r + 176, sum, l0, r0);
        AVX_L2SQR(l + 184, r + 184, sum, l1, r1);

        AVX_L2SQR(l + 192, r + 192, sum, l0, r0);
        AVX_L2SQR(l + 200, r + 200, sum, l1, r1);
        AVX_L2SQR(l + 208, r + 208, sum, l0, r0);
        AVX_L2SQR(l + 216, r + 216, sum, l1, r1);

        AVX_L2SQR(l + 224, r + 224, sum, l0, r0);
        AVX_L2SQR(l + 232, r + 232, sum, l1, r1);
        AVX_L2SQR(l + 240, r + 240, sum, l0, r0);
        AVX_L2SQR(l + 248, r + 248, sum, l1, r1);

        AVX_L2SQR(l + 256, r + 256, sum, l0, r0);
        AVX_L2SQR(l + 264, r + 264, sum, l1, r1);
        AVX_L2SQR(l + 272, r + 272, sum, l0, r0);
        AVX_L2SQR(l + 280, r + 280, sum, l1, r1);

        AVX_L2SQR(l + 288, r + 288, sum, l0, r0);
        AVX_L2SQR(l + 296, r + 296, sum, l1, r1);
        AVX_L2SQR(l + 304, r + 304, sum, l0, r0);
        AVX_L2SQR(l + 312, r + 312, sum, l1, r1);

        AVX_L2SQR(l + 320, r + 320, sum, l0, r0);
        AVX_L2SQR(l + 328, r + 328, sum, l1, r1);
        AVX_L2SQR(l + 336, r + 336, sum, l0, r0);
        AVX_L2SQR(l + 344, r + 344, sum, l1, r1);

        AVX_L2SQR(l + 352, r + 352, sum, l0, r0);
        AVX_L2SQR(l + 360, r + 360, sum, l1, r1);
        AVX_L2SQR(l + 368, r + 368, sum, l0, r0);
        AVX_L2SQR(l + 376, r + 376, sum, l1, r1);

        AVX_L2SQR(l + 384, r + 384, sum, l0, r0);
        AVX_L2SQR(l + 392, r + 392, sum, l1, r1);
        AVX_L2SQR(l + 400, r + 400, sum, l0, r0);
        AVX_L2SQR(l + 408, r + 408, sum, l1, r1);

        AVX_L2SQR(l + 416, r + 416, sum, l0, r0);
        AVX_L2SQR(l + 424, r + 424, sum, l1, r1);
        AVX_L2SQR(l + 432, r + 432, sum, l0, r0);
        AVX_L2SQR(l + 440, r + 440, sum, l1, r1);

        AVX_L2SQR(l + 448, r + 448, sum, l0, r0);
        AVX_L2SQR(l + 456, r + 456, sum, l1, r1);
        AVX_L2SQR(l + 464, r + 464, sum, l0, r0);
        AVX_L2SQR(l + 472, r + 472, sum, l1, r1);
    }

    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
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
#endif

inline void l2_sqr_dist_2(const float* __restrict x, const float* __restrict y, size_t d, float& result) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    result = res;
}

PRAGMA_IMPRECISE_FUNCTION_BEGIN
inline void fvec_L2sqr_batch_4(
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


int64_t exp_l1_sqr_dist(const float* baseVecs, size_t baseDimension, size_t baseNumVectors) {
    auto start = std::chrono::high_resolution_clock::now();
    float res = 0;
    const float* query = baseVecs;
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

        l2_sqr_dist_2(query, baseVecs + (i * baseDimension), baseDimension, res0);
        l2_sqr_dist_2(query, baseVecs + ((i+1) * baseDimension), baseDimension, res1);
        l2_sqr_dist_2(query, baseVecs + ((i+2) * baseDimension), baseDimension, res2);
        l2_sqr_dist_2(query, baseVecs + ((i+3) * baseDimension), baseDimension, res3);
        res += (res0 + res1 + res2 + res3);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Result: %f\n", res);
    return duration;
}


int64_t exp_l2_sqr_dist(const float* baseVecs, size_t baseDimension, size_t baseNumVectors) {
    auto start = std::chrono::high_resolution_clock::now();
    float res = 0;
    const float* query = baseVecs;
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

        l2_sqr_dist_2(query, baseVecs + (i * baseDimension), baseDimension, res0);
        l2_sqr_dist_2(query, baseVecs + ((i+1) * baseDimension), baseDimension, res1);
        l2_sqr_dist_2(query, baseVecs + ((i+2) * baseDimension), baseDimension, res2);
        l2_sqr_dist_2(query, baseVecs + ((i+3) * baseDimension), baseDimension, res3);
        res += (res0 + res1 + res2 + res3);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Result: %f\n", res);
    return duration;
}

int64_t exp_l2_sqr_dist_2(const float* baseVecs, size_t baseDimension, size_t baseNumVectors) {
    auto start = std::chrono::high_resolution_clock::now();
    float res = 0;
    const float* query = baseVecs;
    for (size_t i = 1; i < baseNumVectors - 4; i += 4) {
        float res0, res1, res2, res3;
        fvec_L2sqr_batch_4(
                query,
                baseVecs + (i * baseDimension),
                baseVecs + ((i+1) * baseDimension),
                baseVecs + ((i+2) * baseDimension),
                baseVecs + ((i+3) * baseDimension),
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

void benchmark_simd_distance() {
    auto basePath = "/home/gaurav/vector_index_experiments/vector_index/data/gist_200k";
    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);

    size_t baseDimension, baseNumVectors;
    float* baseVecs = Utils::fvecs_read(baseVectorPath.c_str(),&baseDimension,&baseNumVectors);
    printf("Base dimension: %zu, Base num vectors: %zu\n", baseDimension, baseNumVectors);

    int64_t duration = 0;
    for (int i =0; i < 100; i++) {
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

void build_graph(HNSW& hnsw, const float* baseVecs, size_t baseNumVectors, size_t baseDimension) {
    auto start = std::chrono::high_resolution_clock::now();
    hnsw.build(baseVecs, baseNumVectors);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Building time: " << duration << " ms" << std::endl;
}

void query_graph(
        HNSW& hnsw,
        const float* queryVecs,
        size_t queryNumVectors,
        size_t queryDimension,
        const int* gtVecs,
        size_t k,
        size_t ef_search,
        size_t baseNumVectors) {
    auto start = std::chrono::high_resolution_clock::now();
    auto visited = VisitedTable(baseNumVectors);
    auto recall = 0.0;
    for (size_t i = 0; i < queryNumVectors; i++) {
        std::priority_queue<HNSW::NodeDistCloser> resultSet;
        hnsw.search_v1(queryVecs + (i * queryDimension), k, ef_search, visited, resultSet);
        auto gt = gtVecs + i * 100;
        std::vector<HNSW::NodeDistCloser> res;
        while (!resultSet.empty()) {
            res.push_back(resultSet.top());
            resultSet.pop();
        }
        for (int j = 0; j < res.size(); j++) {
            if (std::find(gt, gt + 100, res[j].id) != (gt + 100)) {
                recall++;
            }
        }
    }
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

void benchmark_hnsw_queries() {
    auto basePath = "/home/g3sehgal/vector_index_exp/gist";

    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);
    auto queryVectorPath = fmt::format("{}/query.fvecs", basePath);
    auto groundTruthPath = fmt::format("{}/groundtruth.ivecs", basePath);

    size_t baseDimension, baseNumVectors;
    float* baseVecs = Utils::fvecs_read(baseVectorPath.c_str(),&baseDimension,&baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = Utils::fvecs_read(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    size_t gtDimension, gtNumVectors;
    int *gtVecs = Utils::ivecs_read(groundTruthPath.c_str(), &gtDimension, &gtNumVectors);

    omp_set_num_threads(32);
    HNSW hnsw(64, 200, 128, baseDimension);
    build_graph(hnsw, baseVecs, baseNumVectors, baseDimension);

//    enable_perf();
    query_graph(hnsw, queryVecs, queryNumVectors, queryDimension, gtVecs, 100, 150, baseNumVectors);
//    disable_perf();
}

int main() {
    benchmark_hnsw_queries();
//    benchmark_simd_distance();
    return 0;
}
