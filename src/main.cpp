#include <iostream>
#include "hnsw.h"
#include "utils.h"
#include "spdlog/fmt/fmt.h"
#include <chrono>

using namespace orangedb;

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

void exp_hnsw() {
    auto basePath = "/home/gaurav/vector_index_experiments/vector_index/data/siftsmall";

    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);
//    auto queryVectorPath = fmt::format("{}/query.fvecs", basePath);
//    auto gtVectorPath = fmt::format("{}/groundtruth.ivecs", basePath);

    size_t baseDimension, baseNumVectors;
    float* baseVecs = Utils::fvecs_read(baseVectorPath.c_str(),&baseDimension,&baseNumVectors);
//    size_t queryDimension, queryNumVectors;
//    float *queryVecs = Utils::fvecs_read(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
//    size_t gtDimension, gtNumVectors;
//    int *gtVecs = Utils::ivecs_read(gtVectorPath.c_str(), &gtDimension, &gtNumVectors);

    omp_set_num_threads(32);
    HNSW hnsw(64, 120, 120, baseDimension);
    auto start = std::chrono::high_resolution_clock::now();
    hnsw.build(baseVecs, baseNumVectors);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Building time: " << duration << " ms" << std::endl;
}

inline void l2_sqr_dist(const float* __restrict x, const float* __restrict y, size_t d, float& result) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    result = res;
}

void exp_l2_sqr_dist() {
    auto basePath = "/home/gaurav/vector_index_experiments/vector_index/data/gist_50k";
    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);

    size_t baseDimension, baseNumVectors;
    float* baseVecs = Utils::fvecs_read(baseVectorPath.c_str(),&baseDimension,&baseNumVectors);
    printf("Base dimension: %zu, Base num vectors: %zu\n", baseDimension, baseNumVectors);

    auto start = std::chrono::high_resolution_clock::now();
    float res = 0;
    for (size_t i = 0; i < baseNumVectors - 1; i++) {
        float result;
        l2_sqr_dist(baseVecs + (i * baseDimension), baseVecs + ((i + 1) * baseDimension), baseDimension, result);
        res += result;
    }
    printf("Result: %f\n", res);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "l2_sqr_dist time: " << duration << " ms" << std::endl;
}

int main() {
    exp_l2_sqr_dist();
//    exp_hnsw();
    return 0;
}
