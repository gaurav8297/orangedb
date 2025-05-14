#include <memory>
#include <vector>
#include "include/clustering.h"
#include <cstring>
#include <omp.h>
#include <queue>
#include <helper_ds.h>
#include <common.h>
#include <limits>

// a bit above machine epsilon for float16
#define EPS (1 / 1024.)

namespace orangedb {
    Clustering::Clustering(int dim, int numCentroids, int nIter, int minCentroidSize, int maxCentroidSize, float lambda)
            : dim(dim),
              numCentroids(numCentroids),
              nIter(nIter),
              minCentroidSize(minCentroidSize),
              maxCentroidSize(maxCentroidSize),
              centroids(std::vector<float>(dim * numCentroids)),
              lambda(lambda) {}

    void Clustering::initCentroids(const float *data, int n) {
        // TODO: Implement divide and conquer k-means++ initialization
        // For now choose the random centroids
        CHECK_ARGUMENT(n > numCentroids, "Number of vectors should be greater than number of centroids");
        auto c = centroids.data();
        std::vector<vector_idx_t> perm(numCentroids);
        RandomGenerator rg(seed + 15486557L);
        rg.randomPerm(n, perm.data(), numCentroids);
        for (int i = 0; i < numCentroids; i++) {
            memcpy(c + i * dim, data + perm[i] * dim, dim * sizeof(float));
        }
    }

    void Clustering::train(float *data, int n) {
        // printf("Training numCentroids: %d, nIter: %d, minCentroidSize: %d, maxCentroidSize: %d\n",
        //        numCentroids, nIter, minCentroidSize, maxCentroidSize);
        // Sample data from the given data
        float *sample = nullptr;
        int nSample = sampleData(n, data, &sample);

        auto *dist = new double[nSample];
        auto *assign = new int32_t[nSample];
        for (int i = 0; i < nIter; i++) {
            // printf("Running iteration: %d\n", i);
            // Initialize the index
            CosineDistanceComputer dc = CosineDistanceComputer(centroids.data(), dim, numCentroids);
            IndexOneNN index = IndexOneNN(&dc, dim, numCentroids, lambda);

            // Phase 1: Assign each vector to the nearest centroid
            index.search(nSample, sample, dist, assign);

            // accumulate objective
            float objective = 0;
            for (int j = 0; j < nSample; j++) {
                objective += dist[j];
            }

            // Phase 2: Update the centroids
            int *hist = new int[numCentroids];
            float *newCentroids = new float[numCentroids * dim];
            computeCentroids(nSample, sample, assign, hist, newCentroids);

            // Phase 3: Split the big clusters
            splitClusters(hist, newCentroids);

            // Copy the centroids
            memcpy(centroids.data(), newCentroids, numCentroids * dim * sizeof(float));
        }
    }

    void Clustering::assignCentroids(const float *data, int n, int32_t *assign) {
        // Initialize the index
        CosineDistanceComputer dc = CosineDistanceComputer(centroids.data(), dim, numCentroids);
        IndexOneNN index = IndexOneNN(&dc, dim, numCentroids);

        // Assign each vector to the nearest centroid
        // TODO: Dist is not needed. We can optimize this.
        auto *dist = new double[n];
        index.search(n, data, dist, assign);
        delete[] dist;
    }

    void Clustering::assignCentroids(const float *data, int n, double *dist, int32_t *assign) {
        CosineDistanceComputer dc = CosineDistanceComputer(centroids.data(), dim, numCentroids);
        IndexOneNN index = IndexOneNN(&dc, dim, numCentroids);
        index.search(n, data, dist, assign);
    }

    void Clustering::computeCentroids(int n, const float *data, const int *assign, int *hist, float *newCentroids) {
        memset(newCentroids, 0, numCentroids * dim * sizeof(float));
        memset(hist, 0, numCentroids * sizeof(int));
#pragma omp parallel
        {
            int nt = omp_get_num_threads();
            int rank = omp_get_thread_num();

            // this thread is taking care of centroids c0:c1
            size_t c0 = (numCentroids * rank) / nt;
            size_t c1 = (numCentroids * (rank + 1)) / nt;

            // compute the new centroids
            for (int i = 0; i < n; i++) {
                int c = assign[i];
                if (c0 <= c && c < c1) {
                    auto centroid = newCentroids + c * dim;
                    auto point = data + i * dim;
                    for (int j = 0; j < dim; j++) {
                        centroid[j] += point[j];
                    }
                    hist[c]++;
                }
            }
        }

        // normalize centroids
        // No need to parallelize this part since the assumption is that numCentroids is small
        for (int i = 0; i < numCentroids; i++) {
            if (hist[i] == 0) {
                continue;
            }
            auto norm = 1.0f / hist[i];
            auto centroid = newCentroids + i * dim;
            for (int j = 0; j < dim; j++) {
                centroid[j] *= norm;
            }
        }
    }

    void Clustering::splitClusters(int *hist, float *newCentroids) {
        // Split the big clusters
        for (int small = 0; small < numCentroids; small++) {
            // skip the clusters that are already large enough
            if (hist[small] > minCentroidSize) {
                continue;
            }

            // find the biggest cluster
            int big = 0;
            for (int j = 1; j < numCentroids; j++) {
                if (hist[j] > hist[big]) {
                    big = j;
                }
            }

            // split the biggest cluster
            // copy the centroid
            memcpy(newCentroids + small * dim, newCentroids + big * dim, dim * sizeof(float));

            // symmetric perturbation
            for (size_t j = 0; j < dim; j++) {
                if (j % 2 == 0) {
                    newCentroids[small * dim + j] *= 1 + EPS;
                    newCentroids[big * dim + j] *= 1 - EPS;
                } else {
                    newCentroids[small * dim + j] *= 1 - EPS;
                    newCentroids[big * dim + j] *= 1 + EPS;
                }
            }

            // update the histogram assuming that equal number of points are moved
            hist[small] = hist[big] / 2;
            hist[big] -= hist[small];
        }
    }

    int Clustering::sampleData(int n, float *data, float **sampleData) {
        // Sample data from the given data
        // We will sample maxCentroidSize * numOfCentroids vectors from the data
        // We will return the sampled data and the number of vectors sampled
        int nSample = std::min(maxCentroidSize * numCentroids, n);
        if (nSample == n) {
            *sampleData = data;
            return nSample;
        }

        *sampleData = new float[nSample * dim];
        std::vector<vector_idx_t> perm(nSample);
        RandomGenerator rg(seed);
        rg.randomPerm(n, perm.data(), nSample);
        for (int i = 0; i < nSample; i++) {
            memcpy(*sampleData + i * dim, data + perm[i] * dim, dim * sizeof(float));
        }
        return nSample;
    }

    void IndexOneNN::search(int n, const float *queries, double *distances, int32_t *resultIds) {
        std::vector<double> hist(numEntries, 0);
#pragma omp parallel
        {
            auto localDc = dc->clone();
#pragma omp for
            for (int i = 0; i < n; i++) {
                localDc->setQuery(queries + i * dim);
                double minDistance = std::numeric_limits<double>::max();
                vector_idx_t j = 0, minId = 0;
                while (j + 4 < numEntries) {
                    double dists[4];
                    vector_idx_t idx[4] = {j, j + 1, j + 2, j + 3};
                    localDc->batchComputeDistances(idx, dists, 4);
                    for (int l = 0; l < 4; l++) {
                        auto recomputedDist = dists[l] + lambda * hist[j + l];
                        if (recomputedDist < minDistance) {
                            minDistance = recomputedDist;
                            minId = j + l;
                        }
                    }
                    j += 4;
                }

                for (vector_idx_t l = j; l < numEntries; l++) {
                    double d;
                    localDc->computeDistance(l, &d);
                    auto recomputedDist = d + lambda * hist[l];
                    if (recomputedDist < minDistance) {
                        minDistance = recomputedDist;
                        minId = l;
                    }
                }

                distances[i] = minDistance;
                resultIds[i] = minId;
                hist[minId]++;
            }
        }
    }


    void IndexOneNN::search(int n, const float *queries, int32_t *resultIds) {
        std::vector<double> hist(numEntries, 0);
#pragma omp parallel
        {
            auto localDc = dc->clone();
#pragma omp for
            for (int i = 0; i < n; i++) {
                localDc->setQuery(queries + i * dim);
                double minDistance = std::numeric_limits<double>::max();
                vector_idx_t j = 0, minId = 0;
                while (j + 4 < numEntries) {
                    double dists[4];
                    vector_idx_t idx[4] = {j, j + 1, j + 2, j + 3};
                    localDc->batchComputeDistances(idx, dists, 4);
                    for (int l = 0; l < 4; l++) {
                        auto recomputedDist = dists[l] + lambda * hist[j + l];
                        if (recomputedDist < minDistance) {
                            minDistance = recomputedDist;
                            minId = j + l;
                        }
                    }
                    j += 4;
                }

                for (vector_idx_t l = j; l < numEntries; l++) {
                    double d;
                    localDc->computeDistance(l, &d);
                    auto recomputedDist = d + lambda * hist[l];
                    if (recomputedDist < minDistance) {
                        minDistance = recomputedDist;
                        minId = l;
                    }
                }
                resultIds[i] = minId;
                hist[minId]++;
            }
        }
    }

    void IndexOneNN::knn(int k, const float *query, double *distance, vector_idx_t *resultIds) {
        // Single threaded implementation
        auto localDc = dc->clone();
        // Find the k nearest neighbors
        localDc->setQuery(query);
        std::priority_queue<NodeDistFarther> res;
        for (int i = 0; i < numEntries; i++) {
            double d;
            localDc->computeDistance(i, &d);
            res.emplace(i, d);
        }
        for (int i = 0; i < k; i++) {
            resultIds[i] = res.top().id;
            distance[i] = res.top().dist;
            res.pop();
        }
    }

    void IndexOneNN::knnFiltered(int k, const float *query, double *distance, vector_idx_t *resultIds, const uint8_t *filteredMask) {
        // Single threaded implementation
        auto localDc = dc->clone();
        // Find the k nearest neighbors
        localDc->setQuery(query);
        std::priority_queue<NodeDistFarther> res;
        for (int i = 0; i < numEntries; i++) {
            if (filteredMask[i]) {
                double d;
                localDc->computeDistance(i, &d);
                // TODO: Optimize this to only keep top k elements
                res.emplace(i, d);
            }
        }
        printf("Size of res: %d\n", res.size());
        for (int i = 0; i < k; i++) {
            resultIds[i] = res.top().id;
            distance[i] = res.top().dist;
            res.pop();
        }
    }
}
