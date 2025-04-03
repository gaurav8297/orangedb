#include "include/partitioned_index.h"

namespace orangedb {
    PartitionedIndex::PartitionedIndex(int dim, const PartitionedIndexConfig &config, RandomGenerator* rg)
            : config(config), numVectors(0), dim(dim) {
        clustering = std::make_unique<Clustering>(dim, config.numCentroids, config.nIter, config.minCentroidSize,
                                                  config.maxCentroidSize);
        for (int i = 0; i < config.numCentroids; i++) {
            indexes.push_back(
                    std::make_unique<HNSW>(
                            HNSWConfig(config.M, config.efConstruction, config.efSearch, config.alpha, 1, 0.01, 30, 30, "none", "",
                                       false, 20, 10, 1, "none"),
                            rg,
                            dim));
        }
        actualIds = std::vector<std::vector<vector_idx_t>>(config.numCentroids);
        centroidIndex = new IndexOneNN(new L2DistanceComputer(clustering->centroids.data(), dim, config.numCentroids), dim, config.numCentroids);
    }

    void PartitionedIndex::build(const float *data, size_t n) {
        clustering->initCentroids(data, n);
        clustering->train(data, n);

        int32_t *assign = new int32_t[n];
        clustering->assignCentroids(data, n, assign);

        // Build hassign for each centroid
        int *hassign = new int[config.numCentroids];
        memset(hassign, 0, config.numCentroids * sizeof(int));
        for (int i = 0; i < n; i++) {
            hassign[assign[i]]++;
        }

        // Print the number of vectors assigned to each centroid
        for (int i = 0; i < config.numCentroids; i++) {
            spdlog::info("Centroid {} has {} vectors", i, hassign[i]);
        }

        std::vector<float *> datePerCentroid;
        std::vector<int> idx;
        for (int i = 0; i < config.numCentroids; i++) {
            datePerCentroid.push_back(new float[hassign[i] * dim]);
            idx.push_back(0);
        }

        for (int i = 0; i < n; i++) {
            actualIds[assign[i]].push_back(i);
        }

        // Copy data to each centroid
        for (int i = 0; i < n; i++) {
            auto centroid = assign[i];
            memcpy(datePerCentroid[centroid] + (idx[centroid]++ * dim), data + (i * dim),
                   dim * sizeof(float));
        }

        // Build index for each centroid
        for (int i = 0; i < config.numCentroids; i++) {
            indexes[i]->build(datePerCentroid[i], hassign[i]);
        }
        this->numVectors = n;
    }

    int PartitionedIndex::search(const float *query, uint16_t k, VisitedTable &visited, std::vector<NodeDistFarther> &results, Stats &stats) {
        double *centroidDistances = new double[config.numCentroids];
        vector_idx_t *centroidIndices = new vector_idx_t[config.numCentroids];
        centroidIndex->knn(config.maxSearchCentroids, query, centroidDistances, centroidIndices);
        std::priority_queue<NodeDistFarther> finalResult;
        int qCentroid = 0;
        for (int i = 0; i < config.maxSearchCentroids; i++) {
            if (centroidDistances[i] - centroidDistances[0] > config.searchThreshold) {
                break;
            }
            auto centroidId = centroidIndices[i];
            std::priority_queue<NodeDistCloser> resultQueue;
            indexes[centroidId]->search(query, k, config.efSearch, visited, resultQueue, stats);
            qCentroid++;
            while (!resultQueue.empty()) {
                finalResult.push(NodeDistFarther(actualIds[centroidId][resultQueue.top().id], resultQueue.top().dist));
                resultQueue.pop();
            }
        }
        // Copy the top k results to the results vector
        int s = 0;
        while (!finalResult.empty() && s < k) {
            results.push_back(finalResult.top());
            finalResult.pop();
            s++;
        }

        return qCentroid;
    }
} // namespace orangedb
