#include "include/reclustering_index.h"

namespace orangedb {
    ReclusteringIndex::ReclusteringIndex(int dim, const ReclusteringIndexConfig &config, RandomGenerator *rg)
        : dim(dim), config(config), rg(rg), size(0) {}

    void ReclusteringIndex::insert(float *data, size_t n) {
        printf("ReclusteringIndex::insert\n");
        // Perform k means
        Clustering clustering(dim, config.numCentroids, config.nIter, config.minCentroidSize, config.maxCentroidSize,
                              config.lambda);

        printf("Initialized clustering\n");
        clustering.initCentroids(data, n);

        printf("Training Kmeans\n");
        clustering.train(data, n);

        printf("Assigning centroids\n");
        // Assign the centroids
        std::vector<int32_t> assign(n);
        clustering.assignCentroids(data, n, assign.data());

        // Get the hist
        std::vector<int> hist(config.numCentroids, 0);
        for (int i = 0; i < n; i++) {
            hist[assign[i]]++;
        }

        // TODO: Try parallelizing this!
        printf("Copying actual data\n");
        // Copy the centroids
        std::vector<std::vector<float>> temp_clusters(config.numCentroids);
        std::vector<std::vector<vector_idx_t>> temp_vectorIds(config.numCentroids);
        for (int i = 0; i < config.numCentroids; i++) {
            std::vector<float> cluster(hist[i] * dim);
            temp_clusters[i] = cluster;
            std::vector<vector_idx_t> vectorId(hist[i]);
            temp_vectorIds[i] = vectorId;
            hist[i] = 0;
        }

        for (int i = 0; i < n; i++) {
            auto assignId = assign[i];
            auto idx = hist[assignId];
            auto cluster = temp_clusters[assignId];
            memcpy(cluster.data() + idx * dim, data + i * dim, dim * sizeof(float));
            temp_vectorIds[assignId][idx] = i + size;
            hist[assign[i]]++;
        }
        // Store the clusters
        for (int i = 0; i < config.numCentroids; i++) {
            std::vector<float> cluster = temp_clusters[i];
            clusters.push_back(std::move(cluster));
            std::vector<vector_idx_t> vectorId = temp_vectorIds[i];
            vectorIds.push_back(std::move(vectorId));
        }
        printf("Copying centroids\n");
        appendCentroids(clustering.centroids.data(), clustering.centroids.size());
        size += n;
    }

    void ReclusteringIndex::performReclustering() {
        printf("ReclusteringIndex::performReclustering\n");

    }

    void ReclusteringIndex::appendCentroids(const float *ctrds, size_t n) {
        auto curSize = centroids.size();
        centroids.resize(curSize + n * dim);
        memcpy(centroids.data() + curSize, ctrds, n * dim * sizeof(float));
    }

    int ReclusteringIndex::search(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results, int nProbes) {
        CHECK_ARGUMENT(centroids.size() > 0, "Centroids not initialized");
        CHECK_ARGUMENT(nProbes < centroids.size(), "Number of probes should be less than number of centroids");

        // TODO: Maybe parallelize it
        auto numCentroids = centroids.size() / dim;
        auto dc = getDistanceComputer(centroids.data(), numCentroids);
        dc->setQuery(query);
        std::vector<double> dists(numCentroids * dim);

        for (int i = 0; i < numCentroids; i++) {
            dc->computeDistance(i, &dists[i]);
        }

        // Sort the distances and get the top k indices
        std::vector<int> indices(numCentroids);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&dists](int i1, int i2) { return dists[i1] < dists[i2]; });

        for (int i = 0; i < nProbes; i++) {
            auto centroidId = indices[i];
            auto cluster = clusters[centroidId];
            auto ids = vectorIds[centroidId];
            CHECK_ARGUMENT((cluster.size() / dim) == ids.size() , "Clusters and vector ids should be same");
            auto clusterSize = ids.size();
            auto clusterDc = getDistanceComputer(cluster.data(), clusterSize);
            clusterDc->setQuery(query);
            for (int j = 0; j < clusterSize; j++) {
                double dist;
                clusterDc->computeDistance(j, &dist);
                if (results.size() < k || dist < results.top().dist) {
                    results.emplace(ids[j], dist);
                    if (results.size() > k) {
                        results.pop();
                    }
                }
            }
        }
    }
}
