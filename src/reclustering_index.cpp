#include "include/reclustering_index.h"

namespace orangedb {
    ReclusteringIndex::ReclusteringIndex(int dim, ReclusteringIndexConfig config, RandomGenerator *rg)
        : dim(dim), config(config), rg(rg), size(0) {}

    ReclusteringIndex::ReclusteringIndex(const std::string &file_path, RandomGenerator *rg) : rg(rg) {
        load_from_disk(file_path);
    }

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
            auto& cluster = temp_clusters[assignId];
            memcpy(cluster.data() + idx * dim, data + i * dim, dim * sizeof(float));
            temp_vectorIds[assignId][idx] = i + size;
            hist[assignId]++;
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

    void ReclusteringIndex::printStats() {
        printf("ReclusteringIndex::printStats\n");
        for (int i = 0; i < config.numCentroids; i++) {
            printf("Centroid %d has %zu vectors and reclustered %d times\n", i, clusters[i].size() / dim, reclusteringCount[i]);
        }
    }

    void ReclusteringIndex::flush_to_disk(const std::string &file_path) const {
        std::ofstream out(file_path, std::ios::binary);
        if (!out) {
            std::cerr << "Error opening file for writing: " << file_path << std::endl;
            return;
        }

        // Write the basic fields
        out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char *>(&size), sizeof(size));

        // Write the config
        out.write(reinterpret_cast<const char *>(&config.numCentroids), sizeof(config.numCentroids));
        out.write(reinterpret_cast<const char *>(&config.nIter), sizeof(config.nIter));
        out.write(reinterpret_cast<const char *>(&config.minCentroidSize), sizeof(config.minCentroidSize));
        out.write(reinterpret_cast<const char *>(&config.maxCentroidSize), sizeof(config.maxCentroidSize));
        out.write(reinterpret_cast<const char *>(&config.lambda), sizeof(config.lambda));
        out.write(reinterpret_cast<const char *>(&config.searchThreshold), sizeof(config.searchThreshold));
        out.write(reinterpret_cast<const char *>(&config.distanceType), sizeof(config.distanceType));
        out.write(reinterpret_cast<const char *>(&config.numReclusterCentroids), sizeof(config.numReclusterCentroids));

        // Write the centroids
        size_t centroidSize = centroids.size();
        out.write(reinterpret_cast<const char *>(&centroidSize), sizeof(centroidSize));
        out.write(reinterpret_cast<const char *>(centroids.data()), centroidSize * sizeof(float));

        // Write the clusters
        for (const auto &cluster : clusters) {
            size_t clusterSize = cluster.size();
            out.write(reinterpret_cast<const char *>(&clusterSize), sizeof(clusterSize));
            out.write(reinterpret_cast<const char *>(cluster.data()), clusterSize * sizeof(float));
        }

        // Write the vector ids
        for (const auto &vectorId : vectorIds) {
            size_t vectorIdSize = vectorId.size();
            out.write(reinterpret_cast<const char *>(&vectorIdSize), sizeof(vectorIdSize));
            out.write(reinterpret_cast<const char *>(vectorId.data()), vectorIdSize * sizeof(vector_idx_t));
        }

        // Write the re-clustering count
        for (const auto &count : reclusteringCount) {
            out.write(reinterpret_cast<const char *>(&count), sizeof(count));
        }
    }

    void ReclusteringIndex::load_from_disk(const std::string &file_path) {
        std::ifstream in(file_path, std::ios::binary);
        if (!in) {
            std::cerr << "Error opening file for reading: " << file_path << std::endl;
            return;
        }

        // Read the basic fields
        in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
        in.read(reinterpret_cast<char *>(&size), sizeof(size));

        // Read the config
        in.read(reinterpret_cast<char *>(&config.numCentroids), sizeof(config.numCentroids));
        in.read(reinterpret_cast<char *>(&config.nIter), sizeof(config.nIter));
        in.read(reinterpret_cast<char *>(&config.minCentroidSize), sizeof(config.minCentroidSize));
        in.read(reinterpret_cast<char *>(&config.maxCentroidSize), sizeof(config.maxCentroidSize));
        in.read(reinterpret_cast<char *>(&config.lambda), sizeof(config.lambda));
        in.read(reinterpret_cast<char *>(&config.searchThreshold), sizeof(config.searchThreshold));
        in.read(reinterpret_cast<char *>(&config.distanceType), sizeof(config.distanceType));
        in.read(reinterpret_cast<char *>(&config.numReclusterCentroids), sizeof(config.numReclusterCentroids));

        // Read the centroids
        size_t centroidSize;
        in.read(reinterpret_cast<char *>(&centroidSize), sizeof(centroidSize));
        centroids.resize(centroidSize);
        in.read(reinterpret_cast<char *>(centroids.data()), centroidSize * sizeof(float));

        // Read the clusters
        auto numClusters = centroidSize / dim;
        clusters.resize(numClusters);
        for (size_t i = 0; i < numClusters; i++) {
            size_t clusterSize;
            in.read(reinterpret_cast<char *>(&clusterSize), sizeof(clusterSize));
            clusters[i].resize(clusterSize);
            in.read(reinterpret_cast<char *>(clusters[i].data()), clusterSize * sizeof(float));
        }

        // Read the vector ids
        vectorIds.resize(numClusters);
        for (size_t i = 0; i < numClusters; i++) {
            size_t vectorIdSize;
            in.read(reinterpret_cast<char *>(&vectorIdSize), sizeof(vectorIdSize));
            vectorIds[i].resize(vectorIdSize);
            in.read(reinterpret_cast<char *>(vectorIds[i].data()), vectorIdSize * sizeof(vector_idx_t));
        }

        // Read the re-clustering count
        reclusteringCount.resize(numClusters);
        for (size_t i = 0; i < numClusters; i++) {
            in.read(reinterpret_cast<char *>(&reclusteringCount[i]), sizeof(reclusteringCount[i]));
        }
    }

    void ReclusteringIndex::performReclustering() {
        printf("ReclusteringIndex::performReclustering\n");
        // Pick a centroid with minimum reclustering count.
        // TODO: Pick a random one from the list of centroids with minimum re-clustering count.
        int centroidId = 0;
        long minCount = std::numeric_limits<long>::max();
        for (int i = 0; i < config.numCentroids; i++) {
            if (reclusteringCount[i] < minCount) {
                centroidId = i;
                minCount = reclusteringCount[i];
            }
        }
        printf("Selected centroid id %d with reclustering count %ld\n", centroidId, minCount);

        // Find numReclusterCentroids closest centroids.
        std::priority_queue<NodeDistFarther> ctrdsToRecluster;
        auto numCentroids = centroids.size() / dim;
        auto dc = getDistanceComputer(centroids.data(), numCentroids);
        dc->setQuery(centroids.data() + centroidId * dim);
        ctrdsToRecluster.emplace(centroidId, 0);
        for (int i = 0; i < numCentroids; i++) {
            if (i == centroidId) continue;
            double dist;
            dc->computeDistance(i, &dist);
            ctrdsToRecluster.emplace(i, dist);
        }
        printf("Computed distances from centroid %d to all others.\n", centroidId);

        // Determine which clusters will be reclustered.
        std::vector<vector_idx_t> clusterIdToRecluster;
        vector_idx_t totalVecToRecluster = 0;
        while (!ctrdsToRecluster.empty()) {
            auto clusterId = ctrdsToRecluster.top();
            ctrdsToRecluster.pop();
            clusterIdToRecluster.push_back(clusterId.id);
            totalVecToRecluster += clusters[clusterId.id].size() / dim;
            if (clusterIdToRecluster.size() >= config.numReclusterCentroids) {
                break;
            }
        }
        printf("Selected %zu clusters for reclustering with a total of %llu vectors\n",
               clusterIdToRecluster.size(), (unsigned long long) totalVecToRecluster);

        // Copy all the vectors and their ids from the selected clusters into
        // contiguous storage for reclustering.
        std::vector<float> reclusterVectors(totalVecToRecluster * dim);
        std::vector<vector_idx_t> reclusterVectorIds(totalVecToRecluster);
        uint64_t currentSize = 0;
        for (auto clusterId: clusterIdToRecluster) {
            auto &cluster = clusters[clusterId];
            auto &vectorId = vectorIds[clusterId];
            memcpy(reclusterVectors.data() + (currentSize * dim),
                   cluster.data(), cluster.size() * sizeof(float));
            memcpy(reclusterVectorIds.data() + currentSize,
                   vectorId.data(), vectorId.size() * sizeof(vector_idx_t));
            currentSize += vectorId.size();
        }
        printf("Copied recluster vectors into contiguous storage.\n");

        // Now perform reclustering on the combined vectors.
        // The number of new clusters is set to the number of selected clusters.
        printf("Performing reclustering on %llu vectors divided into %zu clusters.\n",
               (unsigned long long) totalVecToRecluster, clusterIdToRecluster.size());
        Clustering clustering(dim, clusterIdToRecluster.size(), config.nIter,
                              config.minCentroidSize, config.maxCentroidSize, config.lambda);
        clustering.initCentroids(reclusterVectors.data(), totalVecToRecluster);
        printf("Initialized new centroids for reclustering.\n");
        clustering.train(reclusterVectors.data(), totalVecToRecluster);
        printf("Trained new centroids using k-means.\n");

        // Obtain the new cluster assignments.
        auto reclusterAssign = new int32_t[totalVecToRecluster];
        clustering.assignCentroids(reclusterVectors.data(), totalVecToRecluster, reclusterAssign);
        printf("Assigned new cluster labels for reclustered vectors.\n");

        // Partition the reclustered vectors into new clusters.
        int newClusters = clusterIdToRecluster.size();
        std::vector<int> newClusterSizes(newClusters, 0);
        for (size_t i = 0; i < totalVecToRecluster; i++) {
            int label = reclusterAssign[i];
            newClusterSizes[label]++;
        }

        // Prepare temporary storage for the new clusters and their vector ids.
        std::vector<std::vector<float> > newClustersData(newClusters);
        std::vector<std::vector<vector_idx_t> > newClustersVectorIds(newClusters);
        for (int j = 0; j < newClusters; j++) {
            newClustersData[j].resize(newClusterSizes[j] * dim);
            newClustersVectorIds[j].resize(newClusterSizes[j]);
            newClusterSizes[j] = 0; // Reset as an insertion counter.
        }

        // Split the reclustered vectors into their new cluster assignments.
        for (size_t i = 0; i < totalVecToRecluster; i++) {
            int label = reclusterAssign[i];
            int idx = newClusterSizes[label];
            memcpy(newClustersData[label].data() + idx * dim,
                   reclusterVectors.data() + i * dim,
                   dim * sizeof(float));
            newClustersVectorIds[label][idx] = reclusterVectorIds[i];
            newClusterSizes[label]++;
        }
        printf("Partitioned reclustered vectors into new clusters.\n");

        // Update the original clusters and centroids with the results of reclustering.
        // Each new cluster replaces the original cluster corresponding to its selection.
        for (int j = 0; j < newClusters; j++) {
            int origClusterId = clusterIdToRecluster[j];
            auto oldSize = clusters[origClusterId].size() / dim;
            clusters[origClusterId] = std::move(newClustersData[j]);
            vectorIds[origClusterId] = std::move(newClustersVectorIds[j]);
            // Update the centroid for this original cluster.
            memcpy(centroids.data() + origClusterId * dim,
                   clustering.centroids.data() + j * dim,
                   dim * sizeof(float));
            // Increment the reclustering counter.
            reclusteringCount[origClusterId]++;
            printf("Updated cluster %d: new size = %zu, old size = %zu, reclustering count = %ld\n",
                   origClusterId, clusters[origClusterId].size() / dim, oldSize, reclusteringCount[origClusterId]);
        }

        delete[] reclusterAssign;
        printf("Reclustering completed.\n");
    }

    void ReclusteringIndex::appendCentroids(const float *ctrds, size_t n) {
        auto curSize = centroids.size();
        reclusteringCount.resize(curSize + n);
        centroids.resize(curSize + n);
        memcpy(centroids.data() + curSize, ctrds, n * sizeof(float));
        for (int i = 0; i < n; i++) {
            reclusteringCount[curSize + i] = 0;
        }
    }

    void ReclusteringIndex::search(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results, int nProbes,
                                   ReclusteringIndexStats &stats) {
        CHECK_ARGUMENT(centroids.size() > 0, "Centroids not initialized");
        CHECK_ARGUMENT(nProbes < centroids.size(), "Number of probes should be less than number of centroids");
        stats.numDistanceComp = 0;

        // TODO: Maybe parallelize it
        auto numCentroids = centroids.size() / dim;
        auto dc = getDistanceComputer(centroids.data(), numCentroids);
        dc->setQuery(query);
        std::vector<double> dists(numCentroids);

        for (int i = 0; i < numCentroids; i++) {
            dc->computeDistance(i, &dists[i]);
            stats.numDistanceComp++;
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
                stats.numDistanceComp++;
                if (results.size() <= k || dist < results.top().dist) {
                    results.emplace(ids[j], dist);
                    if (results.size() > k) {
                        results.pop();
                    }
                }
            }
        }
    }
}
