#include "include/incremental_index.h"

namespace orangedb {
    IncrementalIndex::IncrementalIndex(int dim, IncrementalIndexConfig config,
                                       RandomGenerator *rg) : dim(dim), config(config), rg(rg), size(0) {
    }

    IncrementalIndex::IncrementalIndex(const std::string &file_path, RandomGenerator *rg) : rg(rg) {
        load_from_disk(file_path);
    }

    void IncrementalIndex::insert(float *data, size_t n) {
        // Create set of clusters if not already created
        // Otherwise assign vectors to already created clusters
        if (clusters.empty()) {
            // First time inserting data
            insertFirstTime(data, n);
            return;
        }

        // TODO: Figure out how to do it in batch. First let's benchmark the splitting stuff!
        // Find the closest mega centroid
        std::vector<int32_t> megaAssign(n);
        assignMegaCentroids(data, n, megaAssign.data());

        // Assign micro centroids
        std::vector<int32_t> microAssign(n);
        assignMicroCentroids(data, n, megaAssign.data(), microAssign.data());

        // Now we need to assign the vectors to the clusters
        unordered_map<int32_t, int32_t> microHist;
        for (int i = 0; i < n; i++) {
            auto microId = microAssign[i];
            microHist[microId]++;
        }

        // Now resize the clusters
        for (auto &[microId, size]: microHist) {
            auto currSize = clusters[microId].size() / dim;
            clusters[microId].resize((currSize + size) * dim);
            vectorIds[microId].resize(currSize + size);
            microHist[microId] = currSize;
        }

        // Now copy the data to the clusters
        for (int i = 0; i < n; i++) {
            auto microId = microAssign[i];
            auto currSize = microHist[microId];
            memcpy(clusters[microId].data() + currSize * dim, data + i * dim, dim * sizeof(float));
            vectorIds[microId][currSize] = i + size;
            microHist[microId]++;
        }

        size += n;
    }

    void IncrementalIndex::split() {
        // Pick the big mega clusters that violates the threshold
        auto megaClusterToSplit = std::vector<int32_t>();
        auto megaCentroidsSize = megaCentroids.size() / dim;
        for (int i = 0; i < megaCentroidsSize; i++) {
            auto size = getMegaClusterSize(i);
            if (size > config.maxMegaClusterSize) {
                megaClusterToSplit.push_back(i);
            }
        }

        if (megaClusterToSplit.empty()) {
            printf("No mega clusters to split\n");
            return;
        }

        // Split the mega clusters
        for (auto megaClusterId: megaClusterToSplit) {
            // Split the mega cluster
            splitMegaCluster(megaClusterId);
        }
    }

    void IncrementalIndex::printStats() {
        printf("IncrementalIndex::printStats\n");
        // Print the number of mega clusters
        printf("Number of mega clusters: %zu\n", megaCentroids.size() / dim);
        printf("Number of micro clusters: %zu\n", clusters.size());
        for (int i = 0; i < clusters.size(); i++) {
            printf("Centroid %d has %zu vectors\n", i, clusters[i].size() / dim);
        }
    }

    void IncrementalIndex::splitMegaCluster(int megaClusterId) {
        // Split into two using k means
        // Collect all the vectors
        size_t totalVectors = 0;
        for (auto microId: megaCentroidAssignment[megaClusterId]) {
            totalVectors += vectorIds[microId].size();
        }

        // TODO: Ideally instead of copying the data, we should just use the pointers using some sort of view
        std::vector<float> temp_vectors(totalVectors * dim);
        std::vector<vector_idx_t> temp_vectorIds(totalVectors);

        // Copy the data to the temp vectors
        size_t idx = 0;
        for (auto microId: megaCentroidAssignment[megaClusterId]) {
            auto &vectorId = vectorIds[microId];
            auto size = vectorId.size();
            CHECK_ARGUMENT(size * dim == clusters[microId].size(), "Size mismatch");
            memcpy(temp_vectors.data() + idx * dim, clusters[microId].data(), size * dim * sizeof(float));
            memcpy(temp_vectorIds.data() + idx, vectorId.data(), size * sizeof(vector_idx_t));
            idx += size;
        }

        // TODO: Move some vectors based on edge cases
        auto minCentroidSize = totalVectors / 2.1;
        auto maxCentroidSize = totalVectors / 1.5;
        Clustering megaClustering(dim, 2, config.nIter, minCentroidSize, maxCentroidSize,
                                  0);
        megaClustering.initCentroids(temp_vectors.data(), totalVectors);
        megaClustering.train(temp_vectors.data(), totalVectors);

        std::vector<int32_t> megaAssign(totalVectors);
        megaClustering.assignCentroids(temp_vectors.data(), totalVectors, megaAssign.data());

        // Create two temp clusters
        auto megaClusterASize = 0;
        auto megaClusterBSize = 0;
        for (int i = 0; i < totalVectors; i++) {
            if (megaAssign[i] == 0) {
                megaClusterASize++;
            } else {
                megaClusterBSize++;
            }
        }

        // Copy the vectors
        std::vector<float> megaClusterA(megaClusterASize * dim);
        std::vector<float> megaClusterB(megaClusterBSize * dim);
        std::vector<vector_idx_t> megaClusterAIds(megaClusterASize);
        std::vector<vector_idx_t> megaClusterBIds(megaClusterBSize);

        megaClusterASize = 0;
        megaClusterBSize = 0;
        for (int i = 0; i < totalVectors; i++) {
            if (megaAssign[i] == 0) {
                memcpy(megaClusterA.data() + megaClusterASize * dim, temp_vectors.data() + i * dim,
                       dim * sizeof(float));
                megaClusterAIds[megaClusterASize] = temp_vectorIds[i];
                megaClusterASize++;
            } else {
                memcpy(megaClusterB.data() + megaClusterBSize * dim, temp_vectors.data() + i * dim,
                       dim * sizeof(float));
                megaClusterBIds[megaClusterBSize] = temp_vectorIds[i];
                megaClusterBSize++;
            }
        }

        // Cluster mega cluster A
        auto minCentroidSizeA = megaClusterASize / (config.numCentroids * 1.3);
        auto maxCentroidSizeA = megaClusterASize / (config.numCentroids * 0.9);
        Clustering megaClusteringA(dim, config.numCentroids, config.nIter, minCentroidSizeA,
                                   maxCentroidSizeA, 0);
        // TODO: Maybe initialize based on previous clusters
        megaClusteringA.initCentroids(megaClusterA.data(), megaClusterASize);
        megaClusteringA.train(megaClusterA.data(), megaClusterASize);
        storeMegaCluster(megaClusterId, megaClustering.centroids.data(),
                         &megaClusteringA, megaClusterA.data(), megaClusterAIds.data(), megaClusterASize);

        // Cluster mega cluster B
        auto minCentroidSizeB = megaClusterBSize / (config.numCentroids * 1.3);
        auto maxCentroidSizeB = megaClusterBSize / (config.numCentroids * 0.9);
        Clustering megaClusteringB(dim, config.numCentroids, config.nIter, minCentroidSizeB,
                                   maxCentroidSizeB,
                                   0);
        megaClusteringB.initCentroids(megaClusterB.data(), megaClusterBSize);
        megaClusteringB.train(megaClusterB.data(), megaClusterBSize);
        appendMegaCluster(megaClusteringB.centroids.data(),
                          &megaClusteringB, megaClusterB.data(), megaClusterBIds.data(), megaClusterBSize);
    }

    void IncrementalIndex::storeMegaCluster(int oldMegaClusterId, const float *newMegaCentroid,
                                            Clustering *microClustering, const float *data,
                                            const vector_idx_t *vectorIds, size_t n) {
        // 1) Update the mega‐centroid coordinates
        memcpy(
            megaCentroids.data() + oldMegaClusterId * dim,
            newMegaCentroid,
            dim * sizeof(float)
        );

        auto currClusterIds = megaCentroidAssignment[oldMegaClusterId];

        // Obtain the new cluster assignments.
        std::vector<int32_t> reclusterAssign(n);
        microClustering->assignCentroids(data, n, reclusterAssign.data());
        printf("Assigned new cluster labels for reclustered vectors.\n");

        // Partition the reclustered vectors into new clusters.
        int newClusters = microClustering->getNumCentroids();
        std::vector<int> newClusterSizes(newClusters, 0);
        for (size_t i = 0; i < n; i++) {
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
        for (size_t i = 0; i < n; i++) {
            int label = reclusterAssign[i];
            int idx = newClusterSizes[label];
            memcpy(newClustersData[label].data() + idx * dim,
                   data + i * dim,
                   dim * sizeof(float));
            newClustersVectorIds[label][idx] = vectorIds[i];
            newClusterSizes[label]++;
        }
        printf("Partitioned reclustered vectors into new clusters.\n");

        CHECK_ARGUMENT(currClusterIds.size() == newClusters, "Size mismatch");

        // Update the original clusters and centroids with the results of reclustering.
        // Each new cluster replaces the original cluster corresponding to its selection.
        for (int j = 0; j < newClusters; j++) {
            int origClusterId = currClusterIds[j];
            auto oldSize = clusters[origClusterId].size() / dim;
            clusters[origClusterId] = std::move(newClustersData[j]);
            this->vectorIds[origClusterId] = std::move(newClustersVectorIds[j]);
            // Update the centroid for this original cluster.
            memcpy(microCentroids.data() + origClusterId * dim,
                   microClustering->centroids.data() + j * dim,
                   dim * sizeof(float));
            // Increment the reclustering counter.
            printf("Updated cluster %d: new size = %zu, old size = %zu\n",
                   origClusterId, clusters[origClusterId].size() / dim, oldSize);
        }
    }

    void IncrementalIndex::appendMegaCluster(const float *newMegaCentroid,
                                             Clustering *microClustering, const float *data,
                                             const vector_idx_t *vectorIds, size_t n) {
        // Increase the size of mega centroids
        auto curSize = megaCentroids.size() / dim;
        megaCentroids.resize((curSize + 1) * dim);
        memcpy(megaCentroids.data() + (curSize * dim), newMegaCentroid, dim * sizeof(float));

        auto currClusterSize = clusters.size();

        // Obtain the new cluster assignments.
        std::vector<int32_t> reclusterAssign(n);
        microClustering->assignCentroids(data, n, reclusterAssign.data());
        printf("Assigned new cluster labels for reclustered vectors.\n");

        // Partition the reclustered vectors into new clusters.
        int newClusters = microClustering->getNumCentroids();
        std::vector<int> newClusterSizes(newClusters, 0);
        for (size_t i = 0; i < n; i++) {
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
        for (size_t i = 0; i < n; i++) {
            int label = reclusterAssign[i];
            int idx = newClusterSizes[label];
            memcpy(newClustersData[label].data() + idx * dim,
                   data + i * dim,
                   dim * sizeof(float));
            newClustersVectorIds[label][idx] = vectorIds[i];
            newClusterSizes[label]++;
        }
        printf("Partitioned reclustered vectors into new clusters.\n");

        // Append the new clusters to the mega cluster
        std::vector<vector_idx_t> microAssign(config.numCentroids);
        // Store the clusters
        for (int i = 0; i < config.numCentroids; i++) {
            std::vector<float> cluster = newClustersData[i];
            clusters.push_back(std::move(cluster));
            std::vector<vector_idx_t> vectorId = newClustersVectorIds[i];
            this->vectorIds.push_back(std::move(vectorId));
            microAssign[i] = currClusterSize + i;
        }
        appendCentroids(microClustering->centroids.data(), microClustering->centroids.size());
        megaCentroidAssignment.push_back(std::move(microAssign));
    }


    size_t IncrementalIndex::getMegaClusterSize(int megaCentroidId) {
        size_t size = 0;
        for (auto microId: megaCentroidAssignment[megaCentroidId]) {
            size += vectorIds[microId].size();
        }
        return size;
    }

    void IncrementalIndex::assignMegaCentroids(const float *data, int n, int32_t *assign) {
        auto numMegaCentroids = megaCentroids.size() / dim;
        auto dc = getDistanceComputer(megaCentroids.data(), numMegaCentroids);
#pragma omp parallel
        {
            auto localDc = dc->clone();
#pragma omp for
            for (int i = 0; i < n; i++) {
                localDc->setQuery(data + i * dim);
                double minDistance = std::numeric_limits<double>::max();
                vector_idx_t j = 0, minId = 0;
                while (j + 4 < numMegaCentroids) {
                    double dists[4];
                    vector_idx_t idx[4] = {j, j + 1, j + 2, j + 3};
                    localDc->batchComputeDistances(idx, dists, 4);
                    for (int l = 0; l < 4; l++) {
                        auto recomputedDist = dists[l];
                        if (recomputedDist < minDistance) {
                            minDistance = recomputedDist;
                            minId = j + l;
                        }
                    }
                    j += 4;
                }

                for (vector_idx_t l = j; l < numMegaCentroids; l++) {
                    double d;
                    localDc->computeDistance(l, &d);
                    auto recomputedDist = d;
                    if (recomputedDist < minDistance) {
                        minDistance = recomputedDist;
                        minId = l;
                    }
                }
                assign[i] = minId;
            }
        }
    }

    void IncrementalIndex::assignMicroCentroids(const float *data, int n, const int32_t *megaAssign,
                                                int32_t *microAssign) {
        auto numMicroCentroids = microCentroids.size() / dim;
        auto dc = getDistanceComputer(microCentroids.data(), numMicroCentroids);
#pragma omp parallel
        {
            auto localDc = dc->clone();
#pragma omp for
            for (auto i = 0; i < n; i++) {
                localDc->setQuery(data + i * dim);
                double minDistance = std::numeric_limits<double>::max();
                vector_idx_t minId = 0;
                auto megaCentroidId = megaAssign[i];
                auto microCentroidIds = megaCentroidAssignment[megaCentroidId];
                for (auto microCentroidId: microCentroidIds) {
                    double d;
                    localDc->computeDistance(microCentroidId, &d);
                    if (d < minDistance) {
                        minDistance = d;
                        minId = microCentroidId;
                    }
                }
                microAssign[i] = minId;
            }
        }
    }

    void IncrementalIndex::insertFirstTime(float *data, size_t n) {
        printf("IncrementalIndex::insert\n");
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

        printf("Copying actual data\n");
        // Copy the centroids
        std::vector<std::vector<float> > temp_clusters(config.numCentroids);
        std::vector<std::vector<vector_idx_t> > temp_vectorIds(config.numCentroids);
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
            auto &cluster = temp_clusters[assignId];
            memcpy(cluster.data() + idx * dim, data + i * dim, dim * sizeof(float));
            temp_vectorIds[assignId][idx] = i + size;
            hist[assignId]++;
        }

        std::vector<vector_idx_t> microAssign(config.numCentroids);
        // Store the clusters
        for (int i = 0; i < config.numCentroids; i++) {
            std::vector<float> cluster = temp_clusters[i];
            clusters.push_back(std::move(cluster));
            std::vector<vector_idx_t> vectorId = temp_vectorIds[i];
            vectorIds.push_back(std::move(vectorId));
            microAssign[i] = i;
        }

        printf("Copying centroids\n");
        appendCentroids(clustering.centroids.data(), clustering.centroids.size());
        size += n;

        // Calculate the first mega centroid i.e. average of all vectors
        megaCentroids.resize(dim);
        auto megaCentroid = megaCentroids.data();
        memset(megaCentroid, 0, dim * sizeof(float));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                megaCentroid[j] += data[i * dim + j];
            }
        }
        for (int j = 0; j < dim; j++) {
            megaCentroid[j] /= n;
        }
        megaCentroidAssignment.push_back(std::move(microAssign));
    }

    void IncrementalIndex::appendCentroids(const float *ctrds, size_t n) {
        auto curSize = microCentroids.size();
        microCentroids.resize(curSize + n);
        memcpy(microCentroids.data() + curSize, ctrds, n * sizeof(float));
    }

    void IncrementalIndex::flush_to_disk(const std::string &file_path) const {
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
        out.write(reinterpret_cast<const char *>(&config.maxMegaClusterSize), sizeof(config.maxMegaClusterSize));

        // Write the mega centroids
        size_t megaCentroidSize = megaCentroids.size();
        out.write(reinterpret_cast<const char *>(&megaCentroidSize), sizeof(megaCentroidSize));
        out.write(reinterpret_cast<const char *>(megaCentroids.data()), megaCentroidSize * sizeof(float));

        // Write megaCentroidAssignment
        size_t megaCentroidAssignmentSize = megaCentroidAssignment.size();
        out.write(reinterpret_cast<const char *>(&megaCentroidAssignmentSize), sizeof(megaCentroidAssignmentSize));
        for (const auto &assignment: megaCentroidAssignment) {
            size_t assignmentSize = assignment.size();
            out.write(reinterpret_cast<const char *>(&assignmentSize), sizeof(assignmentSize));
            out.write(reinterpret_cast<const char *>(assignment.data()), assignmentSize * sizeof(vector_idx_t));
        }

        // Write the micro centroids
        size_t microCentroidSize = microCentroids.size();
        out.write(reinterpret_cast<const char *>(&microCentroidSize), sizeof(microCentroidSize));
        out.write(reinterpret_cast<const char *>(microCentroids.data()), microCentroidSize * sizeof(float));

        // Write the clusters
        size_t numClusters = clusters.size();
        out.write(reinterpret_cast<const char *>(&numClusters), sizeof(numClusters));
        for (const auto &cluster: clusters) {
            size_t clusterSize = cluster.size();
            out.write(reinterpret_cast<const char *>(&clusterSize), sizeof(clusterSize));
            out.write(reinterpret_cast<const char *>(cluster.data()), clusterSize * sizeof(float));
        }

        // Write the vector ids
        for (const auto &vectorId: vectorIds) {
            size_t vectorIdSize = vectorId.size();
            out.write(reinterpret_cast<const char *>(&vectorIdSize), sizeof(vectorIdSize));
            out.write(reinterpret_cast<const char *>(vectorId.data()), vectorIdSize * sizeof(vector_idx_t));
        }
    }

    void IncrementalIndex::load_from_disk(const std::string &file_path) {
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
        in.read(reinterpret_cast<char *>(&config.maxMegaClusterSize), sizeof(config.maxMegaClusterSize));

        // Read the mega centroids
        size_t megaCentroidSize;
        in.read(reinterpret_cast<char *>(&megaCentroidSize), sizeof(megaCentroidSize));
        megaCentroids.resize(megaCentroidSize);
        in.read(reinterpret_cast<char *>(megaCentroids.data()), megaCentroidSize * sizeof(float));

        // Read the megaCentroidAssignment
        size_t megaCentroidAssignmentSize;
        in.read(reinterpret_cast<char *>(&megaCentroidAssignmentSize), sizeof(megaCentroidAssignmentSize));
        megaCentroidAssignment.clear();
        for (size_t i = 0; i < megaCentroidAssignmentSize; i++) {
            size_t assignmentSize;
            in.read(reinterpret_cast<char *>(&assignmentSize), sizeof(assignmentSize));
            std::vector<vector_idx_t> assignment(assignmentSize);
            in.read(reinterpret_cast<char *>(assignment.data()), assignmentSize * sizeof(vector_idx_t));
            megaCentroidAssignment.push_back(std::move(assignment));
        }

        // Read the micro centroids
        size_t microCentroidSize;
        in.read(reinterpret_cast<char *>(&microCentroidSize), sizeof(microCentroidSize));
        microCentroids.resize(microCentroidSize);
        in.read(reinterpret_cast<char *>(microCentroids.data()), microCentroidSize * sizeof(float));

        // Read the clusters
        size_t numClusters;
        in.read(reinterpret_cast<char *>(&numClusters), sizeof(numClusters));
        clusters.clear();
        clusters.resize(numClusters);
        for (size_t i = 0; i < numClusters; i++) {
            size_t clusterSize;
            in.read(reinterpret_cast<char *>(&clusterSize), sizeof(clusterSize));
            clusters[i].resize(clusterSize);
            in.read(reinterpret_cast<char *>(clusters[i].data()), clusterSize * sizeof(float));
        }

        // Read the vector ids for each cluster
        vectorIds.clear();
        vectorIds.resize(numClusters);
        for (size_t i = 0; i < numClusters; i++) {
            size_t vectorIdSize;
            in.read(reinterpret_cast<char *>(&vectorIdSize), sizeof(vectorIdSize));
            vectorIds[i].resize(vectorIdSize);
            in.read(reinterpret_cast<char *>(vectorIds[i].data()), vectorIdSize * sizeof(vector_idx_t));
        }
        in.close();
    }

    void IncrementalIndex::search(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results,
                                  int nMegaProbes, int nMicroProbes, IncrementalIndexStats &stats) {
        // Find 5 closest mega centroids
        std::vector<int32_t> megaAssign(nMegaProbes);
        std::vector<double> megaDists(nMegaProbes);
        auto numMegaCentroids = megaCentroids.size() / dim;
        auto dc = getDistanceComputer(megaCentroids.data(), numMegaCentroids);
        dc->setQuery(query);
        for (int i = 0; i < numMegaCentroids; i++) {
            double d;
            dc->computeDistance(i, &d);
            stats.numDistanceComp++;
            if (i < nMegaProbes) {
                megaDists[i] = d;
                megaAssign[i] = i;
            } else {
                auto maxId = ranges::max_element(megaDists) - megaDists.begin();
                if (d < megaDists[maxId]) {
                    megaDists[maxId] = d;
                    megaAssign[maxId] = i;
                }
            }
        }

        auto numMicroCentroids = microCentroids.size() / dim;
        dc = getDistanceComputer(microCentroids.data(), numMicroCentroids);
        dc->setQuery(query);

        // Now find the closest micro centroids
        std::priority_queue<NodeDistCloser> closestMicro;
        for (int i = 0; i < nMegaProbes; i++) {
            auto megaId = megaAssign[i];
            auto microIds = megaCentroidAssignment[megaId];
            for (auto microId: microIds) {
                double d;
                dc->computeDistance(microId, &d);
                stats.numDistanceComp++;
                if (closestMicro.size() < nMicroProbes || d < closestMicro.top().dist) {
                    closestMicro.emplace(microId, d);
                    if (closestMicro.size() > nMicroProbes) {
                        closestMicro.pop();
                    }
                }
            }
        }

        // Now we have the closest micro centroids, let's find the closest vectors
        while (!closestMicro.empty()) {
            auto microId = closestMicro.top().id;
            closestMicro.pop();
            auto cluster = clusters[microId];
            auto ids = vectorIds[microId];
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

    // Silhouette metric to measure the quality of the clustering i.e. how well the clusters are separated
    // How it works:
    // 1. For each point in the dataset, compute the average distance to all other points in the same cluster (a(i)).
    // 2. For each point in the dataset, compute the min distance to all points in the nearest cluster (b(i)).
    // 3. For each point in the dataset, compute the silhouette score s(i) = (b(i) - a(i)) / max(a(i), b(i)).
    // How to interpret the results:
    // The silhouette score ranges from -1 to 1, where 1 indicates that the point is well clustered,
    // 0 indicates that the point is on the boundary of two clusters,
    // and -1 indicates that the point is misclassified.
    // Note: For now we are only computing the silhouette score for the micro centroids
    double IncrementalIndex::computeSilhouetteMetricOnMicroCentroids() {
        // TODO: Maybe calculate the silhouette score for each mega cluster
        auto numCentroids = microCentroids.size() / dim;
        CHECK_ARGUMENT(numCentroids == clusters.size(), "Number of centroids must equal number of clusters");

        double totalSilhouette = 0.0;
        long long totalPoints = 0;

        // Parallelize across clusters (and their points).
#pragma omp parallel for reduction(+: totalSilhouette, totalPoints) schedule(dynamic)
        for (int i = 0; i < numCentroids; i++) {
            const auto &cluster = clusters[i];
            auto numPoints = cluster.size() / dim;
            auto dc = getDistanceComputer(cluster.data(), numPoints);

            for (int j = 0; j < numPoints; j++) {
                const float *curCluster = microCentroids.data() + i * dim;
                dc->setQuery(curCluster);

                // 1) a = distance to own centroid
                double a = 0;
                dc->computeDistance(j, &a);

                // 2) b = min distance to any other centroid
                double b = std::numeric_limits<double>::infinity();
                for (int k = 0; k < numCentroids; k++) {
                    if (k == i) continue;
                    const float *cO = microCentroids.data() + k * dim;
                    dc->setQuery(cO);
                    double dist;
                    dc->computeDistance(j, &dist);
                    b = std::min(b, dist);
                }

                // 3) silhouette for this point
                double m = std::max(a, b);
                double s = (m > 0.0) ? (b - a) / m : 0.0;
                totalSilhouette += s;
                totalPoints += 1;
            }
        }

        return (totalPoints > 0)
                   ? float(totalSilhouette / double(totalPoints))
                   : 0.0f;
    }
}
