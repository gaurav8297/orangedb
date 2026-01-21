#include "include/reclustering_index.h"

#include "faiss/IndexFlat.h"

namespace orangedb {
    ReclusteringIndex::ReclusteringIndex(int dim, ReclusteringIndexConfig config, RandomGenerator *rg)
        : dim(dim), config(config), size(0), rg(rg) {
        quantizer = std::make_unique<SQ8Bit>(dim);
    }

    ReclusteringIndex::ReclusteringIndex(const std::string &file_path, RandomGenerator *rg) : rg(rg) {
        load_from_disk(file_path);
    }

    void ReclusteringIndex::insert(float *data, size_t n) {
        printf("ReclusteringIndex::insert\n");
        // Create the vector ids
        std::vector<vector_idx_t> vectorIds(n);
        for (size_t i = 0; i < n; i++) {
            vectorIds[i] = i + size;
        }

        // Run clustering to create mini clusters
        std::vector<float> centroids;
        std::vector<std::vector<float> > clusters;
        std::vector<std::vector<vector_idx_t> > clusterVectorIds;
        clusterData(data, vectorIds.data(), n, config.newMiniCentroidSize, centroids, clusters, clusterVectorIds);

        // Store the mini clusters into the index buffering space
        for (int i = 0; i < clusters.size(); i++) {
            std::vector<float> cluster = clusters[i];
            newMiniClusters.push_back(std::move(cluster));
            std::vector<vector_idx_t> vectorId = clusterVectorIds[i];
            newMiniClusterVectorIds.push_back(std::move(vectorId));
        }

        auto curMiniCtrdSize = newMiniCentroids.size();
        newMiniCentroids.resize(curMiniCtrdSize + centroids.size());
        memcpy(newMiniCentroids.data() + curMiniCtrdSize, centroids.data(), centroids.size() * sizeof(float));
        size += n;
        updateTotalDataWrittenByUser(n);

        printf("Added %lu new mini centroids!\n", newMiniCentroids.size() / dim);
    }

    void ReclusteringIndex::trainQuant(float *data, size_t n) {
        // Allocate for normalized vectors
        std::vector<float> normalizedVector(dim);
        size_t totalTrained = 0;
        // Quantize the new mini centroids
        for (size_t i = 0; i < n; i++) {
            if (rg->randFloat() > config.quantizationTrainPercentage) {
                // Skip this vector
                continue;
            }
            // Train using this vector
            if (config.distanceType == COSINE) {
                normalize_vectors(data + i * dim, dim, 1, normalizedVector.data());
                quantizer->batch_train(1, normalizedVector.data());
            } else {
                quantizer->batch_train(1, data + i * dim);
            }
            totalTrained++;
        }
        printf("ReclusteringIndex::trainQuant trained on %lu vectors\n", totalTrained);
        quantizer->finalize_train();
    }

    void ReclusteringIndex::simpleInsertWithoutClustering(float *data, size_t n) {
        std::vector<vector_idx_t> vectorIds(n);
        for (size_t i = 0; i < n; i++) {
            vectorIds[i] = i + size;
        }

        // Calculate how many mini clusters we can create
        int numMiniClusters = (n + config.miniCentroidSize - 1) / config.miniCentroidSize;

        std::vector<float> newMiniCentroids;
        std::vector<std::vector<float>> newMiniClusters;
        std::vector<std::vector<vector_idx_t>> newMiniClusterVectorIds;
        newMiniCentroids.reserve(numMiniClusters * dim);
        newMiniClusters.reserve(numMiniClusters);
        newMiniClusterVectorIds.reserve(numMiniClusters);

        // Create mini clusters by taking mean of each miniCentroidSize batch
        for (size_t batchStart = 0; batchStart < n; batchStart += config.miniCentroidSize) {
            size_t batchEnd = std::min(batchStart + config.miniCentroidSize, n);
            size_t batchSize = batchEnd - batchStart;

            // Calculate mean centroid for this batch
            std::vector<float> meanCentroid(dim, 0.0f);
            for (size_t i = batchStart; i < batchEnd; i++) {
                for (int j = 0; j < dim; j++) {
                    meanCentroid[j] += data[i * dim + j];
                }
            }
            float norm = 1.0f / batchSize;
            for (int j = 0; j < dim; j++) {
                meanCentroid[j] *= norm;
            }

            // Store the mini centroid
            newMiniCentroids.insert(newMiniCentroids.end(), meanCentroid.begin(), meanCentroid.end());

            // Store the vectors in this mini cluster
            std::vector<float> clusterVectors;
            clusterVectors.reserve(batchSize * dim);
            std::vector<vector_idx_t> clusterVectorIds;
            clusterVectorIds.reserve(batchSize);

            for (size_t i = batchStart; i < batchEnd; i++) {
                clusterVectors.insert(clusterVectors.end(), data + i * dim, data + (i + 1) * dim);
                clusterVectorIds.push_back(vectorIds[i]);
            }

            newMiniClusters.push_back(std::move(clusterVectors));
            newMiniClusterVectorIds.push_back(std::move(clusterVectorIds));
        }

        // Assign mini cluster unique ids
        auto curMiniClusterSize = miniCentroids.size() / dim;
        auto newMiniClusterSize = newMiniCentroids.size() / dim;
        std::vector<vector_idx_t> newMiniClusterIds(newMiniClusterSize);
        for (size_t i = 0; i < newMiniClusterSize; i++) {
            newMiniClusterIds[i] = curMiniClusterSize + i;
        }

        // Copy mini centroids and clusters to main storage
        miniCentroids.resize((curMiniClusterSize + newMiniClusterSize) * dim);
        memcpy(miniCentroids.data() + curMiniClusterSize * dim, newMiniCentroids.data(),
               newMiniCentroids.size() * sizeof(float));
        miniClusters.resize(curMiniClusterSize + newMiniClusterSize);
        miniClusterVectorIds.resize(curMiniClusterSize + newMiniClusterSize);
        for (size_t i = 0; i < newMiniClusterSize; i++) {
            miniClusters[curMiniClusterSize + i] = std::move(newMiniClusters[i]);
            miniClusterVectorIds[curMiniClusterSize + i] = std::move(newMiniClusterVectorIds[i]);
        }

        // Create mega centroids by taking mean of mini centroids when we have enough
        // Group mini centroids in batches of megaCentroidSize
        int numMegaClusters = (newMiniClusterSize + config.megaCentroidSize - 1) / config.megaCentroidSize;
        std::vector<float> newMegaCentroids;
        std::vector<std::vector<vector_idx_t>> newMegaMiniClusterIds;
        newMegaCentroids.reserve(numMegaClusters * dim);
        newMegaMiniClusterIds.reserve(numMegaClusters);

        for (size_t megaBatchStart = 0; megaBatchStart < newMiniClusterSize; megaBatchStart += config.megaCentroidSize) {
            size_t megaBatchEnd = std::min(megaBatchStart + config.megaCentroidSize, newMiniClusterSize);
            size_t megaBatchSize = megaBatchEnd - megaBatchStart;

            // Calculate mean of mini centroids for this mega cluster
            std::vector<float> megaMeanCentroid(dim, 0.0f);
            for (size_t i = megaBatchStart; i < megaBatchEnd; i++) {
                for (int j = 0; j < dim; j++) {
                    megaMeanCentroid[j] += newMiniCentroids[i * dim + j];
                }
            }
            float megaNorm = 1.0f / megaBatchSize;
            for (int j = 0; j < dim; j++) {
                megaMeanCentroid[j] *= megaNorm;
            }

            // Store the mega centroid
            newMegaCentroids.insert(newMegaCentroids.end(), megaMeanCentroid.begin(), megaMeanCentroid.end());

            // Store which mini clusters belong to this mega cluster
            std::vector<vector_idx_t> megaClusterMiniIds;
            megaClusterMiniIds.reserve(megaBatchSize);
            for (size_t i = megaBatchStart; i < megaBatchEnd; i++) {
                megaClusterMiniIds.push_back(newMiniClusterIds[i]);
            }
            newMegaMiniClusterIds.push_back(std::move(megaClusterMiniIds));
        }

        // Copy the new mega centroids
        auto curMegaClusterSize = megaCentroids.size() / dim;
        auto newMegaClusterSize = newMegaCentroids.size() / dim;
        megaCentroids.resize((curMegaClusterSize + newMegaClusterSize) * dim);
        memcpy(megaCentroids.data() + curMegaClusterSize * dim, newMegaCentroids.data(),
               newMegaCentroids.size() * sizeof(float));
        megaMiniCentroidIds.resize(curMegaClusterSize + newMegaClusterSize);
        for (size_t i = 0; i < newMegaClusterSize; i++) {
            megaMiniCentroidIds[curMegaClusterSize + i] = std::move(newMegaMiniClusterIds[i]);
        }
        megaClusteringScore.resize(curMegaClusterSize + newMegaClusterSize);

        size += n;
        updateTotalDataWrittenByUser(n);
    }


    void ReclusteringIndex::naiveInsert(float *data, size_t n, bool use_rebalancing, float rebalancing_ratio, float sampling_ratio) {
        std::vector<vector_idx_t> vectorIds(n);
        for (size_t i = 0; i < n; i++) {
            vectorIds[i] = i + size;
        }

        // Run clustering to create mini clusters
        std::vector<float> newMiniCentroids;
        std::vector<std::vector<float> > newMiniClusters;
        std::vector<std::vector<vector_idx_t> > newMiniClusterVectorIds;
        clusterData(data, vectorIds.data(), n, config.miniCentroidSize, newMiniCentroids, newMiniClusters,
                    newMiniClusterVectorIds, use_rebalancing, false, rebalancing_ratio, sampling_ratio);

        // Assign mini cluster unique ids
        auto curMiniClusterSize = miniCentroids.size() / dim;
        auto newMiniClusterSize = newMiniCentroids.size() / dim;
        std::vector<vector_idx_t> newMiniClusterIds(newMiniClusterSize);
        for (size_t i = 0; i < newMiniClusterSize; i++) {
            newMiniClusterIds[i] = curMiniClusterSize + i;
        }

        miniCentroids.resize((curMiniClusterSize + newMiniClusterSize) * dim);
        memcpy(miniCentroids.data() + curMiniClusterSize * dim, newMiniCentroids.data(),
               newMiniCentroids.size() * sizeof(float));
        miniClusters.resize(curMiniClusterSize + newMiniClusterSize);
        miniClusterVectorIds.resize(curMiniClusterSize + newMiniClusterSize);
        for (size_t i = 0; i < newMiniClusterSize; i++) {
            miniClusters[curMiniClusterSize + i] = std::move(newMiniClusters[i]);
            miniClusterVectorIds[curMiniClusterSize + i] = std::move(newMiniClusterVectorIds[i]);
        }

        // TODO: Fix this
        // Create the mega centroids just by taking the mean
        std::vector<float> newMegaCentroid;
        std::vector<std::vector<vector_idx_t> > miniClusterIds;
        clusterData(newMiniCentroids.data(), newMiniClusterIds.data(), newMiniClusterIds.size(),
                    config.megaCentroidSize, newMegaCentroid, miniClusterIds, -1, use_rebalancing, true, rebalancing_ratio, sampling_ratio);

        // Copy the new mega centroids
        auto curMegaClusterSize = megaCentroids.size() / dim;
        auto newMegaClusterSize = newMegaCentroid.size() / dim;
        megaCentroids.resize((curMegaClusterSize + newMegaClusterSize) * dim);
        memcpy(megaCentroids.data() + curMegaClusterSize * dim, newMegaCentroid.data(),
               newMegaCentroid.size() * sizeof(float));
        megaMiniCentroidIds.resize(curMegaClusterSize + newMegaClusterSize);
        for (size_t i = 0; i < newMegaClusterSize; i++) {
            megaMiniCentroidIds[curMegaClusterSize + i] = std::move(miniClusterIds[i]);
        }
        // TODO: Store the score
        megaClusteringScore.resize(curMegaClusterSize + newMegaClusterSize);
        size += n;
        updateTotalDataWrittenByUser(n);
    }

    void ReclusteringIndex::naiveInsertQuant(float *data, size_t n) {
        std::vector<vector_idx_t> vectorIds(n);
        for (size_t i = 0; i < n; i++) {
            vectorIds[i] = i + size;
        }
        auto dataDim = quantizer->codeSize;

        // Quantize the data
        std::vector<uint8_t> quantizedData;
        quantizeVectors(data, n, quantizedData);

        // Run clustering to create mini clusters
        std::vector<float> newMiniCentroids;
        std::vector<std::vector<uint8_t>> newMiniClusters;
        std::vector<std::vector<vector_idx_t> > newMiniClusterVectorIds;
        clusterDataQuant(quantizedData.data(), vectorIds.data(), n, config.miniCentroidSize, newMiniCentroids,
            newMiniClusters, newMiniClusterVectorIds);

        // Quantize the new mini centroids
        std::vector<uint8_t> quantizedMiniCtrds;
        quantizeVectors(newMiniCentroids.data(), newMiniCentroids.size() / dim, quantizedMiniCtrds);

        // Assign mini cluster unique ids
        auto curMiniClusterSize = quantizedMiniCentroids.size() / dataDim;
        auto newMiniClusterSize = quantizedMiniCtrds.size() / dataDim;
        std::vector<vector_idx_t> newMiniClusterIds(newMiniClusterSize);
        for (size_t i = 0; i < newMiniClusterSize; i++) {
            newMiniClusterIds[i] = curMiniClusterSize + i;
        }

        quantizedMiniCentroids.resize((curMiniClusterSize + newMiniClusterSize) * dataDim);
        memcpy(quantizedMiniCentroids.data() + curMiniClusterSize * dataDim, quantizedMiniCtrds.data(),
               newMiniCentroids.size() * sizeof(uint8_t));
        quantizedMiniClusters.resize(curMiniClusterSize + newMiniClusterSize);
        miniClusterVectorIds.resize(curMiniClusterSize + newMiniClusterSize);
        for (size_t i = 0; i < newMiniClusterSize; i++) {
            quantizedMiniClusters[curMiniClusterSize + i] = std::move(newMiniClusters[i]);
            miniClusterVectorIds[curMiniClusterSize + i] = std::move(newMiniClusterVectorIds[i]);
        }

        // Create the mega centroids just by taking the mean
        std::vector<float> newMegaCentroid;
        std::vector<std::vector<vector_idx_t> > miniClusterIds;
        clusterDataQuant(quantizedMiniCtrds.data(), newMiniClusterIds.data(), newMiniClusterIds.size(),
                    config.megaCentroidSize, newMegaCentroid, miniClusterIds);

        // Copy the new mega centroids
        auto curMegaClusterSize = megaCentroids.size() / dim;
        auto newMegaClusterSize = newMegaCentroid.size() / dim;
        megaCentroids.resize((curMegaClusterSize + newMegaClusterSize) * dim);
        memcpy(megaCentroids.data() + curMegaClusterSize * dim, newMegaCentroid.data(),
               newMegaCentroid.size() * sizeof(float));
        megaMiniCentroidIds.resize(curMegaClusterSize + newMegaClusterSize);
        for (size_t i = 0; i < newMegaClusterSize; i++) {
            megaMiniCentroidIds[curMegaClusterSize + i] = std::move(miniClusterIds[i]);
        }

        megaClusteringScore.resize(curMegaClusterSize + newMegaClusterSize);
        size += n;
        updateTotalDataWrittenByUser(n);
    }

    void ReclusteringIndex::recluster(int n, bool fast) {
        printf("ReclusteringIndex::reclusterFull\n");
        // Do clustering on mega centroids
        auto megaClusterSize = megaCentroids.size() / dim;
        std::vector<vector_idx_t> megaClusterIds(megaClusterSize);
        for (size_t i = 0; i < megaClusterSize; i++) {
            megaClusterIds[i] = i;
        }
        std::vector<float> megaMegaCentroids;
        std::vector<std::vector<vector_idx_t>> megaMegaCentroidIds;
        clusterData(megaCentroids.data(), megaClusterIds.data(), megaClusterSize,
                    n, megaMegaCentroids, megaMegaCentroidIds, false, true);
        for (size_t i = 0; i < megaMegaCentroidIds.size(); i++) {
            if (fast) {
                reclusterFastMegaCentroids(megaMegaCentroidIds[i]);
            } else {
                reclusterFullMegaCentroids(megaMegaCentroidIds[i]);
            }
        }
    }

    void ReclusteringIndex::reclusterFast(int n) {
        auto megaClusterSize = std::min((size_t)n, megaCentroids.size() / dim);
        // List all mega centroids
        std::vector<vector_idx_t> megaClusterIds(megaClusterSize);
        for (size_t i = 0; i < megaClusterIds.size(); i++) {
            megaClusterIds[i] = i;
        }
        reclusterFastMegaCentroids(megaClusterIds);
    }

    void ReclusteringIndex::getMegaClusterIds(std::vector<vector_idx_t> &megaClusterIds) {
        auto megaClusterSize = megaCentroids.size() / dim;
        megaClusterIds.resize(megaClusterSize);
        // List all mega centroids
        for (size_t i = 0; i < megaClusterIds.size(); i++) {
            megaClusterIds[i] = i;
        }
    }


    void ReclusteringIndex::reclusterFastQuant() {
        // List all mega centroids
        std::vector<vector_idx_t> megaClusterIds(megaCentroids.size() / dim);
        for (size_t i = 0; i < megaClusterIds.size(); i++) {
            megaClusterIds[i] = i;
        }
        // Now recluster miniCentroids within the mega centroids
        for (auto megaCentroidId: megaClusterIds) {
            reclusterInternalMegaCentroidQuant(megaCentroidId);
        }
    }

    void ReclusteringIndex::reclusterFull(int numMegaCentroids) {
        // List all mega centroids
        std::vector<vector_idx_t> megaClusterIds(megaCentroids.size() / dim);
        for (size_t i = 0; i < megaClusterIds.size(); i++) {
            megaClusterIds[i] = i;
        }
        // reclusterOnlyMegaCentroids(megaClusterIds);

        // Create Mega Mega centroids
        std::vector<float> megaMegaCentroids;
        std::vector<std::vector<vector_idx_t>> megaMegaCentroidIds;
        clusterData(megaCentroids.data(), megaClusterIds.data(), megaClusterIds.size(),
                    numMegaCentroids, megaMegaCentroids, megaMegaCentroidIds, false, true);

        for (const auto & megaMegaCentroidId : megaMegaCentroidIds) {
            if (megaMegaCentroidId.size() == 0) {
                continue;
            }
            reclusterFullMegaCentroids(megaMegaCentroidId);
        }
    }

    void ReclusteringIndex::reclusterBasedOnScore(int n) {
        auto totalClusterSize = 0;
        auto megaClusterSize = megaCentroids.size() / dim;
        while (totalClusterSize <= megaClusterSize) {
            auto worstMegaClusterId = getWorstMegaCentroid();
            // Find the closest mega centroid
            std::vector<vector_idx_t> megaAssign;
            findKClosestMegaCentroids(megaCentroids.data() + (worstMegaClusterId * dim), n, megaAssign, stats);
            if (megaAssign.empty()) {
                continue;
            }
            auto newMegaIds = reclusterFullMegaCentroids(megaAssign);
            // Recalculate score for megaAssign
            for (auto megaId: newMegaIds) {
                megaClusteringScore[megaId] = calcScoreForMegaCluster(megaId);
            }
            totalClusterSize += megaAssign.size();
        }
    }

    void ReclusteringIndex::reclusterBasedOnMSEScore() {
        // Get megacentroids that need reclustering based on MSE score and centroid change criteria
        std::vector<vector_idx_t> megaClusterIds = getMegaCentroidsToRecluster();
        printf("reclusterBasedOnMSEScore: Reclustering %zu megacentroids\n", megaClusterIds.size());
        reclusterFastMegaCentroids(megaClusterIds);
    }

    void ReclusteringIndex::mergeNewMiniCentroids() {
        printf("ReclusteringIndex::mergeNewMiniCentroids\n");
        if (newMiniCentroids.empty()) {
            return;
        }

        if (megaCentroids.empty()) {
            // Init situation, run reclustering on all miniCentroids and create mini as well as mega centroids
            mergeNewMiniCentroidsInit();
            return;
        }

        auto startTime = std::chrono::high_resolution_clock::now();
        auto numMegaCentroids = megaCentroids.size() / dim;
        // Reclustering on the new mini centroids
        // TODO: Make this process concurrent!!
        std::vector<vector_idx_t> miniCentroidIds(newMiniCentroids.size() / dim);
        for (size_t i = 0; i < miniCentroidIds.size(); i++) {
            miniCentroidIds[i] = i;
        }
        std::vector<float> newMegaCentroids;
        std::vector<std::vector<vector_idx_t> > newMiniClusterIds;
        if (numMegaCentroids > config.numMegaReclusterCentroids * 3) {
            clusterData(newMiniCentroids.data(), miniCentroidIds.data(), miniCentroidIds.size(),
                        config.numNewMiniReclusterCentroids, newMegaCentroids, newMiniClusterIds, false, true);
            for (size_t i = 0; i < (newMegaCentroids.size() / dim); i++) {
                mergeNewMiniCentroidsBatch(newMegaCentroids.data() + i * dim,
                                           newMiniClusterIds[i]);
            }
        } else {
            calcMeanCentroid(newMiniCentroids.data(), miniCentroidIds.data(), miniCentroidIds.size(), dim,
                              newMegaCentroids, newMiniClusterIds);
            mergeNewMiniCentroidsBatch(newMegaCentroids.data(),
                                           newMiniClusterIds[0]);
        }

        // Reset all newMiniCentroids, clusters and vectorIds
        resetInputBuffer();

        auto endTime = std::chrono::high_resolution_clock::now();
        printf("Reclustering took %lld ms\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());
    }

    void ReclusteringIndex::reclusterMegaCentroids(int n) {
        printf("ReclusteringIndex::reclusterMegaCentroids\n");
        if (megaCentroids.empty()) {
            return;
        }

        // Find smallest score mega centroids
        std::vector<vector_idx_t> megaCentroidIds;
        std::vector<double> scores;
        for (int i = 0; i < megaCentroids.size() / dim; i++) {
            megaCentroidIds.push_back(i);
            scores.push_back(megaClusteringScore[i]);
        }

        // Sort mega centroids ids based on score
        std::sort(megaCentroidIds.begin(), megaCentroidIds.end(),
                  [&scores](int a, int b) { return scores[a] < scores[b]; });
    }

    std::vector<vector_idx_t> ReclusteringIndex::reclusterFullMegaCentroids(std::vector<vector_idx_t> megaClusterIds) {
        // // Find the closest mega centroid
        // std::vector<vector_idx_t> megaAssign;
        // findKClosestMegaCentroids(megaCentroids.data() + (megaIdToRecluster * dim),
        // config.numExistingMegaReclusterCentroids, megaAssign);
        // assert(std::find(megaAssign.begin(), megaAssign.end(), megaIdToRecluster) != megaAssign.end());

        // Take all the existing mini centroids and merge them
        printf("ReclusteringIndex::reclusterFullMegaCentroids on %lu mega centroids\n",
               megaClusterIds.size());
        size_t totalVecs = 0;
        for (auto megaCentroidId: megaClusterIds) {
            auto microCentroidIds = megaMiniCentroidIds[megaCentroidId];
            auto miniClusterSize = miniClusters.size();
            for (auto microCentroidId: microCentroidIds) {
                assert(microCentroidId < miniClusterSize);
                auto cluster = miniClusters[microCentroidId];
                totalVecs += (cluster.size() / dim);
            }
        }
        printf("Total vecs: %lu\n", totalVecs);

        // Copy actual vecs and vectorIds here
        std::vector<float> tempData(totalVecs * (size_t)dim);
        std::vector<vector_idx_t> tempVectorIds(totalVecs);
        size_t idx = 0;
        for (auto megaCentroidId: megaClusterIds) {
            auto microCentroidIds = megaMiniCentroidIds[megaCentroidId];
            for (auto microCentroidId: microCentroidIds) {
                auto cluster = miniClusters[microCentroidId];
                auto vectorId = miniClusterVectorIds[microCentroidId];
                size_t numVectors = cluster.size() / dim;
                memcpy(tempData.data() + static_cast<size_t>(idx) * dim, cluster.data(), cluster.size() * sizeof(float));
                memcpy(tempVectorIds.data() + idx, vectorId.data(), numVectors * sizeof(vector_idx_t));
                idx += numVectors;
            }
        }

        // Run mini reclustering
        std::vector<float> newMiniCentroids;
        std::vector<std::vector<float> > newMiniClusters;
        std::vector<std::vector<vector_idx_t> > newMiniClusterVectorIds;
        clusterData(tempData.data(), tempVectorIds.data(), totalVecs, config.miniCentroidSize,
                    newMiniCentroids, newMiniClusters, newMiniClusterVectorIds, config.clusteringMode);

        // Run mega reclustering
        std::vector<vector_idx_t> miniCentroidIds(newMiniCentroids.size() / dim);
        for (size_t i = 0; i < miniCentroidIds.size(); i++) {
            miniCentroidIds[i] = i;
        }
        std::vector<float> newMegaCentroids;
        std::vector<std::vector<vector_idx_t> > newMiniClusterIds;
        clusterData(newMiniCentroids.data(), miniCentroidIds.data(), miniCentroidIds.size(),
                    config.megaCentroidSize, newMegaCentroids, newMiniClusterIds, false, true);

        // Append the new mini and mega centroids to the index
        return appendOrMergeCentroids(megaClusterIds, newMegaCentroids, newMiniClusterIds, newMiniCentroids,
                               newMiniClusters, newMiniClusterVectorIds);
    }

    void ReclusteringIndex::quantizeVectors(float *data, int n, std::vector<uint8_t> &quantizedVectors) {
        quantizedVectors = std::vector<uint8_t>(n * quantizer->codeSize);
        if (config.distanceType == COSINE) {
            std::vector<float> normalizedVector(dim);
            for (size_t i = 0; i < n; i++) {
                normalize_vectors(data + i * dim, dim, 1, normalizedVector.data());
                quantizer->encode(normalizedVector.data(), quantizedVectors.data() + i * quantizer->codeSize, 1);
            }
        } else {
            quantizer->encode(data, quantizedVectors.data(), n);
        }
    }

    void ReclusteringIndex::reclusterFastMegaCentroids(std::vector<vector_idx_t> megaClusterIdsToRecluster) {
        // Now recluster miniCentroids within the mega centroids
        for (auto megaCentroidId: megaClusterIdsToRecluster) {
            reclusterInternalMegaCentroid(megaCentroidId);
        }
    }

    void ReclusteringIndex::reclusterInternalMegaCentroid(vector_idx_t megaClusterId) {
        // Take all the existing mini centroids and merge them
        size_t totalVecs = 0;
        auto microCentroidIds = megaMiniCentroidIds[megaClusterId];
        // std::vector<vector_idx_t> oldVectorIds;
        // for (auto microCentroidId: microCentroidIds) {
        //     if (microCentroidId == nextMiniCentroidId) {
        //         std::unordered_set<vector_idx_t> nearL1Ids;
        //         calcScoreForMiniCluster(microCentroidId, &nearL1Ids);
        //         oldVectorIds = miniClusterVectorIds[microCentroidId];
        //         // Print out of someIds how many part of microCentroidIds
        //         size_t count = 0;
        //         for (auto id: nearL1Ids) {
        //             if (std::find(microCentroidIds.begin(), microCentroidIds.end(), id) != microCentroidIds.end()) {
        //                 count++;
        //             }
        //         }
        //         printf("Fount id %lu in megaCentroidId %llu with %lu/%lu of someIds\n",
        //                nextMiniCentroidId, megaClusterId, count, nearL1Ids.size());
        //         // Now print all microCentroidIds
        //         // printf("microCentroidIds: ");
        //         // for (auto id: microCentroidIds) {
        //         //     printf("%llu,", id);
        //         // }
        //         // printf("\n");
        //         break;
        //     }
        // }

        auto miniClusterSize = miniClusters.size();
        for (auto microCentroidId: microCentroidIds) {
            if (microCentroidId >= miniClusterSize) {
                printf("Error: microCentroidId %llu >= miniClusterSize %lu\n", microCentroidId, miniClusterSize);
            }
            assert(microCentroidId < miniClusterSize);
            auto& cluster = miniClusters[microCentroidId];
            totalVecs += (cluster.size() / dim);
        }
        printf("Running reclusterInternalMegaCentroid on %llu with %lu vectors\n", megaClusterId, totalVecs);
        if (totalVecs == 0) {
            printf("No vectors to recluster for mega centroid %llu\n", megaClusterId);
            return;
        }

        // Copy actual vecs and vectorIds here
        std::vector<float> tempData(totalVecs * dim);
        std::vector<vector_idx_t> tempVectorIds(totalVecs);
        size_t idx = 0;
        for (auto microCentroidId: microCentroidIds) {
            auto& cluster = miniClusters[microCentroidId];
            auto& vectorId = miniClusterVectorIds[microCentroidId];
            size_t numVectors = cluster.size() / dim;
            assert(cluster.size() % dim == 0 && "cluster size must be multiple of dim");
            assert(vectorId.size() == numVectors && "vectorId size must match number of vectors in cluster");
            assert(idx + numVectors <= totalVecs && "tempData/tempVectorIds overflow");
            memcpy(tempData.data() + static_cast<size_t>(idx) * dim, cluster.data(), cluster.size() * sizeof(float));
            memcpy(tempVectorIds.data() + idx, vectorId.data(), numVectors * sizeof(vector_idx_t));
            idx += numVectors;
        }
        assert(idx == totalVecs && "totalVecs calculation mismatch");

        // Run mini reclustering
        //printf("Clustering L1 with mode: %d (0=HARD_LIMIT, 1=REBALANCE_CENTROIDS, 2=REBALANCE_VECTORS)\n", config.clusteringMode);
        std::vector<float> newMiniCentroids;
        std::vector<std::vector<float> > newMiniClusters;
        std::vector<std::vector<vector_idx_t> > newMiniClusterVectorIds;
        clusterData(tempData.data(), tempVectorIds.data(), totalVecs, config.miniCentroidSize,
                    newMiniCentroids, newMiniClusters, newMiniClusterVectorIds, config.clusteringMode);

        std::vector<std::vector<vector_idx_t>> newMiniCentroidIds(1);
        newMiniCentroidIds[0].resize(newMiniCentroids.size() / dim);
        for (size_t i = 0; i < newMiniCentroidIds[0].size(); i++) {
            newMiniCentroidIds[0][i] = i;
        }
        std::vector<float> newMegaCentroids(dim);
        memcpy(newMegaCentroids.data(), megaCentroids.data() + megaClusterId * dim,
               dim * sizeof(float));

        appendOrMergeCentroids({megaClusterId},
                               newMegaCentroids,
                               newMiniCentroidIds,
                               newMiniCentroids,
                               newMiniClusters,
                               newMiniClusterVectorIds);

        // if (oldVectorIds.empty()) {
        //     return;
        // }
        // // Find the new mini centroid that contains oldVectorIds
        // auto max_match_count = 0;
        // auto max_match_id = -1;
        // for (auto miniId: megaMiniCentroidIds[megaClusterId]) {
        //     auto vectorIds = miniClusterVectorIds[miniId];
        //     size_t matchCount = 0;
        //     for (auto oldId: oldVectorIds) {
        //         if (std::find(vectorIds.begin(), vectorIds.end(), oldId) != vectorIds.end()) {
        //             matchCount++;
        //         }
        //     }
        //     if (matchCount > max_match_count) {
        //         max_match_count = matchCount;
        //         max_match_id = miniId;
        //     }
        // }
        // if (max_match_id != -1) {
        //     printf("After reclustering, old mini centroid %llu has max match count %d in new mini centroid %d\n",
        //            nextMiniCentroidId, max_match_count, max_match_id);
        //     nextMiniCentroidId = max_match_id;
        // }
    }

    void ReclusteringIndex::reclusterInternalMegaCentroidQuant(vector_idx_t megaClusterId) {
        // Take all the existing mini centroids and merge them
        auto dataDim = quantizer->codeSize;
        auto totalVecs = 0;
        auto microCentroidIds = megaMiniCentroidIds[megaClusterId];
        auto miniClusterSize = quantizedMiniClusters.size();
        for (auto microCentroidId: microCentroidIds) {
            assert(microCentroidId < miniClusterSize);
            auto cluster = quantizedMiniClusters[microCentroidId];
            totalVecs += (cluster.size() / dataDim);
        }

        // Copy actual vecs and vectorIds here
        std::vector<uint8_t> tempData(totalVecs * dataDim);
        std::vector<vector_idx_t> tempVectorIds(totalVecs);
        size_t idx = 0;
        for (auto microCentroidId: microCentroidIds) {
            auto cluster = quantizedMiniClusters[microCentroidId];
            auto vectorId = miniClusterVectorIds[microCentroidId];
            size_t numVectors = cluster.size() / dataDim;
            memcpy(tempData.data() + idx * dataDim, cluster.data(), cluster.size() * sizeof(uint8_t));
            memcpy(tempVectorIds.data() + idx, vectorId.data(), numVectors * sizeof(vector_idx_t));
            idx += numVectors;
        }

        // Run mini reclustering
        std::vector<float> newMiniCentroids;
        std::vector<std::vector<uint8_t>> newMiniClusters;
        std::vector<std::vector<vector_idx_t>> newMiniClusterVectorIds;
        clusterDataQuant(tempData.data(), tempVectorIds.data(), totalVecs, config.miniCentroidSize,
                    newMiniCentroids, newMiniClusters, newMiniClusterVectorIds);

        // Quantize the new mini centroids
        std::vector<uint8_t> quantizedMiniCtrds;
        quantizeVectors(newMiniCentroids.data(), newMiniCentroids.size() / dim, quantizedMiniCtrds);

        std::vector<std::vector<vector_idx_t>> newMiniCentroidIds(1);
        newMiniCentroidIds[0].resize(quantizedMiniCtrds.size() / dataDim);
        for (size_t i = 0; i < newMiniCentroidIds[0].size(); i++) {
            newMiniCentroidIds[0][i] = i;
        }
        std::vector<float> newMegaCentroids(dim);
        memcpy(newMegaCentroids.data(), megaCentroids.data() + megaClusterId * dim,
               dim * sizeof(float));

        appendOrMergeCentroidsQuant({megaClusterId},
                               newMegaCentroids,
                               newMiniCentroidIds,
                               quantizedMiniCtrds,
                               newMiniClusters,
                               newMiniClusterVectorIds);
    }

    void ReclusteringIndex::computeAllSubCells(int avgSubCellSize) {
        auto miniClusterSize = miniCentroids.size() / dim;
        if (miniClusterSize == 0) {
            return;
        }
        miniClusterSubCells.resize(miniClusterSize);
        printf("ReclusteringIndex::computeAllSubCells\n");
        for (int i = 0; i < miniClusterSize; i++) {
            computeMiniClusterSubcells(i, avgSubCellSize);
        }
    }

    void ReclusteringIndex::computeMiniClusterSubcells(int miniClusterId, int avgSubCellSize) {
        // Try different ideas:
        // 1. Use simple k means / k-means++ to find subcells
        // 2. Normalize vector and then use k means to find better subcells
        // 3. Find k nearest centroids and divide based on mid-points
        // 4. Use PCA and then k means to find subcells (Might be useful)
        SubCells newSubCells;
        auto miniClusterSize = miniClusters[miniClusterId].size() / dim;
        if (miniClusterSize < avgSubCellSize * 2) {
            miniClusterSubCells[miniClusterId] = std::move(newSubCells);
            return;
        }

        std::vector<float> subCellCentroids;
        std::vector<std::vector<float>> newMiniClusters;
        std::vector<std::vector<vector_idx_t>> newMiniClusterVectorIds;
        clusterData(miniClusters[miniClusterId].data(),
                    miniClusterVectorIds[miniClusterId].data(),
                    miniClusterSize, avgSubCellSize,
                    newMiniCentroids, newMiniClusters, newMiniClusterVectorIds);
        // Now we have new mini centroids, we need to update the miniClusters and miniClusterVectorIds

        auto subCellSize = subCellCentroids.size() / dim;
        if (subCellSize == 1) {
            // No subcells created, just return
            miniClusterSubCells[miniClusterId] = std::move(newSubCells);
            return;
        }

        size_t totalVectors = 0;
        for (const auto & cluster : newMiniClusters) {
            totalVectors += (cluster.size() / dim);
        }
        assert(totalVectors == miniClusterSize);

        newSubCells.centroids = std::move(subCellCentroids);
        std::vector<float> sortedMiniCluster;
        std::vector<vector_idx_t> sortedMiniClusterVectorIds;
        sortedMiniCluster.reserve(totalVectors * dim);
        sortedMiniClusterVectorIds.reserve(totalVectors);
        newSubCells.start_end_idxes.resize(newMiniClusters.size());
        auto start = 0;
        for (size_t i = 0; i < newMiniClusters.size(); i++) {
            auto &cluster = newMiniClusters[i];
            auto &vectorIds = newMiniClusterVectorIds[i];
            size_t numVectors = cluster.size() / dim;
            sortedMiniCluster.insert(sortedMiniCluster.end(), cluster.begin(), cluster.end());
            sortedMiniClusterVectorIds.insert(sortedMiniClusterVectorIds.end(), vectorIds.begin(), vectorIds.end());
            newSubCells.start_end_idxes[i] = {start, start + numVectors};
            start += numVectors;
        }
        assert(sortedMiniCluster.size() == totalVectors * dim);
        miniClusterSubCells[miniClusterId] = std::move(newSubCells);
   }

    vector_idx_t ReclusteringIndex::getWorstMegaCentroid() {
        vector_idx_t worstMegaCentroid = 0;
        double worstScore = std::numeric_limits<double>::max();
        for (int i = 0; i < megaClusteringScore.size(); i++) {
            if (megaClusteringScore[i] < worstScore) {
                worstScore = megaClusteringScore[i];
                worstMegaCentroid = i;
            }
        }

        return worstMegaCentroid;
    }

    void ReclusteringIndex::reclusterAllMegaCentroids(int n) {
        auto numMegaCentroids = megaCentroids.size() / dim;
        if (numMegaCentroids == 0) {
            return;
        }
        numMegaCentroids = std::min(numMegaCentroids, (size_t)n);
        std::vector<vector_idx_t> megaCentroidIds(numMegaCentroids);
        for (size_t i = 0; i < numMegaCentroids; i++) {
            megaCentroidIds[i] = i;
        }
        reclusterOnlyMegaCentroids(megaCentroidIds);
    }

    void ReclusteringIndex::reclusterAllMiniCentroidsQuant() {
        auto numMegaCentroids = megaCentroids.size() / dim;
        if (numMegaCentroids == 0) {
            return;
        }
        std::vector<vector_idx_t> megaCentroidIds(numMegaCentroids);
        for (size_t i = 0; i < numMegaCentroids; i++) {
            megaCentroidIds[i] = i;
        }
        reclusterOnlyMegaCentroidsQuant(megaCentroidIds);
    }

    void ReclusteringIndex::fixBoundaryMiniCentroids(int n) {
        // Find the most negative Mini
        std::unordered_set<vector_idx_t> alreadyFixed;
        for (int i = 0; i < n; i++) {
            auto worstMiniCentroid = -1;
            double worstScore = std::numeric_limits<double>::max();
            for (int j = 0; j < miniClusteringScore.size(); j++) {
                if (alreadyFixed.contains(j)) {
                    continue;
                }
                if (miniClusteringScore[j] < worstScore) {
                    worstScore = miniClusteringScore[j];
                    worstMiniCentroid = j;
                }
            }
            if (worstMiniCentroid == -1) {
                printf("No more boundary mini centroids to fix\n");
                break;
            }
            printf("Fixing boundary mini centroid %d with score %f\n", worstMiniCentroid, worstScore);
            fixBoundaryMiniCentroid(worstMiniCentroid, &alreadyFixed);
            // alreadyFixed.emplace(worstMiniCentroid);
        }
        // fixBoundaryMiniCentroid(6541);
    }

    void ReclusteringIndex::fixBoundaryMiniCentroidsV2(int n) {
        // For each mega centroid, find mini centroids with negative score and fix them
        auto numMegaCentroids = megaCentroids.size() / dim;
        for (int megaId = 0; megaId < numMegaCentroids; megaId++) {
            auto miniCentroidIds = megaMiniCentroidIds[megaId];
            for (auto miniId : miniCentroidIds) {
                if (miniClusteringScore[miniId] < -0.009) {
                    printf("Fixing boundary mini centroid %llu in mega centroid %d with score %f\n",
                           miniId, megaId, miniClusteringScore[miniId]);
                    fixBoundaryMiniCentroidV2(miniId);
                }
            }
        }
    }

    void ReclusteringIndex::fixBoundaryMiniCentroidV2(int miniCentroidId) {
        // 1. Find 200 closest mini centroids
        std::vector<vector_idx_t> megaAssign;
        // First find relevant mega centroids to search
        findKClosestMegaCentroids(miniCentroids.data() + miniCentroidId * dim, 10, megaAssign, stats);

        // Then find closest mini centroids
        std::vector<vector_idx_t> closestMiniCentroids;
        findKClosestMiniCentroids(miniCentroids.data() + miniCentroidId * dim, 200, megaAssign, closestMiniCentroids, stats);

        // Add the target mini centroid if not already in the list
        if (std::find(closestMiniCentroids.begin(), closestMiniCentroids.end(), miniCentroidId) == closestMiniCentroids.end()) {
            closestMiniCentroids.push_back(miniCentroidId);
        }

        // 2. Create new empty assignments for each mini centroid
        std::vector<std::vector<float>> newAssignments(closestMiniCentroids.size());
        std::vector<std::vector<vector_idx_t>> newAssignmentVectorIds(closestMiniCentroids.size());

        // 3. For each vector in each mini centroid, find its closest mini centroid and reassign
#pragma omp parallel
        {
            // Each thread gets its own distance computer for thread safety
            auto dc = getDistanceComputer(miniCentroids.data(), miniCentroids.size() / dim);
#pragma omp for schedule(dynamic)
            for (size_t i = 0; i < closestMiniCentroids.size(); i++) {
                auto miniId = closestMiniCentroids[i];
                auto& cluster = miniClusters[miniId];
                auto& vectorIds = miniClusterVectorIds[miniId];
                size_t numVectors = cluster.size() / dim;

                // Local buffers for this thread to avoid contention
                std::vector<std::vector<float>> localAssignments(closestMiniCentroids.size());
                std::vector<std::vector<vector_idx_t>> localAssignmentVectorIds(closestMiniCentroids.size());

                for (size_t j = 0; j < numVectors; j++) {
                    const float* vec = cluster.data() + j * dim;

                    // Find closest mini centroid among the candidates
                    int bestMiniIdx = 0;
                    float bestDist = std::numeric_limits<float>::max();

                    dc->setQuery(vec);
                    for (size_t k = 0; k < closestMiniCentroids.size(); k++) {
                        double dist;
                        dc->computeDistance(closestMiniCentroids[k], &dist);
                        if (dist < bestDist) {
                            bestDist = dist;
                            bestMiniIdx = k;
                        }
                    }

                    // Assign vector to its closest mini centroid in local buffer
                    localAssignments[bestMiniIdx].insert(
                        localAssignments[bestMiniIdx].end(),
                        vec,
                        vec + dim
                    );
                    localAssignmentVectorIds[bestMiniIdx].push_back(vectorIds[j]);
                }

                // Merge local buffers into global buffers with critical section
                for (size_t k = 0; k < closestMiniCentroids.size(); k++) {
                    if (!localAssignments[k].empty()) {
#pragma omp critical
                        {
                            newAssignments[k].insert(
                                newAssignments[k].end(),
                                localAssignments[k].begin(),
                                localAssignments[k].end()
                            );
                            newAssignmentVectorIds[k].insert(
                                newAssignmentVectorIds[k].end(),
                                localAssignmentVectorIds[k].begin(),
                                localAssignmentVectorIds[k].end()
                            );
                        }
                    }
                }
            }
        }

        // 4. Update the mini clusters with the new assignments (keeping centroids as-is)
        for (size_t i = 0; i < closestMiniCentroids.size(); i++) {
            auto miniId = closestMiniCentroids[i];
            miniClusters[miniId] = std::move(newAssignments[i]);
            miniClusterVectorIds[miniId] = std::move(newAssignmentVectorIds[i]);
        }

        // 5. Recalculate clustering score for affected mini centroids
#pragma omp parallel for
        for (auto miniId: closestMiniCentroids) {
            miniClusteringScore[miniId] = calcScoreForMiniCluster(miniId);
        }
    }

    void ReclusteringIndex::fixBoundaryMiniCentroid(int miniCentroidId, std::unordered_set<vector_idx_t> *alreadyFixed) {
        // 1. Find 200 closest mini centroids
        std::vector<vector_idx_t> megaAssign;
        // First find relevant mega centroids to search
        findKClosestMegaCentroids(miniCentroids.data() + miniCentroidId * dim, 10, megaAssign, stats);

        // Then find closest mini centroids
        std::vector<vector_idx_t> closestMiniCentroids;
        findKClosestMiniCentroids(miniCentroids.data() + miniCentroidId * dim, 200, megaAssign, closestMiniCentroids, stats);

        if (std::find(closestMiniCentroids.begin(), closestMiniCentroids.end(), miniCentroidId) == closestMiniCentroids.end()) {
            closestMiniCentroids.push_back(miniCentroidId);
        }

        // 2. Find which mega centroids these mini centroids belong to
        std::unordered_set<vector_idx_t> affectedMegaCentroids;
        for (auto miniId : closestMiniCentroids) {
            for (size_t megaId = 0; megaId < megaMiniCentroidIds.size(); megaId++) {
                auto& miniIds = megaMiniCentroidIds[megaId];
                if (std::find(miniIds.begin(), miniIds.end(), miniId) != miniIds.end()) {
                    affectedMegaCentroids.insert(megaId);
                    break;
                }
            }
        }

        // 3. Collect all vectors from these mini centroids
        size_t totalVecs = 0;
        for (auto miniId : closestMiniCentroids) {
            totalVecs += miniClusters[miniId].size() / dim;
        }

        std::vector<float> tempData(totalVecs * dim);
        std::vector<vector_idx_t> tempVectorIds(totalVecs);
        size_t idx = 0;

        for (auto miniId : closestMiniCentroids) {
            auto& cluster = miniClusters[miniId];
            auto& vectorIds = miniClusterVectorIds[miniId];
            size_t numVectors = cluster.size() / dim;

            memcpy(tempData.data() + idx * dim, cluster.data(), cluster.size() * sizeof(float));
            memcpy(tempVectorIds.data() + idx, vectorIds.data(), numVectors * sizeof(vector_idx_t));
            idx += numVectors;

            // Remove from mega centroids
            for (auto megaId : affectedMegaCentroids) {
                auto& miniIds = megaMiniCentroidIds[megaId];
                auto it = std::find(miniIds.begin(), miniIds.end(), miniId);
                if (it != miniIds.end()) {
                    miniIds.erase(it);
                }
            }
        }

        // 4. Cluster them together to get new mini centroids
        std::vector<float> newMiniCentroids;
        std::vector<std::vector<float>> newMiniClusters;
        std::vector<std::vector<vector_idx_t>> newMiniClusterVectorIds;
        clusterData(tempData.data(), tempVectorIds.data(), totalVecs, config.miniCentroidSize,
                    newMiniCentroids, newMiniClusters, newMiniClusterVectorIds, config.clusteringMode);

        if (newMiniCentroids.size() / dim != newMiniClusters.size()) {
            printf("Error: newMiniCentroids size %lu / dim %d != newMiniClusters size %lu\n",
                   newMiniCentroids.size(), dim, newMiniClusters.size());
        }
        if (newMiniCentroids.size() / dim != newMiniClusterVectorIds.size()) {
            printf("Error: newMiniClusters size %lu != newMiniClusterVectorIds size %lu\n",
                   newMiniCentroids.size() / dim, newMiniClusterVectorIds.size());
        }
        assert(newMiniCentroids.size() / dim == newMiniClusters.size());
        assert(newMiniCentroids.size() / dim == newMiniClusterVectorIds.size());

        // 5. Create mega centroid ids for the new mini centroids and cluster them
        std::vector<vector_idx_t> miniCentroidIds(newMiniCentroids.size() / dim);
        for (size_t i = 0; i < miniCentroidIds.size(); i++) {
            miniCentroidIds[i] = i;
        }

        std::vector<float> newMegaCentroids;
        std::vector<std::vector<vector_idx_t>> newMiniClusterIds;
        clusterData(newMiniCentroids.data(), miniCentroidIds.data(), miniCentroidIds.size(),
                    config.megaCentroidSize, newMegaCentroids, newMiniClusterIds);
        auto oldMiniCentroidSize = miniCentroids.size() / dim;
        // 6. Add them to new mega centroid(s)
        appendOrMergeCentroids({}, newMegaCentroids, newMiniClusterIds,
                              newMiniCentroids, newMiniClusters, newMiniClusterVectorIds, closestMiniCentroids);
        miniClusteringScore.resize(miniCentroids.size() / dim);
        auto newMiniCentroidSize = miniCentroids.size() / dim;
#pragma omp parallel for
        for (size_t i = oldMiniCentroidSize; i < newMiniCentroidSize; i++) {
            miniClusteringScore[i] = calcScoreForMiniCluster(i);
        }
#pragma omp parallel for
        for (auto miniId : closestMiniCentroids) {
            if (miniId < newMiniCentroidSize) {
                miniClusteringScore[miniId] = calcScoreForMiniCluster(miniId);
            }
        }

        if (alreadyFixed != nullptr) {
            for (size_t i = oldMiniCentroidSize; i < newMiniCentroidSize; i++) {
                alreadyFixed->insert(i);
            }
            for (auto miniId : closestMiniCentroids) {
                if (miniId < newMiniCentroidSize) {
                    alreadyFixed->insert(miniId);
                }
            }
        }
    }

    void ReclusteringIndex::mergeNewMiniCentroidsBatch(float *newMegaCentroid,
                                                       std::vector<vector_idx_t> newMiniCentroidBatch) {
        // Find the closest mega centroid
        std::vector<vector_idx_t> megaAssign;
        findKClosestMegaCentroids(newMegaCentroid, config.numMegaReclusterCentroids, megaAssign, stats);

        auto totalVecs = 0;
        for (auto i = 0; i < newMiniCentroidBatch.size(); i++) {
            auto cluster = newMiniClusters[newMiniCentroidBatch[i]];
            totalVecs += (cluster.size() / dim);
        }
        for (auto megaCentroidId: megaAssign) {
            auto microCentroidIds = megaMiniCentroidIds[megaCentroidId];
            for (auto microCentroidId: microCentroidIds) {
                auto cluster = miniClusters[microCentroidId];
                totalVecs += (cluster.size() / dim);
            }
        }

        // Copy actual vecs and vectorIds here
        std::vector<float> tempData(totalVecs * dim);
        std::vector<vector_idx_t> tempVectorIds(totalVecs);
        size_t idx = 0;
        for (auto i = 0; i < newMiniCentroidBatch.size(); i++) {
            auto cluster = newMiniClusters[newMiniCentroidBatch[i]];
            auto vectorId = newMiniClusterVectorIds[newMiniCentroidBatch[i]];
            size_t numVectors = cluster.size() / dim;
            memcpy(tempData.data() + idx * dim, cluster.data(), cluster.size() * sizeof(float));
            memcpy(tempVectorIds.data() + idx, vectorId.data(), numVectors * sizeof(vector_idx_t));
            idx += numVectors;
        }
        for (auto megaCentroidId: megaAssign) {
            auto microCentroidIds = megaMiniCentroidIds[megaCentroidId];
            for (auto microCentroidId: microCentroidIds) {
                auto cluster = miniClusters[microCentroidId];
                auto vectorId = miniClusterVectorIds[microCentroidId];
                size_t numVectors = cluster.size() / dim;
                memcpy(tempData.data() + static_cast<size_t>(idx) * dim, cluster.data(), cluster.size() * sizeof(float));
                memcpy(tempVectorIds.data() + idx, vectorId.data(), numVectors * sizeof(vector_idx_t));
                idx += numVectors;
            }
        }

        // Run mini reclustering
        std::vector<float> newMiniCentroids;
        std::vector<std::vector<float> > newMiniClusters;
        std::vector<std::vector<vector_idx_t> > newMiniClusterVectorIds;
        clusterData(tempData.data(), tempVectorIds.data(), totalVecs, config.miniCentroidSize,
                    newMiniCentroids, newMiniClusters, newMiniClusterVectorIds);

        // Run mega reclustering
        std::vector<vector_idx_t> miniCentroidIds(newMiniCentroids.size() / dim);
        for (size_t i = 0; i < miniCentroidIds.size(); i++) {
            miniCentroidIds[i] = i;
        }
        std::vector<float> newMegaCentroids;
        std::vector<std::vector<vector_idx_t> > newMiniClusterIds;
        clusterData(newMiniCentroids.data(), miniCentroidIds.data(), miniCentroidIds.size(),
                    config.megaCentroidSize, newMegaCentroids, newMiniClusterIds, false, true);

        // Append the new mini and mega centroids to the index
        appendOrMergeCentroids(megaAssign, newMegaCentroids, newMiniClusterIds, newMiniCentroids,
                               newMiniClusters, newMiniClusterVectorIds);
    }

    void ReclusteringIndex::mergeNewMiniCentroidsInit() {
        // Copy all the data to temp vectors
        size_t totalVectors = 0;
        for (const auto &cluster: newMiniClusters) {
            totalVectors += (cluster.size() / dim);
        }

        // Create the clustering object
        std::vector<float> newVectors(totalVectors * dim);
        std::vector<vector_idx_t> newVectorIds(totalVectors);

        // Copy from newMiniClusters to newVectors
        size_t idx = 0;
        for (size_t i = 0; i < newMiniClusters.size(); i++) {
            auto cluster = newMiniClusters[i];
            auto vectorId = newMiniClusterVectorIds[i];
            size_t numVectors = cluster.size() / dim;
            memcpy(newVectors.data() + idx * dim, cluster.data(), cluster.size() * sizeof(float));
            memcpy(newVectorIds.data() + idx, vectorId.data(), numVectors * sizeof(vector_idx_t));
            idx += numVectors;
        }

        // Perform mini clustering
        std::vector<float> tempMiniCentroids;
        std::vector<std::vector<float>> tempMiniClusters;
        std::vector<std::vector<vector_idx_t> > tempMiniClusterVectorIds;
        clusterData(newVectors.data(), newVectorIds.data(), totalVectors, config.miniCentroidSize,
                    tempMiniCentroids, tempMiniClusters, tempMiniClusterVectorIds);

        // Create mega centroids
        std::vector<vector_idx_t> miniCentroidIds(tempMiniCentroids.size() / dim);
        for (size_t i = 0; i < miniCentroidIds.size(); i++) {
            miniCentroidIds[i] = i;
        }
        std::vector<float> tempMegaCentroids;
        std::vector<std::vector<vector_idx_t> > tempMiniClusterIds;
        clusterData(tempMiniCentroids.data(), miniCentroidIds.data(), miniCentroidIds.size(),
                    config.megaCentroidSize, tempMegaCentroids, tempMiniClusterIds, false, true);

        // Move the mini and mega centroids to the index
        megaCentroids = std::move(tempMegaCentroids);
        megaMiniCentroidIds = std::move(tempMiniClusterIds);
        miniCentroids = std::move(tempMiniCentroids);
        miniClusters = std::move(tempMiniClusters);
        miniClusterVectorIds = std::move(tempMiniClusterVectorIds);

        // Reset input buffer!
        resetInputBuffer();
    }

    void ReclusteringIndex::reclusterOnlyMegaCentroids(std::vector<vector_idx_t> oldMegaCentroidIds) {
        printf("Reclustering only mega centroids with size %lu\n", oldMegaCentroidIds.size());

        size_t totalVec = 0;
        for (auto megaId: oldMegaCentroidIds) {
            totalVec += megaMiniCentroidIds[megaId].size();
        }

        // Take all the micro centroids and copy into temp storage
        std::vector<float> tempMiniCentroids(totalVec * dim);
        std::vector<vector_idx_t> tempMiniCentroidIds(totalVec);
        int idx = 0;
        for (auto megaId: oldMegaCentroidIds) {
            for (auto miniId: megaMiniCentroidIds[megaId]) {
                memcpy(tempMiniCentroids.data() + idx * dim, miniCentroids.data() + miniId * dim, sizeof(float) * dim);
                tempMiniCentroidIds[idx] = miniId;
                idx++;
            }
        }

        // Cluster data and write the mega and micro back again
        std::vector<float> tempMegaCentroids;
        std::vector<std::vector<vector_idx_t>> tempMiniClusterIds;
        clusterData(tempMiniCentroids.data(), tempMiniCentroidIds.data(), totalVec, config.megaCentroidSize,
                    tempMegaCentroids, tempMiniClusterIds);

        // Append back to mini centroids
        appendOrMergeMegaCentroids(oldMegaCentroidIds, tempMegaCentroids, tempMiniClusterIds);
    }

    void ReclusteringIndex::reclusterOnlyMegaCentroidsQuant(std::vector<vector_idx_t> oldMegaCentroidIds) {
        auto totalVec = 0;
        for (auto megaId: oldMegaCentroidIds) {
            totalVec += megaMiniCentroidIds[megaId].size();
        }
        auto dataDim = quantizer->codeSize;

        // Take all the micro centroids and copy into temp storage
        std::vector<uint8_t> tempMiniCentroids(totalVec * dataDim);
        std::vector<vector_idx_t> tempMiniCentroidIds(totalVec);
        int idx = 0;
        for (auto megaId: oldMegaCentroidIds) {
            for (auto miniId: megaMiniCentroidIds[megaId]) {
                memcpy(tempMiniCentroids.data() + idx * dataDim, quantizedMiniCentroids.data() + miniId * dataDim, sizeof(uint8_t) * dataDim);
                tempMiniCentroidIds[idx] = miniId;
                idx++;
            }
        }

        // Cluster data and write the mega and micro back again
        std::vector<float> tempMegaCentroids;
        std::vector<std::vector<vector_idx_t>> tempMiniClusterIds;
        clusterDataQuant(tempMiniCentroids.data(), tempMiniCentroidIds.data(), totalVec, config.megaCentroidSize, tempMegaCentroids, tempMiniClusterIds);

        // Append back to mini centroids
        appendOrMergeMegaCentroids(oldMegaCentroidIds, tempMegaCentroids, tempMiniClusterIds);
    }

    void ReclusteringIndex::resetInputBuffer() {
        newMiniCentroids.clear();
        newMiniClusters.clear();
        newMiniClusterVectorIds.clear();
        newMiniCentroids = std::vector<float>();
        newMiniClusters = std::vector<std::vector<float> >();
        newMiniClusterVectorIds = std::vector<std::vector<vector_idx_t> >();
    }

    void ReclusteringIndex::clusterData(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                        std::vector<float> &centroids, std::vector<std::vector<float> > &clusters,
                                        std::vector<std::vector<vector_idx_t> > &clusterVectorIds,
                                        bool use_rebalancing, bool is_clustering_centroids, float rebalancing_ratio, float sampling_ratio) {
        // auto dc = createDistanceComputer(data, dim, n, config.distanceType);
        // clusterData_<float>(data, vectorIds, n, avgClusterSize, centroids, clusters, clusterVectorIds,
        //                     dc.get(), dim, [](const float x, int d) { return x; });
        
        // Only use rebalancing for L1 (mini clusters), not for L2 (mega clusters)
        if (!is_clustering_centroids) {
            if (use_rebalancing==REBALANCE_VECTORS) {
                clusterDataWithRebalancing(data, vectorIds, n, avgClusterSize, centroids, &clusters, clusterVectorIds);
            } else if (use_rebalancing==REBALANCE_CENTROIDS) {
                clusterDataWithCentoidRebalancing(data, vectorIds, n, avgClusterSize, centroids, &clusters, clusterVectorIds, rebalancing_ratio, sampling_ratio);
            } else {
                clusterDataWithFaiss(data, vectorIds, n, avgClusterSize, centroids, &clusters, clusterVectorIds);
            }
        } else {
            clusterDataWithFaiss(data, vectorIds, n, avgClusterSize, centroids, &clusters, clusterVectorIds);
        }       
    }

    void ReclusteringIndex::clusterData(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                        std::vector<float> &centroids,
                                        std::vector<std::vector<vector_idx_t> > &clusterVectorIds, int nClusters,
                                         bool use_rebalancing, bool is_clustering_centroids, float rebalancing_ratio, float sampling_ratio) {
        // auto dc = createDistanceComputer(data, dim, n, config.distanceType);
        // clusterData_<float>(data, vectorIds, n, avgClusterSize, centroids, clusterVectorIds,
        //                     dc.get(), dim, [](const float x, int d) { return x; });
        
        // This overload doesn't have clusters parameter, so we pass nullptr
        // Only use rebalancing for L1 (mini clusters), not for L2 (mega clusters)
        if (!is_clustering_centroids) {
            if (use_rebalancing == REBALANCE_CENTROIDS) {
                clusterDataWithCentoidRebalancing(data, vectorIds, n, avgClusterSize, centroids, nullptr, clusterVectorIds, rebalancing_ratio, sampling_ratio);
            } else if (use_rebalancing == REBALANCE_VECTORS) {
                clusterDataWithRebalancing(data, vectorIds, n, avgClusterSize, centroids, nullptr, clusterVectorIds);
            } else {
                clusterDataWithFaiss(data, vectorIds, n, avgClusterSize, centroids, nullptr, clusterVectorIds, nClusters);
            }
        } else {
            clusterDataWithFaiss(data, vectorIds, n, avgClusterSize, centroids, nullptr, clusterVectorIds, nClusters);
        }
    }
    void ReclusteringIndex::clusterDataQuant(uint8_t *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                             std::vector<float> &centroids,
                                             std::vector<std::vector<uint8_t> > &clusters,
                                             std::vector<std::vector<vector_idx_t> > &clusterVectorIds) {
        auto dc = createQuantizedDistanceComputer(data, dim, n, config.distanceType, quantizer.get());
        auto q = quantizer.get();
        clusterData_<uint8_t>(data, vectorIds, n, avgClusterSize, centroids, clusters, clusterVectorIds,
                              dc.get(), q->codeSize, [&](const uint8_t x, int d) { return q->decode_one(x, d); });
    }

    void ReclusteringIndex::clusterDataQuant(uint8_t *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                             std::vector<float> &centroids,
                                             std::vector<std::vector<vector_idx_t> > &clusterVectorIds) {
        auto dc = createQuantizedDistanceComputer(data, dim, n, config.distanceType, quantizer.get());
        auto q = quantizer.get();
        clusterData_<uint8_t>(data, vectorIds, n, avgClusterSize, centroids, clusterVectorIds,
                              dc.get(), q->codeSize, [&](const uint8_t x, int d) { return q->decode_one(x, d); });
    }

    float ReclusteringIndex::findAppropriateLambda(const float *data, size_t num_rows, int dim, int num_clusters,
                                                   size_t sample_size) {
        auto dc = createDistanceComputer(data, dim, num_rows, config.distanceType);
        auto num_rows_per_cluster = num_rows / num_clusters;
        sample_size = std::min(sample_size, num_rows);
        double lambda = std::numeric_limits<double>::lowest();

        for (int i = 0; i <= sample_size; i++) {
            size_t idx1 = rg->randInt(num_rows);
            size_t idx2 = rg->randInt(num_rows);
            // Take the absolute distance to cover both L2 and Inner Product cases
            double dist;
            dc->computeDistance(idx1, idx2, &dist);
            dist = std::abs(dist);
            lambda = std::max(lambda, dist);
        }
        return (lambda / num_rows_per_cluster);
    }

    // Helper function: Find K nearest clusters to a given cluster
    static std::vector<int64_t> findKNearestClusters(
        const float* centroids, //array of centroids
        int64_t numClusters, //total number of clusters
        int dim, //dimension of the centroids
        int64_t target_cluster_id, //the cluster to find nearest neighbors for
        int k, //number of nearest neighbors to find
        faiss::MetricType metric_type) {
        
        // Create an index with all centroids
        //TODO_GILLI: ALTERNATIVE: use a loop to find the k nearest neighbors - memory efficient
        faiss::IndexFlat index(dim, metric_type);
        index.add(numClusters, centroids);
        
        // Search for k+1 nearest neighbors (including itself)
        std::vector<float> distances(k + 1);
        std::vector<int64_t> labels(k + 1);
        const float* query_centroid = centroids + target_cluster_id * dim;
        
        index.search(1, query_centroid, k + 1, distances.data(), labels.data());
        
        // Return the k nearest neighbors (excluding itself)
        std::vector<int64_t> neighbors;
        neighbors.reserve(k);
        for (int i = 0; i < k + 1; i++) {
            if (labels[i] != target_cluster_id) {
                neighbors.push_back(labels[i]);
                if (neighbors.size() >= k) break;
            }
        }
        
        return neighbors;
    }

    // Helper function: Rebalance a region of clusters
    static void rebalanceClusterRegion(
        float* data, //array of data
        int n, //total number of vectors
        int64_t* assignments, //array of assignments
        std::vector<int64_t>& hist, //histogram of cluster sizes
        float* centroids, //array of centroids
        int64_t numClusters, //total number of clusters
        int dim, //dimension of the centroids
        const std::unordered_set<int64_t>& clusters_to_rebalance, //set of clusters to rebalance
        faiss::MetricType metric_type,
        uint64_t hardClusterSizeLimit) {
        
        
        // Collect all vectors belonging to the clusters to rebalance
        std::vector<int> vector_indices;
        for (int i = 0; i < n; i++) {
            if (clusters_to_rebalance.find(assignments[i]) != clusters_to_rebalance.end()) {
                vector_indices.push_back(i);
            }
        }       
        if (vector_indices.empty()) {
            printf("No vectors to rebalance\n"); // shouldnt reach here - if rebalancing is triggered, there should be vectors to rebalance
            return;
        }
        // printf("Collected %zu vectors for rebalancing\n", vector_indices.size());
        
        // Extract the data for these vectors
        int num_vecs = vector_indices.size();
        std::vector<float> region_data(num_vecs * dim);
        // Copy vectors to contiguous memory - required by FAISS
        // std::copy may be better optimized by compiler than memcpy
        for (int i = 0; i < num_vecs; i++) {
            const float* src = data + vector_indices[i] * dim;
            float* dst = region_data.data() + i * dim;
            std::copy(src, src + dim, dst);
        }
        
        // Setup cluster ID mapping
        int num_region_clusters = clusters_to_rebalance.size(); // TODO_GILLI: why not k+1? can search return less than k+1?
        std::vector<int64_t> cluster_id_mapping(num_region_clusters);
        int idx = 0;
        for (auto cluster_id : clusters_to_rebalance) {
            cluster_id_mapping[idx] = cluster_id;
            idx++;
        }
        
        // Run k-means on the region with equal cluster sizes
        faiss::ClusteringParameters temp_cl;
        temp_cl.niter = 10;  // Fewer iterations for quick rebalancing
        temp_cl.min_points_per_centroid = 1;  // to make sure train does not sub sample
        temp_cl.max_points_per_centroid = num_vecs;  // to make sure train does not sub sample
        temp_cl.verbose = false;
        
        faiss::Clustering temp_clustering(dim, num_region_clusters, temp_cl);
        faiss::IndexFlat temp_index(dim, metric_type);
        
        /* GILLI_DEBUG: we can do better centroid init if we have neighbors and not just split
        // Initialize clustering centroids directly from global centroids - avoid intermediate copy
        clustering.centroids.resize(num_region_clusters * dim);
        idx = 0;
        for (auto cluster_id : clusters_to_rebalance) {
            const float* src = centroids + cluster_id * dim;
            float* dst = clustering.centroids.data() + idx * dim;
            std::copy(src, src + dim, dst);
            idx++;
        }
        */
        // Don't initialize centroids - let FAISS start fresh with random sampling
        // This ensures effective splitting when rebalancing oversized clusters
        // FAISS will randomly sample initial centroids from region_data
        temp_clustering.centroids.clear();
        
        printf("Splitting cluster %d (size %d), into %d clusters \n", cluster_id_mapping[0], num_vecs, num_region_clusters);
        // Train - FAISS will initialize all centroids from scratch
        temp_clustering.train(num_vecs, region_data.data(), temp_index);
        
        // Reassign vectors in the region with hard limit enforcement
        std::vector<int64_t> new_assignments(num_vecs);
        std::vector<float> distances(num_vecs);
        
        // NO HARD LIMIT when rebalancing
        //index.search(num_vecs,region_data.data(),1,distances.data(),new_assignments.data());

        // Enforce hard limit after the rebalnce
        std::vector<int64_t> local_hist(num_region_clusters, 0);
        
        // Use SearchParameters with dist_modifier to enforce hard limit
        faiss::SearchParameters params;
        std::unique_ptr<faiss::ClusterSizeCapDistModifier> hardLimitDistModifier;
        hardLimitDistModifier = std::make_unique<faiss::ClusterSizeCapDistModifier>(num_region_clusters, hardClusterSizeLimit);
        params.dist_modifier = hardLimitDistModifier.get();
        
        temp_index.search(num_vecs, region_data.data(), 1, distances.data(), new_assignments.data(), &params);
        
        // Update the global assignments and histograms
        // First, clear the old histogram counts for these clusters
        for (auto cluster_id : clusters_to_rebalance) {
            hist[cluster_id] = 0;
        }
        
        // Update assignments and recalculate histogram
        for (int i = 0; i < num_vecs; i++) {
            int64_t global_cluster_id = cluster_id_mapping[new_assignments[i]];
            assignments[vector_indices[i]] = global_cluster_id;
            hist[global_cluster_id]++;
        }
        
        // Update the centroids in the original array
        for (int i = 0; i < num_region_clusters; i++) {
            int64_t global_cluster_id = cluster_id_mapping[i];
            const float* src = temp_clustering.centroids.data() + i * dim;
            float* dst = centroids + global_cluster_id * dim;
            std::copy(src, src + dim, dst);
        }
        
        /*
        printf("Rebalancing complete. New cluster sizes in region:\n");
        for (auto cluster_id : clusters_to_rebalance) {
            printf("  Cluster %lld: %d vectors\n", cluster_id, hist[cluster_id]);
        }
        */
    }

    void ReclusteringIndex::clusterDataWithCentoidRebalancing(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                                 std::vector<float> &centroids,
                                                 std::vector<std::vector<float> > *clusters,
                                                 std::vector<std::vector<vector_idx_t> > &clusterVectorIds, 
                                                 float rebalancing_ratio, 
                                                float sampling_ratio) {
        // printf("Clustering %d vectors with avgClusterSize %d\n", n, avgClusterSize);
        if (n == 0) {
            return;
        }
        // Create the clustering object
        auto numClusters = getNumCentroids(n, avgClusterSize);
        //printf("clusterDataWithCentoidRebalancing: n=%d, avgClusterSize=%d, numClusters=%d\n", n, avgClusterSize, numClusters);
        if (numClusters <= 1) {
            calcMeanCentroid(data, vectorIds, n, dim, centroids, clusterVectorIds);
            // Copy all data to the single cluster
            if (clusters != nullptr) {
                clusters->resize(1);
                (*clusters)[0].resize(n * dim);
                memcpy((*clusters)[0].data(), data, n * dim * sizeof(float));
            }
            return;
        }

        auto updated_num_clusters = static_cast<int>(round(numClusters * rebalancing_ratio)); // 85% of the original number of clusters
        faiss::ClusteringParameters cl;
        cl.niter = config.nIter;
        if (config.distanceType == IP) {
            cl.spherical = true;
        }
        cl.min_points_per_centroid = getMinCentroidSize(n, numClusters);
        cl.max_points_per_centroid = getMaxCentroidSize(n, numClusters);
        // cl.seed = -1;
        std::unique_ptr<faiss::BalancedClusteringDistModifier> distModifier;
        cl.verbose = false; // GILLI: I changed this to false to avoid printing the clustering progress
        faiss::Clustering clustering(dim, updated_num_clusters, cl);
        // TODO: This is a hack
        auto metric_type = config.distanceType == L2 ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
        auto index = faiss::IndexFlat(dim, metric_type);
        
        // Initialize the centroids
        clustering.train(n, data, index);

        // find out how many new clusters are oversized 
        std::vector<std::pair<int64_t, int64_t>> clusters_to_rebalance;

        // DEBUG: check how many clusters have size 0
        int empty_counter = 0;
        for (int i = 0; i < updated_num_clusters; i++) {
            if (clustering.init_cluster_sizes[i] ==0) {
                empty_counter++;
            }
        }
        if (empty_counter > 0) {
            printf("Number of clusters with size 0: %d\n", empty_counter);
        }

        // GILLI: print histogram of cluster sizes
        printf("Histogram of cluster sizes:\n");
        for (int i = 0; i < updated_num_clusters; i++) {
            printf("Cluster %d: size = %d\n", i, clustering.init_cluster_sizes[i]);
        }
        printf("\n");

        int total_vectors_assigned = 0;
        for (int i = 0; i < updated_num_clusters; i++) {
            total_vectors_assigned += clustering.init_cluster_sizes[i];
        }
        printf("total vectors assigned = %d\n", total_vectors_assigned);

        for (int i = 0; i < updated_num_clusters; i++) {
            if (clustering.init_cluster_sizes[i] > sampling_ratio * config.hardClusterSizeLimit) {
                clusters_to_rebalance.push_back(std::make_pair(i, clustering.init_cluster_sizes[i]));
            }
        }
        
        // Initialize after clusters_to_rebalance is populated
        std::vector<int64_t> num_of_centroids_to_split_to(clusters_to_rebalance.size());
        
        // decide where to add new clusters - based on the cluster sizes
        // sort from the biggest oversized cluster to smallest
        std::sort(clusters_to_rebalance.begin(), clusters_to_rebalance.end(), 
                [](const std::pair<int64_t, int64_t>& a, const std::pair<int64_t, int64_t>& b)
        {
            return a.second > b.second;
        });
        
        // calculate the number of new clusters to add
        std::transform(clusters_to_rebalance.begin(), clusters_to_rebalance.end(), 
        num_of_centroids_to_split_to.begin(),
        [&](const std::pair<int64_t, int64_t>& cluster) {
            return static_cast<int64_t>(floor(static_cast<double>(cluster.second) / (sampling_ratio * config.hardClusterSizeLimit)));
        });

        int num_of_new_clusters = std::accumulate(num_of_centroids_to_split_to.begin(), num_of_centroids_to_split_to.end(), 0) - num_of_centroids_to_split_to.size();
        // DEBUG: check how many new clusters are added
        //printf("Number of clusters to add: %d\n", num_of_new_clusters);
        //printf("85 percent of clusters: %d (updated_num_clusters: %d, numClusters: %d)\n", updated_num_clusters, updated_num_clusters, numClusters);

        if (updated_num_clusters + num_of_new_clusters > numClusters) {
            int cluster_iter_big = 0;
            while(updated_num_clusters + num_of_new_clusters > numClusters) {
                
                // This should never happen - throw error if it does
                if (clusters_to_rebalance.empty()) {
                    throw std::runtime_error("clusters_to_rebalance is empty during rebalancing - this should not happen");
                }

                // Need at least 2 elements to compare (cluster_iter_big and back must be different)
                if (clusters_to_rebalance.size() == 1) {
                    num_of_centroids_to_split_to[0] = 1 + (numClusters - updated_num_clusters);
                    num_of_new_clusters = numClusters - updated_num_clusters;
                    break;
                }
                
                // If cluster_iter_big is pointing at or past the last element, wrap around to the beginning
                if (cluster_iter_big >= clusters_to_rebalance.size() - 1) {
                    cluster_iter_big = 0;
                }

                //compare biggest cluster with smallest one and remove the extra centroid when it's less needed
                int split_cluster_size_bigger = clusters_to_rebalance[cluster_iter_big].second / (num_of_centroids_to_split_to[cluster_iter_big] - 1);
                int split_cluster_size_smaller = clusters_to_rebalance.back().second / (num_of_centroids_to_split_to.back() - 1);

                if (split_cluster_size_bigger >= split_cluster_size_smaller) {
                    // remove the extra centroid from the smaller cluster
                    num_of_centroids_to_split_to.back()--;
                    // if we no longer need to split the cluster, we can remove it from the list
                    if(num_of_centroids_to_split_to.back() == 1) {
                        clusters_to_rebalance.pop_back();
                        num_of_centroids_to_split_to.pop_back();
                    }
                    cluster_iter_big++;
                } else {
                    // remove the extra centroid from the bigger cluster
                    num_of_centroids_to_split_to[cluster_iter_big]--;
                    cluster_iter_big++;
                }
                num_of_new_clusters--;
            }
        }
        // GILLI_DEBUG: we can do a smarter logic than that
        else if (updated_num_clusters + num_of_new_clusters < numClusters) {
            if (!clusters_to_rebalance.empty()) {
                int cluster_iter = 0;
                while(updated_num_clusters + num_of_new_clusters < numClusters) {
                    // Add one cluster at a time to avoid overshooting
                    num_of_centroids_to_split_to[cluster_iter]++;
                    num_of_new_clusters++;
                    cluster_iter++;
                    // Wrap around to distribute evenly across all clusters
                    if (cluster_iter >= clusters_to_rebalance.size()) {
                        cluster_iter = 0;
                    }
                }
            }
            else {
                // no clusters are oversized. split the biggest clusters into 2 new clusters
                for (int i = 0; i < (numClusters - updated_num_clusters); i++) {
                    clusters_to_rebalance.push_back(std::make_pair(i, clustering.init_cluster_sizes[i]));
                    num_of_centroids_to_split_to.push_back(2);
                }
            }
        }

        clustering.centroids.resize(numClusters * dim); // does resize copy old centroids? GILLI
        // update assign 
        std::vector<int64_t> assign(n,-1);
        std::vector<int64_t> hist(numClusters, 0);
        int max_sampled_idx =0;

        // Handle both subsampled and non-subsampled cases
        if (!clustering.sampled_indices.empty()) {
            // Subsampling occurred - use sampled_indices mapping
            for (int i = 0; i < clustering.sampled_indices.size(); i++) {
                assign[clustering.sampled_indices[i]] = clustering.init_assign.get()[i];
                if (clustering.sampled_indices[i] > max_sampled_idx) {
                    max_sampled_idx = clustering.sampled_indices[i];
                }
            }
        } else {
            // No subsampling - direct mapping
            for (int i = 0; i < n; i++) {
                assign[i] = clustering.init_assign.get()[i];
                if (i > max_sampled_idx) {
                    max_sampled_idx = i;
                }
            }
        }

        // Track which clusters were touched by rebalancing (both original and new)
        std::unordered_set<int64_t> all_rebalanced_clusters;
        
        // rebalance the oversized clusters
        for (int i = 0; i < clusters_to_rebalance.size(); i++) {
            int64_t cluster_id = clusters_to_rebalance[i].first;
            int64_t num_of_centroids_to_split = num_of_centroids_to_split_to[i];

            // Create the set of clusters to rebalance (1 full cluster + k neighbors)
            std::unordered_set<int64_t> clusters_to_rebalance_set;
            clusters_to_rebalance_set.insert(cluster_id);
            all_rebalanced_clusters.insert(cluster_id); // Track the original cluster
            
            for (int j = 0 ; j < (num_of_centroids_to_split - 1); j++) {
                int64_t new_cluster_id = updated_num_clusters + j;
                clusters_to_rebalance_set.insert(new_cluster_id);
                all_rebalanced_clusters.insert(new_cluster_id); // Track the new cluster
            }
            
            // split the cluster into num_of_centroids_to_split clusters
            // Add (num_of_centroids_to_split - 1) new clusters (original cluster remains)
            updated_num_clusters += (num_of_centroids_to_split - 1);
            // Update the index with the new cluster
            clustering.centroids.resize(updated_num_clusters * dim); // does resize copy old centroids? GILLI
            hist.resize(updated_num_clusters, 0);
            //index.reset();
            //index.add(updated_num_clusters, clustering.centroids.data());
                                    
            // Rebalance the cluster region - this will compute fresh centroids
            rebalanceClusterRegion(
                data, max_sampled_idx + 1, assign.data(), hist, 
                clustering.centroids.data(), updated_num_clusters, dim,
                clusters_to_rebalance_set, metric_type, sampling_ratio * config.hardClusterSizeLimit);

            // Update the index with the rebalanced centroids
            index.reset();
            index.add(updated_num_clusters, clustering.centroids.data());

        }

        // Verify that the adjustment logic correctly balanced the number of clusters
        // Lines 1560-1589 should guarantee: updated_num_clusters == numClusters
        if (updated_num_clusters != numClusters) {
            printf("ERROR: Cluster count mismatch after rebalancing! updated_num_clusters=%d, numClusters=%d\n",
                   updated_num_clusters, numClusters);
            throw std::runtime_error("Cluster rebalancing logic error: final cluster count doesn't match expected");
        }
        
        // Assign the centroids
        std::fill(assign.begin(), assign.end(), -1);
        std::vector<float> distances(n);
        std::unique_ptr<faiss::BalancedClusteringDistModifier> hardLimitDistModifier;
        std::fill(hist.begin(), hist.end(), 0);
        faiss::SearchParameters params;

        if (config.hardClusterSizeLimit > 0) {
            hardLimitDistModifier = std::make_unique<faiss::ClusterSizeCapDistModifier>(numClusters, config.hardClusterSizeLimit);
            params.dist_modifier = hardLimitDistModifier.get();
            printf("hard limit = %llu\n", config.hardClusterSizeLimit);
        }
        index.search(n, data, 1, distances.data(), assign.data(), &params);

        for (int i = 0; i < n; i++) {
            assert(assign[i]>=0 && assign[i]<numClusters);
            hist[assign[i]]++;
        }

        // Validate that no histogram is greater than the hard limit
        // Also check for small/empty clusters and identify their source
        int num_empty_clusters = 0;
        int num_singleton_clusters = 0;
        int num_small_clusters = 0;  // size <= 10
        int num_empty_from_original_kmeans = 0;
        int num_empty_from_rebalanced = 0;
        int num_singleton_from_original_kmeans = 0;
        int num_singleton_from_rebalanced = 0;
        int num_oversized_clusters = 0;
        
        // Calculate average sizes for different cluster sources
        long long sum_size_original_kmeans = 0;
        long long sum_size_rebalanced = 0;
        int count_original_kmeans = 0;
        int count_rebalanced = 0;
        
        for (int i=0; i<numClusters; i++) {
            if (config.hardClusterSizeLimit>0 && hist[i]>config.hardClusterSizeLimit) {
                printf("Warning: Cluster %d has size %d greater than %llu\n", i, hist[i], config.hardClusterSizeLimit);
                num_oversized_clusters++;
            }
            
            // Track which source this cluster came from
            bool is_rebalanced = (all_rebalanced_clusters.find(i) != all_rebalanced_clusters.end());
            
            // Accumulate sizes by source
            if (is_rebalanced) {
                sum_size_rebalanced += hist[i];
                count_rebalanced++;
            } else {
                sum_size_original_kmeans += hist[i];
                count_original_kmeans++;
            }
            
            if (hist[i] == 0) {
                num_empty_clusters++;
                if (is_rebalanced) {
                    num_empty_from_rebalanced++;
                    printf("Warning: Cluster %d is EMPTY after final assignment (source: rebalanced)\n", i);
                } else {
                    num_empty_from_original_kmeans++;
                    printf("Warning: Cluster %d is EMPTY after final assignment (source: original k-means)\n", i);
                }
            } else if (hist[i] == 1) {
                num_singleton_clusters++;
                if (is_rebalanced) {
                    num_singleton_from_rebalanced++;
                    printf("Warning: Cluster %d is SINGLETON (1 vector) after final assignment (source: rebalanced)\n", i);
                } else {
                    num_singleton_from_original_kmeans++;
                    printf("Warning: Cluster %d is SINGLETON (1 vector) after final assignment (source: original k-means)\n", i);
                }
            } else if (hist[i] <= 10) {
                num_small_clusters++;
            }
        }
        
        if (num_empty_clusters > 0 || num_singleton_clusters > 0) {
            printf("\n=== CLUSTER SIZE ANALYSIS ===\n");
            if (num_empty_clusters > 0) {
                printf("Total EMPTY clusters: %d out of %d (%.2f%%)\n", 
                       num_empty_clusters, numClusters, 100.0 * num_empty_clusters / numClusters);
                printf("  - From original k-means (never rebalanced): %d\n", num_empty_from_original_kmeans);
                printf("  - From rebalanced clusters: %d\n", num_empty_from_rebalanced);
            }
            if (num_singleton_clusters > 0) {
                printf("Total SINGLETON clusters: %d out of %d (%.2f%%)\n", 
                       num_singleton_clusters, numClusters, 100.0 * num_singleton_clusters / numClusters);
                printf("  - From original k-means (never rebalanced): %d\n", num_singleton_from_original_kmeans);
                printf("  - From rebalanced clusters: %d\n", num_singleton_from_rebalanced);
            }
            if (num_small_clusters > 0) {
                printf("Total SMALL clusters (2-10 vectors): %d out of %d (%.2f%%)\n", 
                       num_small_clusters, numClusters, 100.0 * num_small_clusters / numClusters);
            }
        }
        
        printf("\n=== AVERAGE CLUSTER SIZES BY SOURCE ===\n");
        printf("Original k-means (never rebalanced): %d clusters, avg size = %.2f\n",
               count_original_kmeans, 
               count_original_kmeans > 0 ? (double)sum_size_original_kmeans / count_original_kmeans : 0.0);
        printf("Rebalanced clusters (local k-means): %d clusters, avg size = %.2f\n",
               count_rebalanced,
               count_rebalanced > 0 ? (double)sum_size_rebalanced / count_rebalanced : 0.0);
        printf("Total clusters involved in rebalancing: %zu out of %d\n", 
               all_rebalanced_clusters.size(), numClusters);
        
        if (num_oversized_clusters > 0) {
            printf("Total oversized clusters (exceeding hard limit): %d\n", num_oversized_clusters);
        }

        // Copy the centroids
        centroids.resize(numClusters * dim);
        memcpy(centroids.data(), clustering.centroids.data(), numClusters * dim * sizeof(float));
        if (clusters != nullptr) {
            clusters->resize(numClusters);
            for (int i = 0; i < numClusters; i++) {
                std::vector<float> cluster(hist[i] * dim);
                (*clusters)[i] = std::move(cluster);
            }
        }
        clusterVectorIds.resize(numClusters);
        for (int i = 0; i < numClusters; i++) {
            std::vector<vector_idx_t> vectorId(hist[i]);
            clusterVectorIds[i] = std::move(vectorId);
            hist[i] = 0;
        }

        if (clusters != nullptr) {
            auto total_size = 0;
            for (int i = 0; i < numClusters; i++) {
                total_size += (*clusters)[i].size() / dim;
            }
            assert(total_size == n);
        }

        for (int i = 0; i < n; i++) {
            auto assignId = assign[i];
            auto idx = hist[assignId];
            // Copy cluster data if requested
            if (clusters != nullptr) {
                auto &cluster = (*clusters)[assignId];
                // auto maxClusterSize = cluster.size() / dim;
                memcpy(cluster.data() + static_cast<size_t>(idx) * dim,
                       data + static_cast<size_t>(i) * dim,
                       dim * sizeof(float));
            }
            clusterVectorIds[assignId][idx] = vectorIds[i];
            hist[assignId]++;
        }
        stats.numDistanceCompForRecluster += config.nIter * numClusters * n;
    }

    void ReclusteringIndex::clusterDataWithFaiss(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                                 std::vector<float> &centroids,
                                                 std::vector<std::vector<float> > *clusters,
                                                 std::vector<std::vector<vector_idx_t> > &clusterVectorIds,
                                                 int nClusters) {
        // printf("Clustering %d vectors with avgClusterSize %d\n", n, avgClusterSize);
        if (n == 0) {
            return;
        }
        
        // Create the clustering object
        auto numClusters = nClusters > 0 ? nClusters : getNumCentroids(n, avgClusterSize);
        // printf("Performing reclustering on %d vectors with %d clusters %d avgClusterSize\n", n, numClusters, avgClusterSize);
        
        if (numClusters <= 1) {
            calcMeanCentroid(data, vectorIds, n, dim, centroids, clusterVectorIds);
            // Copy all data to the single cluster if clusters output is requested
            if (clusters != nullptr) {
                clusters->resize(1);
                (*clusters)[0].resize(n * dim);
                memcpy((*clusters)[0].data(), data, n * dim * sizeof(float));
            }
            return;
        }

        faiss::ClusteringParameters cl;
        cl.niter = config.nIter;
        if (config.distanceType == IP) {
            cl.spherical = true;
        }
        cl.min_points_per_centroid = getMinCentroidSize(n, numClusters);
        cl.max_points_per_centroid = getMaxCentroidSize(n, numClusters);
        // cl.seed = -1;
        std::unique_ptr<faiss::BalancedClusteringDistModifier> distModifier;
        if (config.lambda > 0) {
            auto lambda = findAppropriateLambda(data, n, dim, numClusters);
            distModifier = std::make_unique<faiss::LambdaBasedDistModifier>(numClusters, lambda);
            cl.dist_modifier = distModifier.get();
            printf("cl.lambda = %f\n", lambda);
        }
        cl.verbose = false; // GILLI: I changed this to false to avoid printing the clustering progress
        faiss::Clustering clustering(dim, numClusters, cl);
        // TODO: This is a hack
        auto metric_type = config.distanceType == L2 ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
        auto index = faiss::IndexFlat(dim, metric_type);

        // Initialize the centroids
        clustering.train(n, data, index);


        // GILLI: print histogram of cluster sizes
        printf("Histogram of cluster sizes:\n");
        for (int i = 0; i < numClusters; i++) {
            printf("Cluster %d: size = %d\n", i, clustering.init_cluster_sizes[i]);
        }
        printf("\n");

        int total_vectors_assigned = 0;
        for (int i = 0; i < numClusters; i++) {
            total_vectors_assigned += clustering.init_cluster_sizes[i];
        }
        printf("total vectors assigned = %d\n", total_vectors_assigned);

                
        // Assign the centroids
        std::vector<int64_t> assign(n);
        std::vector<float> distances(n);
        std::unique_ptr<faiss::BalancedClusteringDistModifier> hardLimitDistModifier;
        faiss::SearchParameters params;

        if (config.hardClusterSizeLimit > 0) {
            hardLimitDistModifier = std::make_unique<faiss::ClusterSizeCapDistModifier>(numClusters, config.hardClusterSizeLimit);
            params.dist_modifier = hardLimitDistModifier.get();
            printf("hard limit = %llu\n", config.hardClusterSizeLimit);
        }
        index.search(n, data, 1, distances.data(), assign.data(), &params);

        // Build histogram
        std::vector<int> hist(numClusters, 0);
        for (int i = 0; i < n; i++) {
            if (assign[i] >= 0 && assign[i] < numClusters) {
                hist[assign[i]]++;
            } else {
                printf("WARNING: Invalid assignment at i=%d: assign[i]=%ld (numClusters=%d)\n",
                       i, assign[i], numClusters);
            }
        }

        // Validate that no histogram is greater than hard limit
        for (int i = 0; i < numClusters; i++) {
            if (config.hardClusterSizeLimit > 0 && hist[i] > config.hardClusterSizeLimit) {
                printf("Warning: Cluster %d has size %d greater than %llu\n", i, hist[i], config.hardClusterSizeLimit);
            }
        }

        // Copy the centroids
        centroids.resize(numClusters * dim);
        memcpy(centroids.data(), clustering.centroids.data(), numClusters * dim * sizeof(float));
        
        // Allocate space for cluster data if requested
        if (clusters != nullptr) {
            clusters->resize(numClusters);
            for (int i = 0; i < numClusters; i++) {
                std::vector<float> cluster(hist[i] * dim);
                (*clusters)[i] = std::move(cluster);
            }
        }
        
        // Always allocate clusterVectorIds
        clusterVectorIds.resize(numClusters);
        for (int i = 0; i < numClusters; i++) {
            std::vector<vector_idx_t> vectorId(hist[i]);
            clusterVectorIds[i] = std::move(vectorId);
        }
        
        // Reset histogram for use as insertion counters
        std::fill(hist.begin(), hist.end(), 0);

        // Validate total size if clusters are requested
        if (clusters != nullptr) {
            auto total_size = 0;
            for (int i = 0; i < numClusters; i++) {
                total_size += (*clusters)[i].size() / dim;
            }
            assert(total_size == n);
        }

        // Assign vectors to clusters
        for (int i = 0; i < n; i++) {
            auto assignId = assign[i];
            if (assignId < 0 || assignId >= numClusters) {
                printf("ERROR: Invalid assignId = %ld for vector i = %d (numClusters = %d)\n", assignId, i, numClusters);
                continue;  // Skip this vector
            }
            
            auto idx = hist[assignId];
            if (idx >= clusterVectorIds[assignId].size()) {
                printf("ERROR: idx = %d >= clusterVectorIds size = %lu for i = %d, assignId = %ld\n",
                       idx, clusterVectorIds[assignId].size(), i, assignId);
                continue;  // Skip this vector
            }
            
            // Copy cluster data if requested
            if (clusters != nullptr) {
                auto &cluster = (*clusters)[assignId];
                // auto maxClusterSize = cluster.size() / dim;
                memcpy(cluster.data() + static_cast<size_t>(idx) * dim,
                       data + static_cast<size_t>(i) * dim,
                       dim * sizeof(float));
            }
            
            clusterVectorIds[assignId][idx] = vectorIds[i];
            hist[assignId]++;
        }
        
        stats.numDistanceCompForRecluster += config.nIter * numClusters * n;
    }

    void ReclusteringIndex::clusterDataWithRebalancing(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                                        std::vector<float> &centroids,
                                                        std::vector<std::vector<float> > *clusters,
                                                        std::vector<std::vector<vector_idx_t> > &clusterVectorIds) {
        
                                                            //printf("Clustering %d vectors with avgClusterSize %d\n", n, avgClusterSize);
        if (n == 0) {
            return;
        }
        // Create the clustering object
        auto numClusters = getNumCentroids(n, avgClusterSize);
        // printf("Performing mini-reclustering on %d vectors with %d clusters %d avgClusterSize\n", n, numClusters, avgClusterSize);
        if (numClusters <= 1) {
            calcMeanCentroid(data, vectorIds, n, dim, centroids, clusterVectorIds);
            // Copy all data to the single cluster
            if (clusters != nullptr) {
                clusters->resize(1);
                (*clusters)[0].resize(n * dim);
                memcpy((*clusters)[0].data(), data, n * dim * sizeof(float));
            }
            return;
        }

        auto updated_num_clusters = round(numClusters * 0.9); // 90% of the original number of clusters
        faiss::ClusteringParameters cl;
        cl.niter = config.nIter;
        if (config.distanceType == IP) {
        cl.spherical = true;
        }
        cl.min_points_per_centroid = getMinCentroidSize(n, updated_num_clusters);
        cl.max_points_per_centroid = getMaxCentroidSize(n, updated_num_clusters);
        // cl.seed = -1;
        std::unique_ptr<faiss::BalancedClusteringDistModifier> distModifier;
        cl.verbose = false;
        faiss::Clustering clustering(dim, updated_num_clusters, cl); // cluster to only 90% of the original number of clusters
        // TODO: This is a hack
        auto metric_type = config.distanceType == L2 ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
        auto index = faiss::IndexFlat(dim, metric_type);

        // Initialize the centroids
        clustering.train(n, data, index);

        // Assign the centroids with per-vector rebalancing
        std::vector<int64_t> assign(n);
        std::vector<float> distances(n);
        std::unique_ptr<faiss::BalancedClusteringDistModifier> hardLimitDistModifier;
        std::vector<int64_t> hist(numClusters, 0); // take into account additional clusters 
        faiss::SearchParameters params_general;
        
        faiss::SearchParameters params_next_nearest;
        std::unique_ptr<faiss::ClusterSizeCapDistModifier> hardLimitDistModifier_next_nearest;
        hardLimitDistModifier_next_nearest = std::make_unique<faiss::ClusterSizeCapDistModifier>(numClusters, config.hardClusterSizeLimit);

        for (int i = 0; i < n; i++) {
            // Assign current vector to nearest centroid
            index.search(1, data + i * dim, 1, &distances[i], &assign[i], &params_general);

            int64_t assigned_cluster = assign[i];

            // Check if trying to assign to an ALREADY FULL cluster
            if (config.hardClusterSizeLimit > 0 && hist[assigned_cluster] >= config.hardClusterSizeLimit) {
                printf("Vector %d trying to assign to ALREADY FULL cluster %lld (%d/%llu). Triggering rebalancing...\n", 
                i, assigned_cluster, hist[assigned_cluster], config.hardClusterSizeLimit);

                // Find 4 nearest neighbor clusters
                auto neighbor_clusters = findKNearestClusters(
                clustering.centroids.data(), numClusters, dim, assigned_cluster, 5, metric_type);

                // Check if we need to add a new cluster - when all the neighbors are full
                bool add_new_cluster = true;
                for (auto neighbor_id : neighbor_clusters) {
                    if (hist[neighbor_id] < config.hardClusterSizeLimit) {
                        add_new_cluster = false;
                        break;
                    }
                }
                if (add_new_cluster && updated_num_clusters < numClusters) {
                    updated_num_clusters++;
                    neighbor_clusters.push_back(updated_num_clusters-1); // the new cluster is the last cluster
                    clustering.centroids.resize(updated_num_clusters * dim); // does resize copy old centroids? GILLI

                    // Update the index with the new cluster
                    index.reset();
                    index.add(updated_num_clusters, clustering.centroids.data());
                }
                else if (add_new_cluster && updated_num_clusters >= numClusters) {
                    printf("All clusters are full. no rebalancing\n");
                    // set the dist modifier for the next nearest search
                    hardLimitDistModifier_next_nearest->reset();
                    hardLimitDistModifier_next_nearest->populate_weights(hist.data(), updated_num_clusters); 
                    params_next_nearest.dist_modifier = hardLimitDistModifier_next_nearest.get();

                    // assign to the nearest cluster that has space 
                    index.search(1, data + i * dim, 1, &distances[i], &assign[i], &params_next_nearest);
                    assigned_cluster = assign[i];
                    hist[assigned_cluster]++;

                    continue;
                }

                // Create the set of clusters to rebalance (1 full cluster + k neighbors)
                std::unordered_set<int64_t> clusters_to_rebalance;
                clusters_to_rebalance.insert(assigned_cluster);
                for (auto neighbor_id : neighbor_clusters) {
                    clusters_to_rebalance.insert(neighbor_id);
                }

                // Perform immediate rebalancing for this cluster region
                // Only rebalance vectors assigned so far (0 to i-1, not including current vector i)
                rebalanceClusterRegion(
                data, i, assign.data(), hist, 
                clustering.centroids.data(), updated_num_clusters, dim,
                clusters_to_rebalance, metric_type, config.hardClusterSizeLimit);

                // Update the index with the rebalanced centroids
                index.reset();
                index.add(updated_num_clusters, clustering.centroids.data());

                // Re-assign current vector to the updated centroids
                index.search(1, data + i * dim, 1, &distances[i], &assign[i], &params_general);
                assigned_cluster = assign[i];
            }
            // Now increment the histogram
            hist[assigned_cluster]++;
        }

        // Copy the centroids
        centroids.resize(numClusters * dim);
        memcpy(centroids.data(), clustering.centroids.data(), numClusters * dim * sizeof(float));
        if (clusters != nullptr) {
            clusters->resize(numClusters);
            for (int i = 0; i < numClusters; i++) {
                std::vector<float> cluster(hist[i] * dim);
                (*clusters)[i] = std::move(cluster);
            }
        }
        clusterVectorIds.resize(numClusters);
        for (int i = 0; i < numClusters; i++) {
            std::vector<vector_idx_t> vectorId(hist[i]);
            clusterVectorIds[i] = std::move(vectorId);
        }

        if (clusters != nullptr) {
            auto total_size = 0;
            for (int i = 0; i < numClusters; i++) {
                total_size += (*clusters)[i].size() / dim;
            }
            assert(total_size == n);
        }

        for (int i = 0; i < n; i++) {
            auto assignId = assign[i];
            auto idx = hist[assignId];
            // Copy cluster data if requested
            if (clusters != nullptr) {
                auto &cluster = (*clusters)[assignId];
                // auto maxClusterSize = cluster.size() / dim;
                memcpy(cluster.data() + static_cast<size_t>(idx) * dim,
                       data + static_cast<size_t>(i) * dim,
                       dim * sizeof(float));
            }
            clusterVectorIds[assignId][idx] = vectorIds[i];
            hist[assignId]++;
        }
        stats.numDistanceCompForRecluster += config.nIter * numClusters * n;
}

    template <typename T>
    void ReclusteringIndex::clusterData_(T *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                        std::vector<float> &centroids, std::vector<std::vector<T> > &clusters,
                                        std::vector<std::vector<vector_idx_t> > &clusterVectorIds,
                                        DelegateDC<T> *dc, int dataDim, decode_func_t<T> decodeFunc) {
        // Create the clustering object
        auto numClusters = getNumCentroids(n, avgClusterSize);
        // printf("Performing mini-reclustering on %d vectors with %d clusters %d avgClusterSize\n", n, numClusters, avgClusterSize);
        if (numClusters <= 1) {
            calcMeanCentroid(data, vectorIds, n, dataDim, centroids, clusterVectorIds);
            return;
        }

        Clustering<T> clustering(dim, dataDim, numClusters, config.nIter,
                                     getMinCentroidSize(n, numClusters),
                                     getMaxCentroidSize(n, numClusters),
                                     dc,
                                     decodeFunc,
                                     config.lambda);

        // Initialize the centroids
        clustering.initCentroids(data, n);
        clustering.train(data, n);

        // Assign the centroids
        std::vector<int32_t> assign(n);
        clustering.assignCentroids(data, n, assign.data());

        // Get the hist
        std::vector<int> hist(numClusters, 0);
        for (int i = 0; i < n; i++) {
            hist[assign[i]]++;
        }

        // Copy the centroids
        centroids.resize(numClusters * dim);
        memcpy(centroids.data(), clustering.centroids.data(), numClusters * dim * sizeof(float));
        clusters.resize(numClusters);
        clusterVectorIds.resize(numClusters);
        for (int i = 0; i < numClusters; i++) {
            std::vector<T> cluster(hist[i] * dataDim);
            clusters[i] = cluster;
            std::vector<vector_idx_t> vectorId(hist[i]);
            clusterVectorIds[i] = vectorId;
            hist[i] = 0;
        }

        for (int i = 0; i < n; i++) {
            auto assignId = assign[i];
            auto idx = hist[assignId];
            auto &cluster = clusters[assignId];
            memcpy(cluster.data() + idx * dataDim, data + i * dataDim, dataDim * sizeof(T));
            clusterVectorIds[assignId][idx] = vectorIds[i];
            hist[assignId]++;
        }
        stats.numDistanceCompForRecluster += config.nIter * numClusters * n;
    }

    template <typename T>
    void ReclusteringIndex::clusterData_(T *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                        std::vector<float> &centroids,
                                        std::vector<std::vector<vector_idx_t> > &clusterVectorIds,
                                        DelegateDC<T> *dc, int dataDim, decode_func_t<T> decodeFunc) {
        // Create the clustering object
        auto numClusters = getNumCentroids(n, avgClusterSize);
        // printf("Performing mega-reclustering on %d vectors with %d clusters %d avgClusterSize\n", n, numClusters, avgClusterSize);
        if (numClusters <= 1) {
            calcMeanCentroid(data, vectorIds, n, dataDim, centroids, clusterVectorIds);
            return;
        }

        Clustering<T> clustering(dim, dataDim, numClusters, config.nIter,
                                     getMinCentroidSize(n, numClusters),
                                     getMaxCentroidSize(n, numClusters),
                                     dc,
                                     decodeFunc,
                                     config.lambda);

        // Initialize the centroids
        clustering.initCentroids(data, n);
        clustering.train(data, n);

        // Assign the centroids
        std::vector<int32_t> assign(n);
        clustering.assignCentroids(data, n, assign.data());

        // Get the hist
        std::vector<int> hist(numClusters, 0);
        for (int i = 0; i < n; i++) {
            hist[assign[i]]++;
        }

        // Copy the centroids
        centroids.resize(numClusters * dim);
        memcpy(centroids.data(), clustering.centroids.data(), numClusters * dim * sizeof(float));
        clusterVectorIds.resize(numClusters);
        for (int i = 0; i < numClusters; i++) {
            std::vector<vector_idx_t> vectorId(hist[i]);
            clusterVectorIds[i] = vectorId;
            hist[i] = 0;
        }

        for (int i = 0; i < n; i++) {
            auto assignId = assign[i];
            auto idx = hist[assignId];
            clusterVectorIds[assignId][idx] = vectorIds[i];
            hist[assignId]++;
        }
        stats.numDistanceCompForRecluster += config.nIter * numClusters * n;
    }

    template <typename T>
    void ReclusteringIndex::calcMeanCentroid(T *data, vector_idx_t *vectorIds, int n, int dataDim, std::vector<float> &centroids,
                                             std::vector<std::vector<vector_idx_t> > &clusterVectorIds) {
        // Calculate mean over all vectors and copy the vectorIds directly
        centroids.resize(dim);
        memset(centroids.data(), 0, dim * sizeof(float));
        // TODO: Maybe do this using simd at some point
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                // centroids[j] += quantizer->decode_one(data[i * dataDim + j], j);
                centroids[j] += static_cast<float>(data[i * dataDim + j]);
            }
        }
        auto norm = 1.0f / n;
        for (int j = 0; j < dim; j++) {
            centroids[j] *= norm;
        }
        clusterVectorIds.resize(1);
        clusterVectorIds[0].resize(n);
        for (int i = 0; i < n; i++) {
            clusterVectorIds[0][i] = vectorIds[i];
        }
    }

    std::vector<vector_idx_t> ReclusteringIndex::appendOrMergeCentroids(std::vector<vector_idx_t> oldMegaCentroids,
                                                   std::vector<float> &newMegaCentroids,
                                                   std::vector<std::vector<vector_idx_t> > &miniCentroidIds,
                                                   std::vector<float> &newMiniCentroids,
                                                   std::vector<std::vector<float> > &newMiniClusters,
                                                   std::vector<std::vector<vector_idx_t>> &newMiniClusterVectorIds,
                                                   std::vector<vector_idx_t> existingOldMiniClusterIds) {
        updateTotalDataWrittenBySystem({}, newMiniClusterVectorIds);
        // Try to copy inplace if possible otherwise append
        std::vector<vector_idx_t> oldMiniClusterIds;
        for (const int currMegaId: oldMegaCentroids) {
            for (const auto &megaMiniId: megaMiniCentroidIds[currMegaId]) {
                oldMiniClusterIds.push_back(megaMiniId);
            }
        }

        // Add existingOldMiniClusterIds
        if (!existingOldMiniClusterIds.empty()) {
            for (const auto id: existingOldMiniClusterIds) {
                oldMiniClusterIds.push_back(id);
            }
        }

        // Copy the mini centroids, clusters and vector ids and fix the miniClusterIds
        std::unordered_map<vector_idx_t, vector_idx_t> newToOldCentroidIdMap;
        auto newMiniCentroidsSize = newMiniCentroids.size() / dim;
        // assert(oldMiniClusterIds.size() <= newMiniCentroidsSize);
        auto miniCentroidsSize = std::min(newMiniCentroidsSize, oldMiniClusterIds.size());
        if (newMiniCentroidsSize != newMiniClusters.size()) {
            printf("Warning! newMiniCentroidsSize = %lu, newMiniClusters.size() = %lu\n", newMiniCentroidsSize,
                   newMiniClusters.size());
        }
        assert(newMiniCentroidsSize == newMiniClusters.size());
        for (int i = 0; i < miniCentroidsSize; i++) {
            auto oldCentroidId = oldMiniClusterIds[i];
            // Copy the centroid
            memcpy(miniCentroids.data() + static_cast<size_t>(oldCentroidId) * dim,
                   newMiniCentroids.data() + static_cast<size_t>(i) * dim,
                   dim * sizeof(float));
            // Move the cluster
            auto currCluster = newMiniClusters[i];
            auto currVectorId = newMiniClusterVectorIds[i];
            miniClusters[oldCentroidId] = std::move(currCluster);
            miniClusterVectorIds[oldCentroidId] = std::move(currVectorId);
            newToOldCentroidIdMap[i] = oldCentroidId;
        }

        if (newMiniCentroidsSize > miniCentroidsSize) {
            // Append the new mini centroids
            auto currentSize = miniCentroids.size() / dim;
            miniCentroids.resize((currentSize + newMiniCentroidsSize - miniCentroidsSize) * dim);
            memcpy(miniCentroids.data() + static_cast<size_t>(currentSize) * dim,
                   newMiniCentroids.data() + static_cast<size_t>(miniCentroidsSize) * dim,
                   (newMiniCentroidsSize - miniCentroidsSize) * dim * sizeof(float));

            // Append the new clusters
            miniClusters.resize(currentSize + newMiniCentroidsSize - miniCentroidsSize);
            miniClusterVectorIds.resize(currentSize + newMiniCentroidsSize - miniCentroidsSize);
            auto idx = 0;
            for (auto i = miniCentroidsSize; i < newMiniCentroidsSize; i++) {
                auto currCluster = newMiniClusters[i];
                auto currVectorId = newMiniClusterVectorIds[i];
                miniClusters[currentSize + idx] = std::move(currCluster);
                miniClusterVectorIds[currentSize + idx] = std::move(currVectorId);
                newToOldCentroidIdMap[i] = currentSize + idx;
                idx++;
            }
        } else {
            std::unordered_map<vector_idx_t, vector_idx_t> mappedMiniClusterIds;
            auto lastCentroidId = (miniCentroids.size() / dim) - 1;
            // If the new mini centroid smaller than oldMiniClusterIds.size()
            for (int i = newMiniCentroidsSize; i < oldMiniClusterIds.size(); i++) {
                // Copy from last to i
                auto currCentroidId = oldMiniClusterIds[i];
                while (std::find(oldMiniClusterIds.begin() + newMiniCentroidsSize, oldMiniClusterIds.end(), lastCentroidId) != oldMiniClusterIds.end()) {
                    lastCentroidId--;
                }
                if (currCentroidId > lastCentroidId) {
                    // No need to delete from megaMiniCentroidIds since it'll be taken care when we append mega centroids.
                    continue;
                }
                memcpy(miniCentroids.data() + static_cast<size_t>(currCentroidId) * dim,
                       miniCentroids.data() + static_cast<size_t>(lastCentroidId) * dim,
                       dim * sizeof(float));
                miniClusters[currCentroidId] = std::move(miniClusters[lastCentroidId]);
                miniClusterVectorIds[currCentroidId] = std::move(miniClusterVectorIds[lastCentroidId]);
                mappedMiniClusterIds[lastCentroidId] = currCentroidId;
                // printf("Removing mini centroid %d with miniCentroid %d\n", lastCentroidId, currCentroidId);
                lastCentroidId--;
            }

            // Update mega mini centroid ids
            for (auto &ids : megaMiniCentroidIds) {
                for (auto &id: ids) {
                    auto it = mappedMiniClusterIds.find(id);
                    if (it != mappedMiniClusterIds.end()) {
                        id = it->second;
                    }
                }
                // Remove IDs that are greater than lastCentroidId
                ids.erase(std::remove_if(ids.begin(), ids.end(),
                    [lastCentroidId](vector_idx_t id) { return id > lastCentroidId; }),
                    ids.end());
            }

            // Update newToOldCentroidIdMap
            for (auto &ids : newToOldCentroidIdMap) {
                auto it = mappedMiniClusterIds.find(ids.second);
                if (it != mappedMiniClusterIds.end()) {
                    ids.second = it->second;
                }
            }

            // Resize the mini centroids
            miniCentroids.resize((lastCentroidId + 1) * dim);
            miniClusters.resize(lastCentroidId + 1);
            miniClusterVectorIds.resize(lastCentroidId + 1);
        }

        // Upadate the ids in miniCentroidIds using the newToOldCentroidIdMap
        for (auto & ids : miniCentroidIds) {
            for (auto &id: ids) {
                id = newToOldCentroidIdMap[id];
            }
        }

        // Copy the mega clusters
        return appendOrMergeMegaCentroids(oldMegaCentroids, newMegaCentroids, miniCentroidIds);
    }

    std::vector<vector_idx_t> ReclusteringIndex::appendOrMergeCentroidsQuant(
        std::vector<vector_idx_t> oldMegaCentroids,
        std::vector<float> &newMegaCentroids,
        std::vector<std::vector<vector_idx_t>> &miniCentroidIds,
        std::vector<uint8_t> &newMiniCentroids,
        std::vector<std::vector<uint8_t>> &newMiniClusters,
        std::vector<std::vector<vector_idx_t> > &newMiniClusterVectorIds) {
        auto quantDim = quantizer->codeSize;
        updateTotalDataWrittenBySystem({}, newMiniClusterVectorIds);
        // Try to copy inplace if possible otherwise append
        std::vector<vector_idx_t> oldMiniClusterIds;
        for (const int currMegaId : oldMegaCentroids) {
            for (const auto &megaMiniId: megaMiniCentroidIds[currMegaId]) {
                oldMiniClusterIds.push_back(megaMiniId);
            }
        }

        // Copy the mini centroids, clusters and vector ids and fix the miniClusterIds
        std::unordered_map<vector_idx_t, vector_idx_t> newToOldCentroidIdMap;
        auto newMiniCentroidsSize = newMiniCentroids.size() / quantDim;
        // assert(oldMiniClusterIds.size() <= newMiniCentroidsSize);
        auto miniCentroidsSize = std::min(newMiniCentroidsSize, oldMiniClusterIds.size());
        for (int i = 0; i < miniCentroidsSize; i++) {
            auto oldCentroidId = oldMiniClusterIds[i];
            // Copy the centroid
            memcpy(quantizedMiniCentroids.data() + oldCentroidId * quantDim, newMiniCentroids.data() + i * quantDim, quantDim * sizeof(uint8_t));
            // Move the cluster
            auto currCluster = newMiniClusters[i];
            auto currVectorId = newMiniClusterVectorIds[i];
            quantizedMiniClusters[oldCentroidId] = std::move(currCluster);
            miniClusterVectorIds[oldCentroidId] = std::move(currVectorId);
            newToOldCentroidIdMap[i] = oldCentroidId;
        }

        if (newMiniCentroidsSize > miniCentroidsSize) {
            // Append the new mini centroids
            auto currentSize = quantizedMiniCentroids.size() / quantDim;
            quantizedMiniCentroids.resize((currentSize + newMiniCentroidsSize - miniCentroidsSize) * quantDim);
            memcpy(quantizedMiniCentroids.data() + currentSize * quantDim, newMiniCentroids.data() + miniCentroidsSize * quantDim,
                   (newMiniCentroidsSize - miniCentroidsSize) * quantDim * sizeof(uint8_t));

            // Append the new clusters
            quantizedMiniClusters.resize(currentSize + newMiniCentroidsSize - miniCentroidsSize);
            miniClusterVectorIds.resize(currentSize + newMiniCentroidsSize - miniCentroidsSize);
            auto idx = 0;
            printf("miniCentroidsSize: %lu, newMiniCentroidsSize: %lu, newMiniClusters.size: %lu\n", miniCentroidsSize,
                   newMiniCentroidsSize, newMiniClusters.size());
            for (auto i = miniCentroidsSize; i < newMiniCentroidsSize; i++) {
                auto currCluster = newMiniClusters[i];
                auto currVectorId = newMiniClusterVectorIds[i];
                quantizedMiniClusters[currentSize + idx] = std::move(currCluster);
                miniClusterVectorIds[currentSize + idx] = std::move(currVectorId);
                newToOldCentroidIdMap[i] = currentSize + idx;
                idx++;
            }
        } else {
            std::unordered_map<vector_idx_t, vector_idx_t> mappedMiniClusterIds;
            auto lastCentroidId = (quantizedMiniCentroids.size() / quantDim) - 1;
            // If the new mini centroid smaller than oldMiniClusterIds.size()
            for (int i = newMiniCentroidsSize; i < oldMiniClusterIds.size(); i++) {
                // Copy from last to i
                auto currCentroidId = oldMiniClusterIds[i];
                while (std::find(oldMiniClusterIds.begin() + newMiniCentroidsSize, oldMiniClusterIds.end(), lastCentroidId) != oldMiniClusterIds.end()) {
                    lastCentroidId--;
                }
                if (currCentroidId > lastCentroidId) {
                    // No need to delete from megaMiniCentroidIds since it'll be taken care when we append mega centroids.
                    continue;
                }
                memcpy(quantizedMiniCentroids.data() + currCentroidId * quantDim, quantizedMiniCentroids.data() + (lastCentroidId * quantDim), quantDim * sizeof(uint8_t));
                quantizedMiniClusters[currCentroidId] = std::move(quantizedMiniClusters[lastCentroidId]);
                miniClusterVectorIds[currCentroidId] = std::move(miniClusterVectorIds[lastCentroidId]);
                mappedMiniClusterIds[lastCentroidId] = currCentroidId;
                // printf("Removing mini centroid %d with miniCentroid %d\n", lastCentroidId, currCentroidId);
                lastCentroidId--;
            }
            // Update mega mini centroid ids
            for (auto &ids : megaMiniCentroidIds) {
                for (auto &id: ids) {
                    auto it = mappedMiniClusterIds.find(id);
                    if (it != mappedMiniClusterIds.end()) {
                        id = it->second;
                    }
                }
            }
            // Resize the mini centroids
            quantizedMiniCentroids.resize((lastCentroidId + 1) * quantDim);
            miniClusters.resize(lastCentroidId + 1);
            miniClusterVectorIds.resize(lastCentroidId + 1);
        }

        // Upadate the ids in miniCentroidIds using the newToOldCentroidIdMap
        for (auto & ids : miniCentroidIds) {
            for (auto &id: ids) {
                id = newToOldCentroidIdMap[id];
            }
        }

        // Copy the mega clusters
        return appendOrMergeMegaCentroids(oldMegaCentroids, newMegaCentroids, miniCentroidIds);
    }

    void ReclusteringIndex::storeScoreForMegaClusters(int n) {
        printf("ReclusteringIndex::storeScoreForMegaClusters\n");
        auto numMegaCentroids = megaCentroids.size() / dim;
        megaClusteringScore.resize(numMegaCentroids);
        auto numMiniClusters = miniCentroids.size() / dim;
        miniClusteringScore.resize(numMiniClusters);
        auto numToCalc = std::min(n, (int)numMegaCentroids);
        printf("numToCalc: %d\n", numToCalc);
        for (auto i = 0; i < numToCalc; i++) {
            megaClusteringScore[i] = calcScoreForMegaCluster(i);
        }
    }

    void ReclusteringIndex::storeMSEScoreForMegaClusters(int n) {
        printf("ReclusteringIndex::storeScoreForMegaClusters\n");
        auto numMegaCentroids = megaCentroids.size() / dim;
        megaClusteringScore.resize(numMegaCentroids);
        auto numMiniClusters = miniCentroids.size() / dim;
        miniClusteringScore.resize(numMiniClusters);
        auto numToCalc = std::min(n, (int)numMegaCentroids);
        printf("numToCalc: %d\n", numToCalc);
        for (auto i = 0; i < numToCalc; i++) {
            megaClusteringScore[i] = calcMSEScoreForMegaCluster(i);
        }
    }

    void ReclusteringIndex::calculateOverlapScoreForL2(int megaCentroidId) {
        auto &miniIds = megaMiniCentroidIds[megaCentroidId];
        auto numMiniClusters = miniCentroids.size() / dim;
        auto dc = getDistanceComputer(miniCentroids.data(), numMiniClusters, L2);
        
        // For each mini cluster, find the closest mini cluster in this mega cluster
        std::vector<double> approxOverlapScores(miniIds.size());
        std::vector<double> realOverlapScores(miniIds.size(), 0.0);  // Initialize to 0, only compute for worst k
        
        // Store topKMiniIds for each mini cluster (needed for real overlap calculation)
        std::vector<std::vector<vector_idx_t>> allTopKMiniIds(miniIds.size());
        
        // First pass: calculate all approx overlap scores
        for (size_t idx = 0; idx < miniIds.size(); idx++) {
            auto miniId = miniIds[idx];
            std::vector<std::pair<double, vector_idx_t>> overlapRatiosWithIds;
            overlapRatiosWithIds.reserve(miniIds.size() - 1);
            dc->setQuery(miniCentroids.data() + static_cast<size_t>(miniId) * dim);
            for (const auto j : miniIds) {
                if (j == miniId) {
                    continue;
                }
                double dist;
                dc->computeDistance(j, &dist);
                auto radiusSum = std::sqrt(miniClusteringScore[j]) + std::sqrt(miniClusteringScore[miniId]);
                auto overlapRatio = (radiusSum > 1e-9) ? (std::sqrt(dist) / radiusSum) : 0.0;
                overlapRatiosWithIds.emplace_back(overlapRatio, j);
            }

            // Sort overlap ratios and take the 10 lowest
            std::sort(overlapRatiosWithIds.begin(), overlapRatiosWithIds.end(),
                      [](const std::pair<double, size_t> &a, const std::pair<double, size_t> &b) {
                          return a.first < b.first;
                      });
            int k = std::min(10, static_cast<int>(overlapRatiosWithIds.size()));
            std::vector<double> topKOverlapRatios;
            std::vector<vector_idx_t> topKMiniIds;
            for (int i = 0; i < k; i++) {
                topKOverlapRatios.push_back(overlapRatiosWithIds[i].first);
                topKMiniIds.push_back(overlapRatiosWithIds[i].second);
            }
            
            approxOverlapScores[idx] = mergeOverlapScores(topKOverlapRatios);
            allTopKMiniIds[idx] = std::move(topKMiniIds);
        }
        
        // Find k worst mini IDs based on approx overlap scores (lowest approx scores are worst)
        constexpr int kWorstForReal = 10;  // Only calculate real overlap for k worst mini clusters
        int numWorst = std::min(kWorstForReal, static_cast<int>(miniIds.size()));
        
        std::vector<size_t> worstIndices(miniIds.size());
        std::iota(worstIndices.begin(), worstIndices.end(), 0);
        std::partial_sort(worstIndices.begin(), worstIndices.begin() + numWorst, worstIndices.end(),
                          [&approxOverlapScores](size_t a, size_t b) { 
                              return approxOverlapScores[a] < approxOverlapScores[b]; 
                          });
        
        // Second pass: calculate real overlap scores only for the k worst mini clusters
        for (int i = 0; i < numWorst; i++) {
            size_t idx = worstIndices[i];
            auto miniId = miniIds[idx];
            realOverlapScores[idx] = calculateRealOverlapScore(miniId, allTopKMiniIds[idx]);
        }

        auto avgOverlapRatio = computeAvg(approxOverlapScores);
        auto powerAvgOverlapRatio = computePowerAvgOnWorstElement(approxOverlapScores);
        
        // Calculate real overlap stats only from the k worst elements that have real scores
        std::vector<double> worstRealOverlapScores;
        worstRealOverlapScores.reserve(numWorst);
        for (int i = 0; i < numWorst; i++) {
            worstRealOverlapScores.push_back(realOverlapScores[worstIndices[i]]);
        }
        auto avgRealOverlapScore = computeAvg(worstRealOverlapScores);
        auto powerAvgRealOverlapScore = computePowerAvgOnWorstElement(worstRealOverlapScores);

        // Calculate aggregated statistics for worst elements (based on approx scores)
        int k = std::min(config.workElementsForAveraging, numWorst);
        
        // Aggregate statistics for worst k elements
        double worstApproxMin = std::numeric_limits<double>::max();
        double worstApproxMax = std::numeric_limits<double>::lowest();
        double worstRealMin = std::numeric_limits<double>::max();
        double worstRealMax = std::numeric_limits<double>::lowest();
        double worstApproxSum = 0.0;
        double worstRealSum = 0.0;
        
        for (int i = 0; i < k; i++) {
            auto idx = worstIndices[i];
            double approxScore = approxOverlapScores[idx];
            double realScore = realOverlapScores[idx];
            
            worstApproxMin = std::min(worstApproxMin, approxScore);
            worstApproxMax = std::max(worstApproxMax, approxScore);
            worstRealMin = std::min(worstRealMin, realScore);
            worstRealMax = std::max(worstRealMax, realScore);
            worstApproxSum += approxScore;
            worstRealSum += realScore;
        }
        
        double worstApproxAvg = (k > 0) ? worstApproxSum / k : 0.0;
        double worstRealAvg = (k > 0) ? worstRealSum / k : 0.0;

        // Print aggregated statistics
        /*
        printf("Mega Centroid %d [%zu mini clusters]:\n", megaCentroidId, miniIds.size());
        printf("  Overall Stats - Approx Overlap: avg=%.4f, power_avg=%.4f | Real Overlap: avg=%.4f, power_avg=%.4f\n",
               avgOverlapRatio, powerAvgOverlapRatio, avgRealOverlapScore, powerAvgRealOverlapScore);
        printf("  Worst %d Elements - Approx Overlap: min=%.4f, max=%.4f, avg=%.4f, power_avg=%.4f | Real Overlap: min=%.4f, max=%.4f, avg=%.4f, power_avg=%.4f\n",
               k, worstApproxMin, worstApproxMax, worstApproxAvg, powerAvgOverlapRatio,
               worstRealMin, worstRealMax, worstRealAvg, powerAvgRealOverlapScore);
        */
        avgRealOverlapScores[megaCentroidId] = powerAvgRealOverlapScore;
        overlapScores[megaCentroidId] = powerAvgOverlapRatio;
    }

    void ReclusteringIndex::calculateOverlapScoreForAngular(int megaCentroidId) {
        auto &miniIds = megaMiniCentroidIds[megaCentroidId];
        auto numMiniClusters = miniCentroids.size() / dim;
        auto dc = getDistanceComputer(miniCentroids.data(), numMiniClusters, COSINE);

        // For each mini cluster, find the closest mini cluster in this mega cluster
        std::vector<double> approxOverlapScores(miniIds.size());
        std::vector<double> realOverlapScores(miniIds.size(), 0.0);  // Initialize to 0, only compute for worst k

        // First pass: calculate all approx overlap scores
        for (size_t idx = 0; idx < miniIds.size(); idx++) {
            auto miniId = miniIds[idx];
            std::vector<std::pair<double, vector_idx_t>> overlapRatiosWithIds;
            overlapRatiosWithIds.reserve(miniIds.size() - 1);
            dc->setQuery(miniCentroids.data() + static_cast<size_t>(miniId) * dim);

            // Precompute angular radius for miniId
            // miniClusteringScore stores average cosine distance (1 - cos_sim)
            // Convert to angular distance: acos(cos_sim) = acos(1 - cosine_dist)
            double angularRadiusMiniId = std::acos(std::clamp(1.0 - miniClusteringScore[miniId], -1.0, 1.0));

            for (const auto j : miniIds) {
                if (j == miniId) {
                    continue;
                }

                double cosineDist;
                dc->computeDistance(j, &cosineDist);

                // Convert cosine distance to angular distance
                // cosineDist = 1 - cos_sim, so cos_sim = 1 - cosineDist
                // angular distance = acos(cos_sim) = acos(1 - cosineDist)
                double angularDist = std::acos(std::clamp(1.0 - cosineDist, -1.0, 1.0));

                // Convert cluster spread (avg cosine distance) to angular radius
                double angularRadiusJ = std::acos(std::clamp(1.0 - miniClusteringScore[j], -1.0, 1.0));

                // Sum of angular radii - this is geometrically correct on the hypersphere
                double angularRadiusSum = angularRadiusJ + angularRadiusMiniId;

                // Overlap ratio: distance between centroids / sum of radii
                // < 1 means clusters overlap, > 1 means they don't
                double overlapRatio = (angularRadiusSum > 1e-9) ? (angularDist / angularRadiusSum) : 0.0;

                overlapRatiosWithIds.emplace_back(overlapRatio, j);
            }

            // Sort overlap ratios and take the 10 lowest (most overlapping)
            std::sort(overlapRatiosWithIds.begin(), overlapRatiosWithIds.end(),
                      [](const std::pair<double, size_t> &a, const std::pair<double, size_t> &b) {
                          return a.first < b.first;
                      });
            int k = std::min(10, static_cast<int>(overlapRatiosWithIds.size()));
            std::vector<double> topKOverlapRatios;
            std::vector<vector_idx_t> topKMiniIds;
            for (int i = 0; i < k; i++) {
                topKOverlapRatios.push_back(overlapRatiosWithIds[i].first);
                topKMiniIds.push_back(overlapRatiosWithIds[i].second);
            }

            approxOverlapScores[idx] = mergeOverlapScores(topKOverlapRatios);
        }
        
        // Find k worst mini IDs based on approx overlap scores (lowest approx scores are worst)
        constexpr int kWorstForReal = 10;  // Only calculate real overlap for k worst mini clusters
        int numWorst = std::min(kWorstForReal, static_cast<int>(miniIds.size()));
        
        std::vector<size_t> worstIndices(miniIds.size());
        std::iota(worstIndices.begin(), worstIndices.end(), 0);
        std::partial_sort(worstIndices.begin(), worstIndices.begin() + numWorst, worstIndices.end(),
                          [&approxOverlapScores](size_t a, size_t b) { 
                              return approxOverlapScores[a] < approxOverlapScores[b]; 
                          });
        
        // Second pass: calculate real overlap scores only for the k worst mini clusters
        for (int i = 0; i < numWorst; i++) {
            size_t idx = worstIndices[i];
            auto miniId = miniIds[idx];
            realOverlapScores[idx] = calculateRealOverlapScoreForAngular(miniId, miniIds);
        }

        auto avgOverlapRatio = computeAvg(approxOverlapScores);
        auto powerAvgOverlapRatio = computePowerAvgOnWorstElement(approxOverlapScores);
        
        // Calculate real overlap stats only from the k worst elements that have real scores
        std::vector<double> worstRealOverlapScores;
        worstRealOverlapScores.reserve(numWorst);
        for (int i = 0; i < numWorst; i++) {
            worstRealOverlapScores.push_back(realOverlapScores[worstIndices[i]]);
        }
        auto avgRealOverlapScore = computeAvg(worstRealOverlapScores);
        auto powerAvgRealOverlapScore = computePowerAvgOnWorstElement(worstRealOverlapScores);

        // Calculate aggregated statistics for worst elements (based on approx scores)
        int k = std::min(config.workElementsForAveraging, numWorst);

        // Aggregate statistics for worst k elements
        double worstApproxMin = std::numeric_limits<double>::max();
        double worstApproxMax = std::numeric_limits<double>::lowest();
        double worstRealMin = std::numeric_limits<double>::max();
        double worstRealMax = std::numeric_limits<double>::lowest();
        double worstApproxSum = 0.0;
        double worstRealSum = 0.0;

        for (int i = 0; i < k; i++) {
            auto idx = worstIndices[i];
            double approxScore = approxOverlapScores[idx];
            double realScore = realOverlapScores[idx];

            worstApproxMin = std::min(worstApproxMin, approxScore);
            worstApproxMax = std::max(worstApproxMax, approxScore);
            worstRealMin = std::min(worstRealMin, realScore);
            worstRealMax = std::max(worstRealMax, realScore);
            worstApproxSum += approxScore;
            worstRealSum += realScore;
        }

        double worstApproxAvg = (k > 0) ? worstApproxSum / k : 0.0;
        double worstRealAvg = (k > 0) ? worstRealSum / k : 0.0;

        // Print aggregated statistics
        printf("Mega Centroid %d [%zu mini clusters] (Angular):\n", megaCentroidId, miniIds.size());
        printf("  Overall Stats - Approx Overlap: avg=%.4f, power_avg=%.4f | Real Overlap: avg=%.4f, power_avg=%.4f\n",
               avgOverlapRatio, powerAvgOverlapRatio, avgRealOverlapScore, powerAvgRealOverlapScore);
        printf("  Worst %d Elements - Approx Overlap: min=%.4f, max=%.4f, avg=%.4f, power_avg=%.4f | Real Overlap: min=%.4f, max=%.4f, avg=%.4f, power_avg=%.4f\n",
               k, worstApproxMin, worstApproxMax, worstApproxAvg, powerAvgOverlapRatio,
               worstRealMin, worstRealMax, worstRealAvg, powerAvgRealOverlapScore);

        avgRealOverlapScores[megaCentroidId] = powerAvgRealOverlapScore;
        overlapScores[megaCentroidId] = powerAvgOverlapRatio;
    }

    double ReclusteringIndex::calculateRealOverlapScoreForAngular(vector_idx_t miniCentroidId,
                                                               std::vector<vector_idx_t> &closestMiniIds) {
        auto numMiniClusters = miniCentroids.size() / dim;
        auto dc = getDistanceComputer(miniCentroids.data(), numMiniClusters, COSINE);
        double avgScore = 0.0;
        auto &miniClusterVectors = miniClusters[miniCentroidId];
        auto miniClusterSize = miniClusterVectors.size() / dim;

        for (int i = 0; i < miniClusterSize; i++) {
            dc->setQuery(miniClusterVectors.data() + static_cast<size_t>(i) * dim);

            // Distance to own centroid
            double ownCosineDist = 0.0;
            dc->computeDistance(miniCentroidId, &ownCosineDist);
            double ownAngularDist = std::acos(std::clamp(1.0 - ownCosineDist, -1.0, 1.0));

            // Find minimum distance to other centroids
            double minAngularDist = std::numeric_limits<double>::max();
            double minCosineDist = std::numeric_limits<double>::max();
            for (const auto &closestMiniCentroidId : closestMiniIds) {
                if (closestMiniCentroidId == miniCentroidId) {
                    continue;
                }
                double cosineDist;
                dc->computeDistance(closestMiniCentroidId, &cosineDist);
                double angularDist = std::acos(std::clamp(1.0 - cosineDist, -1.0, 1.0));
                if (angularDist < minAngularDist) {
                    minAngularDist = angularDist;
                }
                if (cosineDist < minCosineDist) {
                    minCosineDist = cosineDist;
                }
            }

            // Silhouette-like score using angular distances
            // Positive = point is closer to own centroid, Negative = closer to other centroid
            // avgScore += (minAngularDist - ownAngularDist) / std::max(minAngularDist, ownAngularDist);
            avgScore += (minCosineDist - ownCosineDist) / std::max(minCosineDist, ownCosineDist);
        }

        avgScore /= static_cast<double>(miniClusterSize);
        return avgScore;
    }

    double ReclusteringIndex::calculateRealOverlapScore(vector_idx_t miniCentroidId,
                                                        std::vector<vector_idx_t> &closestMiniIds) {
        auto numMiniClusters = miniCentroids.size() / dim;
        auto dc = getDistanceComputer(miniCentroids.data(), numMiniClusters);
        double avgScore = 0.0;
        auto &miniClusterVectors = miniClusters[miniCentroidId];
        auto miniClusterSize = miniClusterVectors.size() / dim;
        for (int i = 0; i < miniClusterSize; i++) {
            dc->setQuery(miniClusterVectors.data() + static_cast<size_t>(i) * dim);
            double ownDist = 0.0;
            dc->computeDistance(miniCentroidId, &ownDist);
            double minDistance = std::numeric_limits<double>::max();
            for (const auto &closestMiniCentroidId : closestMiniIds) {
                double dist;
                dc->computeDistance(closestMiniCentroidId, &dist);
                if (dist < minDistance) {
                    minDistance = dist;
                }
            }
            avgScore += (minDistance - ownDist) / std::max(minDistance, ownDist);
        }
        avgScore /= static_cast<double>(miniClusterSize);
        return avgScore;
    }

    void ReclusteringIndex::computeOverlapScores() {
        auto numMegaCentroids = megaCentroids.size() / dim;
        overlapScores.resize(numMegaCentroids);
        avgRealOverlapScores.resize(numMegaCentroids);
#pragma omp parallel for
        for (auto i = 0; i < numMegaCentroids; i++) {
            if (config.distanceType == COSINE || config.distanceType == IP) {
                calculateOverlapScoreForAngular(i);
            } else {
                // L2 and IP (IP should ideally use normalized vectors)
                calculateOverlapScoreForL2(i);
            }
        }
    }

    void ReclusteringIndex::saveOldScoreForMegaClusters() {
        printf("ReclusteringIndex::saveOldScoreForMegaClusters\n");
        auto numMegaCentroids = megaCentroids.size() / dim;
        oldMegaClusteringScore.resize(numMegaCentroids);
        oldMegaCentroids.resize(megaCentroids.size());
        for (auto i = 0; i < numMegaCentroids; i++) {
            oldMegaClusteringScore[i] = megaClusteringScore[i];
        }
        memcpy(oldMegaCentroids.data(), megaCentroids.data(), megaCentroids.size() * sizeof(float));
    }

    std::vector<vector_idx_t> ReclusteringIndex::getMegaCentroidsToRecluster() const {
        std::vector<vector_idx_t> megaCentroidsToRecluster;

        if (oldMegaCentroids.empty() || oldMegaClusteringScore.empty()) {
            printf("No old mega centroid or score to compare! Returning all megacentroids.\n");
            // Return all mega centroids if no old data exists
            auto numMegaCentroids = megaCentroids.size() / dim;
            for (size_t i = 0; i < numMegaCentroids; i++) {
                megaCentroidsToRecluster.push_back(i);
            }
            return megaCentroidsToRecluster;
        }

        auto numMegaCentroids = megaCentroids.size() / dim;
        auto numOldMegaCentroids = oldMegaCentroids.size() / dim;
        auto dc = getDistanceComputer(oldMegaCentroids.data(), numOldMegaCentroids);

        // Calculate centroid of all old centroids
        std::vector<float> oldCentroidsMean(dim, 0.0f);
        for (size_t i = 0; i < numOldMegaCentroids; i++) {
            const float* centroid = oldMegaCentroids.data() + i * dim;
            for (size_t d = 0; d < dim; d++) {
                oldCentroidsMean[d] += centroid[d];
            }
        }
        for (size_t d = 0; d < dim; d++) {
            oldCentroidsMean[d] /= static_cast<float>(numOldMegaCentroids);
        }

        // Check each mega centroid against the criteria
        for (auto i = 0; i < numMegaCentroids; i++) {
            // Find closest old mega centroid
            dc->setQuery(megaCentroids.data() + static_cast<size_t>(i) * dim);
            double minDistance = std::numeric_limits<double>::max();
            int oldCentroidId = -1;
            for (size_t j = 0; j < numOldMegaCentroids; j++) {
                double dist;
                dc->computeDistance(j, &dist);
                if (dist < minDistance) {
                    minDistance = dist;
                    oldCentroidId = j;
                }
            }

            if (oldCentroidId != -1) {
                const float* oldCentroid = oldMegaCentroids.data() + static_cast<size_t>(oldCentroidId) * dim;
                // Calculate distance from old centroid to centroid of all old centroids
                double oldDistFromCentroid = 0.0;
                for (size_t d = 0; d < dim; d++) {
                    double diff = oldCentroid[d] - oldCentroidsMean[d];
                    oldDistFromCentroid += diff * diff;
                }
                oldDistFromCentroid = std::sqrt(oldDistFromCentroid);

                // Calculate relative change (centroid-based)
                double relativeChangeCentroid = (oldDistFromCentroid > 1e-9) ?
                    (std::sqrt(minDistance) / oldDistFromCentroid) : 0.0;

                // Calculate relative score change
                double scoreChange = megaClusteringScore[i] - oldMegaClusteringScore[oldCentroidId];
                double relativeScoreChange = (std::abs(oldMegaClusteringScore[oldCentroidId]) > 1e-9) ?
                    (scoreChange / std::abs(oldMegaClusteringScore[oldCentroidId])) : 0.0;

                // Check criteria
                if (std::abs(relativeScoreChange) > config.scoreChangeThreshold || relativeChangeCentroid > config.
                    centroidChangeThreshold) {
                    megaCentroidsToRecluster.push_back(i);
                }
            } else {
                megaCentroidsToRecluster.push_back(i);
            }
        }

        printf("getMegaCentroidsToRecluster: %zu out of %zu megacentroids meet criteria\n",
               megaCentroidsToRecluster.size(), numMegaCentroids);

        return megaCentroidsToRecluster;
    }

    void ReclusteringIndex::printChangeClusterStats() {
        if (oldMegaCentroids.empty() || oldMegaClusteringScore.empty()) {
            printf("No old mega centroid or score to compare!\n");
            return;
        }

        // printf("ReclusteringIndex::printChangeClusterStats\n");
        auto numMegaCentroids = megaCentroids.size() / dim;
        auto numOldMegaCentroids = oldMegaCentroids.size() / dim;
        auto dc = getDistanceComputer(oldMegaCentroids.data(), numOldMegaCentroids);

        // Calculate centroid of all old centroids
        std::vector<float> oldCentroidsMean(dim, 0.0f);
        for (size_t i = 0; i < numOldMegaCentroids; i++) {
            const float* centroid = oldMegaCentroids.data() + i * dim;
            for (size_t d = 0; d < dim; d++) {
                oldCentroidsMean[d] += centroid[d];
            }
        }
        for (size_t d = 0; d < dim; d++) {
            oldCentroidsMean[d] /= numOldMegaCentroids;
        }

        // Define thresholds
        const std::vector<double> changeThresholds = {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0};
        const std::vector<double> scoreThresholds = {0.01, 0.1, 0.15, 0.2, 0.25, 1.0};

        // Statistics counters using vectors
        std::vector<int> countRelativeChange(changeThresholds.size(), 0);
        std::vector<int> countRelativeChangeCentroid(changeThresholds.size(), 0);
        std::vector<int> countRelativeScoreChange(scoreThresholds.size(), 0);

        double totalRelativeChange = 0.0;
        double totalRelativeChangeCentroid = 0.0;
        double totalRelativeScoreChange = 0.0;
        int validCentroids = 0;
        int shouldReclusterCount = 0;
        int shouldReclusterCountWithOverlapScore = 0;

        for (auto i = 0; i < numMegaCentroids; i++) {
            // Find closest old mega centroid
            dc->setQuery(megaCentroids.data() + static_cast<size_t>(i) * dim);
            double minDistance = std::numeric_limits<double>::max();
            int oldCentroidId = -1;
            for (size_t j = 0; j < numOldMegaCentroids; j++) {
                double dist;
                dc->computeDistance(j, &dist);
                if (dist < minDistance) {
                    minDistance = dist;
                    oldCentroidId = j;
                }
            }
            if (oldCentroidId != -1) {
                // Calculate the norm of the old centroid for relative change (from origin)
                const float* oldCentroid = oldMegaCentroids.data() + static_cast<size_t>(oldCentroidId) * dim;
                double oldNorm = 0.0;
                for (size_t d = 0; d < dim; d++) {
                    oldNorm += oldCentroid[d] * oldCentroid[d];
                }
                oldNorm = std::sqrt(oldNorm);

                // Calculate distance from old centroid to centroid of all old centroids
                double oldDistFromCentroid = 0.0;
                for (size_t d = 0; d < dim; d++) {
                    double diff = oldCentroid[d] - oldCentroidsMean[d];
                    oldDistFromCentroid += diff * diff;
                }
                oldDistFromCentroid = std::sqrt(oldDistFromCentroid);

                // Calculate relative change (origin-based): ||new - old|| / ||old||
                // minDistance is squared distance, so take sqrt
                double relativeChange = (oldNorm > 0) ? (std::sqrt(minDistance) / oldNorm) : 0.0;

                // Calculate relative change (centroid-based): ||new - old|| / ||old - centroid||
                double relativeChangeCentroid = (oldDistFromCentroid > 1e-9) ?
                    (std::sqrt(minDistance) / oldDistFromCentroid) : 0.0;

                // Calculate relative score change: (new - old) / |old| (preserves sign)
                double scoreChange = megaClusteringScore[i] - oldMegaClusteringScore[oldCentroidId];
                double relativeScoreChange = (std::abs(oldMegaClusteringScore[oldCentroidId]) > 1e-9) ?
                    (scoreChange / std::abs(oldMegaClusteringScore[oldCentroidId])) : 0.0;

                /*
                printf("Mega Centroid %d: Old id = %d, Dist to Old = %.6f, RelChange(origin) = %.6f, RelChange(centroid) = %.6f, Score = %.4f->%.4f, RelScoreChange = %.6f\n",
                       i,
                       oldCentroidId,
                       std::sqrt(minDistance),
                       relativeChange,
                       relativeChangeCentroid,
                       oldMegaClusteringScore[oldCentroidId],
                       megaClusteringScore[i],
                       relativeScoreChange);
                */

                // Update statistics
                validCentroids++;
                totalRelativeChange += relativeChange;
                totalRelativeChangeCentroid += relativeChangeCentroid;
                totalRelativeScoreChange += relativeScoreChange;

                // Count for each threshold using loops
                for (size_t t = 0; t < changeThresholds.size(); t++) {
                    if (relativeChange < changeThresholds[t]) countRelativeChange[t]++;
                    if (relativeChangeCentroid < changeThresholds[t]) countRelativeChangeCentroid[t]++;
                }

                for (size_t t = 0; t < scoreThresholds.size(); t++) {
                    if (std::abs(relativeScoreChange) < scoreThresholds[t]) countRelativeScoreChange[t]++;
                }

                if (std::abs(relativeScoreChange) < config.scoreChangeThreshold && relativeChangeCentroid < config.
                    centroidChangeThreshold) {
                    shouldReclusterCount++;
                    
                    // Check overlap score if available
                    if (i < overlapScores.size() && overlapScores[i] > config.overlappingScoreThreshold) {
                        shouldReclusterCountWithOverlapScore++;
                    }
                }

            } else {
                // printf("Mega Centroid %d: No old centroid found!\n", i);
            }
        }

        // Print aggregated statistics
        if (validCentroids > 0) {
            // printf("\n=== Aggregated Statistics ===\n");
            // printf("Total centroids: %d\n", validCentroids);
            // printf("Average relative change (origin-based): %.6f\n", totalRelativeChange / validCentroids);
            // printf("Average relative change (centroid-based): %.6f\n", totalRelativeChangeCentroid / validCentroids);
            //printf("Average relative score change: %.6f\n", totalRelativeScoreChange / validCentroids);

            /*
            printf("\n=== Origin-Based Relative Change Distribution ===\n");
            for (size_t t = 0; t < changeThresholds.size(); t++) {
                printf("  < %.2f (%.0f%%):\t%d (%.1f%%)\n",
                       changeThresholds[t],
                       changeThresholds[t] * 100,
                       countRelativeChange[t],
                       100.0 * countRelativeChange[t] / validCentroids);
            }

            printf("\n=== Centroid-Based Relative Change Distribution ===\n");
            for (size_t t = 0; t < changeThresholds.size(); t++) {
                printf("  < %.2f (%.0f%%):\t%d (%.1f%%)\n",
                       changeThresholds[t],
                       changeThresholds[t] * 100,
                       countRelativeChangeCentroid[t],
                       100.0 * countRelativeChangeCentroid[t] / validCentroids);
            }

            printf("\n=== Relative Score Change Distribution (absolute) ===\n");
            for (size_t t = 0; t < scoreThresholds.size(); t++) {
                printf("  < %.2f:\t%d (%.1f%%)\n",
                       scoreThresholds[t],
                       countRelativeScoreChange[t],
                       100.0 * countRelativeScoreChange[t] / validCentroids);
            }
            */
            
            // Print overlap score statistics if available
            /*if (overlapScores.size() == numMegaCentroids) {
                double overlapMin = std::numeric_limits<double>::max();
                double overlapMax = std::numeric_limits<double>::lowest();
                double overlapSum = 0.0;
                int overlapValidCount = 0;
                
                for (size_t i = 0; i < overlapScores.size(); i++) {
                    double score = overlapScores[i];
                    overlapMin = std::min(overlapMin, score);
                    overlapMax = std::max(overlapMax, score);
                    overlapSum += score;
                    overlapValidCount++;
                }
                
                double overlapAvg = (overlapValidCount > 0) ? overlapSum / overlapValidCount : 0.0;
                double overlapPowerAvg = computePowerAvgOnWorstElement(overlapScores);
                printf("\n=== Overlap Score Statistics ===\n");
                printf("Overlap scores available for %d centroids: ", overlapValidCount);
                printf("Min: %.6f, Max: %.6f, Avg: %.6f, Power Avg (worst): %.6f, ",
                       overlapMin, overlapMax, overlapAvg, overlapPowerAvg);
                printf("Threshold: %.6f\n", config.overlappingScoreThreshold);
            } else {
                printf("\n=== Overlap Score Statistics ===\n");
                printf("Overlap scores not available (expected %lu, got %zu)\n", numMegaCentroids, overlapScores.size());
            }*/

            // Print MSE score of L2 clusters (mega) statistics if available
            if (megaClusteringScore.size() == numMegaCentroids) {
                double mseMin = std::numeric_limits<double>::max();
                double mseMax = std::numeric_limits<double>::lowest();
                double mseSum = 0.0;
                int mseValidCount = 0;
                
                for (size_t i = 0; i < megaClusteringScore.size(); i++) {
                    double score = megaClusteringScore[i];
                    mseMin = std::min(mseMin, score);
                    mseMax = std::max(mseMax, score);
                    mseSum += score;
                    mseValidCount++;
                }
                
                double mseAvg = (mseValidCount > 0) ? mseSum / mseValidCount : 0.0;
                printf("\n=== MSE Score Statistics (L2 Clusters) ===\n");
                printf("MSE scores available for %d centroids: ", mseValidCount);
                printf("Min: %.6f, Max: %.6f, Avg: %.6f\n",
                       mseMin, mseMax, mseAvg);
            } else {
                printf("\n=== MSE Score Statistics (L2 Clusters) ===\n");
                printf("MSE scores not available (expected %lu, got %zu)\n", numMegaCentroids, megaClusteringScore.size());
            }

            // Print MSE score of L1 clusters (mini) statistics if available
            auto numMiniCentroids = miniCentroids.size() / dim;
            if (miniClusteringScore.size() == numMiniCentroids) {
                double mseMin = std::numeric_limits<double>::max();
                double mseMax = std::numeric_limits<double>::lowest();
                double mseSum = 0.0;
                int mseValidCount = 0;
                
                for (size_t i = 0; i < miniClusteringScore.size(); i++) {
                    double score = miniClusteringScore[i];
                    mseMin = std::min(mseMin, score);
                    mseMax = std::max(mseMax, score);
                    mseSum += score;
                    mseValidCount++;
                }
                
                double mseAvg = (mseValidCount > 0) ? mseSum / mseValidCount : 0.0;
                printf("\n=== MSE Score Statistics (L1 Clusters) ===\n");
                printf("MSE scores available for %d centroids: ", mseValidCount);
                printf("Min: %.6f, Max: %.6f, Avg: %.6f\n",
                       mseMin, mseMax, mseAvg);
            } else {
                printf("\n=== MSE Score Statistics (L1 Clusters) ===\n");
                printf("MSE scores not available (expected %lu, got %zu)\n", numMiniCentroids, miniClusteringScore.size());
            }
            
            /*
            printf("\nNumber of mega centroids that should NOT be reclustered (RelScoreChange < %.6f and RelChange(centroid) < %.6f): %d (%.1f%%)\n",
                config.scoreChangeThreshold,
                config.centroidChangeThreshold,
                shouldReclusterCount,
                100.0 * shouldReclusterCount / validCentroids);
            
            if (overlapScores.size() == numMegaCentroids) {
                printf("Number of mega centroids that should NOT be reclustered (with overlap score > %.6f): %d (%.1f%%)\n",
                       config.overlappingScoreThreshold,
                       shouldReclusterCountWithOverlapScore,
                       100.0 * shouldReclusterCountWithOverlapScore / validCentroids);
            }
            */
        }
    }


    void ReclusteringIndex::quantizeVectors() {
        printf("ReclusteringIndex::quantizeVectors\n");
        if (miniCentroids.empty()) {
            return;
        }

        // Calculate max miniClusterSize
        size_t maxMiniClusterSize = 0;
        for (const auto &miniCluster : miniClusters) {
            maxMiniClusterSize = std::max(maxMiniClusterSize, miniCluster.size() / dim);
        }
        // Allocate for normalized vectors
        std::vector<float> normalizedVectors(maxMiniClusterSize * dim);
        auto miniCentroidsSize = miniClusters.size();
        if (config.quantizationTrainPercentage >= 1) {
            // Quantize the new mini centroids
            for (size_t i = 0; i < miniCentroidsSize; i++) {
                auto &miniCluster = miniClusters[i];
                auto miniClusterSize = miniCluster.size() / dim;
                if (miniCluster.empty()) {
                    continue;
                }
                if (config.distanceType == COSINE) {
                    normalize_vectors(miniCluster.data(), dim, miniClusterSize, normalizedVectors.data());
                    quantizer->batch_train(miniClusterSize, normalizedVectors.data());
                } else {
                    quantizer->batch_train(miniClusterSize, miniCluster.data());
                }
            }
        } else {
            auto vectorsTrainedOn = 0;
            // Quantize the new mini centroids
            for (size_t i = 0; i < miniCentroidsSize; i++) {
                auto &miniCluster = miniClusters[i];
                auto miniClusterSize = miniCluster.size() / dim;
                if (miniCluster.empty()) {
                    continue;
                }
                for (size_t j = 0; j < miniClusterSize; j++) {
                    if (rg->randFloat() > config.quantizationTrainPercentage) {
                        // Skip this vector
                        continue;
                    }

                    // Train using this vector
                    if (config.distanceType == COSINE) {
                        normalize_vectors(miniCluster.data() + j * dim, dim, 1, normalizedVectors.data());
                        quantizer->batch_train(1, normalizedVectors.data());
                    } else {
                        quantizer->batch_train(1, miniCluster.data() + j * dim);
                    }
                    vectorsTrainedOn++;
                }
            }
            printf("ReclusteringIndex::quantizeVectors trained on %d vectors\n", vectorsTrainedOn);
        }

        // Finalize the quantizer
        quantizer->finalize_train();

        // Resize the quantized mini clusters
        quantizedMiniClusters.resize(miniCentroidsSize);

        // Quantize the mini clusters
        for (size_t i = 0; i < miniCentroidsSize; i++) {
            auto &miniCluster = miniClusters[i];
            auto miniClusterSize = miniCluster.size() / dim;
            quantizedMiniClusters[i].resize(miniClusterSize * quantizer->codeSize);
            if (miniClusterSize == 0) {
                continue;
            }
            if (config.distanceType == COSINE) {
                // Normalize the vectors before quantization
                normalize_vectors(miniCluster.data(), dim, miniClusterSize, normalizedVectors.data());
                quantizer->encode(normalizedVectors.data(), quantizedMiniClusters[i].data(), miniClusterSize);
            } else {
                // Directly quantize the vectors without normalization
                quantizer->encode(miniCluster.data(), quantizedMiniClusters[i].data(), miniClusterSize);
            }
        }

        // Quantize the mega centroids
        // auto numMegaCentroids = megaCentroids.size() / dim;
        // quantizedMegaCentroids.resize(numMegaCentroids * dim);
        // Copy the mega centroids to quantizedMegaCentroids
        // memcpy(quantizedMegaCentroids.data(), megaCentroids.data(), numMegaCentroids * dim * sizeof(float));
        // if (config.distanceType == COSINE) {
        //     normalizedVectors.resize(numMegaCentroids * dim);
        //     normalize_vectors(megaCentroids.data(), dim, numMegaCentroids, normalizedVectors.data());
        //     quantizer->encode(normalizedVectors.data(), quantizedMegaCentroids.data(), numMegaCentroids);
        // } else {
        //     quantizer->encode(megaCentroids.data(), quantizedMegaCentroids.data(), numMegaCentroids);
        // }

        // Quantize the mini centroids
        auto numMiniCentroids = miniCentroids.size() / dim;
        quantizedMiniCentroids.resize(numMiniCentroids * quantizer->codeSize);
        if (config.distanceType == COSINE) {
            normalizedVectors.resize(numMiniCentroids * dim);
            normalize_vectors(miniCentroids.data(), dim, numMiniCentroids, normalizedVectors.data());
            quantizer->encode(normalizedVectors.data(), quantizedMiniCentroids.data(), numMiniCentroids);
        } else {
            quantizer->encode(miniCentroids.data(), quantizedMiniCentroids.data(), numMiniCentroids);
        }
    }

    double ReclusteringIndex::calcScoreForMegaCluster(int megaClusterId) {
        auto miniCentroidIds = megaMiniCentroidIds[megaClusterId];
        double avgMiniScore = 0.0;
#pragma omp parallel for reduction(+: avgMiniScore) schedule(dynamic)
        for (auto miniCentroidId : miniCentroidIds) {
            double s = calcScoreForMiniCluster(miniCentroidId);
            miniClusteringScore[miniCentroidId] = s;
            avgMiniScore += s;
        }

        double avgMegaScore = 0.0;
        auto numMegaCentroids = megaCentroids.size() / dim;
        auto dc = getDistanceComputer(megaCentroids.data(), numMegaCentroids);
#pragma omp parallel for reduction(+: avgMegaScore) schedule(dynamic)
        for (auto miniCentroidId: miniCentroidIds) {
            dc->setQuery(miniCentroids.data() + miniCentroidId * dim);

            // 1) a = distance to own centroid
            double a = 0;
            dc->computeDistance(megaClusterId, &a);

            // 2) b = min distance to any other centroid
            double b = std::numeric_limits<double>::infinity();
            for (int j = 0; j < numMegaCentroids; j++) {
                if (j == megaClusterId) continue;
                double dist;
                dc->computeDistance(j, &dist);
                b = std::min(b, dist);
            }

            // 3) silhouette for this point
            double m = std::max(a, b);
            if (m < 0) {
                m = std::max(-a, -b);
            }
            double s = (m != 0.0) ? (b - a) / m : 0.0;
            avgMegaScore += s;
        }

        avgMegaScore /= miniCentroidIds.size();
        avgMiniScore /= miniCentroidIds.size();

        // Weight the mega silhouette score more than the mini silhouette score
        return avgMiniScore;
    }

    double ReclusteringIndex::calcScoreForMiniCluster(int miniClusterId, std::unordered_set<vector_idx_t> *closerL1s) {
        // Find 5 closest mega centroids
        std::vector<vector_idx_t> megaAssign;
        findKClosestMegaCentroids(miniCentroids.data() + miniClusterId * dim, 100, megaAssign, stats);

        // Collect centroids to check for silhouette
        std::vector<vector_idx_t> closestMiniCentroidIds;
        findKClosestMiniCentroids(miniCentroids.data() + miniClusterId * dim, 1000, megaAssign, closestMiniCentroidIds, stats);

        // Calculate the silhouette score
        double totalSilhouette = 0.0;
        long long totalPoints = 0;
        auto numMiniCentroids = miniCentroids.size() / dim;
        auto dc = getDistanceComputer(miniCentroids.data(), numMiniCentroids);
        auto& curMiniCluster = miniClusters[miniClusterId];
        auto numPoints = curMiniCluster.size() / dim;

// #pragma omp parallel for reduction(+: totalSilhouette, totalPoints) schedule(dynamic)
        for (int i = 0; i < numPoints; i++) {
            const float *curPoint = curMiniCluster.data() + i * dim;
            dc->setQuery(curPoint);

            // 1) a = distance to own centroid
            double a = 0;
            dc->computeDistance(miniClusterId, &a);

            // 2) b = min distance to any other centroid
            vector_idx_t minCentroid;
            double b = std::numeric_limits<double>::infinity();
            for (auto closestMiniCentroidId : closestMiniCentroidIds) {
                if (closestMiniCentroidId == miniClusterId) continue;
                double dist;
                dc->computeDistance(closestMiniCentroidId, &dist);
                b = std::min(b, dist);
                if (b == dist) {
                    minCentroid = closestMiniCentroidId;
                }
            }

            // 3) silhouette for this point
            double m = std::max(a, b);
            if (m < 0) {
                m = std::max(-a, -b);
            }
            double s = (m != 0.0) ? (b - a) / m : 0.0;

            if (closerL1s != nullptr && s < -0.009) {
                closerL1s->insert(minCentroid);
            }

            totalSilhouette += s;
            totalPoints += 1;
        }
        return (totalPoints > 0)
                   ? totalSilhouette / double(totalPoints)
                   : 0.0;
    }

    double ReclusteringIndex::calcMSEScoreForMegaCluster(int megaClusterId) {
        auto miniCentroidIds = megaMiniCentroidIds[megaClusterId];
        double avgMiniScore = 0.0;
#pragma omp parallel for reduction(+: avgMiniScore) schedule(dynamic)
        for (auto miniCentroidId : miniCentroidIds) {
            double s = calcMSEScoreForMiniCluster(miniCentroidId);
            miniClusteringScore[miniCentroidId] = s;
            avgMiniScore += s;
        }
        return avgMiniScore / miniCentroidIds.size();
    }

    double ReclusteringIndex::calcMSEScoreForMiniCluster(int miniClusterId) {
        auto centroid = miniCentroids.data() + miniClusterId * dim;
        auto& curMiniCluster = miniClusters[miniClusterId];
        auto numPoints = curMiniCluster.size() / dim;
        double totalMSE = 0.0;
        auto metric = config.distanceType;
        if (config.distanceType == IP) {
            metric = COSINE;
        }
        auto dc = getDistanceComputer(curMiniCluster.data(), numPoints, metric);
        dc->setQuery(centroid);
        for (int i = 0; i < numPoints; i++) {
            double dist;
            dc->computeDistance(i, &dist);
            totalMSE += dist;
        }
        
        double mse = (numPoints > 0) ? totalMSE / numPoints : 0.0;
        
        // Diagnostic: print cluster info when MSE is 0
        if (mse == 0.0 || numPoints <= 1) {
            printf("Mini cluster %d has MSE=%.6f with %lu vectors (totalMSE=%.6f)\n", 
                   miniClusterId, mse, numPoints, totalMSE);
        }
        
        return mse;
    }

    std::vector<vector_idx_t> ReclusteringIndex::appendOrMergeMegaCentroids(std::vector<vector_idx_t> oldMegaCentroidIds,
                                                      std::vector<float> &newMegaCentroids,
                                                      std::vector<std::vector<vector_idx_t> > &newMiniClusterIds) {
        updateTotalDataWrittenBySystem(newMiniClusterIds, {});
        std::vector<vector_idx_t> updatedMegaCentroids;
        auto numNewMegaCentroids = newMegaCentroids.size() / dim;
        // printf("numNewMegaCentroids: %zu, oldMegaCentroidIds: %zu\n", numNewMegaCentroids, oldMegaCentroidIds.size());
        auto oldMegaCentroidSize = oldMegaCentroidIds.size();
        auto centroidSize = std::min(oldMegaCentroidSize, numNewMegaCentroids);
        for (int i = 0; i < centroidSize; i++) {
            auto currMegaId = oldMegaCentroidIds[i];
            memcpy(megaCentroids.data() + static_cast<size_t>(currMegaId) * dim,
                   newMegaCentroids.data() + static_cast<size_t>(i) * dim,
                   dim * sizeof(float));

            // Move the miniCentroidIds
            megaMiniCentroidIds[currMegaId] = std::move(newMiniClusterIds[i]);
            updatedMegaCentroids.push_back(currMegaId);
        }

        if (numNewMegaCentroids > oldMegaCentroidSize) {
            // Append the new mega centroids
            auto currentSize = megaCentroids.size() / dim;
            megaCentroids.resize((currentSize + numNewMegaCentroids - oldMegaCentroidSize) * dim);
            memcpy(megaCentroids.data() + static_cast<size_t>(currentSize) * dim,
                   newMegaCentroids.data() + static_cast<size_t>(oldMegaCentroidSize) * dim,
                   (numNewMegaCentroids - oldMegaCentroidSize) * dim * sizeof(float));

            // Move the miniCentroidIds
            megaMiniCentroidIds.resize(currentSize + numNewMegaCentroids - oldMegaCentroidSize);
            auto idx = 0;
            for (auto i = oldMegaCentroidSize; i < numNewMegaCentroids; i++) {
                megaMiniCentroidIds[currentSize + idx] = std::move(newMiniClusterIds[i]);
                updatedMegaCentroids.push_back(currentSize + idx);
                idx++;
            }
            megaClusteringScore.resize(currentSize + numNewMegaCentroids - oldMegaCentroidSize);
        } else {
            auto lastCentroidId = (megaCentroids.size() / dim) - 1;
            // If the new mega centroid smaller than oldMegaCentroidIds.size()
            for (int i = numNewMegaCentroids; i < oldMegaCentroidSize; i++) {
                // Copy from last to i
                auto currMegaId = oldMegaCentroidIds[i];
                // Fix the lastCentroidId before fixing currMegaId
                while (std::find(oldMegaCentroidIds.begin() + numNewMegaCentroids, oldMegaCentroidIds.end(), lastCentroidId) != oldMegaCentroidIds.end()) {
                    lastCentroidId--;
                }
                if (currMegaId > lastCentroidId) {
                    continue;
                }

                memcpy(megaCentroids.data() + static_cast<size_t>(currMegaId) * dim,
                       megaCentroids.data() + static_cast<size_t>(lastCentroidId) * dim,
                       dim * sizeof(float));
                megaMiniCentroidIds[currMegaId] = std::move(megaMiniCentroidIds[lastCentroidId]);
                megaClusteringScore[currMegaId] = megaClusteringScore[lastCentroidId];
                lastCentroidId--;
            }
            // Resize the mega centroids
            megaCentroids.resize((lastCentroidId + 1) * dim);
            megaMiniCentroidIds.resize(lastCentroidId + 1);
            megaClusteringScore.resize(lastCentroidId + 1);
        }

        return updatedMegaCentroids;
    }

    void ReclusteringIndex::updateTotalDataWrittenBySystem(const std::vector<std::vector<vector_idx_t>> &newMiniClusterIds,
                                                           const std::vector<std::vector<vector_idx_t>>
                                                           &newMiniClusterVectorIds) {
        auto totalVecsWritten = newMiniClusterIds.size();
        for (const auto& ids: newMiniClusterIds) {
            totalVecsWritten += ids.size();
        }
        for (const auto& ids: newMiniClusterVectorIds) {
            totalVecsWritten += ids.size();
        }
        stats.totalDataWrittenBySystem += totalVecsWritten * dim * sizeof(float);
    }

    void ReclusteringIndex::updateTotalDataWrittenByUser(const size_t n) {
        stats.totalDataWrittenByUser += n * dim * sizeof(float);
    }

    void ReclusteringIndex::search(const float *query, uint16_t k, std::priority_queue<NodeDistCloser> &results,
                                   int nMegaProbes, int nMicroProbes, ReclusteringIndexStats &stats, int queryId) {
        auto numMegaCentroids = megaCentroids.size() / dim;
        auto numMiniCentroids = miniCentroids.size() / dim;
        nMegaProbes = std::min(nMegaProbes, (int)numMegaCentroids);
        nMicroProbes = std::min(nMicroProbes, (int)numMiniCentroids);

        // Find 5 closest mega centroids
        std::vector<vector_idx_t> megaAssign;
        findKClosestMegaCentroids(query, nMegaProbes, megaAssign, stats);
        // printf("Total mega centroids to search: %zu\n", megaAssign.size());

        // Now find the closest micro centroids
        std::vector<vector_idx_t> miniAssign;
        findKClosestMiniCentroids(query, nMicroProbes, megaAssign, miniAssign, stats);
        // printf("Total mini centroids to search: %zu\n", miniAssign.size());

        /*    
        // Print cluster assignments for query 49
        if (queryId == 49) {
            printf("\n=== List of all cluster Assignments ===\n");
            for(size_t i = 0; i < numMegaCentroids; i++) {
                printf("Mega Cluster %llu (MSE = %.2f):", i, megaClusteringScore[i]);
                for(size_t j = 0; j < megaMiniCentroidIds[i].size(); j++) {
                    printf("%llu, ", megaMiniCentroidIds[i][j]);
                }
                printf("\n");
            }

            printf("\n=== Query 49 Cluster Assignments ===\n");
            printf("L2 (Mega) Clusters chosen (%zu total): ", megaAssign.size());
            for (size_t i = 0; i < megaAssign.size(); i++) {
                printf("%llu (MSE = %.2f)", megaAssign[i], megaClusteringScore[megaAssign[i]]);
                if (i < megaAssign.size() - 1) printf(", ");
            }
            printf("\nL1 (Mini) Clusters chosen (%zu total): ", miniAssign.size());
            for (size_t i = 0; i < miniAssign.size(); i++) {
                printf("%llu (MSE = %.2f)", miniAssign[i], miniClusteringScore[miniAssign[i]]);
                if (i < miniAssign.size() - 1) printf(", ");
            }
            
            printf("\n\n=== Query 49 Distances to All L2 Centroids ===\n");
            auto dc_mega = getDistanceComputer(megaCentroids.data(), numMegaCentroids);
            dc_mega->setQuery(query);
            std::vector<std::pair<vector_idx_t, double>> centroidDistances_mega;
            for (size_t i = 0; i < numMegaCentroids; i++) {
                double dist;
                dc_mega->computeDistance(i, &dist);
                centroidDistances_mega.push_back({i, dist});
            }
            // Sort by distance
            std::sort(centroidDistances_mega.begin(), centroidDistances_mega.end(), 
                     [](const auto& a, const auto& b) { return a.second < b.second; });
            
            for (size_t i = 0; i < numMegaCentroids; i++) {
                printf("  Centroid %llu: distance = %.6f\n", centroidDistances_mega[i].first, centroidDistances_mega[i].second);
            }

            // Calculate distances from query 49 to all L1 centroids
            printf("\n\n=== Query 49 Distances to All L1 Centroids ===\n");
            auto dc_mini = getDistanceComputer(miniCentroids.data(), numMiniCentroids);
            dc_mini->setQuery(query);
            std::vector<std::pair<vector_idx_t, double>> centroidDistances_mini;
            for (size_t i = 0; i < numMiniCentroids; i++) {
                double dist;
                dc_mini->computeDistance(i, &dist);
                centroidDistances_mini.push_back({i, dist});
            }
            // Sort by distance
            std::sort(centroidDistances_mini.begin(), centroidDistances_mini.end(), 
                     [](const auto& a, const auto& b) { return a.second < b.second; });
            
            for (size_t i = 0; i < numMiniCentroids; i++) {
                printf("  Centroid %llu: distance = %.6f\n", centroidDistances_mini[i].first, centroidDistances_mini[i].second);
            }
            
            printf("=====================================\n\n");
        }
        */

        // auto dc = getDistanceComputer(miniCentroids.data(), numMiniCentroids);
        // dc->setQuery(query);
        // // Find the min and max distance from miniAssign
        // for (auto miniId : miniAssign) {
        //     double dist;
        //     dc->computeDistance(miniId, &dist);
        //     printf("Mini centroid %llu distance: %f\n", miniId, dist);
        // }

        // Print the shilloute score for each mini centroid
        // auto num_of_negative_silhouette = 0;
//         double most_negative_silhouette = 0.0;
//         auto most_neg_id = -1;
//
// #pragma omp parallel for schedule(dynamic)
//         for (std::size_t i = 0; i < miniAssign.size(); ++i) {
//             auto miniId = miniAssign[i];
//             auto score = calcScoreForMiniCluster(miniId);
//
//             if (score < most_negative_silhouette) {
// #pragma omp critical
//                 {
//                     if (score < most_negative_silhouette) {
//                         most_negative_silhouette = score;
//                         most_neg_id = miniId;
//                     }
//                 }
//             }
//         }

        // printf("Most negative silhouette mini centroid id: %d with score: %f\n", most_neg_id, most_negative_silhouette);

        // // Now we want to print the L1s and L2s cz of which it's negative silhouette
        // if (most_neg_id != -1) {
        //     auto dc = getDistanceComputer(megaCentroids.data(), numMegaCentroids);
        //     std::unordered_set<vector_idx_t> closerL1s;
        //     calcScoreForMiniCluster(most_neg_id, &closerL1s);
        //     std::unordered_map<vector_idx_t, std::unordered_set<vector_idx_t>> closerL2s;
        //     auto mega_most_neg_id = -1;
        //     // Find which mega centroid it belongs to
        //     for (int megaId = 0; megaId < megaMiniCentroidIds.size(); megaId++) {
        //         auto &miniIds = megaMiniCentroidIds[megaId];
        //         if (std::find(miniIds.begin(), miniIds.end(), most_neg_id) != miniIds.end()) {
        //             mega_most_neg_id = megaId;
        //             break;
        //         }
        //     }
        //     double most_neg_dist = 0.0;
        //     dc->setQuery(miniCentroids.data() + most_neg_id * dim);
        //     dc->computeDistance(mega_most_neg_id, &most_neg_dist);
        //
        //     for (const auto &l1 : closerL1s) {
        //         for (int megaId = 0; megaId < megaMiniCentroidIds.size(); megaId++) {
        //             auto &miniIds = megaMiniCentroidIds[megaId];
        //             if (std::find(miniIds.begin(), miniIds.end(), l1) != miniIds.end()) {
        //                 closerL2s[megaId].insert(l1);
        //                 break;
        //             }
        //         }
        //     }
        //     printf("Mega centroid id for mini centroid %d is [%d, %f]\n", most_neg_id, mega_most_neg_id, most_neg_dist);
        //     printf("L1 centroids closer than own mini centroid:\n");
        //     for (const auto &l2s : closerL2s) {
        //         double l2_dist = 0.0;
        //         dc->computeDistance(l2s.first, &l2_dist);
        //         printf("Mega centroid [%llu, %f]: ", l2s.first, l2_dist);
        //         for (const auto &l1 : l2s.second) {
        //             printf("%llu ", l1);
        //         }
        //         printf("\n");
        //         printf("count of L1s: %zu\n", l2s.second.size());
        //         printf("\n");
        //     }
        //     printf("Total count of L1s: %zu\n", closerL1s.size());
        // }

        // printf("Number of negative silhouette mini centroids in search: %d out of %d\n", num_of_negative_silhouette, (int)miniAssign.size());

        // Now find the closest vectors
        findKClosestVectors(query, k, miniAssign, results, stats);
    }

    void ReclusteringIndex::printStatsForTrackId() {
        auto score = calcScoreForMiniCluster(nextMiniCentroidId);
        printf("Most negative silhouette mini centroid id: %llu with score: %f\n", nextMiniCentroidId, score);
        auto numMegaCentroids = megaCentroids.size() / dim;
        // Now we want to print the L1s and L2s cz of which it's negative silhouette
        if (nextMiniCentroidId != -1) {
            auto dc = getDistanceComputer(megaCentroids.data(), numMegaCentroids);
            std::unordered_set<vector_idx_t> closerL1s;
            calcScoreForMiniCluster(nextMiniCentroidId, &closerL1s);
            std::unordered_map<vector_idx_t, std::unordered_set<vector_idx_t>> closerL2s;
            auto mega_most_neg_id = -1;
            // Find which mega centroid it belongs to
            for (int megaId = 0; megaId < megaMiniCentroidIds.size(); megaId++) {
                auto &miniIds = megaMiniCentroidIds[megaId];
                if (std::find(miniIds.begin(), miniIds.end(), nextMiniCentroidId) != miniIds.end()) {
                    mega_most_neg_id = megaId;
                    break;
                }
            }
            double most_neg_dist = 0.0;
            dc->setQuery(miniCentroids.data() + nextMiniCentroidId * dim);
            dc->computeDistance(mega_most_neg_id, &most_neg_dist);

            for (const auto &l1 : closerL1s) {
                for (int megaId = 0; megaId < megaMiniCentroidIds.size(); megaId++) {
                    auto &miniIds = megaMiniCentroidIds[megaId];
                    if (std::find(miniIds.begin(), miniIds.end(), l1) != miniIds.end()) {
                        closerL2s[megaId].insert(l1);
                        break;
                    }
                }
            }
            printf("Mega centroid id for mini centroid %llu is [%d, %f]\n", nextMiniCentroidId, mega_most_neg_id, most_neg_dist);
            printf("L1 centroids closer than own mini centroid:\n");
            for (const auto &l2s : closerL2s) {
                double l2_dist = 0.0;
                dc->computeDistance(l2s.first, &l2_dist);
                printf("Mega centroid [%llu, %f]: ", l2s.first, l2_dist);
                for (const auto &l1 : l2s.second) {
                    printf("%llu ", l1);
                }
                printf("\n");
                printf("count of L1s: %zu\n", l2s.second.size());
                printf("\n");
            }
            printf("Total count of L1s: %zu\n", closerL1s.size());
        }
    }

    // Simplest Idea: Based on sillouhette score, we can find the bad clusters and search them separately.
    void ReclusteringIndex::searchWithBadClusters(const float *query, uint16_t k,
                                                  std::priority_queue<NodeDistCloser> &results,
                                                  int nMegaProbes, int nMicroProbes, int nMiniProbesForBadClusters,
                                                  ReclusteringIndexStats &stats, bool searchEachBadCluster) {
        auto numMegaCentroids = megaCentroids.size() / dim;
        auto numMiniCentroids = miniCentroids.size() / dim;
        nMegaProbes = std::min(nMegaProbes, (int)numMegaCentroids);
        nMicroProbes = std::min(nMicroProbes, (int)numMiniCentroids);
        nMiniProbesForBadClusters = std::min(nMiniProbesForBadClusters, (int)numMiniCentroids);

        std::vector<vector_idx_t> megaAssign;
        findKClosestMegaCentroids(query, nMegaProbes, megaAssign, stats, true);

        if (!searchEachBadCluster) {
            for (int i = 0; i < numMegaCentroids; i++) {
                if (megaClusteringScore[i] >= 0.01) {
                    continue;
                }
                if (std::find(megaAssign.begin(), megaAssign.end(), i) != megaAssign.end()) {
                    continue;
                }
                megaAssign.push_back(i);
            }
        }

        // Now find the closest micro centroids
        std::vector<vector_idx_t> miniAssign;
        findKClosestMiniCentroids(query, nMicroProbes, megaAssign, miniAssign, stats);

        // Now find the closest vectors
        findKClosestVectors(query, k, miniAssign, results, stats);

        if (!searchEachBadCluster) {
            return;
        }

        // Now iterate through mega clusters
        for (int i = 0; i < numMegaCentroids; i++) {
            if (megaClusteringScore[i] >= 0.01) {
                continue;
            }
            searchMegaCluster(query, k, results, i, nMiniProbesForBadClusters, stats);
        }
    }

    void ReclusteringIndex::searchMegaCluster(const float *query, uint16_t k,
                                              std::priority_queue<NodeDistCloser> &results, int megaClusterId,
                                              int nMiniProbes, ReclusteringIndexStats &stats) {
        std::vector<vector_idx_t> megaClusterIds;
        megaClusterIds.emplace_back(megaClusterId);
        std::vector<vector_idx_t> miniClusterIds;
        findKClosestMiniCentroids(query, nMiniProbes, megaClusterIds, miniClusterIds, stats);
        // Now find the closest vectors
        findKClosestVectors(query, k, miniClusterIds, results, stats);
    }

    void ReclusteringIndex::searchQuantized(const float *query, uint16_t k,
                                            std::priority_queue<NodeDistCloser> &results, int nMegaProbes,
                                            int nMicroProbes, ReclusteringIndexStats &stats) {
        if (quantizedMiniClusters.size() == 0) {
            // If quantizedMiniClusters is empty, we cannot search
            return;
        }

        auto numMegaCentroids = megaCentroids.size() / dim;
        auto numMiniCentroids = quantizedMiniCentroids.size() / quantizer->codeSize;
        nMegaProbes = std::min(nMegaProbes, (int)numMegaCentroids);
        nMicroProbes = std::min(nMicroProbes, (int)numMiniCentroids);

        // Find 5 closest mega centroids
        std::vector<vector_idx_t> megaAssign;
        findKClosestMegaCentroids(query, nMegaProbes, megaAssign, stats);

        auto numMicroCentroids = quantizedMiniCentroids.size() / quantizer->codeSize;
        auto dc = getQuantizedDistanceComputer(quantizedMiniCentroids.data(), numMicroCentroids);
        dc->setQuery(query);

        // Now find the closest micro centroids
        std::priority_queue<NodeDistCloser> closestMicro;
        for (auto megaId : megaAssign) {
            auto microIds = megaMiniCentroidIds[megaId];
            for (auto microId: microIds) {
                double d;
                dc->computeDistance(microId, &d);
                stats.numDistanceCompForSearch++;
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
            auto cluster = quantizedMiniClusters[microId];
            auto ids = miniClusterVectorIds[microId];
            auto clusterSize = ids.size();
            auto clusterDc = getQuantizedDistanceComputer(cluster.data(), clusterSize);
            clusterDc->setQuery(query);
            for (int j = 0; j < clusterSize; j++) {
                double dist;
                clusterDc->computeDistance(j, &dist);
                // printf("dist: %f\n", dist);
                stats.numDistanceCompForSearch++;
                if (results.size() <= k || dist < results.top().dist) {
                    results.emplace(ids[j], dist);
                    if (results.size() > k) {
                        results.pop();
                    }
                }
            }
        }
    }

    void ReclusteringIndex::checkDuplicateMiniClusters() {
        auto numMegaCentroids = megaCentroids.size() / dim;
        for (int i = 0; i < numMegaCentroids; i++) {
            auto miniIds = megaMiniCentroidIds[i];
            // Check if there are duplicates
            auto actualSize = miniIds.size();
            auto uniqueSize = std::unordered_set<vector_idx_t>(miniIds.begin(), miniIds.end()).size();
            printf("Duplicate mini clusters in mega cluster %d size: %lu actual: %llu\n", i, uniqueSize, actualSize);
        }
    }

    void ReclusteringIndex::findKClosestMegaCentroids(const float *query, int minK, std::vector<vector_idx_t> &ids, ReclusteringIndexStats &stats, bool onlyGoodClusters) {
        std::priority_queue<NodeDistCloser> closestMicro;
        auto numMegaCentroids = megaCentroids.size() / dim;
        auto dc = getDistanceComputer(megaCentroids.data(), numMegaCentroids);
        dc->setQuery(query);
        // auto k = std::max(minK, 100);
        auto k = minK;
        auto minDistance = std::numeric_limits<double>::infinity();
        for (int i = 0; i < numMegaCentroids; i++) {
            if (onlyGoodClusters && megaClusteringScore[i] < 0.01) {
                continue;
            }
            double d;
            stats.numDistanceCompForSearch++;
            dc->computeDistance(i, &d);
            minDistance = std::min(minDistance, d);
            if (closestMicro.size() < k || d < closestMicro.top().dist) {
                closestMicro.emplace(i, d);
                if (closestMicro.size() > k) {
                    closestMicro.pop();
                }
            }
        }
        // reverse the pq
        std::priority_queue<NodeDistFarther> results;
        while (!closestMicro.empty()) {
            results.emplace(closestMicro.top().id, closestMicro.top().dist);
            closestMicro.pop();
        }

        int inserted = 0;
        // Copy the ids to vector
        ids.reserve(results.size());
        while (!results.empty()) {
            auto microId = results.top().id;
            // auto dist = results.top().dist;
            results.pop();
            if (std::find(ids.begin(), ids.end(), microId) != ids.end()) {
                continue;
            }
            // if (inserted >= minK && dist > minDistance * 1.7) {
            //     break;
            // }
            ids.push_back(microId);
            inserted++;
        }
    }

    void ReclusteringIndex::findKClosestMegaCentroids(const float *query, int k, std::vector<vector_idx_t> &ids, std::vector<float> &distances) {
        std::priority_queue<NodeDistCloser> closestMicro;
        auto numMegaCentroids = megaCentroids.size() / dim;
        auto dc = getDistanceComputer(megaCentroids.data(), numMegaCentroids);
        dc->setQuery(query);
        for (int i = 0; i < numMegaCentroids; i++) {
            double d;
            stats.numDistanceCompForSearch++;
            dc->computeDistance(i, &d);
            if (closestMicro.size() < k || d < closestMicro.top().dist) {
                closestMicro.emplace(i, d);
                if (closestMicro.size() > k) {
                    closestMicro.pop();
                }
            }
        }

        // Copy the ids to vector
        while (!closestMicro.empty()) {
            auto microId = closestMicro.top().id;
            auto dist = closestMicro.top().dist;
            closestMicro.pop();
            if (std::find(ids.begin(), ids.end(), microId) != ids.end()) {
                continue;
            }
            ids.push_back(microId);
            distances.push_back(dist);
        }
    }

    

    void ReclusteringIndex::findKClosestMiniCentroids(const float *query, int minK,
                                                      std::vector<vector_idx_t> megaCentroids,
                                                      std::vector<vector_idx_t> &ids, ReclusteringIndexStats &stats) {
        std::priority_queue<NodeDistCloser> closestMini;
        auto numMiniCentroids = miniCentroids.size() / dim;
        auto dc = getDistanceComputer(miniCentroids.data(), numMiniCentroids);
        dc->setQuery(query);
        // auto k = std::max(minK, 2000);
        auto k = minK;
        auto minDistance = std::numeric_limits<double>::infinity();

        // Iterate through the specified mega centroids
        for (auto megaId : megaCentroids) {
            // Get all mini centroids belonging to this mega centroid
            auto& miniIds = megaMiniCentroidIds[megaId];
            for (auto miniId : miniIds) {
                double d;
                stats.numDistanceCompForSearch++;
                dc->computeDistance(miniId, &d);
                minDistance = std::min(minDistance, d);
                if (closestMini.size() < k || d < closestMini.top().dist) {
                    closestMini.emplace(miniId, d);
                    if (closestMini.size() > k) {
                        closestMini.pop();
                    }
                }
            }
        }

        // reverse the pq
        std::priority_queue<NodeDistFarther> results;
        while (!closestMini.empty()) {
            results.emplace(closestMini.top().id, closestMini.top().dist);
            closestMini.pop();
        }

        int inserted = 0;
        // Copy the ids to vector (in reverse order to get closest first)
        ids.reserve(results.size());
        ids.clear();
        while (!results.empty()) {
            auto miniId = results.top().id;
            // auto dist = results.top().dist;
            results.pop();
            if (std::find(ids.begin(), ids.end(), miniId) != ids.end()) {
                continue;
            }
            // if (inserted >= minK && dist > minDistance * 1.7) {
            //     break;
            // }
            ids.push_back(miniId);
            inserted++;
        }
    }

    void ReclusteringIndex::findKClosestVectors(const float *query, int k, std::vector<vector_idx_t> miniCentroids,
                                                std::priority_queue<NodeDistCloser> &results, ReclusteringIndexStats &stats) {

        // Now we have the closest micro centroids, let's find the closest vectors
        for (auto miniId : miniCentroids) {
            auto cluster = miniClusters[miniId];
            auto ids = miniClusterVectorIds[miniId];
            auto clusterSize = ids.size();
            auto clusterDc = getDistanceComputer(cluster.data(), clusterSize);
            clusterDc->setQuery(query);
            for (int j = 0; j < clusterSize; j++) {
                double dist;
                clusterDc->computeDistance(j, &dist);
                stats.numDistanceCompForSearch++;
                if (results.size() <= k || dist < results.top().dist) {
                    results.emplace(ids[j], dist);
                    if (results.size() > k) {
                        results.pop();
                    }
                }
            }
        }
    }

    bool ReclusteringIndex::isAtBoundary(vector_idx_t miniClusterId) {
        std::vector<vector_idx_t> ids;
        std::vector<float> distances;
        findKClosestMegaCentroids(miniCentroids.data() + miniClusterId * dim, 2, ids, distances);
        if (distances.size() < 2) {
            return false;
        }
        auto distDiff = std::abs(distances[1] - distances[0]);
        auto maxDist = std::max(std::abs(distances[0]), std::abs(distances[1]));
        if (maxDist == 0) {
            return false;
        }
        // printf("Mini cluster %llu is at boundary: distDiff: %f, maxDist: %f, ratio: %f\n",
        //        miniClusterId, distDiff, maxDist, distDiff / maxDist);
        if ((distDiff / maxDist) <= 0.2) {
            return true;
        }
        return false;
    }


    void ReclusteringIndex::printStats() {
        printf("ReclusteringIndex::printStats\n");
        // Print the number of mega clusters
        printf("Number of mega clusters: %zu, number of mini clusters: %zu\n", megaCentroids.size() / dim, miniCentroids.size() / dim);
        // print min, max, avg size of the clusters
        auto minSize = std::numeric_limits<size_t>::max();
        size_t maxSize = 0;
        size_t avgSize = 0;
        std::vector<size_t> clusterSizes;
        for (const auto &cluster: miniClusters) {
            auto size = cluster.size() / dim;
            minSize = std::min(minSize, size);
            maxSize = std::max(maxSize, size);
            avgSize += size;
            clusterSizes.push_back(size);
        }
        printf("L1 cluster size: Min: %zu, Max: %zu, Avg: %zu\n", minSize, maxSize, avgSize / miniClusters.size());
        // Print top 10 largest clusters and smallest clusters
        std::sort(clusterSizes.begin(), clusterSizes.end());
       /*
        printf("Top 10 smallest clusters sizes: ");
        for (int i = 0; i < 100 && i < clusterSizes.size(); i++) {
            printf("%zu ", clusterSizes[i]);
        }
        printf("\n");

        printf("Top 10 largest clusters sizes: ");
        for (int i = 0; i < 100 && i < clusterSizes.size(); i++) {
            printf("%zu ", clusterSizes[clusterSizes.size() - 1 - i]);
        }
        printf("\n");
        */
        //printf("Total number of vectors: %zu/%zu\n", avgSize, size);
        // Print min, max and avg for mega clusters
        auto megaMinSize = std::numeric_limits<size_t>::max();
        size_t megaMaxSize = 0;
        size_t megaAvgSize = 0;
        for (const auto &miniIds : megaMiniCentroidIds) {
            auto size = miniIds.size();
            megaMinSize = std::min(megaMinSize, size);
            megaMaxSize = std::max(megaMaxSize, size);
            megaAvgSize += size;
        }
        printf("L2 cluster size: Min: %zu, Max: %zu, Avg: %zu\n", megaMinSize, megaMaxSize, megaAvgSize / megaMiniCentroidIds.size());

        // Print total number of mini clusters with bad silhouette score
        auto totalBadScore = 0;
        for (int i = 0; i < miniClusteringScore.size(); i++) {
            if (miniClusteringScore[i] < -0.009) {
                totalBadScore++;
            }
        }
        printf("Number of mini clusters with bad silhouette score: %d out of %zu\n", totalBadScore, miniClusteringScore.size());

//         auto numMiniCentroids = miniCentroids.size() / dim;
//         auto totalWithBadScore = 0;
//         auto totalWithBadScoreAtBoundary = 0;
//         auto totalAtBoundary = 0;
// #pragma omp parallel for reduction(+: totalWithBadScore, totalAtBoundary) schedule(dynamic)
//         for (int miniCentroidId = 0; miniCentroidId < numMiniCentroids; miniCentroidId++) {
//             double s = calcScoreForMiniCluster(miniCentroidId);
//             bool isB = isAtBoundary(miniCentroidId);
//             if (s < -0.009) {
//                 if (isB) {
//                     totalWithBadScoreAtBoundary++;
//                 }
//                 totalWithBadScore++;
//                 printf("MiniCluster %d, Silhouette Score: %f\n", miniCentroidId, s);
//             } else {
//                 if (isB) {
//                     totalAtBoundary++;
//                 }
//             }
//         }
//         printf("Number of mini clusters with bad silhouette score: %d out of %zu\n", totalWithBadScore, numMiniCentroids);
//         printf("Number of mini clusters at boundary: %d out of %d\n", totalWithBadScoreAtBoundary, totalWithBadScore);
//         printf("Number of mini clusters at boundary (w/o bad score): %d out of %zu\n", totalAtBoundary, numMiniCentroids - totalWithBadScore);

        //
        // // Print vectors
        // auto numMegaCentroids = megaCentroids.size() / dim;
        // for (int i = 0; i < numMegaCentroids; i++) {
        //     printf("Mega cluster %d centroid: ", i);
        //     // Print mega centroid
        //     for (int d = 0; d < dim; d++) {
        //         printf("%f ", megaCentroids[i * dim + d]);
        //     }
        //     printf("\n");
        //     for (auto miniId : megaMiniCentroidIds[i]) {
        //         printf("Mini cluster %llu centroid: ", miniId);
        //         // Print mini centroid
        //         for (int d = 0; d < dim; d++) {
        //             printf("%f ", miniCentroids[miniId * dim + d]);
        //         }
        //         printf("\n");
        //         printf("Vectors in mini cluster %llu:\n", miniId);
        //         // Print the vectors in the mini cluster
        //         auto &miniCluster = miniClusters[miniId];
        //         auto numVectors = miniCluster.size() / dim;
        //         for (int j = 0; j < numVectors; j++) {
        //             for (int d = 0; d < dim; d++) {
        //                 printf("%f ", miniCluster[j * dim + d]);
        //             }
        //             printf("\n");
        //         }
        //     }
        //     printf("\n\n");
        // }

        /*
        if (!miniClusterSubCells.empty()) {
            // Print stats for subcells
            size_t totalSubCells = 0;
            size_t avgSubCells = 0;
            size_t maxSubCells = 0;
            for (auto& subcell: miniClusterSubCells) {
                totalSubCells += subcell.centroids.size() / dim;
                avgSubCells += subcell.centroids.size() / dim;
                maxSubCells = std::max(maxSubCells, subcell.centroids.size() / dim);
            }
            avgSubCells /= miniClusterSubCells.size();
            printf("Total number of subcells: %zu\n", totalSubCells);
            printf("Avg number of subcells: %zu\n", avgSubCells);
            printf("Max number of subcells: %lu\n", maxSubCells);
        }
        */

        // printf("Number of quantized mini clusters: %zu\n", quantizedMiniCentroids.size() / quantizer->codeSize);
        // // print min, max, avg size of the quantized clusters
        // size_t quantizedMinSize = std::numeric_limits<size_t>::max();
        // size_t quantizedMaxSize = 0;
        // size_t quantizedAvgSize = 0;
        // for (const auto &cluster: quantizedMiniClusters) {
        //     auto size = cluster.size() / quantizer->codeSize;
        //     quantizedMinSize = std::min(quantizedMinSize, size);
        //     quantizedMaxSize = std::max(quantizedMaxSize, size);
        //     quantizedAvgSize += size;
        // }
        // printf("Min size of quantized clusters: %zu\n", quantizedMinSize);
        // printf("Max size of quantized clusters: %zu\n", quantizedMaxSize);
        // printf("Avg size of quantized clusters: %zu\n", quantizedAvgSize / quantizedMiniClusters.size());
        // printf("Total number of mini clusters: %zu/%zu\n", quantizedAvgSize, size);

        // Print score for mega clusters
        // int i = 0;
        // for (const auto &megaScore: megaClusteringScore) {
        //     printf("Mega cluster %d score: %f\n", i++, megaScore);
        // }

        // Print avg score for mega clusters
        double avgMegaScore = 0.0;
        for (const auto &megaScore: megaClusteringScore) {
            avgMegaScore += megaScore;
        }
        avgMegaScore /= megaClusteringScore.size();
        printf("Avg mega cluster score: %f\n", avgMegaScore);

        // Print top 5 scores for mega clusters in increasing order
        // std::vector<std::pair<double, int>> scores;

        
        /*
        // Uncomment!
        for (int i = 0; i < megaClusteringScore.size(); i++) {
            printf("Mega cluster %d score: %f\n", i, megaClusteringScore[i]);
            // scores.push_back(std::make_pair(megaClusteringScore[i], i));
        }
        */

        // std::sort(scores.begin(), scores.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        // printf("Top 5 mega cluster scores:\n");
        // for (int i = 0; i < 5; i++) {
        //     printf("Mega cluster %d score: %f\n", scores[i].second, scores[i].first);
        // }

        // Print stats
        // printf("Write amplification: %f\n", static_cast<double>(stats.totalDataWrittenBySystem) / stats.totalDataWrittenByUser);
        //printf("Total Distance Computations for reclustering: %lld\n", stats.numDistanceCompForRecluster);
        printChangeClusterStats();
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
        out.write(reinterpret_cast<const char *>(&config.nIter), sizeof(config.nIter));
        out.write(reinterpret_cast<const char *>(&config.megaCentroidSize), sizeof(config.megaCentroidSize));
        out.write(reinterpret_cast<const char *>(&config.miniCentroidSize), sizeof(config.miniCentroidSize));
        out.write(reinterpret_cast<const char *>(&config.newMiniCentroidSize), sizeof(config.newMiniCentroidSize));
        out.write(reinterpret_cast<const char *>(&config.lambda), sizeof(config.lambda));
        out.write(reinterpret_cast<const char *>(&config.searchThreshold), sizeof(config.searchThreshold));
        out.write(reinterpret_cast<const char *>(&config.distanceType), sizeof(config.distanceType));
        out.write(reinterpret_cast<const char *>(&config.numMegaReclusterCentroids),
                  sizeof(config.numMegaReclusterCentroids));
        out.write(reinterpret_cast<const char *>(&config.numNewMiniReclusterCentroids),
                  sizeof(config.numNewMiniReclusterCentroids));
        out.write(reinterpret_cast<const char *>(&config.quantizationTrainPercentage),
                  sizeof(config.quantizationTrainPercentage));
        out.write(reinterpret_cast<const char *>(&config.hardClusterSizeLimit), sizeof(config.hardClusterSizeLimit));
        out.write(reinterpret_cast<const char *>(&config.kmeansSamplingRatio), sizeof(config.kmeansSamplingRatio));

        // Write mega centroids
        size_t megaCentroidSize = megaCentroids.size();
        out.write(reinterpret_cast<const char *>(&megaCentroidSize), sizeof(megaCentroidSize));
        out.write(reinterpret_cast<const char *>(megaCentroids.data()), megaCentroidSize * sizeof(float));

        // Write megaMiniCentroidIds
        size_t megaMiniCentroidIdsSize = megaMiniCentroidIds.size();
        out.write(reinterpret_cast<const char *>(&megaMiniCentroidIdsSize), sizeof(megaMiniCentroidIdsSize));
        for (const auto &ids: megaMiniCentroidIds) {
            size_t idsSize = ids.size();
            out.write(reinterpret_cast<const char *>(&idsSize), sizeof(idsSize));
            out.write(reinterpret_cast<const char *>(ids.data()), idsSize * sizeof(vector_idx_t));
        }

        // Write megaClusteringScore
        size_t megaClusteringScoreSize = megaClusteringScore.size();
        out.write(reinterpret_cast<const char *>(&megaClusteringScoreSize), sizeof(megaClusteringScoreSize));
        out.write(reinterpret_cast<const char *>(megaClusteringScore.data()), megaClusteringScoreSize * sizeof(double));

        // Write the miniCentroids
        size_t miniCentroidSize = miniCentroids.size();
        out.write(reinterpret_cast<const char *>(&miniCentroidSize), sizeof(miniCentroidSize));
        if (miniCentroidSize > 0) {
            // Verify invariants before writing
            assert(miniCentroidSize % dim == 0 && "miniCentroids size must be multiple of dim");
            size_t expectedClusterCount = miniCentroidSize / dim;
            assert(miniClusters.size() == expectedClusterCount && "miniClusters size must match centroid count");
            assert(miniClusterVectorIds.size() == expectedClusterCount && "miniClusterVectorIds size must match cluster count");

            out.write(reinterpret_cast<const char *>(miniCentroids.data()), miniCentroidSize * sizeof(float));
            // Write the miniClusters
            for (const auto &cluster: miniClusters) {
                size_t clusterSize = cluster.size();
                assert(clusterSize % dim == 0 && "cluster size must be multiple of dim");
                out.write(reinterpret_cast<const char *>(&clusterSize), sizeof(clusterSize));
                out.write(reinterpret_cast<const char *>(cluster.data()), clusterSize * sizeof(float));
            }

            // Write the mini cluster vector ids
            for (const auto &vectorId: miniClusterVectorIds) {
                size_t vectorIdSize = vectorId.size();
                out.write(reinterpret_cast<const char *>(&vectorIdSize), sizeof(vectorIdSize));
                out.write(reinterpret_cast<const char *>(vectorId.data()), vectorIdSize * sizeof(vector_idx_t));
            }
        }

        // Write quantized mini centroids
        size_t quantizedMiniCentroidsSize = quantizedMiniCentroids.size();
        out.write(reinterpret_cast<const char *>(&quantizedMiniCentroidsSize), sizeof(quantizedMiniCentroidsSize));
        if (quantizedMiniCentroidsSize > 0) {
            out.write(reinterpret_cast<const char *>(quantizedMiniCentroids.data()), quantizedMiniCentroidsSize * sizeof(uint8_t));

            // Write quantized mini clusters
            size_t quantizedMiniClustersSize = quantizedMiniClusters.size();
            out.write(reinterpret_cast<const char *>(&quantizedMiniClustersSize), sizeof(quantizedMiniClustersSize));
            for (const auto &cluster: quantizedMiniClusters) {
                size_t clusterSize = cluster.size();
                out.write(reinterpret_cast<const char *>(&clusterSize), sizeof(clusterSize));
                out.write(reinterpret_cast<const char *>(cluster.data()), clusterSize * sizeof(uint8_t));
            }

            // Write the mini cluster vector ids
            for (const auto &vectorId: miniClusterVectorIds) {
                size_t vectorIdSize = vectorId.size();
                out.write(reinterpret_cast<const char *>(&vectorIdSize), sizeof(vectorIdSize));
                out.write(reinterpret_cast<const char *>(vectorId.data()), vectorIdSize * sizeof(vector_idx_t));
            }
        }

        // Write new mini centroids
        size_t newMiniCentroidSize = newMiniCentroids.size();
        out.write(reinterpret_cast<const char *>(&newMiniCentroidSize), sizeof(newMiniCentroidSize));
        if (newMiniCentroidSize > 0) {
            // Verify invariants before writing
            assert(newMiniCentroidSize % dim == 0 && "newMiniCentroids size must be multiple of dim");
            size_t expectedNewClusterCount = newMiniCentroidSize / dim;
            assert(newMiniClusters.size() == expectedNewClusterCount && "newMiniClusters size must match centroid count");
            assert(newMiniClusterVectorIds.size() == expectedNewClusterCount && "newMiniClusterVectorIds size must match cluster count");

            out.write(reinterpret_cast<const char *>(newMiniCentroids.data()), newMiniCentroidSize * sizeof(float));

            // Write newMiniClusters
            for (const auto &cluster: newMiniClusters) {
                size_t clusterSize = cluster.size();
                assert(clusterSize % dim == 0 && "new cluster size must be multiple of dim");
                out.write(reinterpret_cast<const char *>(&clusterSize), sizeof(clusterSize));
                out.write(reinterpret_cast<const char *>(cluster.data()), clusterSize * sizeof(float));
            }

            // Write newMiniClusterVectorIds
            for (const auto &vectorId: newMiniClusterVectorIds) {
                size_t vectorIdSize = vectorId.size();
                out.write(reinterpret_cast<const char *>(&vectorIdSize), sizeof(vectorIdSize));
                out.write(reinterpret_cast<const char *>(vectorId.data()), vectorIdSize * sizeof(vector_idx_t));
            }
        }

        // Write quantizer
        quantizer->flush_to_disk(out);

        // Write stats
        out.write(reinterpret_cast<const char *>(&stats.numDistanceCompForSearch), sizeof(stats.numDistanceCompForSearch));
        out.write(reinterpret_cast<const char *>(&stats.totalQueries), sizeof(stats.totalQueries));
        out.write(reinterpret_cast<const char *>(&stats.numDistanceCompForRecluster), sizeof(stats.numDistanceCompForRecluster));
        out.write(reinterpret_cast<const char *>(&stats.totalReclusters), sizeof(stats.totalReclusters));
        out.write(reinterpret_cast<const char *>(&stats.totalDataWrittenBySystem), sizeof(stats.totalDataWrittenBySystem));
        out.write(reinterpret_cast<const char *>(&stats.totalDataWrittenByUser), sizeof(stats.totalDataWrittenByUser));
        out.close();
    }

    void ReclusteringIndex::load_from_disk(const std::string &file_path) {
        std::ifstream in(file_path, std::ios::binary);
        if (!in) {
            std::cerr << "Error opening file for reading: " << file_path << std::endl;
            return;
        }

        // Read basic fields
        in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
        in.read(reinterpret_cast<char *>(&size), sizeof(size));

        // Read config (order same as flush_to_disk)
        in.read(reinterpret_cast<char *>(&config.nIter), sizeof(config.nIter));
        in.read(reinterpret_cast<char *>(&config.megaCentroidSize), sizeof(config.megaCentroidSize));
        in.read(reinterpret_cast<char *>(&config.miniCentroidSize), sizeof(config.miniCentroidSize));
        in.read(reinterpret_cast<char *>(&config.newMiniCentroidSize), sizeof(config.newMiniCentroidSize));
        in.read(reinterpret_cast<char *>(&config.lambda), sizeof(config.lambda));
        in.read(reinterpret_cast<char *>(&config.searchThreshold), sizeof(config.searchThreshold));
        in.read(reinterpret_cast<char *>(&config.distanceType), sizeof(config.distanceType));
        in.read(reinterpret_cast<char *>(&config.numMegaReclusterCentroids), sizeof(config.numMegaReclusterCentroids));
        in.read(reinterpret_cast<char *>(&config.numNewMiniReclusterCentroids),
                sizeof(config.numNewMiniReclusterCentroids));
        in.read(reinterpret_cast<char *>(&config.quantizationTrainPercentage),
                sizeof(config.quantizationTrainPercentage));
        in.read(reinterpret_cast<char *>(&config.hardClusterSizeLimit), sizeof(config.hardClusterSizeLimit));
        in.read(reinterpret_cast<char *>(&config.kmeansSamplingRatio), sizeof(config.kmeansSamplingRatio));

        // Read mega centroids
        size_t megaCentroidsCount;
        in.read(reinterpret_cast<char *>(&megaCentroidsCount), sizeof(megaCentroidsCount));
        megaCentroids.resize(megaCentroidsCount);
        in.read(reinterpret_cast<char *>(megaCentroids.data()), megaCentroidsCount * sizeof(float));

        // Read megaMiniCentroidIds
        size_t megaMiniCentroidIdsCount;
        in.read(reinterpret_cast<char *>(&megaMiniCentroidIdsCount), sizeof(megaMiniCentroidIdsCount));
        megaMiniCentroidIds.resize(megaMiniCentroidIdsCount);
        for (size_t i = 0; i < megaMiniCentroidIdsCount; i++) {
            size_t idsSize;
            in.read(reinterpret_cast<char *>(&idsSize), sizeof(idsSize));
            megaMiniCentroidIds[i].resize(idsSize);
            in.read(reinterpret_cast<char *>(megaMiniCentroidIds[i].data()), idsSize * sizeof(vector_idx_t));
        }

        // Read megaClusteringScore
        size_t megaClusteringScoreCount;
        in.read(reinterpret_cast<char *>(&megaClusteringScoreCount), sizeof(megaClusteringScoreCount));
        megaClusteringScore.resize(megaClusteringScoreCount);
        in.read(reinterpret_cast<char *>(megaClusteringScore.data()), megaClusteringScoreCount * sizeof(double));

        // Read mini centroids
        size_t miniCentroidsCount;
        in.read(reinterpret_cast<char *>(&miniCentroidsCount), sizeof(miniCentroidsCount));
        if (miniCentroidsCount > 0) {
            assert(miniCentroidsCount % dim == 0 && "miniCentroids count must be multiple of dim");
            miniCentroids.resize(miniCentroidsCount);
            in.read(reinterpret_cast<char *>(miniCentroids.data()), miniCentroidsCount * sizeof(float));

            // Derive mini clusters count from mini centroids (each cluster is one centroid)
            size_t miniClustersCount = miniCentroidsCount / dim;
            miniClusters.resize(miniClustersCount);
            for (size_t i = 0; i < miniClustersCount; i++) {
                size_t clusterSize;
                in.read(reinterpret_cast<char *>(&clusterSize), sizeof(clusterSize));
                assert(clusterSize % dim == 0 && "cluster size must be multiple of dim");
                miniClusters[i].resize(clusterSize);
                in.read(reinterpret_cast<char *>(miniClusters[i].data()), clusterSize * sizeof(float));
            }

            // Read mini cluster vector ids (same count as mini clusters)
            miniClusterVectorIds.resize(miniClustersCount);
            for (size_t i = 0; i < miniClustersCount; i++) {
                size_t vectorIdSize;
                in.read(reinterpret_cast<char *>(&vectorIdSize), sizeof(vectorIdSize));
                miniClusterVectorIds[i].resize(vectorIdSize);
                in.read(reinterpret_cast<char *>(miniClusterVectorIds[i].data()), vectorIdSize * sizeof(vector_idx_t));
            }

            // Verify loaded data consistency
            assert(miniClusters.size() == miniClustersCount && "miniClusters size mismatch after load");
            assert(miniClusterVectorIds.size() == miniClustersCount && "miniClusterVectorIds size mismatch after load");
        }

        // Read quantized mini centroids
        size_t quantizedMiniCentroidsCount;
        in.read(reinterpret_cast<char *>(&quantizedMiniCentroidsCount), sizeof(quantizedMiniCentroidsCount));
        if (quantizedMiniCentroidsCount > 0) {
            quantizedMiniCentroids.resize(quantizedMiniCentroidsCount);
            in.read(reinterpret_cast<char *>(quantizedMiniCentroids.data()), quantizedMiniCentroidsCount * sizeof(uint8_t));

            // Read quantized mini clusters
            size_t quantizedMiniClustersCount;
            in.read(reinterpret_cast<char *>(&quantizedMiniClustersCount), sizeof(quantizedMiniClustersCount));
            quantizedMiniClusters.resize(quantizedMiniClustersCount);
            for (size_t i = 0; i < quantizedMiniClustersCount; i++) {
                size_t clusterSize;
                in.read(reinterpret_cast<char *>(&clusterSize), sizeof(clusterSize));
                quantizedMiniClusters[i].resize(clusterSize);
                in.read(reinterpret_cast<char *>(quantizedMiniClusters[i].data()), clusterSize * sizeof(uint8_t));
            }

            // Read mini cluster vector ids (same count as mini clusters)
            miniClusterVectorIds.resize(quantizedMiniClustersCount);
            for (size_t i = 0; i < quantizedMiniClustersCount; i++) {
                size_t vectorIdSize;
                in.read(reinterpret_cast<char *>(&vectorIdSize), sizeof(vectorIdSize));
                miniClusterVectorIds[i].resize(vectorIdSize);
                in.read(reinterpret_cast<char *>(miniClusterVectorIds[i].data()), vectorIdSize * sizeof(vector_idx_t));
            }
        }

        // Read new mini centroids
        size_t newMiniCentroidsCount;
        in.read(reinterpret_cast<char *>(&newMiniCentroidsCount), sizeof(newMiniCentroidsCount));
        if (newMiniCentroidsCount > 0) {
            assert(newMiniCentroidsCount % dim == 0 && "newMiniCentroids count must be multiple of dim");
            newMiniCentroids.resize(newMiniCentroidsCount);
            in.read(reinterpret_cast<char *>(newMiniCentroids.data()), newMiniCentroidsCount * sizeof(float));

            // Derive new mini clusters count from new mini centroids
            size_t newMiniClustersCount = newMiniCentroidsCount / dim;
            newMiniClusters.resize(newMiniClustersCount);
            for (size_t i = 0; i < newMiniClustersCount; i++) {
                size_t clusterSize;
                in.read(reinterpret_cast<char *>(&clusterSize), sizeof(clusterSize));
                assert(clusterSize % dim == 0 && "new cluster size must be multiple of dim");
                newMiniClusters[i].resize(clusterSize);
                in.read(reinterpret_cast<char *>(newMiniClusters[i].data()), clusterSize * sizeof(float));
            }

            // Read new mini cluster vector ids (same count as new mini clusters)
            newMiniClusterVectorIds.resize(newMiniClustersCount);
            for (size_t i = 0; i < newMiniClustersCount; i++) {
                size_t vectorIdSize;
                in.read(reinterpret_cast<char *>(&vectorIdSize), sizeof(vectorIdSize));
                newMiniClusterVectorIds[i].resize(vectorIdSize);
                in.read(reinterpret_cast<char *>(newMiniClusterVectorIds[i].data()), vectorIdSize * sizeof(vector_idx_t));
            }

            // Verify loaded data consistency
            assert(newMiniClusters.size() == newMiniClustersCount && "newMiniClusters size mismatch after load");
            assert(newMiniClusterVectorIds.size() == newMiniClustersCount && "newMiniClusterVectorIds size mismatch after load");
        }

        // Read quantizer
        quantizer = std::make_unique<SQ8Bit>(dim);
        quantizer->load_from_disk(in);

        // Read stats
        in.read(reinterpret_cast<char *>(&stats.numDistanceCompForSearch), sizeof(stats.numDistanceCompForSearch));
        in.read(reinterpret_cast<char *>(&stats.totalQueries), sizeof(stats.totalQueries));
        in.read(reinterpret_cast<char *>(&stats.numDistanceCompForRecluster), sizeof(stats.numDistanceCompForRecluster));
        in.read(reinterpret_cast<char *>(&stats.totalReclusters), sizeof(stats.totalReclusters));
        in.read(reinterpret_cast<char *>(&stats.totalDataWrittenBySystem), sizeof(stats.totalDataWrittenBySystem));
        in.read(reinterpret_cast<char *>(&stats.totalDataWrittenByUser), sizeof(stats.totalDataWrittenByUser));
        in.close();
    }

    void ReclusteringIndex::getVectorClusterAssignments(
        const float *query,
        const vector_idx_t *vectorIds,
        int n,
        std::unordered_map<vector_idx_t, std::tuple<vector_idx_t, vector_idx_t, vector_idx_t, std::vector<float>,
            std::vector<float>>> &results) const {

        results.clear();

        // Convert input vector IDs to a set for O(1) lookup
        std::unordered_set<vector_idx_t> requestedVectors(vectorIds, vectorIds + n);

        // Iterate through mega clusters
        for (size_t megaId = 0; megaId < megaMiniCentroidIds.size(); megaId++) {
            // Extract mega centroid
            std::vector<float> megaCentroid(dim);
            if (megaId < megaCentroids.size() / dim) {
                std::copy(megaCentroids.begin() + megaId * dim,
                         megaCentroids.begin() + (megaId + 1) * dim,
                         megaCentroid.begin());
            }

            // Iterate through mini clusters in this mega cluster
            const auto& miniIds = megaMiniCentroidIds[megaId];
            for (auto miniId : miniIds) {
                if (miniId >= miniClusterVectorIds.size()) {
                    continue;
                }

                const auto& clusterVectorIds = miniClusterVectorIds[miniId];
                const auto& clusterVectors = miniClusters[miniId];

                // Extract mini centroid
                std::vector<float> miniCentroid(dim);
                if (miniId < miniCentroids.size() / dim) {
                    std::copy(miniCentroids.begin() + miniId * dim,
                             miniCentroids.begin() + (miniId + 1) * dim,
                             miniCentroid.begin());
                }

                // Search for requested vectors in this mini cluster
                for (size_t i = 0; i < clusterVectorIds.size(); i++) {
                    vector_idx_t vectorId = clusterVectorIds[i];

                    // Check if this vector was requested
                    if (requestedVectors.find(vectorId) != requestedVectors.end()) {
                        auto [megaRank, miniInMegaRank, miniRank] = getClusterRanks(query, megaId, miniId);  // Optionally get ranks if needed
                        // Store the result: (miniClusterId, megaClusterId, vector_data, miniCentroid, megaCentroid)
                        results[vectorId] = std::make_tuple(megaRank, miniInMegaRank, miniRank, miniCentroid, megaCentroid);

                        // Remove from requested set for efficiency
                        requestedVectors.erase(vectorId);

                        // Early exit if we found all requested vectors
                        if (requestedVectors.empty()) {
                            return;
                        }
                    }
                }
            }
        }

        // For any vectors not found, store with invalid cluster IDs and empty data
        for (auto vectorId : requestedVectors) {
            results[vectorId] = std::make_tuple(INVALID_VECTOR_ID, INVALID_VECTOR_ID, INVALID_VECTOR_ID, std::vector<float>(),
                                                std::vector<float>());
        }
    }

    std::tuple<vector_idx_t, vector_idx_t, vector_idx_t> ReclusteringIndex::getClusterRanks(
        const float *query, vector_idx_t megaId, vector_idx_t miniId) const {
        int megaRank = -1;
        int miniRankInMega = -1;
        int miniRankOverall = -1;

        // Validate megaId
        size_t numMegaClusters = megaCentroids.size() / dim;
        if (megaId >= numMegaClusters) {
            return std::make_tuple(-1, -1, -1);
        }

        // Calculate distances from query to all mega centroids and find rank
        auto megaDc = getDistanceComputer(megaCentroids.data(), numMegaClusters);
        megaDc->setQuery(query);

        std::vector<std::pair<double, vector_idx_t>> megaDistances;
        megaDistances.reserve(numMegaClusters);

        for (size_t i = 0; i < numMegaClusters; i++) {
            double dist;
            megaDc->computeDistance(i, &dist);
            megaDistances.emplace_back(dist, i);
        }

        // Sort by distance
        std::sort(megaDistances.begin(), megaDistances.end());

        // Find rank of the given megaId
        for (size_t rank = 0; rank < megaDistances.size(); rank++) {
            if (megaDistances[rank].second == megaId) {
                megaRank = rank;
                break;
            }
        }

        // Calculate overall mini cluster rank and rank within mega cluster
        size_t numMiniClusters = miniCentroids.size() / dim;
        auto miniDc = getDistanceComputer(miniCentroids.data(), numMiniClusters);
        miniDc->setQuery(query);

        // Calculate distances to all mini clusters for overall rank
        std::vector<std::pair<double, vector_idx_t>> allMiniDistances;
        allMiniDistances.reserve(numMiniClusters);

        for (size_t i = 0; i < numMiniClusters; i++) {
            double dist;
            miniDc->computeDistance(i, &dist);
            allMiniDistances.emplace_back(dist, i);
        }

        // Sort by distance
        std::sort(allMiniDistances.begin(), allMiniDistances.end());

        // Find overall rank of the given miniId
        for (size_t rank = 0; rank < allMiniDistances.size(); rank++) {
            if (allMiniDistances[rank].second == miniId) {
                miniRankOverall = rank;
                break;
            }
        }

        // Now find the rank of miniId within the mega cluster
        if (megaId < megaMiniCentroidIds.size()) {
            const auto& miniIds = megaMiniCentroidIds[megaId];

            // Check if miniId is in this mega cluster
            if (std::find(miniIds.begin(), miniIds.end(), miniId) == miniIds.end()) {
                return std::make_tuple(megaRank, -1, miniRankOverall);
            }

            // Find rank within the mega cluster
            std::vector<std::pair<double, vector_idx_t>> miniDistancesInMega;
            miniDistancesInMega.reserve(miniIds.size());

            for (auto mId : miniIds) {
                if (mId < numMiniClusters) {
                    // Find the distance from allMiniDistances
                    for (const auto& [dist, id] : allMiniDistances) {
                        if (id == mId) {
                            miniDistancesInMega.emplace_back(dist, mId);
                            break;
                        }
                    }
                }
            }

            // Sort by distance
            std::sort(miniDistancesInMega.begin(), miniDistancesInMega.end());

            // Find rank of the given miniId within mega cluster
            for (size_t rank = 0; rank < miniDistancesInMega.size(); rank++) {
                if (miniDistancesInMega[rank].second == miniId) {
                    miniRankInMega = rank;
                    break;
                }
            }
        }

        return std::make_tuple(megaRank, miniRankInMega, miniRankOverall);
    }

    const float* ReclusteringIndex::getVectorData(vector_idx_t vectorId) const {
        for (size_t miniId = 0; miniId < miniClusterVectorIds.size(); miniId++) {
            const auto& clusterVectorIds = miniClusterVectorIds[miniId];
            const auto& clusterVectors = miniClusters[miniId];

            for (size_t i = 0; i < clusterVectorIds.size(); i++) {
                if (clusterVectorIds[i] == vectorId) {
                    // Return pointer to the vector data
                    return const_cast<float *>(clusterVectors.data() + i * dim);
                }
            }
        }
        return nullptr; // Not found
    }

    void ReclusteringIndex::analyzeQueryClusterChanges(
        const float *query,
        const vector_idx_t *groundTruthVectorIds,
        int nGroundTruth,
        bool onlyStoreChanges) {
        // Get current cluster assignments for ground truth vectors
        std::unordered_map<vector_idx_t, std::tuple<vector_idx_t, vector_idx_t, vector_idx_t, std::vector<float>, std::vector<float>>> currentAssignments;
        getVectorClusterAssignments(query, groundTruthVectorIds, nGroundTruth, currentAssignments);

        if (onlyStoreChanges) {
            prevQueryState = std::move(currentAssignments);
            return;
        }

        printf("\n=== Analyzing Query Cluster Changes ===\n");
        printf("Ground truth vectors: %d\n", nGroundTruth);

        // If prevQueryState is empty, this is the first call - store current state
        if (prevQueryState.empty()) {
            printf("First analysis - storing initial state\n");
            prevQueryState = std::move(currentAssignments);
            printf("=== End Analysis ===\n\n");
            return;
        }

        // Compare previous and current states
        printf("\n=== Comparing Before and After Reclustering ===\n\n");

        int clusterChanges = 0;
        int megaChanges = 0;
        int miniChanges = 0;
        int kClosest = 5; // Number of closest centroids to show

        ReclusteringIndexStats tempStats;

        // Analyze each ground truth vector
        for (int i = 0; i < std::min(nGroundTruth, 100); i++) {
            vector_idx_t vectorId = groundTruthVectorIds[i];

            auto currentIt = currentAssignments.find(vectorId);
            auto prevIt = prevQueryState.find(vectorId);

            if (currentIt == currentAssignments.end() || prevIt == prevQueryState.end()) {
                continue;
            }

            auto [currMegaRank, currMiniInMegaRank, currMiniRank, currMiniCentroid, currMegaCentroid] = currentIt->second;
            auto [prevMegaRank, prevMiniInMegaRank, prevMiniRank, prevMiniCentroid, prevMegaCentroid] = prevIt->second;

            bool megaRankChanged = (currMegaRank != prevMegaRank);
            bool miniRankChanged = (currMiniRank != prevMiniRank);

            if (megaRankChanged || miniRankChanged) {
                clusterChanges++;
                if (megaRankChanged) megaChanges++;
                if (miniRankChanged) miniChanges++;

                printf("Vector %lu (GT rank %d):\n", vectorId, i);

                if (megaRankChanged) {
                    printf("  Mega rank: %lu -> %lu (delta: %+ld)\n",
                           prevMegaRank, currMegaRank, (long)(currMegaRank - prevMegaRank));
                } else {
                    printf("  Mega rank: %lu (unchanged)\n", currMegaRank);
                }

                if (miniRankChanged) {
                    printf("  Mini rank (in mega): %lu -> %lu (delta: %+ld)\n",
                           prevMiniInMegaRank, currMiniInMegaRank, (long)(currMiniInMegaRank - prevMiniInMegaRank));
                    printf("  Mini rank (overall): %lu -> %lu (delta: %+ld)\n",
                           prevMiniRank, currMiniRank, (long)(currMiniRank - prevMiniRank));
                } else {
                    printf("  Mini rank (overall): %lu (unchanged)\n", currMiniRank);
                }
                auto dc = getDistanceComputer(nullptr, 0);

                // Calculate centroid distance change using distance computer
                if (prevMiniCentroid.size() == currMiniCentroid.size() && prevMiniCentroid.size() > 0) {
                    double miniDistChange;
                    dc->computeSymDistance(prevMiniCentroid.data(), currMiniCentroid.data(), &miniDistChange);
                    printf("  Mini centroid distance change: %.6f\n", miniDistChange);
                }

                if (prevMegaCentroid.size() == currMegaCentroid.size() && prevMegaCentroid.size() > 0) {
                    double megaDistChange;
                    dc->computeSymDistance(prevMegaCentroid.data(), currMegaCentroid.data(), &megaDistChange);
                    printf("  Mega centroid distance change: %.6f\n", megaDistChange);
                }

                // Calculate distance from query to previous and current mini centroids
                if (prevMiniCentroid.size() > 0) {
                    double dist;
                    dc->computeSymDistance(query, prevMiniCentroid.data(), &dist);
                    printf("  Distance from query to previous mini centroid: %.6f\n", dist);
                }

                if (currMiniCentroid.size() > 0) {
                    double dist;
                    dc->computeSymDistance(query, currMiniCentroid.data(), &dist);
                    printf("  Distance from query to current mini centroid: %.6f\n", dist);
                }

                // Same for mega centroids
                if (prevMegaCentroid.size() > 0) {
                    double dist;
                    dc->computeSymDistance(query, prevMegaCentroid.data(), &dist);
                    printf("  Distance from query to previous mega centroid: %.6f\n", dist);
                }

                if (currMegaCentroid.size() > 0) {
                    double dist;
                    dc->computeSymDistance(query, currMegaCentroid.data(), &dist);
                    printf("  Distance from query to current mega centroid: %.6f\n", dist);
                }

                // Calculate against vector
                const float *vectorData = getVectorData(vectorId);
                if (vectorData != nullptr) {
                    double distToPrevMini, distToCurrMini;
                    dc->computeSymDistance(vectorData, prevMiniCentroid.data(), &distToPrevMini);
                    dc->computeSymDistance(vectorData, currMiniCentroid.data(), &distToCurrMini);
                    printf("  Distance from vector to previous mini centroid: %.6f\n", distToPrevMini);
                    printf("  Distance from vector to current mini centroid: %.6f\n", distToCurrMini);

                    double distToPrevMega, distToCurrMega;
                    dc->computeSymDistance(vectorData, prevMegaCentroid.data(), &distToPrevMega);
                    dc->computeSymDistance(vectorData, currMegaCentroid.data(), &distToCurrMega);
                    printf("  Distance from vector to previous mega centroid: %.6f\n", distToPrevMega);
                    printf("  Distance from vector to current mega centroid: %.6f\n", distToCurrMega);

                    printf("\n  --- Detailed Distance Analysis ---\n");

                    // Top k closest mega centroids from query
                    printf("  Top %d closest mega centroids from query:\n", kClosest);
                    std::vector<vector_idx_t> megaIdsFromQuery;
                    std::vector<float> megaDistsFromQuery;
                    findKClosestMegaCentroids(query, kClosest, megaIdsFromQuery, megaDistsFromQuery);

                    for (int j = 0; j < std::min(kClosest, (int) megaIdsFromQuery.size()); j++) {
                        printf("    Rank %d: Mega %lu, distance: %.6f%s\n",
                               j, megaIdsFromQuery[j], megaDistsFromQuery[j],
                               (j == currMegaRank) ? " <- vector's mega" : "");
                    }

                    // Top k closest mega centroids from vector
                    printf("\n  Top %d closest mega centroids from vector:\n", kClosest);
                    std::vector<vector_idx_t> megaIdsFromVector;
                    std::vector<float> megaDistsFromVector;
                    findKClosestMegaCentroids(vectorData, kClosest, megaIdsFromVector, megaDistsFromVector);

                    for (int j = 0; j < std::min(kClosest, (int) megaIdsFromVector.size()); j++) {
                        printf("    Rank %d: Mega %lu, distance: %.6f%s\n",
                               j, megaIdsFromVector[j], megaDistsFromVector[j],
                               (j == currMegaRank) ? " <- vector's mega" : "");
                    }

                    // Build map from mini cluster ID to mega cluster ID
                    std::unordered_map<vector_idx_t, vector_idx_t> miniToMegaMap;
                    for (size_t megaId = 0; megaId < megaMiniCentroidIds.size(); megaId++) {
                        for (auto miniId : megaMiniCentroidIds[megaId]) {
                            miniToMegaMap[miniId] = megaId;
                        }
                    }

                    std::vector<vector_idx_t> allMegaIds;
                    getMegaClusterIds(allMegaIds);
                    // Top k closest mini centroids from query
                    printf("\n  Top %d closest mini centroids from query:\n", kClosest);
                    std::vector<vector_idx_t> miniIdsFromQuery;
                    findKClosestMiniCentroids(query, kClosest * 2, megaIdsFromQuery, allMegaIds, tempStats);

                    // Get distances for these mini centroids
                    auto miniDc = getDistanceComputer(miniCentroids.data(), miniCentroids.size() / dim);
                    miniDc->setQuery(query);
                    std::vector<std::pair<double, vector_idx_t> > miniDistsFromQuery;
                    for (auto miniId: miniIdsFromQuery) {
                        double dist;
                        miniDc->computeDistance(miniId, &dist);
                        miniDistsFromQuery.emplace_back(dist, miniId);
                    }
                    std::sort(miniDistsFromQuery.begin(), miniDistsFromQuery.end());

                    for (int j = 0; j < std::min(kClosest, (int) miniDistsFromQuery.size()); j++) {
                        vector_idx_t miniId = miniDistsFromQuery[j].second;
                        vector_idx_t megaId = miniToMegaMap[miniId];
                        auto [megaRank, miniRankInMega, miniRankOverall] = getClusterRanks(query, megaId, miniId);
                        printf("    Rank %d: Mini %lu (overall rank: %llu, mega: %lu), distance: %.6f%s\n",
                               j, miniId, miniRankOverall, megaId, miniDistsFromQuery[j].first,
                               (miniRankOverall == (size_t) currMiniRank) ? " <- vector's mini" : "");
                    }

                    // Top k closest mini centroids from vector
                    printf("\n  Top %d closest mini centroids from vector:\n", kClosest);
                    std::vector<vector_idx_t> miniIdsFromVector;
                    findKClosestMiniCentroids(vectorData, kClosest * 2, allMegaIds, miniIdsFromVector,
                                              tempStats);

                    // Get distances for these mini centroids from vector
                    miniDc->setQuery(vectorData);
                    std::vector<std::pair<double, vector_idx_t> > miniDistsFromVector;
                    for (auto miniId: miniIdsFromVector) {
                        double dist;
                        miniDc->computeDistance(miniId, &dist);
                        miniDistsFromVector.emplace_back(dist, miniId);
                    }
                    std::sort(miniDistsFromVector.begin(), miniDistsFromVector.end());

                    for (int j = 0; j < std::min(kClosest, (int) miniDistsFromVector.size()); j++) {
                        vector_idx_t miniId = miniDistsFromVector[j].second;
                        vector_idx_t megaId = miniToMegaMap[miniId];
                        auto [megaRank, miniRankInMega, miniRankOverall] = getClusterRanks(query, megaId, miniId);
                        printf("    Rank %d: Mini %lu (overall rank: %llu, mega: %lu), distance: %.6f%s\n",
                               j, miniId, miniRankOverall, megaId, miniDistsFromVector[j].first,
                               (miniRankOverall == (size_t) currMiniRank) ? " <- vector's mini" : "");
                    }
                }
                printf("\n");
            }
        }

        // Print summary
        printf("=== Summary ===\n");
        printf("Vectors with rank changes: %d / %d (%.1f%%)\n",
               clusterChanges, nGroundTruth, 100.0 * clusterChanges / nGroundTruth);
        printf("Mega rank changes: %d\n", megaChanges);
        printf("Mini rank changes: %d\n", miniChanges);

        // Update prevQueryState with current assignments
        // prevQueryState = std::move(currentAssignments);

        printf("\n=== End Analysis ===\n\n");
    }
}
