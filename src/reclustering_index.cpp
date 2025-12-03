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

    void ReclusteringIndex::naiveInsert(float *data, size_t n) {
        std::vector<vector_idx_t> vectorIds(n);
        for (size_t i = 0; i < n; i++) {
            vectorIds[i] = i + size;
        }

        // Run clustering to create mini clusters
        std::vector<float> newMiniCentroids;
        std::vector<std::vector<float> > newMiniClusters;
        std::vector<std::vector<vector_idx_t> > newMiniClusterVectorIds;
        clusterData(data, vectorIds.data(), n, config.miniCentroidSize, newMiniCentroids, newMiniClusters,
                    newMiniClusterVectorIds);

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
                    n, megaMegaCentroids, megaMegaCentroidIds);
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
                    numMegaCentroids, megaMegaCentroids, megaMegaCentroidIds);

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
                        config.numNewMiniReclusterCentroids, newMegaCentroids, newMiniClusterIds);
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
                    newMiniCentroids, newMiniClusters, newMiniClusterVectorIds);

        // Run mega reclustering
        std::vector<vector_idx_t> miniCentroidIds(newMiniCentroids.size() / dim);
        for (size_t i = 0; i < miniCentroidIds.size(); i++) {
            miniCentroidIds[i] = i;
        }
        std::vector<float> newMegaCentroids;
        std::vector<std::vector<vector_idx_t> > newMiniClusterIds;
        clusterData(newMiniCentroids.data(), miniCentroidIds.data(), miniCentroidIds.size(),
                    config.megaCentroidSize, newMegaCentroids, newMiniClusterIds);

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
        std::vector<float> newMiniCentroids;
        std::vector<std::vector<float> > newMiniClusters;
        std::vector<std::vector<vector_idx_t> > newMiniClusterVectorIds;
        clusterData(tempData.data(), tempVectorIds.data(), totalVecs, config.miniCentroidSize,
                    newMiniCentroids, newMiniClusters, newMiniClusterVectorIds);

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
                    newMiniCentroids, newMiniClusters, newMiniClusterVectorIds);

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
                    config.megaCentroidSize, newMegaCentroids, newMiniClusterIds);

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
                    config.megaCentroidSize, tempMegaCentroids, tempMiniClusterIds);

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
                                        std::vector<std::vector<vector_idx_t> > &clusterVectorIds) {
        // auto dc = createDistanceComputer(data, dim, n, config.distanceType);
        // clusterData_<float>(data, vectorIds, n, avgClusterSize, centroids, clusters, clusterVectorIds,
        //                     dc.get(), dim, [](const float x, int d) { return x; });
        clusterDataWithFaiss(data, vectorIds, n, avgClusterSize, centroids, clusters, clusterVectorIds);
    }

    void ReclusteringIndex::clusterData(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                        std::vector<float> &centroids,
                                        std::vector<std::vector<vector_idx_t> > &clusterVectorIds, int nClusters) {
        // auto dc = createDistanceComputer(data, dim, n, config.distanceType);
        // clusterData_<float>(data, vectorIds, n, avgClusterSize, centroids, clusterVectorIds,
        //                     dc.get(), dim, [](const float x, int d) { return x; });
        clusterDataWithFaiss(data, vectorIds, n, avgClusterSize, centroids, clusterVectorIds, nClusters);
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

    void ReclusteringIndex::clusterDataWithFaiss(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                                 std::vector<float> &centroids,
                                                 std::vector<std::vector<float> > &clusters,
                                                 std::vector<std::vector<vector_idx_t> > &clusterVectorIds) {
        printf("Clustering %d vectors with avgClusterSize %d\n", n, avgClusterSize);
        if (n == 0) {
            return;
        }
        // Create the clustering object
        auto numClusters = getNumCentroids(n, avgClusterSize);
        // printf("Performing mini-reclustering on %d vectors with %d clusters %d avgClusterSize\n", n, numClusters, avgClusterSize);
        if (numClusters <= 1) {
            calcMeanCentroid(data, vectorIds, n, dim, centroids, clusterVectorIds);
            // Copy all data to the single cluster
            clusters.resize(1);
            clusters[0].resize(n * dim);
            memcpy(clusters[0].data(), data, n * dim * sizeof(float));
            return;
        }

        faiss::ClusteringParameters cl;
        cl.niter = config.nIter;
        if (config.distanceType == IP) {
            cl.spherical = true;
        }
        cl.min_points_per_centroid = getMinCentroidSize(n, numClusters);
        cl.max_points_per_centroid = getMaxCentroidSize(n, numClusters);
        cl.seed = -1;
        std::unique_ptr<faiss::BalancedClusteringDistModifier> distModifier;
        if (config.lambda > 0) {
            auto lambda = findAppropriateLambda(data, n, dim, numClusters);
            distModifier = std::make_unique<faiss::LambdaBasedDistModifier>(numClusters, lambda);
            cl.dist_modifier = distModifier.get();
            printf("cl.lambda = %f\n", lambda);
        }
        cl.verbose = true;
        faiss::Clustering clustering(dim, numClusters, cl);
        // TODO: This is a hack
        auto metric_type = config.distanceType == L2 ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
        auto index = faiss::IndexFlat(dim, metric_type);

        // Initialize the centroids
        clustering.train(n, data, index);

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

        // Get the hist
        std::vector<int> hist(numClusters, 0);
        for (int i = 0; i < n; i++) {
            assert(assign[i] >= 0 && assign[i] < numClusters);
            hist[assign[i]]++;
        }

        // Validate that no histo is greating than 4500
        for (int i = 0; i < numClusters; i++) {
            if (config.hardClusterSizeLimit > 0 && hist[i] >= config.hardClusterSizeLimit) {
                printf("Warning: Cluster %d has size %d greater than %llu\n", i, hist[i], config.hardClusterSizeLimit);
            }
        }

        // Copy the centroids
        centroids.resize(numClusters * dim);
        memcpy(centroids.data(), clustering.centroids.data(), numClusters * dim * sizeof(float));
        clusters.resize(numClusters);
        clusterVectorIds.resize(numClusters);
        for (int i = 0; i < numClusters; i++) {
            std::vector<float> cluster(hist[i] * dim);
            clusters[i] = std::move(cluster);
            std::vector<vector_idx_t> vectorId(hist[i]);
            clusterVectorIds[i] = std::move(vectorId);
            hist[i] = 0;
        }

        auto total_size = 0;
        for (int i = 0; i < numClusters; i++) {
            total_size += clusters[i].size() / dim;
        }
        assert(total_size == n);

        for (int i = 0; i < n; i++) {
            auto assignId = assign[i];
            auto idx = hist[assignId];
            auto &cluster = clusters[assignId];
            auto maxClusterSize = cluster.size() / dim;
            memcpy(cluster.data() + static_cast<size_t>(idx) * dim,
                   data + static_cast<size_t>(i) * dim,
                   dim * sizeof(float));
            clusterVectorIds[assignId][idx] = vectorIds[i];
            hist[assignId]++;
        }
        stats.numDistanceCompForRecluster += config.nIter * numClusters * n;
    }

    void ReclusteringIndex::clusterDataWithFaiss(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                                 std::vector<float> &centroids,
                                                 std::vector<std::vector<vector_idx_t> > &clusterVectorIds, int nClusters) {
        // Create the clustering object
        auto numClusters = nClusters > 0 ? nClusters : getNumCentroids(n, avgClusterSize);
        // printf("Performing mega-reclustering on %d vectors with %d clusters %d avgClusterSize\n", n, numClusters, avgClusterSize);
        if (numClusters <= 1) {
            calcMeanCentroid(data, vectorIds, n, dim, centroids, clusterVectorIds);
            return;
        }

        faiss::ClusteringParameters cl;
        cl.niter = config.nIter;
        cl.seed = -1;
        if (config.distanceType == IP) {
            cl.spherical = true;
        }
        cl.min_points_per_centroid = getMinCentroidSize(n, numClusters);
        cl.max_points_per_centroid = getMaxCentroidSize(n, numClusters);
        std::unique_ptr<faiss::BalancedClusteringDistModifier> distModifier;
        if (config.lambda > 0) {
            auto lambda = findAppropriateLambda(data, n, dim, numClusters);
            distModifier = std::make_unique<faiss::LambdaBasedDistModifier>(numClusters, lambda);
            cl.dist_modifier = distModifier.get();
            printf("cl.lambda = %f\n", lambda);
        }
        cl.verbose = true;
        faiss::Clustering clustering(dim, numClusters, cl);
        // TODO: This is a hack
        auto metric_type = config.distanceType == L2 ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
        auto index = faiss::IndexFlat(dim, metric_type);

        // Initialize the centroids
        clustering.train(n, data, index);

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

        // Get the hist
        std::vector<int> hist(numClusters, 0);
        for (int i = 0; i < n; i++) {
            if (assign[i] >= 0 && assign[i] < numClusters) {
                hist[assign[i]]++;
            } else {
                printf("WARNING: Invalid assignment at i=%d: assign[i]=%ld (numClusters=%d)\n",
                       i, assign[i], numClusters);
            }
        }

        // Validate that no histo is greating than 4500
        for (int i = 0; i < numClusters; i++) {
            if (config.hardClusterSizeLimit > 0 && hist[i] >= config.hardClusterSizeLimit) {
                printf("Warning: Cluster %d has size %d greater than 2500\n", i, hist[i]);
            }
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
                centroids[j] += quantizer->decode_one(data[i * dataDim + j], j);
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

    void ReclusteringIndex::printChangeClusterStats() {
        if (oldMegaCentroids.empty() || oldMegaClusteringScore.empty()) {
            printf("No old mega centroid or score to compare!\n");
            return;
        }

        printf("ReclusteringIndex::printChangeClusterStats\n");
        auto numMegaCentroids = megaCentroids.size() / dim;
        auto numOldMegaCentroids = oldMegaCentroids.size() / dim;
        auto dc = getDistanceComputer(oldMegaCentroids.data(), numOldMegaCentroids);
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
                // Print the old score, distance, new score, and change in score
                double scoreChange = megaClusteringScore[i] - oldMegaClusteringScore[oldCentroidId];
                printf("Mega Centroid %d: Old id = %d, Old Score = %f, Distance to Old = %f, New Score = %f, Change in Score = %f\n",
                       i,
                       oldCentroidId,
                       oldMegaClusteringScore[oldCentroidId],
                       minDistance,
                       megaClusteringScore[i],
                       scoreChange);
            } else {
                printf("Mega Centroid %d: No old centroid found!\n", i);
            }
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
            // miniClusteringScore[miniCentroidId] = s;
            avgMiniScore += s;
        }
        return avgMiniScore / miniCentroidIds.size();
    }

    double ReclusteringIndex::calcMSEScoreForMiniCluster(int miniClusterId) {
        auto centroid = miniCentroids.data() + miniClusterId * dim;
        auto& curMiniCluster = miniClusters[miniClusterId];
        auto numPoints = curMiniCluster.size() / dim;
        double totalMSE = 0.0;
        auto dc = getDistanceComputer(curMiniCluster.data(), numPoints);
        dc->setQuery(centroid);
        for (int i = 0; i < numPoints; i++) {
            double dist;
            dc->computeDistance(i, &dist);
            totalMSE += dist;
        }
        return (numPoints > 0) ? totalMSE / numPoints : 0.0;
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
                                   int nMegaProbes, int nMicroProbes, ReclusteringIndexStats &stats) {
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
            auto miniIds = megaMiniCentroidIds[megaId];
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
        printf("Number of mega clusters: %zu\n", megaCentroids.size() / dim);
        printf("Number of mini clusters: %zu\n", miniCentroids.size() / dim);
        // print min, max, avg size of the clusters
        auto minSize = std::numeric_limits<size_t>::max();
        size_t maxSize = 0;
        size_t avgSize = 0;
        for (const auto &cluster: miniClusters) {
            auto size = cluster.size() / dim;
            minSize = std::min(minSize, size);
            maxSize = std::max(maxSize, size);
            avgSize += size;
        }
        printf("Min size of clusters: %zu\n", minSize);
        printf("Max size of clusters: %zu\n", maxSize);
        printf("Avg size of clusters: %zu\n", avgSize / miniClusters.size());
        printf("Total number of vectors: %zu/%zu\n", avgSize, size);

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
        for (int i = 0; i < megaClusteringScore.size(); i++) {
            printf("Mega cluster %d score: %f\n", i, megaClusteringScore[i]);
            // scores.push_back(std::make_pair(megaClusteringScore[i], i));
        }
        // std::sort(scores.begin(), scores.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        // printf("Top 5 mega cluster scores:\n");
        // for (int i = 0; i < 5; i++) {
        //     printf("Mega cluster %d score: %f\n", scores[i].second, scores[i].first);
        // }

        // Print stats
        printf("Write amplification: %f\n", static_cast<double>(stats.totalDataWrittenBySystem) / stats.totalDataWrittenByUser);
        printf("Total Distance Computations for reclustering: %lld\n", stats.numDistanceCompForRecluster);
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
}
