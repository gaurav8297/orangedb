#include "include/reclustering_index.h"

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
                memcpy(tempData.data() + idx * dim, cluster.data(), cluster.size() * sizeof(float));
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
        auto totalVecs = 0;
        auto microCentroidIds = megaMiniCentroidIds[megaClusterId];
        auto miniClusterSize = miniClusters.size();
        for (auto microCentroidId: microCentroidIds) {
            assert(microCentroidId < miniClusterSize);
            auto cluster = miniClusters[microCentroidId];
            totalVecs += (cluster.size() / dim);
        }

        // Copy actual vecs and vectorIds here
        std::vector<float> tempData(totalVecs * dim);
        std::vector<vector_idx_t> tempVectorIds(totalVecs);
        size_t idx = 0;
        for (auto microCentroidId: microCentroidIds) {
            auto cluster = miniClusters[microCentroidId];
            auto vectorId = miniClusterVectorIds[microCentroidId];
            size_t numVectors = cluster.size() / dim;
            memcpy(tempData.data() + idx * dim, cluster.data(), cluster.size() * sizeof(float));
            memcpy(tempVectorIds.data() + idx, vectorId.data(), numVectors * sizeof(vector_idx_t));
            idx += numVectors;
        }

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

    void ReclusteringIndex::reclusterAllMegaCentroids() {
        auto numMegaCentroids = megaCentroids.size() / dim;
        if (numMegaCentroids == 0) {
            return;
        }

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
                memcpy(tempData.data() + idx * dim, cluster.data(), cluster.size() * sizeof(float));
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
        auto totalVec = 0;
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
        clusterData(tempMiniCentroids.data(), tempMiniCentroidIds.data(), totalVec, config.megaCentroidSize, tempMegaCentroids, tempMiniClusterIds);

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
        auto dc = createDistanceComputer(data, dim, n, config.distanceType);
        clusterData_<float>(data, vectorIds, n, avgClusterSize, centroids, clusters, clusterVectorIds,
                            dc.get(), dim, [](const float x, int d) { return x; });
    }

    void ReclusteringIndex::clusterData(float *data, vector_idx_t *vectorIds, int n, int avgClusterSize,
                                        std::vector<float> &centroids,
                                        std::vector<std::vector<vector_idx_t> > &clusterVectorIds) {
        auto dc = createDistanceComputer(data, dim, n, config.distanceType);
        clusterData_<float>(data, vectorIds, n, avgClusterSize, centroids, clusterVectorIds,
                            dc.get(), dim, [](const float x, int d) { return x; });
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
                                                   std::vector<std::vector<vector_idx_t> > &newMiniClusterVectorIds) {
        updateTotalDataWrittenBySystem({}, newMiniClusterVectorIds);
        // Try to copy inplace if possible otherwise append
        std::vector<vector_idx_t> oldMiniClusterIds;
        for (const int currMegaId: oldMegaCentroids) {
            for (const auto &megaMiniId: megaMiniCentroidIds[currMegaId]) {
                oldMiniClusterIds.push_back(megaMiniId);
            }
        }

        // Copy the mini centroids, clusters and vector ids and fix the miniClusterIds
        std::unordered_map<vector_idx_t, vector_idx_t> newToOldCentroidIdMap;
        auto newMiniCentroidsSize = newMiniCentroids.size() / dim;
        // assert(oldMiniClusterIds.size() <= newMiniCentroidsSize);
        auto miniCentroidsSize = std::min(newMiniCentroidsSize, oldMiniClusterIds.size());
        for (int i = 0; i < miniCentroidsSize; i++) {
            auto oldCentroidId = oldMiniClusterIds[i];
            // Copy the centroid
            memcpy(miniCentroids.data() + oldCentroidId * dim, newMiniCentroids.data() + i * dim, dim * sizeof(float));
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
            memcpy(miniCentroids.data() + currentSize * dim, newMiniCentroids.data() + miniCentroidsSize * dim,
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
                memcpy(miniCentroids.data() + currCentroidId * dim, miniCentroids.data() + (lastCentroidId * dim), dim * sizeof(float));
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
        auto numToCalc = std::min(n, (int)numMegaCentroids);
        for (auto i = 0; i < numToCalc; i++) {
            megaClusteringScore[i] = calcScoreForMegaCluster(i);
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
            double s = (m > 0.0) ? (b - a) / m : 0.0;
            avgMegaScore += s;
        }

        avgMegaScore /= miniCentroidIds.size();
        avgMiniScore /= miniCentroidIds.size();

        // Weight the mega silhouette score more than the mini silhouette score
        return avgMiniScore;
    }

    double ReclusteringIndex::calcScoreForMiniCluster(int miniClusterId) {
        // Find 5 closest mega centroids
        std::vector<vector_idx_t> megaAssign;
        findKClosestMegaCentroids(miniCentroids.data() + miniClusterId * dim, 50, megaAssign, stats);

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
            double b = std::numeric_limits<double>::infinity();
            for (auto closestMiniCentroidId : closestMiniCentroidIds) {
                if (closestMiniCentroidId == miniClusterId) continue;
                double dist;
                dc->computeDistance(closestMiniCentroidId, &dist);
                b = std::min(b, dist);
            }

            // 3) silhouette for this point
            double m = std::max(a, b);
            double s = (m > 0.0) ? (b - a) / m : 0.0;
            totalSilhouette += s;
            totalPoints += 1;
        }
        return (totalPoints > 0)
                   ? totalSilhouette / double(totalPoints)
                   : 0.0;
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
            memcpy(megaCentroids.data() + currMegaId * dim, newMegaCentroids.data() + i * dim, dim * sizeof(float));

            // Move the miniCentroidIds
            megaMiniCentroidIds[currMegaId] = std::move(newMiniClusterIds[i]);
            updatedMegaCentroids.push_back(currMegaId);
        }

        if (numNewMegaCentroids > oldMegaCentroidSize) {
            // Append the new mega centroids
            auto currentSize = megaCentroids.size() / dim;
            megaCentroids.resize((currentSize + numNewMegaCentroids - oldMegaCentroidSize) * dim);
            memcpy(megaCentroids.data() + currentSize * dim, newMegaCentroids.data() + oldMegaCentroidSize * dim,
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

                memcpy(megaCentroids.data() + currMegaId * dim, megaCentroids.data() + (lastCentroidId * dim), dim * sizeof(float));
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

        // Now find the closest micro centroids
        std::vector<vector_idx_t> miniAssign;
        findKClosestMiniCentroids(query, nMicroProbes, megaAssign, miniAssign, stats);

        // Now find the closest vectors
        findKClosestVectors(query, k, miniAssign, results, stats);
    }

    // Simplest Idea: Based on sillouhette score, we can find the bad clusters and search them separately.
    void ReclusteringIndex::searchWithBadClusters(const float *query, uint16_t k,
                                                  std::priority_queue<NodeDistCloser> &results,
                                                  int nMegaProbes, int nMicroProbes, int nMiniProbesForBadClusters,
                                                  ReclusteringIndexStats &stats, bool skipBadClusters) {
        auto numMegaCentroids = megaCentroids.size() / dim;
        auto numMiniCentroids = miniCentroids.size() / dim;
        nMegaProbes = std::min(nMegaProbes, (int)numMegaCentroids);
        nMicroProbes = std::min(nMicroProbes, (int)numMiniCentroids);
        nMiniProbesForBadClusters = std::min(nMiniProbesForBadClusters, (int)numMiniCentroids);

        std::vector<vector_idx_t> megaAssign;
        findKClosestMegaCentroids(query, nMegaProbes, megaAssign, stats, true);

        // Now find the closest micro centroids
        std::vector<vector_idx_t> miniAssign;
        findKClosestMiniCentroids(query, nMicroProbes, megaAssign, miniAssign, stats);

        // Now find the closest vectors
        findKClosestVectors(query, k, miniAssign, results, stats);

        if (skipBadClusters) {
            return;
        }
        std::vector<vector_idx_t> badMegaClusters;
        // Now iterate through mega clusters
        for (int i = 0; i < numMegaCentroids; i++) {
            if (megaClusteringScore[i] >= 0.01) {
                continue;
            }

            badMegaClusters.push_back(i);
            // searchMegaCluster(query, k, results, i, nMiniProbesForBadClusters, stats);
        }

        printf("Found %lu bad mega clusters\n", badMegaClusters.size());

        std::vector<vector_idx_t> badMiniAssign;
        findKClosestMiniCentroids(query, nMiniProbesForBadClusters, badMegaClusters, badMiniAssign, stats);

        printf("Found %lu bad mini clusters\n", badMiniAssign.size());

        // Now find the closest vectors
        findKClosestVectors(query, k, badMiniAssign, results, stats);
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
            if (actualSize != uniqueSize) {
                printf("Duplicate mini clusters in mega cluster %d size: %lu actual: %llu\n", i, uniqueSize, actualSize);
            }
        }
    }

    void ReclusteringIndex::findKClosestMegaCentroids(const float *query, int k, std::vector<vector_idx_t> &ids, ReclusteringIndexStats &stats, bool onlyGoodClusters) {
        std::priority_queue<NodeDistCloser> closestMicro;
        auto numMegaCentroids = megaCentroids.size() / dim;
        auto dc = getDistanceComputer(megaCentroids.data(), numMegaCentroids);
        dc->setQuery(query);
        for (int i = 0; i < numMegaCentroids; i++) {
            if (onlyGoodClusters && megaClusteringScore[i] < 0.01) {
                continue;
            }
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
            closestMicro.pop();
            ids.push_back(microId);
        }
    }
    

    void ReclusteringIndex::findKClosestMiniCentroids(const float *query, int k,
                                                      std::vector<vector_idx_t> megaCentroids,
                                                      std::vector<vector_idx_t> &ids, ReclusteringIndexStats &stats) {
        std::priority_queue<NodeDistCloser> closestMini;
        auto numMiniCentroids = miniCentroids.size() / dim;
        auto dc = getDistanceComputer(miniCentroids.data(), numMiniCentroids);
        dc->setQuery(query);

        // Iterate through the specified mega centroids
        for (auto megaId : megaCentroids) {
            // Get all mini centroids belonging to this mega centroid
            auto miniIds = megaMiniCentroidIds[megaId];
            for (auto miniId : miniIds) {
                double d;
                stats.numDistanceCompForSearch++;
                dc->computeDistance(miniId, &d);
                if (closestMini.size() < k || d < closestMini.top().dist) {
                    closestMini.emplace(miniId, d);
                    if (closestMini.size() > k) {
                        closestMini.pop();
                    }
                }
            }
        }

        // Copy the ids to vector (in reverse order to get closest first)
        ids.clear();
        ids.reserve(closestMini.size());
        while (!closestMini.empty()) {
            auto miniId = closestMini.top().id;
            closestMini.pop();
            ids.push_back(miniId);
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
            out.write(reinterpret_cast<const char *>(miniCentroids.data()), miniCentroidSize * sizeof(float));
            // Write the miniClusters
            for (const auto &cluster: miniClusters) {
                size_t clusterSize = cluster.size();
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
            out.write(reinterpret_cast<const char *>(newMiniCentroids.data()), newMiniCentroidSize * sizeof(float));

            // Write newMiniClusters
            for (const auto &cluster: newMiniClusters) {
                size_t clusterSize = cluster.size();
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
            miniCentroids.resize(miniCentroidsCount);
            in.read(reinterpret_cast<char *>(miniCentroids.data()), miniCentroidsCount * sizeof(float));

            // Derive mini clusters count from mini centroids (each cluster is one centroid)
            size_t miniClustersCount = miniCentroidsCount / dim;
            miniClusters.resize(miniClustersCount);
            for (size_t i = 0; i < miniClustersCount; i++) {
                size_t clusterSize;
                in.read(reinterpret_cast<char *>(&clusterSize), sizeof(clusterSize));
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
            newMiniCentroids.resize(newMiniCentroidsCount);
            in.read(reinterpret_cast<char *>(newMiniCentroids.data()), newMiniCentroidsCount * sizeof(float));

            // Derive new mini clusters count from new mini centroids
            size_t newMiniClustersCount = newMiniCentroidsCount / dim;
            newMiniClusters.resize(newMiniClustersCount);
            for (size_t i = 0; i < newMiniClustersCount; i++) {
                size_t clusterSize;
                in.read(reinterpret_cast<char *>(&clusterSize), sizeof(clusterSize));
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
