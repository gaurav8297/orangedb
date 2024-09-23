#include "include/hnsw.h"
#include <omp.h>
#include <algorithm>
#include "include/prefetch.h"
#include "spdlog/spdlog.h"
#include <memory>
#include <fastQ/pair_wise.h>
#include <fastQ/scalar_8bit.h>

namespace orangedb {
    HNSW::HNSW(HNSWConfig config, RandomGenerator *rg, uint16_t dim) : config(config),
                                                                       rg(rg), stats(Stats()) {
        // Initialize probabilities to save computation time later.
        initProbabs(config.M, 1.0 / log(config.M));

        auto code_size = 0;
        // Initialize the quantizer
        if (config.compressionType == "scalar_8bit") {
            quantizer = std::make_unique<fastq::scalar_8bit::SQ8Bit>(dim);
            code_size = quantizer->codeSize;
        } else if (config.compressionType == "pair_wise") {
            quantizer = std::make_unique<fastq::pair_wise::PairWise2Bit>(dim);
            code_size = quantizer->codeSize;
        }

        if (config.loadStorage) {
            storage = new Storage(config.storagePath, levelProbabs.size(), code_size);
        } else {
            storage = new Storage(dim, config.M, levelProbabs.size(), code_size);
        }
    }

    void HNSW::initProbabs(uint16_t M, double levelMult) {
        for (int level = 0;; level++) {
            // levelMult helps to control how many level there are going to be. Higher levelMult means more levels.
            // This is just reverse of [-ln(uniform(0, 1)) * levelMult] which is a random variable with
            // exponential distribution.
            double prob = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
            if (prob < 1e-9) {
                break;
            }
            levelProbabs.push_back(prob);
        }
    }

    void HNSW::searchNearestOnLevel(
            DistanceComputer *dc,
            level_t level,
            vector_idx_t &nearest,
            double &nearestDist,
            Stats &stats) {
        CHECK_ARGUMENT(level > 0, "Level should be greater than 0");
        auto neighbors = storage->get_neighbors(level);
        // Can we do distance computation on 4 vectors at a time?
        while (true) {
            vector_idx_t prev_nearest = nearest;
            size_t begin, end;
            storage->get_neighbors_offsets(nearest, level, begin, end);
            for (size_t i = begin; i < end; i++) {
                vector_idx_t neighbor = neighbors[i];
                if (neighbor == INVALID_VECTOR_ID) {
                    break;
                }
                double dist;
                stats.totalDistCompDuringSearch++;
                dc->computeDistance(getActualId(level, neighbor), &dist);
                if (dist < nearestDist) {
                    nearest = neighbor;
                    nearestDist = dist;
                }
            }
            if (prev_nearest == nearest) {
                break;
            }
        }
    }

    void HNSW::searchNeighbors(
            DistanceComputer *dc,
            level_t level,
            std::priority_queue<NodeDistCloser> &results,
            vector_idx_t entrypoint,
            double entrypointDist,
            VisitedTable &visited,
            uint64_t efSearch,
            Stats &stats) {
        std::priority_queue<NodeDistFarther> candidates;
        candidates.emplace(entrypoint, entrypointDist);
        results.emplace(entrypoint, entrypointDist);
        visited.set(entrypoint);
        auto neighbors = storage->get_neighbors(level);
        while (!candidates.empty()) {
            auto candidate = candidates.top();
            if (candidate.dist > results.top().dist) {
                break;
            }
            candidates.pop();
            size_t begin, end;
            storage->get_neighbors_offsets(candidate.id, level, begin, end);

            // Prefetch the vectors and the visited table
            size_t jmax = begin;
            for (size_t i = begin; i < end; i++) {
                int neighbor = neighbors[i];
                if (neighbor == INVALID_VECTOR_ID)
                    break;
                prefetch_L3(visited.data() + neighbor);
                jmax += 1;
            }

            for (size_t i = begin; i < jmax; i++) {
                vector_idx_t neighbor = neighbors[i];
                if (neighbor == INVALID_VECTOR_ID) {
                    break;
                }
                if (visited.get(neighbor)) {
                    continue;
                }
                visited.set(neighbor);
                double dist;
                dc->computeDistance(getActualId(level, neighbor), &dist);
                stats.totalDistCompDuringSearch++;
                if (results.size() < efSearch || dist < results.top().dist) {
                    candidates.emplace(neighbor, dist);
                    results.emplace(neighbor, dist);
                    if (results.size() > efSearch) {
                        results.pop();
                    }
                }
            }
        }
        // Reset the visited table
        visited.reset();
    }

    void HNSW::searchNeighborsOnLastLevel(
            DistanceComputer *dc,
            std::priority_queue<NodeDistCloser> &results,
            vector_idx_t entrypoint,
            double entrypointDist,
            VisitedTable &visited,
            uint64_t efSearch,
            int distCompBatchSize,
            Stats &stats) {
        std::priority_queue<NodeDistFarther> candidates;
        candidates.emplace(entrypoint, entrypointDist);
        results.emplace(entrypoint, entrypointDist);
        visited.set(entrypoint);
        auto neighbors = storage->get_neighbors(0);
        printf("efSearch %d\n", efSearch);
        while (!candidates.empty()) {
            auto candidate = candidates.top();
            if (results.size() >= efSearch && candidate.dist > results.top().dist) {
                break;
            }
            candidates.pop();
            size_t begin, end;
            storage->get_neighbors_offsets(candidate.id, 0, begin, end);

            // the following version processes 4 neighbors at a time
            size_t jmax = begin;
            for (size_t i = begin; i < end; i++) {
                vector_idx_t neighbor = neighbors[i];
                if (neighbor == INVALID_VECTOR_ID)
                    break;
                prefetch_L3(visited.data() + neighbor);
                jmax += 1;
            }

            // Perform distance computation on 4 vectors at a time
            int counter = 0;
            vector_idx_t vectorIds[distCompBatchSize];
            for (size_t j = begin; j < jmax; j++) {
                vector_idx_t neighbor = neighbors[j];
                if (neighbor == INVALID_VECTOR_ID)
                    break;
                bool vget = visited.get(neighbor);
                // TODO: Try to set visited in the end of loop
                visited.set(neighbor);
                vectorIds[counter] = neighbor;
                counter += vget ? 0 : 1;
                if (counter == distCompBatchSize) {
                    double distances[distCompBatchSize];
                    dc->batchComputeDistances(vectorIds, distances, distCompBatchSize);
                    stats.totalDistCompDuringSearch += distCompBatchSize;
                    for (int k = 0; k < distCompBatchSize; k++) {
                        if (results.size() < efSearch || distances[k] < results.top().dist) {
                            candidates.emplace(vectorIds[k], distances[k]);
                            results.emplace(vectorIds[k], distances[k]);
                            if (results.size() > efSearch) {
                                results.pop();
                            }
                        }
                    }
                    counter = 0;
                }
            }

            // For the left out nodes
            for (int k = 0; k < counter; k++) {
                double dist;
                dc->computeDistance(vectorIds[k], &dist);
                stats.totalDistCompDuringSearch++;
                if (results.size() < efSearch || dist < results.top().dist) {
                    candidates.emplace(vectorIds[k], dist);
                    results.emplace(vectorIds[k], dist);
                    if (results.size() > efSearch) {
                        results.pop();
                    }
                }
            }
        }
        // Reset the visited table
        visited.reset();
    }

    void HNSW::searchNeighborsOnLastLevel1(
            DistanceComputer *dc,
            std::priority_queue<NodeDistCloser> &results,
            vector_idx_t entrypoint,
            double entrypointDist,
            AtomicVisitedTable &visited,
            uint64_t efSearch,
            int distCompBatchSize,
            Stats &stats) {
        std::priority_queue<NodeDistFarther> candidates;
        candidates.emplace(entrypoint, entrypointDist);
        results.emplace(entrypoint, entrypointDist);
        visited.set(entrypoint);
        auto neighbors = storage->get_neighbors(0);
        while (!candidates.empty()) {
            auto candidate = candidates.top();
            if (candidate.dist > results.top().dist) {
                break;
            }
            candidates.pop();
            size_t begin, end;
            storage->get_neighbors_offsets(candidate.id, 0, begin, end);

            // the following version processes 4 neighbors at a time
            size_t jmax = begin;
            for (size_t i = begin; i < end; i++) {
                vector_idx_t neighbor = neighbors[i];
                if (neighbor == INVALID_VECTOR_ID)
                    break;
//                prefetch_L3(visited.data() + neighbor);
                jmax += 1;
            }

            // Perform distance computation on 4 vectors at a time
            int counter = 0;
            vector_idx_t vectorIds[distCompBatchSize];
            for (size_t j = begin; j < jmax; j++) {
                vector_idx_t neighbor = neighbors[j];
                if (neighbor == INVALID_VECTOR_ID)
                    break;
                bool vget = visited.get(neighbor);
                // TODO: Try to set visited in the end of loop
                visited.set(neighbor);
                vectorIds[counter] = neighbor;
                counter += vget ? 0 : 1;
                if (counter == distCompBatchSize) {
                    double distances[distCompBatchSize];
                    dc->batchComputeDistances(vectorIds, distances, distCompBatchSize);
                    stats.totalDistCompDuringSearch += distCompBatchSize;
                    for (int k = 0; k < distCompBatchSize; k++) {
                        if (results.size() < efSearch || distances[k] < results.top().dist) {
                            candidates.emplace(vectorIds[k], distances[k]);
                            results.emplace(vectorIds[k], distances[k]);
                            if (results.size() > efSearch) {
                                results.pop();
                            }
                        }
                    }
                    counter = 0;
                }
            }

            // For the left out nodes
            for (int k = 0; k < counter; k++) {
                double dist;
                dc->computeDistance(vectorIds[k], &dist);
                stats.totalDistCompDuringSearch++;
                if (results.size() < efSearch || dist < results.top().dist) {
                    candidates.emplace(vectorIds[k], dist);
                    results.emplace(vectorIds[k], dist);
                    if (results.size() > efSearch) {
                        results.pop();
                    }
                }
            }
        }
    }

    // TODO: Run shrink with approx distance (compressed vectors)
    //       Another idea: Run shrink with smaller alpha, if the nodes reduced significantly, use it with higher alpha
    //       Another idea: Use centroids to check the equality condition
    void HNSW::shrinkNeighbors(
            DistanceComputer *dc,
            vector_idx_t id,
            std::priority_queue<NodeDistCloser> &results,
            int maxSize,
            level_t level,
            int dim,
            Stats &stats) {
        if (results.size() <= maxSize) {
            return;
        }

//        if (stats.shrinkCallsPerNode.contains(id)) {
//            stats.shrinkCallsPerNode[id]++;
//        } else {
//            stats.shrinkCallsPerNode[id] = 1;
//        }
//        stats.totalShrinkCalls++;
//        auto currentVal = storage->elementShrinkCalls->at(id).fetch_add(1);
//        auto alpha = std::max(config.maxAlpha - (config.alphaDecay * currentVal), config.minAlpha);
        std::priority_queue<NodeDistFarther> temp;
        std::vector<NodeDistFarther> result;
        while (!results.empty()) {
            temp.emplace(results.top().id, results.top().dist);
            results.pop();
        }

        // Push nodes to result based on a heuristic that distance to the query node is smallest against all other nodes
        // in the neighbors list. We add it in decreasing order of distance such that farthest nodes are always added.
        while (!temp.empty()) {
            auto nodeA = temp.top();
            temp.pop();
            auto distNodeAQ = nodeA.dist;
            bool good = true;
            for (NodeDistFarther &nodeB: result) {
                double distNodeAB;
                dc->computeDistance(getActualId(level, nodeA.id), getActualId(level, nodeB.id), dim, &distNodeAB);
                stats.totalDistCompDuringShrink++;
                if ((config.minAlpha * distNodeAB) < distNodeAQ) {
                    good = false;
                    break;
                }
            }
            if (good) {
                result.push_back(nodeA);
                if (result.size() >= maxSize) {
                    break;
                }
            }
        }
        for (auto &node: result) {
            results.emplace(node.id, node.dist);
        }

//        if (level == 0) {
//            std::vector<double> oldDistances;
//            // read old distances till the last blocker
//            for (int i = storage->afterShrinkDistances[id].size() - 2; i >= 0; i--) {
//                if (storage->afterShrinkDistances[id][i] == -1) {
//                    break;
//                }
//                oldDistances.push_back(storage->afterShrinkDistances[id][i]);
//            }
//
//            // check if the new distances are same as the old distances sort and check
//            if (!oldDistances.empty()) {
//                std::sort(oldDistances.begin(), oldDistances.end());
//                for (int i = 0; i < result.size(); i++) {
//                    if (oldDistances[i] != result[i].dist) {
//                        stats.totalShrinkNoUse++;
//                        break;
//                    }
//                }
//            } else {
//                stats.totalShrinkNoUse++;
//            }
//
//            for (auto &node: result) {
//                storage->afterShrinkDistances[id].push_back(node.dist);
//            }
//            // Add a blocker
//            storage->afterShrinkDistances[id].push_back(-1);
//        }
    }

    // It is assumed that the node is already locked.
    void HNSW::makeConnection(
            DistanceComputer *dc,
            vector_idx_t src,
            vector_idx_t dest,
            double distSrcDest,
            level_t level,
            Stats &stats) {
        auto neighbors = storage->get_neighbors(level);
        size_t begin, end;
        storage->get_neighbors_offsets(src, level, begin, end);
        // TODO: Optimize this code using bulk shrink. Basically accumulate the neighbors in a vector and then shrink
        //  in a single go.
        if (neighbors[end - 1] == INVALID_VECTOR_ID) {
            if (neighbors[begin] == INVALID_VECTOR_ID) {
                neighbors[begin] = dest;
                return;
            }
            // do loop in reverse order, it could yield faster results
            for (size_t i = end; i > begin; i--) {
                if (neighbors[i - 1] != INVALID_VECTOR_ID) {
                    neighbors[i] = dest;
                    return;
                }
            }
        }

        // Otherwise we need to shrink the neighbors list
        std::priority_queue<NodeDistCloser> results;
        results.emplace(dest, distSrcDest);
        for (size_t i = begin; i < end; i++) {
            auto neighbor = neighbors[i];
            double distSrcNbr;
            dc->computeDistance(getActualId(level, src), getActualId(level, neighbor), &distSrcNbr);
            stats.totalDistCompDuringMakeConnection++;
            results.emplace(neighbor, distSrcNbr);
        }
        shrinkNeighbors(dc, src, results, storage->max_neighbors_per_level[level], level, storage->dim, stats);
        size_t i = begin;
        while (!results.empty()) {
            neighbors[i++] = results.top().id;
            results.pop();
        }
        while (i < end) {
            neighbors[i++] = INVALID_VECTOR_ID;
        }
    }

    void HNSW::addNodeOnLevel(
            DistanceComputer *dc,
            vector_idx_t id,
            level_t level,
            vector_idx_t entrypoint,
            double entrypointDist,
            VisitedTable &visited,
            std::vector<NodeDistCloser> &neighbors,
            Stats &stats) {
        std::priority_queue<NodeDistCloser> linkTargets;
        searchNeighbors(dc, level, linkTargets, entrypoint, entrypointDist, visited, config.efConstruction, stats);
//        if (level == 0) {
//            printf("size of linkTargets %d\n", linkTargets.size());
//        }
        shrinkNeighbors(dc, id, linkTargets, storage->max_neighbors_per_level[level], level, storage->dim, stats);

        neighbors.reserve(linkTargets.size());
        while (!linkTargets.empty()) {
            auto neighborNode = linkTargets.top();
            linkTargets.pop();
            if (neighborNode.id == id) {
                continue;
            }
            makeConnection(dc, id, neighborNode.id, neighborNode.dist, level, stats);
            neighbors.emplace_back(neighborNode.id, neighborNode.dist);
        }
    }

    void HNSW::addNode(
            DistanceComputer *dc,
            std::vector<vector_idx_t> node_id,
            level_t node_level,
            std::vector<omp_lock_t> &locks,
            VisitedTable &visited,
            Stats &stats) {
        vector_idx_t nearestId;
        int level;
#pragma omp critical
        {
            if (storage->entryPoint == INVALID_VECTOR_ID) {
                storage->maxLevel = node_level;
                // Fix this maybe
                storage->entryPoint = node_id[node_level];
            }
            nearestId = storage->entryPoint;
            level = storage->maxLevel;
        }

        omp_set_lock(&locks[node_id[0]]);
        double nearestDist;
        std::vector<std::vector<NodeDistCloser>> neighbors(node_level + 1);
        dc->computeDistance(getActualId(level, nearestId), &nearestDist);

        // Update the nearest node
        for (; level > node_level; level--) {
            searchNearestOnLevel(dc, level, nearestId, nearestDist, stats);
            nearestId = storage->next_level_ids[level][nearestId];
        }

        // Add the node to the graph
        for (; level >= 0; level--) {
            auto node_level_id = node_id[level];
            addNodeOnLevel(dc, node_level_id, level, nearestId, nearestDist, visited, neighbors[level], stats);
            if (level != 0) {
                nearestId = storage->next_level_ids[level][nearestId];
            }
        }
        omp_unset_lock(&locks[node_id[0]]);

        for (int i = 0; i < neighbors.size(); i++) {
            for (auto neighbor: neighbors[i]) {
                omp_set_lock(&locks[getActualId(i, neighbor.id)]);
                makeConnection(dc, neighbor.id, node_id[i], neighbor.dist, i, stats);
                omp_unset_lock(&locks[getActualId(i, neighbor.id)]);
            }
        }

#pragma omp critical
        {
            // Update the entry point
            if (node_level > storage->maxLevel) {
                storage->maxLevel = node_level;
                storage->entryPoint = node_id[node_level];
            }
        }
    }

    // TODO: Add caching, thread pinning, live locks, queue with locks etc
    void HNSW::build(const float *data, size_t n) {
        if (config.loadStorage) {
            storage->data = data;
//            if (config.compressionType == "scalar_8bit") {
//                storage->codes = new uint8_t[n * quantizer->codeSize];
//                quantizer->batch_train(n, data);
//                quantizer->encode(data, storage->codes, n);
//            } else if (config.compressionType == "pair_wise") {
//                // Pair wise quantization
//                storage->codes = new uint8_t[n * quantizer->codeSize];
//                quantizer->batch_train(n, data);
//                quantizer->encode(data, storage->codes, n);
//            }
            return;
        }


        std::vector<omp_lock_t> locks(n);
        for (int i = 0; i < n; i++) {
            omp_init_lock(&locks[i]);
        }
        // Initialize ids
        std::vector<std::vector<vector_idx_t>> node_ids(n);
        uint8_t max_level = 0;
        for (int i = 0; i < n; i++) {
            uint8_t level = randomLevel();
            auto id = storage->getFastNextNodeId(level);
            node_ids[i] = id;
            max_level = std::max(max_level, level);
        }

        // Set data to storage
        // Todo: Copy data to storage
        storage->data = data;

        // Quantize the data
        if (config.compressionType == "scalar_8bit") {
            storage->codes = new uint8_t[n * quantizer->codeSize];
            quantizer->batch_train(n, data);
            quantizer->encode(data, storage->codes, n);
        } else if (config.compressionType == "pair_wise") {
            // Pair wise quantization
            storage->codes = new uint8_t[n * quantizer->codeSize];
            quantizer->batch_train(n, data);
            quantizer->encode(data, storage->codes, n);
        }

        // Set the size for storage
        // Todo: Figure out if ordering is important!! Basically we are adding vectors from highest to lowest level.
        for (int i = 0; i <= max_level; i++) {
            storage->resize(storage->fast_level_counters[i], i);
        }

        for (int i = 0; i < n; i++) {
            auto node_id = node_ids[i];
            for (int j = node_id.size() - 1; j > 0; j--) {
                storage->next_level_ids[j][node_id[j]] = node_id[j - 1];
                storage->actual_ids[j][node_id[j]] = node_id[0];
            }
        }

        printf("Building the graph %d\n", max_level);
#pragma omp parallel
        {
//            auto asym_dc = quantizer->get_asym_distance_computer(fastq::DistanceType::L2);
//            auto sym_dc = quantizer->get_sym_distance_computer(fastq::DistanceType::L2);
            DistanceComputer *localDc = new CosineDistanceComputer(data, storage->dim, n);
//            DistanceComputer *localDc = new QuantizedDistanceComputer(storage->codes, asym_dc.get(), sym_dc.get(),
//                                                                      storage->code_size);
            VisitedTable visited(n);
            Stats localStats = Stats();
#pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                localDc->setQuery(storage->data + (i * storage->dim));
                addNode(localDc, node_ids[i], node_ids[i].size() - 1, locks, visited, localStats);
                if (i % 10000 == 0) {
                    spdlog::warn("Inserted 10000!!");
                }
            }
            // Merge stats
            stats.merge(localStats);
        }

        // Destroy locks
        for (int i = 0; i < n; i++) {
            omp_destroy_lock(&locks[i]);
        }
    }

    uint8_t HNSW::randomLevel() {
        double f = rg->randFloat();
        for (int level = 0; level < levelProbabs.size(); level++) {
            if (f < levelProbabs[level]) {
                return level;
            }
            f -= levelProbabs[level];
        }
        // happens with exponentially low probability
        return levelProbabs.size() - 1;
    }

    void HNSW::search(
            const float *query,
            uint64_t k,
            uint64_t efSearch,
            VisitedTable &visited,
            std::priority_queue<NodeDistCloser> &results,
            Stats &stats) {
        CosineDistanceComputer dc = CosineDistanceComputer(storage->data, storage->dim, storage->numPoints);
        dc.setQuery(query);
        int newEfSearch = std::max(k, efSearch);
        vector_idx_t nearestID = storage->entryPoint;
        double nearestDist;
        int level = storage->maxLevel;
        dc.computeDistance(getActualId(level, nearestID), &nearestDist);
        // Update the nearest node
        for (; level > 0; level--) {
            searchNearestOnLevel(&dc, level, nearestID, nearestDist, stats);
            nearestID = storage->next_level_ids[level][nearestID];
        }
        searchNeighborsOnLastLevel(
                &dc,
                results,
                nearestID,
                nearestDist,
                visited,
                newEfSearch,
                4,
                stats);
    }

    void HNSW::searchNearestOnLevelWithQuantizer(const float *query, fastq::DistanceComputer<float, uint8_t> *dc, orangedb::level_t level,
                                                 orangedb::vector_idx_t &nearest, double &nearestDist,
                                                 orangedb::Stats &stats) {
        CHECK_ARGUMENT(level > 0, "Level should be greater than 0");
        auto neighbors = storage->get_neighbors(level);
        // Can we do distance computation on 4 vectors at a time?
        while (true) {
            vector_idx_t prev_nearest = nearest;
            size_t begin, end;
            storage->get_neighbors_offsets(nearest, level, begin, end);
            for (size_t i = begin; i < end; i++) {
                vector_idx_t neighbor = neighbors[i];
                if (neighbor == INVALID_VECTOR_ID) {
                    break;
                }
                double dist;
                stats.totalDistCompDuringSearch++;
                const uint8_t *code = storage->codes + getActualId(level, neighbor) * quantizer->codeSize;
                dc->compute_distance(query, code, &dist);
                if (dist < nearestDist) {
                    nearest = neighbor;
                    nearestDist = dist;
                }
            }
            if (prev_nearest == nearest) {
                break;
            }
        }

    }

    void HNSW::searchNeighborsOnLastLevelWithQuantizer(const float *query, fastq::DistanceComputer<float, uint8_t> *dc,
                                                       std::priority_queue<NodeDistCloser> &results,
                                                       orangedb::vector_idx_t entrypoint, double entrypointDist,
                                                       orangedb::VisitedTable &visited, uint64_t efSearch,
                                                       int distCompBatchSize, orangedb::Stats &stats) {
        std::priority_queue<NodeDistFarther> candidates;
        candidates.emplace(entrypoint, entrypointDist);
        results.emplace(entrypoint, entrypointDist);
        visited.set(entrypoint);
        auto neighbors = storage->get_neighbors(0);
        while (!candidates.empty()) {
            auto candidate = candidates.top();
            if (candidate.dist > results.top().dist) {
                break;
            }
            candidates.pop();
            size_t begin, end;
            storage->get_neighbors_offsets(candidate.id, 0, begin, end);

            // the following version processes 4 neighbors at a time
            size_t jmax = begin;
            for (size_t i = begin; i < end; i++) {
                vector_idx_t neighbor = neighbors[i];
                if (neighbor == INVALID_VECTOR_ID)
                    break;
                prefetch_L3(visited.data() + neighbor);
                jmax += 1;
            }

            // Perform distance computation on 4 vectors at a time
            for (size_t i = begin; i < jmax; i++) {
                vector_idx_t neighbor = neighbors[i];
                if (neighbor == INVALID_VECTOR_ID) {
                    break;
                }
                if (visited.get(neighbor)) {
                    continue;
                }
                visited.set(neighbor);
                double dist;
                const uint8_t *code = storage->codes + neighbor * quantizer->codeSize;
                dc->compute_distance(query, code, &dist);
                stats.totalDistCompDuringSearch++;
                if (results.size() < efSearch || dist < results.top().dist) {
                    candidates.emplace(neighbor, dist);
                    results.emplace(neighbor, dist);
                    if (results.size() > efSearch) {
                        results.pop();
                    }
                }
            }
        }
        // Reset the visited table
        visited.reset();
    }

    void HNSW::searchParallel(
            const float *query,
            uint64_t k,
            uint64_t efSearch,
            AtomicVisitedTable &visited,
            std::priority_queue<NodeDistCloser> &results,
            Stats &stats,
            PocTaskScheduler *scheduler) {
        CosineDistanceComputer dc = CosineDistanceComputer(storage->data, storage->dim, storage->numPoints);
        dc.setQuery(query);
        int newEfSearch = std::max(k, efSearch);
        vector_idx_t nearestID = storage->entryPoint;
        double nearestDist;
        int level = storage->maxLevel;
        dc.computeDistance(getActualId(level, nearestID), &nearestDist);
        // Update the nearest node
        for (; level > 0; level--) {
            searchNearestOnLevel(&dc, level, nearestID, nearestDist, stats);
            nearestID = storage->next_level_ids[level][nearestID];
        }
        VisitedTable localVisited(storage->numPoints);

        if (config.searchParallelAlgorithm == "none") {
            searchNeighborsOnLastLevel(
                    &dc,
                    results,
                    nearestID,
                    nearestDist,
                    localVisited,
                    newEfSearch,
                    4,
                    stats);
            return;
        }

        // search atleast 50 Local minima ANN
        searchNeighborsOnLastLevel(
                &dc,
                results,
                nearestID,
                nearestDist,
                localVisited,
                60,
                4,
                stats);

        // Parallel search
        if (config.searchParallelAlgorithm == "et") {
            searchParallelSyncAfterEveryIter(&dc, results, visited, newEfSearch, stats, scheduler);
        } else if (config.searchParallelAlgorithm == "pq") {
            searchParallelWithParallelQueue(&dc, results, visited, newEfSearch, stats);
        } else if (config.searchParallelAlgorithm == "part") {
            searchParallelWithPartitioning(&dc, results, visited, newEfSearch, stats);
        } else if (config.searchParallelAlgorithm == "deltaStepping") {
            searchParallelWithDeltaStepping(&dc, results, visited, newEfSearch, stats);
        } else {
            throw std::runtime_error("Unknown search parallel algorithm");
        }
    }

    int HNSW::findNextKNeighbours(
            vector_idx_t entrypoint,
            NodeDistCloser *nbrs,
            AtomicVisitedTable &visited,
            int maxK,
            int maxNeighboursCheck) {
        auto neighbors = storage->get_neighbors(0);
        std::queue<vector_idx_t> candidates;
        candidates.push(entrypoint);
        auto neighboursChecked = 0;
        int m = 0;
        std::unordered_set<vector_idx_t> visitedSet;
        while (neighboursChecked <= maxNeighboursCheck && !candidates.empty()) {
            auto candidate = candidates.front();
            candidates.pop();
            size_t begin, end;
            if (visitedSet.contains(candidate)) {
                continue;
            }
            visitedSet.insert(candidate);
            storage->get_neighbors_offsets(candidate, 0, begin, end);
            neighboursChecked += 1;
            stats.totalGetNbrsCall++;
            // TODO: Maybe make it prioritized, might help in correlated cases
            for (size_t i = begin; i < end; i++) {
                auto neighbor = neighbors[i];
                if (neighbor == INVALID_VECTOR_ID) {
                    break;
                }
                // getAndSet has to be atomic
                // Add to visited set
                if (visited.getAndSet(neighbor)) {
                    nbrs[m] = NodeDistCloser(neighbor, std::numeric_limits<double>::max());
                    m++;
                    if (m >= maxK) {
                        return m;
                    }
                }
                candidates.push(neighbor);
            }
        }
        return m;
    }

    static void mergeSortCandidates(
            std::vector<NodeDistCloser> &candidates,
            std::vector<NodeDistCloser> &nextFrontier,
            int M,
            int partitionSize,
            int numPartitions) {
        CHECK_ARGUMENT(candidates.size() >= M, "Candidates size should be greater than M");
        CHECK_ARGUMENT(nextFrontier.size() >= partitionSize * numPartitions, "Next frontier size should be greater than partitionSize * numPartitions");
        int m = std::min(M, partitionSize);
        // candidates is already sorted
        // Next frontier each partition is sorted
        // Merge sort the values from nextFrontier each partition to candidates until M
        for (int part = 0; part < numPartitions; part++) {
            int start = part * partitionSize;
            int end = std::min(start + m, (int) nextFrontier.size());
            int ci = 0;
            // Merge sort the values from nextFrontier each partition to candidates until M
            for (int i = start; i < end; i++) {
                while (ci < M && candidates[ci] < nextFrontier[i]) {
                    ci++;
                }
                if (ci == M) {
                    break;
                }
                candidates[ci] = nextFrontier[i];
                ci++;
            }
        }
    }

    void HNSW::searchParallelSyncAfterEveryIter(
            DistanceComputer *dc,
            std::priority_queue<NodeDistCloser> &results,
            AtomicVisitedTable &visited,
            uint64_t efSearch,
            Stats &stats,
            PocTaskScheduler *scheduler) {
        int nodesToExplore = std::max(config.nodesToExplore, config.numSearchThreads);
        int nodeExpansionPerNode = config.nodeExpansionPerNode;
        std::vector<NodeDistCloser> nextFrontier(nodesToExplore * nodeExpansionPerNode, NodeDistCloser());
        std::vector<NodeDistCloser> candidates(nodesToExplore * nodeExpansionPerNode + 500, NodeDistCloser());
        // Since the check is not parallelized, we can include all threads queue for it.
        ParallelMultiQueue<NodeDistFarther> resultPq(config.numSearchThreads * 2, efSearch);
        // Init the scheduler
        scheduler->resultPq = &resultPq;
        scheduler->candidates = candidates.data();
        scheduler->nextFrontier = nextFrontier.data();
        scheduler->dc = dc;

        // Fill the candidates from results
        int numCandidates = 0;
        while (!results.empty()) {
            auto res = results.top();
            results.pop();
            resultPq.push(NodeDistFarther(res.id, res.dist));
            visited.getAndSet(res.id);
            if (results.size() < nodesToExplore) {
                candidates[numCandidates] = res;
                numCandidates++;
            }
        }

        printf("Number of candidates %d\n", numCandidates);
        int iter = 0;

        // sort the candidates
        sort(candidates.begin(), candidates.begin() + numCandidates);
        int maxNextFrontierSize = nodesToExplore * nodeExpansionPerNode;

        auto startTime = std::chrono::high_resolution_clock::now();

        while (numCandidates > 0) {
            // Reset the nextFrontier
            for (int i = 0; i < maxNextFrontierSize; i++) {
                nextFrontier[i] = NodeDistCloser();
            }

            int expandedCount = scheduler->parallelize_and_wait(0, numCandidates, false);
            int currentMaxNextFrontierSize = numCandidates * nodeExpansionPerNode;
            int numDistComp = 0;
            for (int i = 0; i < currentMaxNextFrontierSize; i++) {
                if (numDistComp == expandedCount) {
                    break;
                }
                auto neighbor = nextFrontier[i];
                if (neighbor.isInvalid()) {
                    continue;
                }
                nextFrontier[i] = NodeDistCloser();
                nextFrontier[numDistComp] = neighbor;
                numDistComp++;
            }

            int distCompBatchSize = numDistComp / config.numSearchThreads;
            stats.totalDistCompDuringSearch += numDistComp;
            scheduler->parallelize_and_wait(0, numDistComp, true);

            // Reset the top numCandidates values
            for (int i = 0; i < numCandidates; i++) {
                candidates[i] = NodeDistCloser();
            }

            auto start = std::chrono::high_resolution_clock::now();
            // Merge sort the candidates and nextFrontier
            sort(candidates.begin(), candidates.end());
            mergeSortCandidates(candidates, nextFrontier, (int) candidates.size(), distCompBatchSize, config.numSearchThreads);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            printf("Time to merge sort %lld us\n", duration.count());

            numCandidates = 0;
            for (int i = 0; i < nodesToExplore; i++) {
                if (candidates[i].isInvalid()) {
                    break;
                }
                numCandidates++;
            }
            iter++;
            if (resultPq.top()->dist < candidates[0].dist) {
                break;
            }
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        printf("Time to search %lld ms\n", duration.count());

        printf("iter %d\n", iter);
        auto start = std::chrono::high_resolution_clock::now();
        // Put resultPq to results
        for (auto& queue : resultPq.queues) {
            while (queue->size() > 0) {
                auto min = queue->popMin();
                results.push(NodeDistCloser(min.id, min.dist));
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Time to put to results %lld ms\n", duration.count());
    }

    void HNSW::searchParallelWithParallelQueue(
            DistanceComputer *dc,
            std::priority_queue<NodeDistCloser> &results,
            AtomicVisitedTable &visited,
            uint64_t efSearch,
            Stats &stats) {
        // Let's try super simple
        auto W = config.nodesToExplore;
        printf("efSearch %llu\n", efSearch);
        ParallelMultiQueue<NodeDistFarther> resultsPq(config.numSearchThreads * 2, efSearch);
        std::vector<std::vector<NodeDistCloser>> candidates(config.numSearchThreads );
        // init the candidates
        for (int i = 0; i < config.numSearchThreads ; i++) {
            candidates[i] = std::vector<NodeDistCloser>();
        }

        // use results to fill the candidates
        int numCandidates = 0;
        while (!results.empty()) {
            auto res = results.top();
            results.pop();
            resultsPq.push(NodeDistFarther(res.id, res.dist));
            visited.getAndSet(res.id);
            if (results.size() < W) {
                candidates[numCandidates].push_back(res);
                numCandidates = (numCandidates + 1) % config.numSearchThreads;
            }
        }

#pragma omp parallel
        {
            int tId = omp_get_thread_num();
            auto &localCandidates = candidates[tId];
            std::priority_queue<NodeDistFarther> localC;
            for (auto &c : localCandidates) {
                localC.emplace(c.id, c.dist);
            }

            // Implement local search
            std::vector<NodeDistCloser> nextFrontier(config.nodeExpansionPerNode);
            while (!localC.empty()) {
                auto candidate = localC.top();
                localC.pop();
                int nextFSize = findNextKNeighbours(candidate.id, nextFrontier.data(), visited, config.nodeExpansionPerNode, 128);

                // Compute the distances
                for (size_t j = 0; j < nextFSize; j++) {
                    vector_idx_t neighbor = nextFrontier[j].id;
                    double dist;
                    dc->computeDistance(neighbor, &dist);
                    if (resultsPq.size() < efSearch || dist < resultsPq.top()->dist) {
                        localC.emplace(neighbor, dist);
                        resultsPq.push(NodeDistFarther(neighbor, dist));
                    }
                }

                if (localC.top().dist > resultsPq.top()->dist) {
                    break;
                }
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        // Put resultPq to results
        for (auto& queue : resultsPq.queues) {
            while (queue->size() > 0) {
                auto min = queue->popMin();
                results.emplace(min.id, min.dist);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Time to put to results %lld ms\n", duration.count());
    }

    void HNSW::searchParallelWithPartitioning(
            DistanceComputer *dc,
            std::priority_queue<NodeDistCloser> &results,
            AtomicVisitedTable &visited,
            uint64_t efSearch,
            Stats &stats) {
        auto W = config.nodesToExplore;
        printf("efSearch %llu\n", efSearch);
        std::vector<std::vector<NodeDistCloser>> candidates(config.numSearchThreads);
        // init the candidates
        for (int i = 0; i < config.numSearchThreads; i++) {
            candidates[i] = std::vector<NodeDistCloser>();
        }

        // use results to fill the candidates
        int numCandidates = 0;
        while (!results.empty()) {
            auto res = results.top();
            results.pop();
            if (results.size() < W) {
                visited.getAndSet(res.id);
                candidates[numCandidates].push_back(res);
                numCandidates = (numCandidates + 1) % config.numSearchThreads;
            }
        }

#pragma omp parallel
        {
            auto localEfSearch = efSearch / config.numSearchThreads;
            printf("localEfSearch %d\n", localEfSearch);
            int tId = omp_get_thread_num();
            auto &cands = candidates[tId];
            std::priority_queue<NodeDistFarther> localCandidates;
            std::priority_queue<NodeDistCloser> localResults;
            for (auto &c : cands) {
                localCandidates.emplace(c.id, c.dist);
                localResults.emplace(c.id, c.dist);
            }
            // Implement local search
            std::vector<NodeDistCloser> nextFrontier(config.nodeExpansionPerNode);
            while (!localCandidates.empty()) {
                auto candidate = localCandidates.top();
                localCandidates.pop();
                int nextFSize = findNextKNeighbours(candidate.id, nextFrontier.data(), visited, config.nodeExpansionPerNode, 128);
                printf("nextFSize %d\n", nextFSize);

                // Compute the distances
                for (size_t j = 0; j < nextFSize; j++) {
                    vector_idx_t neighbor = nextFrontier[j].id;
                    double dist;
                    dc->computeDistance(neighbor, &dist);
                    if (localResults.size() < localEfSearch || dist < localResults.top().dist) {
                        localCandidates.emplace(neighbor, dist);
                        localResults.emplace(neighbor, dist);
                        if (localResults.size() > localEfSearch) {
                            localResults.pop();
                        }
                    }
                }

                if (localCandidates.top().dist > localResults.top().dist) {
                    printf("Breaking\n");
                    break;
                }
            }

            // Put the localResults to results
#pragma omp critical
            {
                while (!localResults.empty()) {
                    results.push(localResults.top());
                    localResults.pop();
                }
            }
        };
    }

    void HNSW::searchParallelWithDeltaStepping(
            DistanceComputer *dc,
            std::priority_queue<NodeDistCloser> &results,
            AtomicVisitedTable &visited,
            uint64_t efSearch,
            Stats &stats) {
        auto W = config.nodesToExplore;
        printf("efSearch %llu\n", efSearch);
        std::vector<std::vector<NodeDistCloser>> candidates(config.numSearchThreads);
        ParallelMultiQueue<NodeDistFarther> resultsPq(config.numSearchThreads * 2, efSearch);
        // init the candidates
        for (int i = 0; i < config.numSearchThreads; i++) {
            candidates[i] = std::vector<NodeDistCloser>();
        }

        // use results to fill the candidates
        int numCandidates = 0;
        while (!results.empty()) {
            auto res = results.top();
            results.pop();
            if (results.size() < W) {
                visited.getAndSet(res.id);
                candidates[numCandidates].push_back(res);
                numCandidates = (numCandidates + 1) % config.numSearchThreads;
            }
        }

        std::priority_queue<NodeDistCloser> syncCandidates;

#pragma omp parallel
        {
            int tId = omp_get_thread_num();
            std::vector<NodeDistCloser> nextFrontier(config.nodeExpansionPerNode);
            auto localEfSearch = 64;
            auto &cands = candidates[tId];

            while (localEfSearch <= efSearch) {
                std::priority_queue<NodeDistFarther> localCandidates;
                for (auto &c: cands) {
                    localCandidates.emplace(c.id, c.dist);
                }

                while (!localCandidates.empty()) {
                    auto candidate = localCandidates.top();
                    localCandidates.pop();
                    int nextFSize = findNextKNeighbours(candidate.id, nextFrontier.data(), visited, config.nodeExpansionPerNode, 128);

                    // Compute the distances
                    for (size_t j = 0; j < nextFSize; j++) {
                        vector_idx_t neighbor = nextFrontier[j].id;
                        double dist;
                        dc->computeDistance(neighbor, &dist);
                        if (localCandidates.size() < localEfSearch || dist < resultsPq.top()->dist) {
                            localCandidates.emplace(neighbor, dist);
                            resultsPq.push(NodeDistFarther(neighbor, dist));
                        }
                    }

                    if (localCandidates.top().dist > resultsPq.top()->dist) {
                        break;
                    }
                }

#pragma omp critical
                {
                    int k = 0;
                    while (!localCandidates.empty()) {
                        auto c = localCandidates.top();
                        localCandidates.pop();
                        syncCandidates.emplace(c.id, c.dist);
                        if (k++ == 20) {
                            break;
                        }
                    }
                }

#pragma omp barrier

                // Sync the candidates
#pragma omp critical
                {
                    int k = 0;
                    cands.clear();
                    while (!syncCandidates.empty()) {
                        auto c = syncCandidates.top();
                        syncCandidates.pop();
                        cands.push_back(c);
                        if (k++ == 2) {
                            break;
                        }
                    }
                }
                localEfSearch += 64;
            }
        };
    }

    void HNSW::searchWithQuantizer(const float *query, uint64_t k, uint64_t efSearch, orangedb::VisitedTable &visited,
                                   std::priority_queue<NodeDistCloser> &results, orangedb::Stats &stats) {
        auto dc = quantizer->get_asym_distance_computer(fastq::DistanceType::COSINE);
        int newEfSearch = std::max(k, efSearch);
        vector_idx_t nearestID = storage->entryPoint;
        double nearestDist;
        int level = storage->maxLevel;
        const uint8_t *code = storage->codes + getActualId(level, nearestID) * quantizer->codeSize;
//        std::vector<uint8_t> queryCode(quantizer->codeSize);
//        quantizer->encode(query, queryCode.data(), 1);

        dc->compute_distance(query, code, &nearestDist);
        // Update the nearest node
        for (; level > 0; level--) {
            searchNearestOnLevelWithQuantizer(query, dc.get(), level, nearestID, nearestDist, stats);
            nearestID = storage->next_level_ids[level][nearestID];
        }
        searchNeighborsOnLastLevelWithQuantizer(
                query,
                dc.get(),
                results,
                nearestID,
                nearestDist,
                visited,
                newEfSearch,
                4,
                stats);
    }

    void HNSW::deleteNodes(const vector_idx_t *deletedIds, size_t n, int dim, Stats &stats) {
        L2DistanceComputer dc = L2DistanceComputer(storage->data, storage->dim, storage->numPoints);
        std::vector<omp_lock_t> locks(storage->numPoints);
        for (int i = 0; i < storage->numPoints; i++) {
            omp_init_lock(&locks[i]);
        }
        VisitedTable visited(storage->numPoints);
        for (int i = 0; i < n; i++) {
            visited.set(deletedIds[i]);
        }

#pragma omp parallel
        {
            Stats localStats = Stats();
            auto *infVector = new float[storage->dim];
            for (int i = 0; i < storage->dim; i++) {
                infVector[i] = MAXFLOAT;
            }

#pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                deleteNode(&dc, deletedIds[i], locks, infVector, dim, visited, localStats);
                if (i % 10000 == 0) {
                    spdlog::warn("Deleted 10000!!");
                }
            }

            delete[] infVector;
            stats.merge(localStats);
        }

        // Destroy the locks
        for (int i = 0; i < storage->numPoints; i++) {
            omp_destroy_lock(&locks[i]);
        }
    }

    void HNSW::searchNeighborsOnLastLevelWithFilterA(
            orangedb::DistanceComputer *dc,
            std::priority_queue<NodeDistCloser> &results,
            orangedb::vector_idx_t entrypoint,
            double entrypointDist,
            orangedb::VisitedTable &visited,
            uint64_t efSearch,
            int distCompBatchSize,
            const uint8_t *filterMask,
            orangedb::Stats &stats) {
        std::priority_queue<NodeDistFarther> candidates;
        candidates.emplace(entrypoint, entrypointDist);
        results.emplace(entrypoint, entrypointDist);
        visited.set(entrypoint);
        auto neighbors = storage->get_neighbors(0);
        while (!candidates.empty()) {
            auto candidate = candidates.top();
            // TODO: Do we need to check if the results are above efSearch?
            if (candidate.dist > results.top().dist) {
                break;
            }
            candidates.pop();
            size_t begin, end;
            storage->get_neighbors_offsets(candidate.id, 0, begin, end);
            stats.totalGetNbrsCall++;

            // the following version processes 4 neighbors at a time
            size_t jmax = begin;
            for (size_t i = begin; i < end; i++) {
                vector_idx_t neighbor = neighbors[i];
                if (neighbor == INVALID_VECTOR_ID)
                    break;
                prefetch_L3(visited.data() + neighbor);
                jmax += 1;
            }

            // Perform distance computation on 4 vectors at a time
            int counter = 0;
            vector_idx_t vectorIds[distCompBatchSize];
            for (size_t j = begin; j < jmax; j++) {
                vector_idx_t neighbor = neighbors[j];
                if (neighbor == INVALID_VECTOR_ID)
                    break;
                bool vget = visited.get(neighbor);
                // TODO: Try to set visited in the end of loop
                visited.set(neighbor);
                vectorIds[counter] = neighbor;
                counter += vget ? 0 : 1;
                if (counter == distCompBatchSize) {
                    double distances[distCompBatchSize];
                    dc->batchComputeDistances(vectorIds, distances, distCompBatchSize);
                    stats.totalDistCompDuringSearch += distCompBatchSize;
                    for (int k = 0; k < distCompBatchSize; k++) {
                        if (results.size() < efSearch || distances[k] < results.top().dist) {
                            candidates.emplace(vectorIds[k], distances[k]);
                            if (filterMask[vectorIds[k]]) {
                                results.emplace(vectorIds[k], distances[k]);
                                if (results.size() > efSearch) {
                                    results.pop();
                                }
                            }
                        }
                    }
                    counter = 0;
                }
            }

            // For the left out nodes
            for (int k = 0; k < counter; k++) {
                double dist;
                dc->computeDistance(vectorIds[k], &dist);
                stats.totalDistCompDuringSearch++;
                if (results.size() < efSearch || dist < results.top().dist) {
                    candidates.emplace(vectorIds[k], dist);
                    if (filterMask[vectorIds[k]]) {
                        results.emplace(vectorIds[k], dist);
                        if (results.size() > efSearch) {
                            results.pop();
                        }
                    }
                }
            }
        }
        // Reset the visited table
        visited.reset();
    }

    void HNSW::findNextFilteredKNeighbours(
            DistanceComputer *dc,
            vector_idx_t entrypoint,
            std::vector<vector_idx_t> &nbrs,
            const uint8_t *filterMask,
            orangedb::VisitedTable &visited,
            int maxK,
            int maxNeighboursCheck,
            Stats &stats) {
        auto neighbors = storage->get_neighbors(0);
        std::queue<vector_idx_t> candidates;
        candidates.push(entrypoint);
        auto neighboursChecked = 0;
        std::unordered_set<vector_idx_t> visitedSet;
        while (neighboursChecked <= maxNeighboursCheck && !candidates.empty()) {
            auto candidate = candidates.front();
            candidates.pop();
            size_t begin, end;
            if (visitedSet.contains(candidate)) {
                continue;
            }
            visitedSet.insert(candidate);
            visited.set(candidate);
            storage->get_neighbors_offsets(candidate, 0, begin, end);
            neighboursChecked += 1;
            stats.totalGetNbrsCall++;
            // TODO: Maybe make it prioritized, might help in correlated cases
            for (size_t i = begin; i < end; i++) {
                auto neighbor = neighbors[i];
                if (neighbor == INVALID_VECTOR_ID) {
                    break;
                }
                if (visited.get(neighbor)) {
                    continue;
                }
                // Add to visited
                if (filterMask[neighbor]) {
                    nbrs.push_back(neighbor);
                    visited.set(neighbor);
                    if (nbrs.size() >= maxK) {
                        return;
                    }
                }
                candidates.push(neighbor);
            }
        }
    }

    void HNSW::searchNeighborsOnLastLevelWithFilterB(
            DistanceComputer *dc,
            std::priority_queue<NodeDistCloser> &results,
            vector_idx_t entrypoint,
            double entrypointDist,
            VisitedTable &visited,
            uint64_t efSearch,
            int distCompBatchSize,
            const uint8_t *filterMask,
            orangedb::Stats &stats) {
        std::priority_queue<NodeDistFarther> candidates;
        candidates.emplace(entrypoint, entrypointDist);
        results.emplace(entrypoint, entrypointDist);
        visited.set(entrypoint);
        while (!candidates.empty()) {
            auto candidate = candidates.top();
            if (candidate.dist > results.top().dist && results.size() >= efSearch) {
                break;
            }
            candidates.pop();
            std::vector<vector_idx_t> nbrs;
            findNextFilteredKNeighbours(dc, candidate.id, nbrs, filterMask, visited, config.filterMinK, config.maxNeighboursCheck, stats);
            if (nbrs.empty()) {
                 // TODO: Maybe change the entrypoint in case no results
                 //     calculate the unvisited filtered neighbors, if above some threshold, then change the entrypoint
//                 spdlog::warn("Nbrs turn out empty!!!");
            }
            for (auto neighbor: nbrs) {
                double dist;
                dc->computeDistance(neighbor, &dist);
                stats.totalDistCompDuringSearch++;
                if (results.size() < efSearch || dist < results.top().dist) {
                    candidates.emplace(neighbor, dist);
                    results.emplace(neighbor, dist);
                    if (results.size() > efSearch) {
                        results.pop();
                    }
                }
            }
        }
        visited.reset();
    }

    void HNSW::searchWithFilter(
            const float *query,
            uint64_t k,
            uint64_t efSearch,
            orangedb::VisitedTable &visited,
            std::priority_queue<NodeDistCloser> &results,
            const uint8_t *filterMask,
            orangedb::Stats &stats) {
        L2DistanceComputer dc = L2DistanceComputer(storage->data, storage->dim, storage->numPoints);
        dc.setQuery(query);
        int newEfSearch = std::max(k, efSearch);
        vector_idx_t nearestID = storage->entryPoint;
        double nearestDist;
        int level = storage->maxLevel;
        dc.computeDistance(getActualId(level, nearestID), &nearestDist);
        // Update the nearest node
        for (; level > 0; level--) {
            searchNearestOnLevel(&dc, level, nearestID, nearestDist, stats);
            nearestID = storage->next_level_ids[level][nearestID];
        }
        searchNeighborsOnLastLevelWithFilterA(
                &dc,
                results,
                nearestID,
                nearestDist,
                visited,
                newEfSearch,
                4,
                filterMask,
                stats);
    }

    void HNSW::deleteNode(
            DistanceComputer* dc,
            orangedb::vector_idx_t deletedId,
            std::vector<omp_lock_t> &locks,
            const float *infVector,
            int dim,
            VisitedTable &visited,
            Stats &stats) {
        // Update the vector in the storage.data at id position
        auto neigbours = storage->get_neighbors(0);
        size_t begin, end;
        storage->get_neighbors_offsets(deletedId, 0, begin, end);
        for (size_t i = begin; i < end; i++) {
            auto nbr = neigbours[i];
            if (nbr == INVALID_VECTOR_ID) {
                break;
            }
            if (visited.get(nbr)) {
                continue;
            }
            // Shrink the neighbors list of the neighbor
            std::unordered_set<vector_idx_t> unionNodes;
            // Add the neighbors of the neighbor to the shrinkNodes
            size_t beginNbr, endNbr;
            storage->get_neighbors_offsets(nbr, 0, beginNbr, endNbr);
            for (size_t j = beginNbr; j < endNbr; j++) {
                auto nbrNbr = neigbours[j];
                if (nbrNbr == INVALID_VECTOR_ID) {
                    break;
                }
                if (nbrNbr == deletedId) {
                    continue;
                }
                unionNodes.insert(nbrNbr);
            }

            // Add the neighbours of deleted node
            for (size_t j = begin; j < end; j++) {
                auto nbrNbr = neigbours[j];
                if (nbrNbr == INVALID_VECTOR_ID) {
                    break;
                }
                if (nbrNbr == nbr) {
                    continue;
                }
                unionNodes.insert(nbrNbr);
            }

            if (unionNodes.size() > storage->max_neighbors_per_level[0]) {
                std::priority_queue<NodeDistCloser> shrinkNodes;
                for (auto &node: unionNodes) {
                    double dist;
                    dc->computeDistance(getActualId(0, nbr), getActualId(0, node), &dist);
                    stats.totalDistCompDuringShrink++;
                    shrinkNodes.emplace(node, dist);
                }
                shrinkNeighbors(dc, nbr, shrinkNodes, storage->max_neighbors_per_level[0], 0, dim, stats);

                // Push result into union nodes
                unionNodes.clear();
                while (!shrinkNodes.empty()) {
                    unionNodes.insert(shrinkNodes.top().id);
                    shrinkNodes.pop();
                }
            }

            omp_set_lock(&locks[nbr]);
            size_t j = beginNbr;
            for (auto &node: unionNodes) {
                neigbours[j++] = node;
            }
            while (j < endNbr) {
                neigbours[j++] = INVALID_VECTOR_ID;
            }
            omp_unset_lock(&locks[nbr]);
            stats.totalNodesShrinkDuringDelete++;
        }

        omp_set_lock(&locks[deletedId]);
        auto replacementNode = neigbours[begin];
        // Reset neighbour of deleted node
        for (size_t i = begin; i < end; i++) {
            neigbours[i] = INVALID_VECTOR_ID;
        }
        memcpy((void *) (storage->data + (deletedId * storage->dim)), infVector, storage->dim * sizeof(float));

        // update next_level_ids for level 1
        for (int i = 0; i < storage->fast_level_counters[1]; i++) {
            if (storage->next_level_ids[1][i] == deletedId) {
                storage->next_level_ids[1][i] = replacementNode;
            }
        }

        // update actual_ids for all level
        for (int level = 0; level < storage->maxLevel; level++) {
            for (auto &actualId: storage->actual_ids[level]) {
                if (actualId == deletedId) {
                    actualId = replacementNode;
                }
            }
        }
        omp_unset_lock(&locks[deletedId]);
    }

    void HNSW::deleteNodeV2(
            DistanceComputer *dc,
            vector_idx_t deletedId,
            std::vector<omp_lock_t> &locks,
            const float *infVector,
            VisitedTable &visited,
            Stats &stats) {
        // Simply set the value to inf without changing graph
        memcpy((void *) (storage->data + (deletedId * storage->dim)), infVector, storage->dim * sizeof(float));
    }

    // IDEA2: Maintain array of list of nodes for each changed neighbour
    //        Delete vector
    //        Run shrink in the end & update
    void HNSW::deleteNodeV3(
            DistanceComputer *dc,
            vector_idx_t deletedId,
            std::vector<omp_lock_t> &locks,
            const float *infVector,
            orangedb::Stats &stats) {

    }

    void HNSW::logStats() {
        stats.logStats();
        // Print avg number of nodes in last layer
        auto nbrs = storage->get_neighbors(0);
        int count = 0;
        for (int i = 0; i < storage->fast_level_counters[0]; i++) {
            for (int j = 0; j < storage->max_neighbors_per_level[0]; j++) {
                if (nbrs[i * storage->max_neighbors_per_level[0] + j] != INVALID_VECTOR_ID) {
                    count++;
                }
            }
        }
        spdlog::warn("Average number of nodes in last layer: {}", count / storage->fast_level_counters[0]);

        // print how neighbors changed after shrink for node with highest shrink calls
//        int maxShrinkCalls = 0;
//        vector_idx_t maxShrinkCallsId = 0;
//        for (auto &node: stats.shrinkCallsPerNode) {
//            if (node.second > maxShrinkCalls) {
//                maxShrinkCalls = node.second;
//                maxShrinkCallsId = node.first;
//            }
//        }

//        spdlog::warn("============================================================");
//        if (maxShrinkCalls > 1) {
//            spdlog::warn("Node with highest shrink calls: {}", maxShrinkCallsId);
//            std::string distances;
//            for (auto &dist: storage->afterShrinkDistances[maxShrinkCallsId]) {
//                if (dist == -1) {
//                    printf("%s\n", distances.c_str());
//                    distances = "";
//                }
//                distances += std::to_string(dist) + ",";
//            }
//        }
    }
} // namespace orangedb
