#include "include/hnsw.h"
#include <omp.h>
#include <algorithm>
#include "include/prefetch.h"
#include "spdlog/spdlog.h"
#include <memory>
#include <unistd.h>

namespace orangedb {
    HNSW::HNSW(HNSWConfig config, RandomGenerator *rg, uint16_t dim) : config(config), entryPoint(INVALID_VECTOR_ID),
                                                                       maxLevel(0),
                                                                       rg(rg), stats(Stats()) {
        // Initialize probabilities to save computation time later.
        initProbabs(config.M, 1.0 / log(config.M));
        storage = new Storage(dim, config.M, levelProbabs.size());
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
            uint16_t efSearch,
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
            uint16_t efSearch,
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
            if (entryPoint == INVALID_VECTOR_ID) {
                maxLevel = node_level;
                // Fix this maybe
                entryPoint = node_id[node_level];
            }
            nearestId = entryPoint;
            level = maxLevel;
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
            if (node_level > maxLevel) {
                maxLevel = node_level;
                entryPoint = node_id[node_level];
            }
        }
    }

    // TODO: Add caching, thread pinning, live locks, queue with locks etc
    void HNSW::build(const float *data, size_t n) {
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
            DistanceComputer *localDc = new L2DistanceComputer(data, storage->dim, n);
            VisitedTable visited(n);
            Stats localStats = Stats();
#pragma omp for schedule(dynamic, 1000)
            for (int i = 0; i < n; i++) {
                localDc->setQuery(storage->data + (i * storage->dim));
                addNode(localDc, node_ids[i], node_ids[i].size() - 1, locks, visited, localStats);
                if (i % 100000 == 0) {
                    spdlog::warn("Inserted 100000!!");
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
            uint16_t k,
            uint16_t efSearch,
            VisitedTable &visited,
            std::priority_queue<NodeDistCloser> &results,
            Stats &stats) {
        L2DistanceComputer dc = L2DistanceComputer(storage->data, storage->dim, storage->numPoints);
        dc.setQuery(query);
        int newEfSearch = std::max(k, efSearch);
        vector_idx_t nearestID = entryPoint;
        double nearestDist;
        int level = maxLevel;
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
            uint16_t efSearch,
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
        while (neighboursChecked <= maxNeighboursCheck && !candidates.empty()) {
            auto candidate = candidates.front();
            candidates.pop();
            size_t begin, end;
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
            uint16_t efSearch,
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
            uint16_t k,
            uint16_t efSearch,
            orangedb::VisitedTable &visited,
            std::priority_queue<NodeDistCloser> &results,
            const uint8_t *filterMask,
            orangedb::Stats &stats) {
        L2DistanceComputer dc = L2DistanceComputer(storage->data, storage->dim, storage->numPoints);
        dc.setQuery(query);
        int newEfSearch = std::max(k, efSearch);
        vector_idx_t nearestID = entryPoint;
        double nearestDist;
        int level = maxLevel;
        dc.computeDistance(getActualId(level, nearestID), &nearestDist);
        // Update the nearest node
        for (; level > 0; level--) {
            searchNearestOnLevel(&dc, level, nearestID, nearestDist, stats);
            nearestID = storage->next_level_ids[level][nearestID];
        }
        searchNeighborsOnLastLevelWithFilterB(
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
        for (int level = 0; level < maxLevel; level++) {
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
