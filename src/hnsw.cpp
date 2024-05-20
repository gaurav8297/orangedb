#include "include/hnsw.h"
#include <omp.h>
#include <algorithm>
#include "include/prefetch.h"
#include "spdlog/spdlog.h"

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
            for (size_t j = begin; j < end; j++) {
                int v1 = neighbors[j];
                if (v1 == INVALID_VECTOR_ID)
                    break;
                prefetch_L3(visited.data() + v1);
                jmax += 1;
            }

            // Perform distance computation on 4 vectors at a time
            int counter = 0;
            vector_idx_t vectorIds[distCompBatchSize];
            for (size_t j = begin; j < jmax; j++) {
                int v1 = neighbors[j];
                if (v1 == INVALID_VECTOR_ID)
                    break;
                bool vget = visited.get(v1);
                // TODO: Try to set visited in the end of loop
                visited.set(v1);
                vectorIds[counter] = v1;
                counter += vget ? 0 : 1;
                if (counter == distCompBatchSize) {
                    double distances[distCompBatchSize];
                    dc->batchComputeDistances(vectorIds, distances, distCompBatchSize);
                    stats.totalDistCompDuringSearch += 4;
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
        visited.reset();
    }

    void HNSW::shrinkNeighbors(
            DistanceComputer *dc,
            std::priority_queue<NodeDistCloser> &results,
            int maxSize,
            level_t level,
            Stats &stats) {
        if (results.size() <= maxSize) {
            return;
        }
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
                dc->computeDistance(getActualId(level, nodeA.id), getActualId(level, nodeB.id), &distNodeAB);
                stats.totalDistCompDuringShrink++;
                if ((config.alpha * distNodeAB) < distNodeAQ) {
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
        return;
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
        shrinkNeighbors(dc, results, storage->max_neighbors_per_level[level], level, stats);
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
        shrinkNeighbors(dc, linkTargets, storage->max_neighbors_per_level[level], level, stats);

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
#pragma omp critical
        {
            if (entryPoint == INVALID_VECTOR_ID) {
                maxLevel = node_level;
                // Fix this maybe
                entryPoint = node_id[node_level];
            }
        }

        omp_set_lock(&locks[node_id[0]]);
        vector_idx_t nearestId = entryPoint;
        double nearestDist;
        std::vector<std::vector<NodeDistCloser>> neighbors(node_level + 1);
        dc->computeDistance(nearestId, &nearestDist);
        int level = maxLevel;

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
#pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                localDc->setQuery(storage->data + (i * storage->dim));
                addNode(localDc, node_ids[i], node_ids[i].size() - 1, locks, visited, localStats);
                if (i % 100000 == 0) {
                    spdlog::warn("Done with 100000!!");
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
        dc.computeDistance(nearestID, &nearestDist);
        int level = maxLevel;
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
                4 /* distCompBatchSize */,
                stats);
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
    }
} // namespace orangedb
