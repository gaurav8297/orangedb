#include <cassert>
#include "include/hnsw.h"
#include <omp.h>
#include <algorithm>
#include "include/prefetch.h"
#include "spdlog/spdlog.h"

// Constants
#define MORSEL_SIZE 1000
#define BEAM_SIZE 8

namespace orangedb {
    HNSW::HNSW(
            uint16_t M,
            uint16_t ef_construction,
            uint16_t dim,
            float explore_factor,
            float alpha,
            int beam_size,
            int beam_thrsh):
            mt(1026), ef_construction(ef_construction), explore_factor(explore_factor), alpha(alpha),
            beam_size(beam_size), beam_thrsh(beam_thrsh), stats(Stats()) {
        // Initialize probabilities to save computation time later.
        assert(explore_factor <= 1 && explore_factor >= 0);
        init_probabs(M, 1.0 / log(M));
        storage = new Storage(dim, M, level_probabs.size());
        sq = new ScalarQuantizer(dim, storage->vmin, storage->vdiff);
    }

    void HNSW::init_probabs(uint16_t M, double levelMult) {
        for (int level = 0;; level++) {
            // levelMult helps to control how many level there are going to be. Higher levelMult means more levels.
            // This is just reverse of [-ln(uniform(0, 1)) * levelMult] which is a random variable with
            // exponential distribution.
            double prob = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
            if (prob < 1e-9) {
                break;
            }
            level_probabs.push_back(prob);
        }
    }

    void HNSW::search_nearest_on_level(
            DistanceComputer *dc, level_t level, storage_idx_t &nearest, float &nearestDist) {
        assert(level > 0);
        auto neighbors = storage->get_neighbors(level);
        // Can we do distance computation on 4 vectors at a time?
        while (true) {
            storage_idx_t prev_nearest = nearest;
            size_t begin, end;
            storage->get_neighbors_offsets(nearest, level, begin, end);
            for (size_t i = begin; i < end; i++) {
                storage_idx_t neighbor = neighbors[i];
                if (neighbor < 0) {
                    break;
                }
                float dist;
                dc->compute_distance(getActualId(level, neighbor), dist);
//                stats.totalDistComp++;
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

    void HNSW::search_neighbors(
            DistanceComputer *dc,
            level_t level,
            std::priority_queue<NodeDistCloser> &results,
            storage_idx_t entrypoint,
            float entrypointDist,
            VisitedTable &visited,
            uint16_t ef) {
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
                if (neighbor < 0)
                    break;
                prefetch_L3(visited.data() + neighbor);
//                prefetch_L3(storage->data + (getActualId(level, neighbor) * storage->dim));
                jmax += 1;
            }

            bool search_improved = false;
            int j = 0;
            size_t max_j = (end - begin) * 1;
            for (size_t i = begin; i < jmax; i++) {
                storage_idx_t neighbor = neighbors[i];
                if (neighbor < 0 || (j > max_j && search_improved)) {
                    break;
                }
                j++;
                if (visited.get(neighbor)) {
                    continue;
                }
                visited.set(neighbor);
                float dist;
                dc->compute_distance(getActualId(level, neighbor), dist);
//                stats.totalDistComp++;
                if (results.size() < ef || dist < results.top().dist) {
                    candidates.emplace(neighbor, dist);
                    results.emplace(neighbor, dist);
                    search_improved = true;
                    if (results.size() > ef) {
                        results.pop();
                    }
                }
            }
        }
        visited.reset();
    }

    void HNSW::search_neighbors_optimized(
            DistanceComputer *dc,
            level_t level,
            std::priority_queue<NodeDistCloser> &results,
            storage_idx_t entrypoint,
            float entrypointDist,
            VisitedTable &visited,
            uint16_t ef) {
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

            // the following version processes 4 neighbors at a time
            size_t jmax = begin;
            for (size_t j = begin; j < end; j++) {
                int v1 = neighbors[j];
                if (v1 < 0)
                    break;

                prefetch_L3(visited.data() + v1);
                jmax += 1;
            }

            int counter = 0;
            size_t cached_ids[4];
            bool search_improved = false;
            int u = 0;
            size_t max_u = (jmax - begin) * explore_factor;

            for (size_t j = begin; j < jmax; j++) {
                u++;
                if (u > max_u && search_improved) {
                    break;
                }
                int v1 = neighbors[j];
                if (v1 < 0)
                    break;
                bool vget = visited.get(v1);
                // TODO: Try to set visited in the end of loop
                visited.set(v1);
                cached_ids[counter] = v1;
                counter += vget ? 0 : 1;
                if (counter == 4) {
                    float dis[4];
                    dc->compute_distance_four_vecs(
                            getActualId(level, cached_ids[0]),
                            getActualId(level, cached_ids[1]),
                            getActualId(level, cached_ids[2]),
                            getActualId(level, cached_ids[3]),
                            dis[0],
                            dis[1],
                            dis[2],
                            dis[3]);
                    stats.totalDistComp += 4;
                    for (int k = 0; k < 4; k++) {
                        if (results.size() < ef || dis[k] < results.top().dist) {
                            candidates.emplace(cached_ids[k], dis[k]);
                            results.emplace(cached_ids[k], dis[k]);
                            search_improved = true;
                            if (results.size() > ef) {
                                results.pop();
                            }
                        }
                    }
                    counter = 0;
                }
            }

            // For the left out nodes
            for (int k = 0; k < counter; k++) {
                float dist;
                dc->compute_distance(getActualId(level, cached_ids[k]), dist);
                stats.totalDistComp++;
                if (results.size() < ef || dist < results.top().dist) {
                    candidates.emplace(cached_ids[k], dist);
                    results.emplace(cached_ids[k], dist);
                    if (results.size() > ef) {
                        results.pop();
                    }
                }
            }
        }
        visited.reset();
    }

    void HNSW::search_neighbors_more_optimized(
            DistanceComputer *dc,
            level_t level,
            std::priority_queue<NodeDistCloser> &results,
            storage_idx_t entrypoint,
            float entrypointDist,
            VisitedTable &visited,
            uint16_t ef) {
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
            std::vector<NodeDistFarther> cds;
            cds.push_back(candidate);
            if (candidates.size() > beam_thrsh) {
                for (int i = 0; i < beam_size; i++) {
                    auto c = candidates.top();
                    candidates.pop();
                    cds.push_back(c);
                }
            }
            std::vector<storage_idx_t> nbrs_not_visited;
            for (auto c: cds) {
                size_t begin, end;
                storage->get_neighbors_offsets(c.id, level, begin, end);
                for (size_t j = begin; j < end; j++) {
                    int v1 = neighbors[j];
                    if (v1 < 0)
                        break;
                    if (!visited.get(v1)) {
                        visited.set(v1);
                        nbrs_not_visited.push_back(v1);
                    }
                }
            }

//            stats.totalLoops++;

            // caclulate distances in batch of 4
            std::vector<float> dists(nbrs_not_visited.size());
//            int jMax = nbrs_not_visited.size() >> 2;
//            jMax = jMax << 2;
//            for (int j = 0; j < jMax; j += 4) {
//                dc->compute_distance_four_vecs(
//                        getActualId(level, nbrs_not_visited[j]),
//                        getActualId(level, nbrs_not_visited[j + 1]),
//                        getActualId(level, nbrs_not_visited[j + 2]),
//                        getActualId(level, nbrs_not_visited[j + 3]),
//                        dists[j],
//                        dists[j + 1],
//                        dists[j + 2],
//                        dists[j + 3]);
////                stats.totalDistComp += 4;
//                stats.total4mul++;
//            }

            // calculate the remaining distances
            for (int j = 0; j < nbrs_not_visited.size(); j++) {
                dc->compute_distance(getActualId(level, nbrs_not_visited[j]), dists[j]);
//                stats.totalDistComp++;
            }

            // add the neighbors to the results
            for (int j = 0; j < nbrs_not_visited.size(); j++) {
                if (results.size() < ef || dists[j] < results.top().dist) {
                    candidates.emplace(nbrs_not_visited[j], dists[j]);
                    results.emplace(nbrs_not_visited[j], dists[j]);
                    if (results.size() > ef) {
                        results.pop();
                    }
                }
            }
        }
        visited.reset();
    }

    int HNSW::shrink_neighbors(
            DistanceComputer *dc, std::priority_queue<NodeDistCloser> &resultSet, int max_size, uint8_t level) {
        if (resultSet.size() <= max_size) {
            return 0;
        }
        int total_dist_comp = 0;
        std::priority_queue<NodeDistFarther> temp;
        std::vector<NodeDistFarther> result;
        while (!resultSet.empty()) {
            temp.emplace(resultSet.top().id, resultSet.top().dist);
            resultSet.pop();
        }
//        std::vector<int> distComp;

        // Push nodes to result based on a heuristic that distance to the query node is smallest against all other nodes
        // in the neighbors list. We add it in decreasing order of distance such that farthest nodes are always added.
        while (!temp.empty()) {
            auto node = temp.top();
            temp.pop();
            auto dist_q_node = node.dist;
            bool good = true;
//            auto dist_comp = 0;
            for (NodeDistFarther &node_2: result) {
                float dist_node_node_2;
                dc->compute_distance(getActualId(level, node.id), getActualId(level, node_2.id), dist_node_node_2);
//                total_dist_comp++;
//                stats.totalDistComp++;
//                stats.totalDistCompInShrink++;
//                dist_comp++;
                if (dist_node_node_2 < dist_q_node) {
                    good = false;
                    break;
                }
            }
//            distComp.push_back(dist_comp);
            if (good) {
                result.push_back(node);
                if (result.size() >= max_size) {
                    stats.totalShrinkNotReduce++;
                    break;
                }
            }
        }

//        if (total_dist_comp > 400) {
//#pragma omp critical
//            {
//                int i = 0;
//                for (auto dist: distComp) {
//                    spdlog::warn("[{}] Distance computations: {}", i, dist);
//                    i++;
//                }
//            }
//        }

        for (auto &node: result) {
            resultSet.emplace(node.id, node.dist);
        }
        return total_dist_comp;
    }

    int HNSW::shrink_neighbors_2(
            DistanceComputer *dc, std::priority_queue<NodeDistCloser> &resultSet, int max_size, uint8_t level) {
        int total_dist_comp = 0;
        std::priority_queue<NodeDistFarther> temp;
        std::vector<NodeDistFarther> result;
        while (!resultSet.empty()) {
            temp.emplace(resultSet.top().id, resultSet.top().dist);
            resultSet.pop();
        }
        max_size = max_size / 2;

//        std::vector<int> distComp;

        // Push nodes to result based on a heuristic that distance to the query node is smallest against all other nodes
        // in the neighbors list. We add it in decreasing order of distance such that farthest nodes are always added.
        while (!temp.empty()) {
            auto node = temp.top();
            temp.pop();
            auto dist_q_node = node.dist;
            bool good = true;
//            auto dist_comp = 0;
            for (NodeDistFarther &node_2: result) {
                float dist_node_node_2;
                dc->compute_distance(getActualId(level, node.id), getActualId(level, node_2.id), dist_node_node_2);
//                total_dist_comp++;
//                stats.totalDistComp++;
//                stats.totalDistCompInShrink++;
//                dist_comp++;
                if (alpha * dist_node_node_2 < dist_q_node) {
                    good = false;
                    break;
                }
            }
//            distComp.push_back(dist_comp);
            if (good) {
                result.push_back(node);
                if (result.size() >= max_size) {
                    stats.totalShrinkNotReduce++;
                    break;
                }
            }
        }

//        if (total_dist_comp > 400) {
//#pragma omp critical
//            {
//                int i = 0;
//                for (auto dist: distComp) {
//                    spdlog::warn("[{}] Distance computations: {}", i, dist);
//                    i++;
//                }
//            }
//        }

        for (auto &node: result) {
            resultSet.emplace(node.id, node.dist);
        }
        return total_dist_comp;
    }

    // It is assumed that the node is already locked.
    void HNSW::make_connection(
            DistanceComputer *dc, storage_idx_t src, storage_idx_t dest, float dist_src_dest, level_t level) {
        auto neighbors = storage->get_neighbors(level);
        size_t begin, end;
        storage->get_neighbors_offsets(src, level, begin, end);
        if (neighbors[end - 1] == -1) {
            if (neighbors[begin] == -1) {
                neighbors[begin] = dest;
                return;
            }
            // do loop in reverse order, it could yield faster results
            for (size_t i = end; i > begin; i--) {
                if (neighbors[i - 1] != -1) {
                    neighbors[i] = dest;
                    return;
                }
            }
        }

        // Otherwise we need to shrink the neighbors list
        std::priority_queue<NodeDistCloser> resultSet;
        resultSet.emplace(dest, dist_src_dest);
        for (size_t i = begin; i < end; i++) {
            storage_idx_t neighbor = neighbors[i];
            float dist_src_nbr;
            dc->compute_distance(getActualId(level, src), getActualId(level, neighbor), dist_src_nbr);
//            stats.totalDistComp++;
//            stats.totalDistCompInShrink++;

            resultSet.emplace(neighbor, dist_src_nbr);
        }
        shrink_neighbors(dc, resultSet, storage->max_neighbors_per_level[level], level);
//        stats.totalShrinkCalls1++;
//        spdlog::warn("[make_connection] Total distance computations in shrink: {}", totalDist);
        size_t i = begin;
        while (!resultSet.empty()) {
            auto top_id = resultSet.top().id;
            neighbors[i++] = top_id;
            resultSet.pop();
        }
        while (i < end) {
            neighbors[i++] = -1;
        }
    }

    void HNSW::shrink(orangedb::DistanceComputer *dc, orangedb::storage_idx_t src, orangedb::level_t level) {
        auto neighbors = storage->get_neighbors(level);
        size_t begin, end;
        storage->get_neighbors_offsets(src, level, begin, end);
        std::priority_queue<NodeDistCloser> resultSet;
        for (size_t i = begin; i < end; i++) {
            storage_idx_t neighbor = neighbors[i];
            if (neighbor == -1) {
                continue;
            }
            float dist_src_nbr;
            dc->compute_distance(getActualId(level, src), getActualId(level, neighbor), dist_src_nbr);
//            stats.totalDistComp++;
//            stats.totalDistCompInShrink++;

            resultSet.emplace(neighbor, dist_src_nbr);
        }

        shrink_neighbors_2(dc, resultSet, storage->max_neighbors_per_level[level], level);
//        spdlog::warn("[shrink] Shrinked neighbors for node {}: {}", src, resultSet.size());
        size_t i = begin;
        while (!resultSet.empty()) {
            auto top_id = resultSet.top().id;
            neighbors[i++] = top_id;
            resultSet.pop();
        }
        while (i < end) {
            neighbors[i++] = -1;
        }
    }

    void HNSW::add_node_on_level(
            orangedb::DistanceComputer *dc,
            orangedb::storage_idx_t id,
            orangedb::level_t level,
            orangedb::storage_idx_t entrypoint,
            float entrypoint_dist,
            VisitedTable &visited,
            std::vector<pair<storage_idx_t, float>> &neighbors) {
        // This is a blocking call.
        std::priority_queue<NodeDistCloser> link_targets;
//        stats.totalShrinkCalls2++;
        search_neighbors(dc, level, link_targets, entrypoint, entrypoint_dist, visited, ef_construction);
        shrink_neighbors(dc, link_targets, storage->max_neighbors_per_level[level], level);
//        spdlog::warn("[add_node_on_level] Total distance computations in shrink: {}", totalDist);

        neighbors.reserve(link_targets.size());

        while (!link_targets.empty()) {
            auto neighbor_node = link_targets.top();
            link_targets.pop();
            if (neighbor_node.id == id) {
                continue;
            }
            make_connection(dc, id, neighbor_node.id, neighbor_node.dist, level);
            neighbors.emplace_back(neighbor_node.id, neighbor_node.dist);
        }
    }

    void HNSW::add_node(
            orangedb::DistanceComputer *dc,
            std::vector<storage_idx_t> node_id,
            orangedb::level_t node_level,
            std::vector<omp_lock_t> &locks,
            VisitedTable &visited) {
#pragma omp critical
        {
            if (entry_point == -1) {
                max_level = node_level;
                // Fix this maybe
                entry_point = node_id[node_level];
            }
        }

        omp_set_lock(&locks[node_id[0]]);
        storage_idx_t nearest_id = entry_point;
        float nearest_dist;
        std::vector<std::vector<pair<storage_idx_t, float>>> neighbors(node_level + 1);
        dc->compute_distance(nearest_id, nearest_dist);
//        stats.totalDistComp++;
        int level = max_level;

        // Update the nearest node
        for (; level > node_level; level--) {
            search_nearest_on_level(dc, level, nearest_id, nearest_dist);
            nearest_id = storage->next_level_ids[level][nearest_id];
        }

        // Add the node to the graph
        for (; level >= 0; level--) {
            auto node_level_id = node_id[level];
            add_node_on_level(dc, node_level_id, level, nearest_id, nearest_dist, visited, neighbors[level]);
            if (level != 0) {
                nearest_id = storage->next_level_ids[level][nearest_id];
            }
        }
        omp_unset_lock(&locks[node_id[0]]);

        for (int i = 0; i < neighbors.size(); i++) {
            for (auto neighbor: neighbors[i]) {
                omp_set_lock(&locks[getActualId(i, neighbor.first)]);
                make_connection(dc, neighbor.first, node_id[i], neighbor.second, i);
                omp_unset_lock(&locks[getActualId(i, neighbor.first)]);
            }
        }

#pragma omp critical
        {
            // Update the entry point
            if (node_level > max_level) {
                max_level = node_level;
                entry_point = node_id[node_level];
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
        std::vector<std::vector<storage_idx_t>> node_ids(n);
        uint8_t max_level = 0;
        for (int i = 0; i < n; i++) {
            uint8_t level = random_level();
            auto id = storage->getFastNextNodeId(level);
            node_ids[i] = id;
            max_level = std::max(max_level, level);
        }

        // Set data to storage
        // Todo: Copy data to storage
        storage->data = data;
        // print vmin and vdiff
        for (int i = 0; i < 2; i++) {
            spdlog::warn("vmin[{}]: {}", i, storage->vmin[i]);
            spdlog::warn("vdiff[{}]: {}", i, storage->vdiff[i]);
        }
        sq->train(n, storage->data);
        storage->vmin = sq->vmin;
        storage->vdiff = sq->vdiff;
        Utils::alloc_aligned((void**)&storage->codes, storage->dim * n, 64);
        sq->compute_codes(storage->data, storage->codes, n);

//        // Print first 10 codes
//        for (int i = 0; i < 1000; i++) {
//            spdlog::warn("Code for {}: {}", i, storage->codes[i]);
//        }

        // print vmin and vdiff
        for (int i = 0; i < 2; i++) {
            spdlog::warn("vmin[{}]: {}", i, storage->vmin[i]);
            spdlog::warn("vdiff[{}]: {}", i, storage->vdiff[i]);
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
            VisitedTable visited(n);
            SQDistanceComputer dc(storage);
#pragma omp for schedule(dynamic, MORSEL_SIZE)
            for (int i = 0; i < n; i++) {
                dc.set_query(storage->data + (i * storage->dim), node_ids[i][0]);
                add_node(&dc, node_ids[i], node_ids[i].size() - 1, locks, visited);

                if (i % 100000 == 0) {
                    spdlog::warn("Done with 100000!!");
                }
            }
        }

        // Create buckets of n with bucket_size
//        for (size_t j = 0; j < n; j += bucket_size) {
//            auto start = j;
//            auto end = std::min(j + bucket_size, n);
//#pragma omp parallel
//            {
//                VisitedTable visited(n);
//                L2DistanceComputer dc(storage);
//#pragma omp for schedule(static) nowait
//                for (int i = start; i < end; i++) {
//                    dc.set_query(storage->data + (i * storage->dim), node_ids[i][0]);
//                    add_node(&dc, node_ids[i], node_ids[i].size() - 1, locks, visited);
//
//                    if (i % 100000 == 0) {
//                        spdlog::warn("Done with 100000!!");
//                    }
//                }
//#pragma omp for schedule(static) nowait
//                for (int i = 0; i < n; i++) {
//                    omp_set_lock(&locks[node_ids[i][0]]);
//                    dc.set_query(storage->data + (i * storage->dim), node_ids[i][0]);
//                    shrink(&dc, node_ids[i][0], 0);
//                    omp_unset_lock(&locks[node_ids[i][0]]);
//                }
//            }
//        }

        for (int i = 0; i < n; i++) {
            omp_destroy_lock(&locks[i]);
        }
    }

    uint8_t HNSW::random_level() {
        double f = rand_float();
        for (int level = 0; level < level_probabs.size(); level++) {
            if (f < level_probabs[level]) {
                return level;
            }
            f -= level_probabs[level];
        }
        // happens with exponentially low probability
        return level_probabs.size() - 1;
    }

    void HNSW::search_v1(
            const float* query,
            uint16_t k,
            uint16_t ef_search,
            VisitedTable& visited,
            std::priority_queue<NodeDistCloser>& result) {
        L2DistanceComputer dc(storage);
        dc.set_query(query, 0);
        int ef = std::max(k, ef_search);
        storage_idx_t nearest_id = entry_point;
        float nearest_dist;
        dc.compute_distance(nearest_id, nearest_dist);
        int level = max_level;
        // Update the nearest node
        for (; level > 0; level--) {
            search_nearest_on_level(&dc, level, nearest_id, nearest_dist);
            nearest_id = storage->next_level_ids[level][nearest_id];
        }
        search_neighbors(&dc, level, result, nearest_id, nearest_dist, visited, ef);
    }

    void HNSW::print_stats() {
        spdlog::warn("Total distance computations: {}", stats.totalDistComp);
        spdlog::warn("Total distance computations in shrink: {}", stats.totalDistCompInShrink);
        spdlog::warn("Total shrink calls 1: {}", stats.totalShrinkCalls1);
        spdlog::warn("Total shrink calls 2: {}", stats.totalShrinkCalls2);
        spdlog::warn("Total shrink not reduce: {}", stats.totalShrinkNotReduce);
        spdlog::warn("Total 4 multiplications: {}", stats.total4mul);
        spdlog::warn("Total loops: {}", stats.totalLoops);

        // Print avg number of nodes in last layer
        auto nbrs = storage->get_neighbors(0);
        int count = 0;
        for (int i = 0; i < storage->fast_level_counters[0]; i++) {
            for (int j = 0; j < storage->max_neighbors_per_level[0]; j++) {
                if (nbrs[i * storage->max_neighbors_per_level[0] + j] != -1) {
                    count++;
                }
            }
        }
        spdlog::warn("Average number of nodes in last layer: {}", count / storage->fast_level_counters[0]);
    }
} // namespace orangedb
