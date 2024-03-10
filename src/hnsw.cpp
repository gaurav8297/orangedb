#include <cassert>
#include "include/hnsw.h"
#include <omp.h>

namespace orangedb {
    HNSW::HNSW(uint16_t M, uint16_t ef_construction, uint16_t ef_search, uint16_t dim) :
            mt(1026), M(M), ef_construction(ef_construction), ef_search(ef_search) {
        // Initialize probabilities to save computation time later.
        init_probabs(M, 1.0 / log(M));
        storage = new Storage(dim, M, level_probabs.size());
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
            max_neighbors_per_level.push_back(level == 0 ? M * 2 : M);
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
            VisitedTable &visited) {
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
            for (size_t i = begin; i < end; i++) {
                storage_idx_t neighbor = neighbors[i];
                if (neighbor < 0) {
                    break;
                }
                if (visited.get(neighbor)) {
                    continue;
                }
                visited.set(neighbor);
                float dist;
                dc->compute_distance(getActualId(level, neighbor), dist);
                if (results.size() < ef_construction || dist < results.top().dist) {
                    candidates.emplace(neighbor, dist);
                    results.emplace(neighbor, dist);
                    if (results.size() > ef_construction) {
                        results.pop();
                    }
                }
            }
        }
        visited.reset();
    }

    void HNSW::shrink_neighbors(
            DistanceComputer *dc, std::priority_queue<NodeDistCloser> &resultSet, int max_size, uint8_t level) {
        if (resultSet.size() <= max_size) {
            return;
        }
        std::priority_queue<NodeDistFarther> temp;
        std::vector<NodeDistFarther> result;
        while (!resultSet.empty()) {
            temp.emplace(resultSet.top().id, resultSet.top().dist);
            resultSet.pop();
        }

        // Push nodes to result based on a heuristic that distance to the query node is smallest against all other nodes
        // in the neighbors list. We add it in decreasing order of distance such that farthest nodes are always added.
        while (!temp.empty()) {
            auto node = temp.top();
            temp.pop();
            auto dist_q_node = node.dist;
            bool good = true;
            for (NodeDistFarther &node_2: result) {
                float dist_node_node_2;
                dc->compute_distance(getActualId(level, node.id), getActualId(level, node_2.id), dist_node_node_2);
                if (dist_node_node_2 < dist_q_node) {
                    good = false;
                    break;
                }
            }

            if (good) {
                result.push_back(node);
                if (result.size() >= max_size) {
                    break;
                }
            }
        }

        for (auto &node: result) {
            resultSet.emplace(node.id, node.dist);
        }
    }

    // It is assumed that the node is already locked.
    void HNSW::make_connection(
            DistanceComputer *dc, storage_idx_t src, storage_idx_t dest, level_t level) {
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
        float dist_src_dest;
        dc->compute_distance(getActualId(level, src), getActualId(level, dest), dist_src_dest);
        resultSet.emplace(src, dist_src_dest);
        for (size_t i = begin; i < end; i++) {
            storage_idx_t neighbor = neighbors[i];
            float dist_src_nbr;
            dc->compute_distance(getActualId(level, src), getActualId(level, neighbor), dist_src_nbr);
            resultSet.emplace(neighbor, dist_src_nbr);
        }
        shrink_neighbors(dc, resultSet, storage->max_neighbors_per_level[level], level);
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
            std::vector<omp_lock_t> &locks,
            VisitedTable &visited,
            std::vector<storage_idx_t> &neighbors) {
        // This is a blocking call.
        std::priority_queue<NodeDistCloser> link_targets;
        search_neighbors(dc, level, link_targets, entrypoint, entrypoint_dist, visited);
        auto max_no_of_edges = storage->max_neighbors_per_level[level];
        shrink_neighbors(dc, link_targets, max_no_of_edges, level);

        neighbors.reserve(link_targets.size());

        while (!link_targets.empty()) {
            auto neighbor_id = link_targets.top().id;
            link_targets.pop();
            if (neighbor_id == id) {
                continue;
            }
            make_connection(dc, id, neighbor_id, level);
            neighbors.push_back(neighbor_id);
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
        std::vector<std::vector<storage_idx_t>> neighbors(node_level + 1);
        dc->compute_distance(nearest_id, nearest_dist);
        int level = max_level;

        // Update the nearest node
        for (; level > node_level; level--) {
            search_nearest_on_level(dc, level, nearest_id, nearest_dist);
            nearest_id = storage->next_level_ids[level][nearest_id];
        }

        // Add the node to the graph
        for (; level >= 0; level--) {
            auto node_level_id = node_id[level];
            add_node_on_level(dc, node_level_id, level, nearest_id, nearest_dist, locks, visited, neighbors[level]);
            if (level != 0) {
                nearest_id = storage->next_level_ids[level][nearest_id];
            }
        }
        omp_unset_lock(&locks[node_id[0]]);

        for (int i = 0; i < neighbors.size(); i++) {
            for (auto neighbor_id: neighbors[i]) {
                omp_set_lock(&locks[getActualId(i, neighbor_id)]);
                make_connection(dc, neighbor_id, node_id[i], i);
                omp_unset_lock(&locks[getActualId(i, neighbor_id)]);
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

    void HNSW::build(const float *data, size_t n) {
        std::vector<omp_lock_t> locks(n);
        for (int i = 0; i < n; i++) {
            omp_init_lock(&locks[i]);
        }
        levels.resize(n);
        // Initialize ids
        std::vector<std::vector<storage_idx_t>> node_ids(n);
        uint8_t max_level = 0;
        for (int i = 0; i < n; i++) {
            uint8_t level = random_level();
            auto id = storage->getFastNextNodeId(level);
            levels[id[0]] = level;
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

        printf("Building the graph\n");

#pragma omp parallel
        {
            VisitedTable visited(n);
            L2DistanceComputer dc(storage);

#pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                dc.set_query(storage->data + (i * storage->dim));
                add_node(&dc, node_ids[i], levels[i], locks, visited);
            }
        }

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
} // namespace orangedb


