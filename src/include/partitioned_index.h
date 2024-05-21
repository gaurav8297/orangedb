#pragma once

#include <unistd.h>
#include <vector>
#include <clustering.h>
#include <hnsw.h>

namespace orangedb {
    struct PartitionedIndexConfig {
        // The number of neighbors to keep for each node
        uint16_t M = 16;
        // The number of neighbors to explore during index construction
        uint16_t efConstruction = 200;
        // The number of neighbors to explore during search
        uint16_t efSearch = 50;
        // RNG alpha parameter
        float alpha = 1.0;
        // The number of centroids
        int numCentroids = 10;
        // The number of iterations
        int nIter = 25;
        // The minimum size of a centroid
        int minCentroidSize;
        // The maximum size of a centroid
        int maxCentroidSize;
        // The maximum number of centroids at search time
        int maxSearchCentroids = 3;
        // The distance threshold for searching in the centroids
        float searchThreshold = 0.4;

        PartitionedIndexConfig(uint16_t M, uint16_t efConstruction, uint16_t efSearch, float alpha, int numCentroids,
                               int nIter, int minCentroidSize, int maxCentroidSize, int maxSearchCentroids,
                               float searchThreshold)
                : M(M), efConstruction(efConstruction), efSearch(efSearch), alpha(alpha), numCentroids(numCentroids),
                  nIter(nIter), minCentroidSize(minCentroidSize), maxCentroidSize(maxCentroidSize),
                  maxSearchCentroids(maxSearchCentroids), searchThreshold(searchThreshold) {};
    };

    class PartitionedIndex {
    public:
        explicit PartitionedIndex(int dim, const PartitionedIndexConfig &config, RandomGenerator* rg);

        void build(const float *data, size_t n);

        void search(const float *query, uint16_t k, VisitedTable &visited, std::vector<NodeDistFarther> &results, Stats &stats);

    private:
        std::unique_ptr<Clustering> clustering;
        std::vector<std::unique_ptr<HNSW>> indexes;
        std::vector<std::vector<vector_idx_t>> actualIds;
        PartitionedIndexConfig config;
        IndexOneNN* centroidIndex;
        int numVectors;

        int dim;
    };
} // namespace orangedb
