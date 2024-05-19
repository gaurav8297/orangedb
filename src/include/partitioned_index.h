#pragma once

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


        PartitionedIndexConfig(uint16_t M, uint16_t efConstruction, uint16_t efSearch, float alpha)
                : M(M), efConstruction(efConstruction), efSearch(efSearch), alpha(alpha) {};
    };


    class PartitionedIndex {
    public:



    };
} // namespace orangedb
