#pragma once

namespace orangedb {

    // TODO: Add serialization and deserialization from disk
    struct Config {
        // The number of neighbors to keep for each node
        int M = 16;
        // The number of neighbors to explore during index construction
        int efConstruction = 200;
        // The number of neighbors to explore during search
        int efSearch = 50;
        // RNG alpha parameter
        float alpha = 1.1;
        // The number of centroids
        int numCentroids = 10;
        // The number of iterations
        int nIter = 60;
        // The minimum size of a centroid
        int minSizePerCentroid = 2000;
        // The maximum size of a centroid
        int sampleSizePerCentroid = 5000;
        // The maximum number of centroids at search time
        int maxCentroidsToSearch = 3;
        // The distance threshold for searching in the centroids
        float searchDistThreshold = 0.4;

        Config(int M, int efConstruction, int efSearch, float alpha, int numCentroids,
               int nIter, int minSizePerCentroid, int sampleSizePerCentroid, int maxCentroidsToSearch,
               float searchDistThreshold)
                : M(M), efConstruction(efConstruction), efSearch(efSearch), alpha(alpha), numCentroids(numCentroids),
                  nIter(nIter), minSizePerCentroid(minSizePerCentroid), sampleSizePerCentroid(sampleSizePerCentroid),
                  maxCentroidsToSearch(maxCentroidsToSearch), searchDistThreshold(searchDistThreshold) {};

        Config() = default;
    };
}
