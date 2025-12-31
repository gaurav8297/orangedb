#!/bin/bash

# MY COMPUTER
#-----------------------------------
# no hard limit
./build/release/bin/orangedb_main -run benchmarkFastReclustering -baseVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/base.fvecs -queryVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/query.fvecs -groundTruthPath /Users/gilli.hadayo/orangedb/data/siftsmall/gt.bin -k 100 -numInserts 1 -numVectors 10000 -numIters 20 -megaCentroidSize 10 -miniCentroidSize 500 -iterations 0 -lambda 0 -nMegaProbes 4 -nMiniProbes 40 -numQueries 50 -readFromDisk 0 -storagePath /Users/gilli.hadayo/orangedb/data/siftsmall/reclustering_fast_index.bin -isParquet 0 -numMegaReclusterCentroids 1 -reclusterOnScore 0 -numThreads 1 -useIP 0 -quantTrainPercentage 0.0 -quantBuild 0 -nMegaRecluster 0 -nFiles 1 -hardClusterSizeLimit 10000 -kmeansSamplingRatio 1.0 -numFixBoundaries 0 -scoreChangeThreshold 0.01 -centroidChangeThreshold 0.01 -useMSEToRecluster 0

# with hard limit 600
./build/release/bin/orangedb_main -run benchmarkFastReclustering -baseVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/base.fvecs -queryVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/query.fvecs -groundTruthPath /Users/gilli.hadayo/orangedb/data/siftsmall/gt.bin -k 100 -numInserts 1 -numVectors 10000 -numIters 20 -megaCentroidSize 10 -miniCentroidSize 500 -iterations 0 -lambda 0 -nMegaProbes 4 -nMiniProbes 40 -numQueries 50 -readFromDisk 0 -storagePath /Users/gilli.hadayo/orangedb/data/siftsmall/reclustering_fast_index.bin -isParquet 0 -numMegaReclusterCentroids 1 -reclusterOnScore 0 -numThreads 1 -useIP 0 -quantTrainPercentage 0.0 -quantBuild 0 -nMegaRecluster 0 -nFiles 1 -hardClusterSizeLimit 600 -kmeansSamplingRatio 1.0 -numFixBoundaries 0 -scoreChangeThreshold 0.01 -centroidChangeThreshold 0.01 -useMSEToRecluster 0

# run dataset 2D projection
./build/release/bin/orangedb_main -run run_umap_2D_without_clustering -baseVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/base.fvecs -numVectors 10000 -outputPath /Users/gilli.hadayo/orangedb/umap_2D_without_clustering.csv
# run dataset 3D projection
./build/release/bin/orangedb_main -run run_umap_3D_without_clustering -baseVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/base.fvecs -numVectors 10000 -outputPath /Users/gilli.hadayo/orangedb/umap_3D_without_clustering.csv


# VM
#-----------------------------------
# no hard limit
./build/release/bin/orangedb_main -run benchmarkFastReclustering -baseVectorPath /home/centos/orangedb/data/siftsmall/base.fvecs -queryVectorPath /home/centos/orangedb/data/siftsmall/query.fvecs -groundTruthPath /home/centos/orangedb/data/siftsmall/gt.bin -k 100 -numInserts 1 -numVectors 10000 -numIters 20 -megaCentroidSize 10 -miniCentroidSize 500 -iterations 0 -lambda 0 -nMegaProbes 4 -nMiniProbes 40 -numQueries 50 -readFromDisk 0 -storagePath /home/centos/orangedb/data/siftsmall/reclustering_fast_index.bin -isParquet 0 -numMegaReclusterCentroids 1 -reclusterOnScore 0 -numThreads 1 -useIP 0 -quantTrainPercentage 0.0 -quantBuild 0 -nMegaRecluster 0 -nFiles 1 -hardClusterSizeLimit 10000 -kmeansSamplingRatio 1.0 -numFixBoundaries 0 -scoreChangeThreshold 0.01 -centroidChangeThreshold 0.01 -useMSEToRecluster 0

# with hard limit 600
./build/release/bin/orangedb_main -run benchmarkFastReclustering -baseVectorPath /home/centos/orangedb/data/siftsmall/base.fvecs -queryVectorPath /home/centos/orangedb/data/siftsmall/query.fvecs -groundTruthPath /home/centos/orangedb/data/siftsmall/gt.bin -k 100 -numInserts 1 -numVectors 10000 -numIters 20 -megaCentroidSize 10 -miniCentroidSize 500 -iterations 0 -lambda 0 -nMegaProbes 4 -nMiniProbes 40 -numQueries 50 -readFromDisk 0 -storagePath /home/centos/orangedb/data/siftsmall/reclustering_fast_index.bin -isParquet 0 -numMegaReclusterCentroids 1 -reclusterOnScore 0 -numThreads 1 -useIP 0 -quantTrainPercentage 0.0 -quantBuild 0 -nMegaRecluster 0 -nFiles 1 -hardClusterSizeLimit 600 -kmeansSamplingRatio 1.0 -numFixBoundaries 0 -scoreChangeThreshold 0.01 -centroidChangeThreshold 0.01 -useMSEToRecluster 0

