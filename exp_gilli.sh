#!/bin/bash

# MY COMPUTER
#-----------------------------------
# no hard limit - live UMAP
./build/release/bin/orangedb_main \
  -run benchmarkFastReclustering \
  -baseVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/base.fvecs \
  -queryVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/query.fvecs \
  -groundTruthPath /Users/gilli.hadayo/orangedb/data/siftsmall/gt.bin \
  -k 100 \
  -numInserts 1 \
  -numVectors 10000 \
  -numIters 20 \
  -megaCentroidSize 10 \
  -miniCentroidSize 500 \
  -iterations 1 \
  -lambda 0 \
  -nMegaProbes 4 \
  -nMiniProbes 40 \
  -numQueries 50 \
  -readFromDisk 0 \
  -storagePath /Users/gilli.hadayo/orangedb/data/siftsmall/reclustering_fast_index.bin \
  -isParquet 0 \
  -numMegaReclusterCentroids 1 \
  -reclusterOnScore 0 \
  -numThreads 1 \
  -useIP 0 \
  -quantTrainPercentage 0.0 \
  -quantBuild 0 \
  -nMegaRecluster 0 \
  -nFiles 1 \
  -hardClusterSizeLimit 10000 \
  -kmeansSamplingRatio 1.0 \
  -numFixBoundaries 0 \
  -scoreChangeThreshold 0.01 \
  -centroidChangeThreshold 0.01 \
  -useMSEToRecluster 0 \
  -umap_mode 1 \
  -clustering_mode 0

# with hard limit 600 - live UMAP
./build/release/bin/orangedb_main \
  -run benchmarkFastReclustering \
  -baseVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/base.fvecs \
  -queryVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/query.fvecs \
  -groundTruthPath /Users/gilli.hadayo/orangedb/data/siftsmall/gt.bin \
  -k 100 \
  -numInserts 1 \
  -numVectors 10000 \
  -numIters 20 \
  -megaCentroidSize 10 \
  -miniCentroidSize 500 \
  -iterations 1 \
  -lambda 0 \
  -nMegaProbes 4 \
  -nMiniProbes 40 \
  -numQueries 50 \
  -readFromDisk 0 \
  -storagePath /Users/gilli.hadayo/orangedb/data/siftsmall/reclustering_fast_index.bin \
  -isParquet 0 \
  -numMegaReclusterCentroids 1 \
  -reclusterOnScore 0 \
  -numThreads 1 \
  -useIP 0 \
  -quantTrainPercentage 0.0 \
  -quantBuild 0 \
  -nMegaRecluster 0 \
  -nFiles 1 \
  -hardClusterSizeLimit 600 \
  -kmeansSamplingRatio 1.0 \
  -numFixBoundaries 0 \
  -scoreChangeThreshold 0.01 \
  -centroidChangeThreshold 0.01 \
  -useMSEToRecluster 0 \
  -umap_mode 1 \
  -clustering_mode 0

# no hard limit - offline UMAP
./build/release/bin/orangedb_main \
  -run benchmarkFastReclustering \
  -baseVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/base.fvecs \
  -queryVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/query.fvecs \
  -groundTruthPath /Users/gilli.hadayo/orangedb/data/siftsmall/gt.bin \
  -k 100 \
  -numInserts 1 \
  -numVectors 10000 \
  -numIters 20 \
  -megaCentroidSize 10 \
  -miniCentroidSize 500 \
  -iterations 1 \
  -lambda 0 \
  -nMegaProbes 4 \
  -nMiniProbes 40 \
  -numQueries 50 \
  -readFromDisk 0 \
  -storagePath /Users/gilli.hadayo/orangedb/data/siftsmall/reclustering_fast_index.bin \
  -isParquet 0 \
  -numMegaReclusterCentroids 1 \
  -reclusterOnScore 0 \
  -numThreads 1 \
  -useIP 0 \
  -quantTrainPercentage 0.0 \
  -quantBuild 0 \
  -nMegaRecluster 0 \
  -nFiles 1 \
  -hardClusterSizeLimit 10000 \
  -kmeansSamplingRatio 1.0 \
  -numFixBoundaries 0 \
  -scoreChangeThreshold 0.01 \
  -centroidChangeThreshold 0.01 \
  -useMSEToRecluster 0 \
  -umap_mode 2 \
  -clustering_mode 0

# with hard limit 600 - offline UMAP
./build/release/bin/orangedb_main \
  -run benchmarkFastReclustering \
  -baseVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/base.fvecs \
  -queryVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/query.fvecs \
  -groundTruthPath /Users/gilli.hadayo/orangedb/data/siftsmall/gt.bin \
  -k 100 \
  -numInserts 1 \
  -numVectors 10000 \
  -numIters 20 \
  -megaCentroidSize 10 \
  -miniCentroidSize 500 \
  -iterations 1 \
  -lambda 0 \
  -nMegaProbes 4 \
  -nMiniProbes 40 \
  -numQueries 50 \
  -readFromDisk 0 \
  -storagePath /Users/gilli.hadayo/orangedb/data/siftsmall/reclustering_fast_index.bin \
  -isParquet 0 \
  -numMegaReclusterCentroids 1 \
  -reclusterOnScore 0 \
  -numThreads 1 \
  -useIP 0 \
  -quantTrainPercentage 0.0 \
  -quantBuild 0 \
  -nMegaRecluster 0 \
  -nFiles 1 \
  -hardClusterSizeLimit 600 \
  -kmeansSamplingRatio 1.0 \
  -numFixBoundaries 0 \
  -scoreChangeThreshold 0.01 \
  -centroidChangeThreshold 0.01 \
  -useMSEToRecluster 0 \
  -umap_mode 2 \
  -clustering_mode 0 \
  > output_hard_limit_600_offline_umap.txt 2>&1



# run dataset 2D projection
./build/release/bin/orangedb_main \
  -run run_umap_2D_without_clustering \
  -baseVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/base.fvecs \
  -numVectors 10000 \
  -outputPath /Users/gilli.hadayo/orangedb/umap_2D_without_clustering.bin

# run dataset 3D projection
./build/release/bin/orangedb_main \
  -run run_umap_3D_without_clustering \
  -baseVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/base.fvecs \
  -numVectors 10000 \
  -outputPath /Users/gilli.hadayo/orangedb/umap_3D_without_clustering.bin

# run dataset 3D projection
./build/release/bin/orangedb_main \
  -run run_umap_3D_without_clustering \
  -baseVectorPath /home/centos/orangedb/data/sift_bill/bigann_base.bvecs \
  -numVectors 1000000000 \
  -outputPath /home/centos/orangedb/umap_3D_without_clustering.bin

# VM
#-----------------------------------
# no hard limit
./build/release/bin/orangedb_main \
  -run benchmarkFastReclustering \
  -baseVectorPath /home/centos/orangedb/data/siftsmall/base.fvecs \
  -queryVectorPath /home/centos/orangedb/data/siftsmall/query.fvecs \
  -groundTruthPath /home/centos/orangedb/data/siftsmall/gt.bin \
  -k 100 \
  -numInserts 1 \
  -numVectors 10000 \
  -numIters 20 \
  -megaCentroidSize 10 \
  -miniCentroidSize 500 \
  -iterations 1 \
  -lambda 0 \
  -nMegaProbes 4 \
  -nMiniProbes 40 \
  -numQueries 50 \
  -readFromDisk 0 \
  -storagePath /home/centos/orangedb/data/siftsmall/reclustering_fast_index.bin \
  -isParquet 0 \
  -numMegaReclusterCentroids 1 \
  -reclusterOnScore 0 \
  -numThreads 1 \
  -useIP 0 \
  -quantTrainPercentage 0.0 \
  -quantBuild 0 \
  -nMegaRecluster 0 \
  -nFiles 1 \
  -hardClusterSizeLimit 10000 \
  -kmeansSamplingRatio 1.0 \
  -numFixBoundaries 0 \
  -scoreChangeThreshold 0.01 \
  -centroidChangeThreshold 0.01 \
  -useMSEToRecluster 0 \
  -umap_mode 1 \
  -clustering_mode 0


./build/release/bin/orangedb_main \
  -run benchmarkFastReclustering \
  -baseVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/base.fvecs \
  -queryVectorPath /Users/gilli.hadayo/orangedb/data/siftsmall/query.fvecs \
  -groundTruthPath /Users/gilli.hadayo/orangedb/data/siftsmall/gt.bin \
  -k 100 \
  -numInserts 1 \
  -numVectors 10000 \
  -numIters 5 \
  -megaCentroidSize 10 \
  -miniCentroidSize 500 \
  -iterations 20 \
  -lambda 0 \
  -nMegaProbes 1 \
  -nMiniProbes 3 \
  -numQueries 50 \
  -readFromDisk 0 \
  -storagePath /Users/gilli.hadayo/orangedb/data/siftsmall/reclustering_fast_index.bin \
  -isParquet 0 \
  -numMegaReclusterCentroids 1 \
  -reclusterOnScore 0 \
  -numThreads 1 \
  -useIP 0 \
  -quantTrainPercentage 0.0 \
  -quantBuild 0 \
  -nMegaRecluster 0 \
  -nFiles 1 \
  -hardClusterSizeLimit 600 \
  -kmeansSamplingRatio 1.0 \
  -numFixBoundaries 0 \
  -scoreChangeThreshold 0.01 \
  -centroidChangeThreshold 0.01 \
  -useMSEToRecluster 0 \
  -umap_mode 2 \
  -clustering_mode 0 





  # with hard limit 600
./build/release/bin/orangedb_main \
  -run benchmarkFastReclustering \
  -baseVectorPath /home/centos/orangedb/data/sift_bill/bigann_base.bvecs \
  -queryVectorPath /home/centos/orangedb/data/sift_bill/bigann_query.bvecs \
  -groundTruthPath /home/centos/orangedb/data/sift_bill/sift_gt_10M.bin \
  -k 100 \
  -numInserts 10 \
  -numVectors 10000000 \
  -numIters 20 \
  -megaCentroidSize 1000 \
  -miniCentroidSize 1000 \
  -iterations 10 \
  -lambda 0 \
  -nMegaProbes 4 \
  -nMiniProbes 100 \
  -numQueries 50 \
  -readFromDisk 0 \
  -storagePath /home/centos/orangedb/data/sift_bill/reclustering_fast_index.bin \
  -isParquet 0 \
  -numMegaReclusterCentroids 1 \
  -reclusterOnScore 0 \
  -numThreads 32 \
  -useIP 0 \
  -quantTrainPercentage 0.0 \
  -quantBuild 0 \
  -nMegaRecluster 0 \
  -nFiles 1 \
  -hardClusterSizeLimit 0 \
  -kmeansSamplingRatio 1.0 \
  -numFixBoundaries 0 \
  -scoreChangeThreshold 0.01 \
  -centroidChangeThreshold 0.01 \
  -useMSEToRecluster 0 \
  -umap_mode 2 \
  -clustering_mode 0 > no_hard_limit_log.txt 2>&1


./build/release/bin/orangedb_main \
  -run run_umap_3D_without_clustering \
  -baseVectorPath /home/centos/orangedb/data/sift_bill/bigann_base.bvecs \
  -numVectors 10000000 \
  -outputPath /home/centos/orangedb/umap_3D_without_clustering.bin