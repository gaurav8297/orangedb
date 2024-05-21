

./orangedb_main -basePath /home/g3sehgal/vector_index_exp/gist -nCentroids 10 -nIter 20 -minCentroidSize 10000 -maxCentroidSize 20000 -M 64 -K 100 -efConstruction 200 -efSearch 200 -nThreads 32 --minCentroids 4 -centroidThreshold 1

./orangedb_main -basePath /home/g3sehgal/vector_index_exp/gist -M 64 -K 100 -efConstruction 200 -efSearch 200 -nThreads 16 -numVectors 1000000 -alpha 1.0 -nCentroids 10 -nIter 30 -minCentroidSize 50000 -maxCentroidSize 50000 -maxSearchCentroids 3 -searchThreshold 0.5

./orangedb_main -basePath /home/g3sehgal/vector_index/vector_index/data/siftsmall -M 64 -K 100 -efConstruction 200 -efSearch 200 -nThreads 16 -numVectors 10000 -alpha 1.0

./orangedb_main -basePath /Users/gauravsehgal/work/vector_index/data/siftsmall -M 64 -K 100 -efConstruction 200 -efSearch 200 -nThreads 16 -numVectors 10000 -alpha 1.0 -nCentroids 10 -nIter 20 -minCentroidSize 500 -maxCentroidSize 500 -maxSearchCentroids 3 -searchThreshold 0.5

