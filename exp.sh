#!/bin/bash

# Function to execute a command with retries
function execute_with_retry {
    local command="$1"
    local output_file="$2"
    local max_retries=3
    local retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        $command > "$output_file"  # Execute the command and write output to the specified file

        if [ $? -eq 0 ]; then
            echo "Command executed successfully. Output written to $output_file."
            break
        else
            echo "Command failed. Retrying..."
            ((retry_count++))
        fi
    done

    if [ $retry_count -eq $max_retries ]; then
        echo "Command failed after $max_retries retries. Exiting."
        exit 1
    fi
}

# Retry the below commands if they fail
echo "Starting the experiment..."

#echo "Running the baseline with 1.2"
#execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/openai_dbpedia -M 16 -k 100 -efConstruction 40 -efSearch 300 -nThreads 32 -minAlpha 1 -maxAlpha 1 -alphaDecay 0" "baseline_40.txt"

echo "Running the experiment with minAlpha 0.95, maxAlpha 1.0 and alphaDecay 0.02"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/openai_dbpedia -M 24 -k 100 -efConstruction 50 -efSearch 300 -nThreads 32 -minAlpha 1.1 -maxAlpha 1.1 -alphaDecay 0" "out_40_1.1.txt"

echo "Running the experiment with minAlpha 0.9, maxAlpha 1.0 and alphaDecay 0.04"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/openai_dbpedia -M 24 -k 100 -efConstruction 50 -efSearch 300 -nThreads 32 -minAlpha 1.2 -maxAlpha 1.2 -alphaDecay 0" "out_40_1.2.txt"

echo "Running the experiment with minAlpha 0.9, maxAlpha 1.2 and alphaDecay 0.06"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/openai_dbpedia -M 24 -k 100 -efConstruction 50 -efSearch 300 -nThreads 32 -minAlpha 0.95 -maxAlpha 1.1 -alphaDecay 0.02" "out_40_0.02.txt"

echo "Running the experiment with minAlpha 0.95, maxAlpha 1.2 and alphaDecay 0.08"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/openai_dbpedia -M 24 -k 100 -efConstruction 50 -efSearch 300 -nThreads 32 -minAlpha 0.95 -maxAlpha 1.1 -alphaDecay 0.04" "out_40_0.04.txt"

echo "Running the experiment with minAlpha 0.95, maxAlpha 1.2 and alphaDecay 0.1"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/openai_dbpedia -M 24 -k 100 -efConstruction 50 -efSearch 300 -nThreads 32 -minAlpha 0.95 -maxAlpha 1.1 -alphaDecay 0.08" "out_40_0.08.txt"


perf record -e cycles:ppp -- ./orangedb_main -basePath /home/g3sehgal/projects/def-ssalihog/g3sehgal/gist_50k -M 64 -k 100 -efConstruction 100 -efSearch 100 -nThreads 32 -minAlpha 1.0 -maxAlpha 1.0 -alphaDecay 0


./orangedb/build/release/bin/orangedb_main -run benchmark -basePath /localscratch/g3sehgal.41095149.0/gist_1M -k 100 -M 64 -K 100 -efConstruction 200 -efSearch 300 -nThreads 32 -minAlpha 1 -alphaDecay 0.1 -maxAlpha 1.1 -filterMinK 10 -maxNeighboursCheck 128

./orangedb/build/release/bin/orangedb_main -run generateGT -basePath /localscratch/g3sehgal.41095149.0/gist_1M -k 100 -gtPath /localscratch/g3sehgal.41095149.0/gist_1M/gt.bin -filteredMaskPath /localscratch/g3sehgal.41095149.0/gist_1M/mask.bin -selectivity 0.05


./orangedb/build/release/bin/orangedb_main -run generateGT -basePath /localscratch/g3sehgal.41104075.0/gist_1M -k 100 -gtPath /localscratch/g3sehgal.41104075.0/gist_1M/gt.bin -filteredMaskPath /localscratch/g3sehgal.41104075.0/gist_1M/mask.bin -selectivity 0.1

./orangedb/build/release/bin/orangedb_main -run benchmark -basePath /localscratch/g3sehgal.41104075.0/gist_1M -k 100 -M 64 -efConstruction 200 -efSearch 200 -nThreads 32 -minAlpha 1 -alphaDecay 0.1 -maxAlpha 1.1 -filterMinK 40 -maxNeighboursCheck 60

./build/release/bin/orangedb_main -run generateGT -basePath /home/centos/orangedb/data/gist -k 100 -numVectors 1000000 -gtPath /home/centos/orangedb/data/gist/gt.bin

./build/release/bin/orangedb_main -run benchmarkReclusteringIndex -baseVectorPath /home/centos/orangedb/data/sift10m/base.fvecs -queryVectorPath /home/centos/orangedb/data/sift10m/query.fvecs -groundTruthPath /home/centos/orangedb/data/sift10m/gt.bin -k 100 -numInserts 1 -numCentroids 40000 -numIters 30 -minCentroidSize 150 -maxCentroidSize 300 -nProbes 100 -lambda 0 -numReclusters 0 -readFromDisk 0 -storagePath /home/centos/orangedb/data/sift10m/index.bin
./build/release/bin/orangedb_main -run benchmarkReclusteringIndex -baseVectorPath /home/centos/orangedb/data/sift10m/base.fvecs -queryVectorPath /home/centos/orangedb/data/sift10m/query.fvecs -groundTruthPath /home/centos/orangedb/data/sift10m/gt.bin -k 100 -numInserts 1 -numCentroids 40000 -numIters 30 -minCentroidSize 150 -maxCentroidSize 300 -nProbes 100 -lambda 0 -numReclusters 0 -readFromDisk 0 -storagePath /home/centos/orangedb/data/sift10m/index.bin
./build/release/bin/orangedb_main -run benchmarkReclusteringIndex -baseVectorPath /home/centos/orangedb/data/sift10m/base.fvecs -queryVectorPath /home/centos/orangedb/data/sift10m/query.fvecs -groundTruthPath /home/centos/orangedb/data/sift10m/gt.bin -k 100 -numInserts 1 -numCentroids 40000 -numIters 30 -minCentroidSize 150 -maxCentroidSize 300 -nProbes 100 -lambda 0 -numReclusters 0 -readFromDisk 0 -storagePath /home/centos/orangedb/data/sift10m/index.bin

./build/release/bin/orangedb_main -run benchmarkSplitting -baseVectorPath /home/centos/orangedb/data/sift10m/base.fvecs -queryVectorPath /home/centos/orangedb/data/sift10m/query.fvecs -groundTruthPath /home/centos/orangedb/data/sift10m/gt.bin -k 100 -numInserts 10 -numCentroids 4000 -numIters 30 -minCentroidSize 150 -maxCentroidSize 300 -maxMegaClusterSize 8000 -nMegaProbes 3 -nMicroProbes 3 -lambda 0 -numReclusters 0 -readFromDisk 0 -storagePath /home/centos/orangedb/data/sift10m/incremental_index.bin

./build/release/bin/orangedb_main -run benchmarkReclusteringIndex -baseVectorPath /home/centos/orangedb/data/sift10m/base.fvecs -queryVectorPath /home/centos/orangedb/data/sift10m/query.fvecs -groundTruthPath /home/centos/orangedb/data/sift10m/gt.bin -k 100 -numInserts 20 -numVectors 1000000 -numIters 40 -megaCentroidSize 100 -miniCentroidSize 500 -newMiniCentroidSize 100 -lambda 0 -numMegaReclusterCentroids 5 -numNewMiniReclusterCentroids 100 -nMegaProbes 3 -nMiniProbes 20 -readFromDisk 0 -storagePath /home/centos/orangedb/data/sift10m/incremental_index.bin

./build/release/bin/orangedb_main -run benchmarkQuantized -baseVectorPath /home/centos/gist/base.fvecs -queryVectorPath /home/centos/gist/query.fvecs -n 1

./build/release/bin/orangedb_main -run benchmarkReclusteringIndex -baseVectorPath /home/centos/orangedb/data/sift10m/base.fvecs -queryVectorPath /home/centos/orangedb/data/sift10m/query.fvecs -groundTruthPath /home/centos/orangedb/data/sift10m/gt.bin -k 100 -numInserts 50 -numVectors 10000000 -numIters 30 -megaCentroidSize 100 -miniCentroidSize 500 -newMiniCentroidSize 100 -lambda 0 -numMegaReclusterCentroids 5 -numNewMiniReclusterCentroids 100 -nMegaProbes 3 -nMiniProbes 20 -readFromDisk 0 -storagePath /home/centos/orangedb/data/sift10m/incremental_index.bin
./build/release/bin/orangedb_main -run benchmarkClustering -baseVectorPath /home/centos/orangedb/data/gist/base.fvecs -queryVectorPath /home/centos/orangedb/data/gist/query.fvecs -groundTruthPath /home/centos/orangedb/data/gist/gt.bin -k 100 -numVectors 10000000 -nIter 30 -clusterSize 500 -lambda 0 -nProbes 20
./build/release/bin/orangedb_main -run benchmarkReclusteringAlgorithm -baseVectorPath /Users/gaurav.sehgal/work/orangedb/data/gist/base.fvecs -queryVectorPath /Users/gaurav.sehgal/work/orangedb/data/gist/query.fvecs -groundTruthPath /Users/gaurav.sehgal/work/orangedb/data/gist/gt.bin -k 100 -numInserts 10 -numVectors 100000 -numIters 10 -megaCentroidSize 50 -miniCentroidSize 250 -maxLevels 10 -fast 0 -lambda 0 -numMegaReclusterCentroids 3 -nMegaProbes 4 -nMiniProbes 40 -readFromDisk 1 -storagePath /Users/gaurav.sehgal/work/orangedb/data/gist/index.bin

./build/release/bin/orangedb_main -run benchmarkReclusteringAlgorithm -baseVectorPath /Users/gaurav.sehgal/work/orangedb/data/gist/base.fvecs -queryVectorPath /Users/gaurav.sehgal/work/orangedb/data/gist/query.fvecs -groundTruthPath /Users/gaurav.sehgal/work/orangedb/data/gist/gt.bin -k 100 -numInserts 10 -numVectors 100000 -numIters 10 -megaCentroidSize 50 -miniCentroidSize 250 -maxLevels 10 -fast 0 -lambda 0 -numMegaReclusterCentroids 3 -nMegaProbes 4 -nMiniProbes 40 -readFromDisk 1 -storagePath /Users/gaurav.sehgal/work/orangedb/data/gist/index.bin


./build/release/bin/orangedb_main -run benchmarkNavix -baseVectorPath /Users/gaurav.sehgal/work/orangedb/data/gist/base.fvecs -queryVectorPath /Users/gaurav.sehgal/work/orangedb/data/gist/query.fvecs -gtPath /Users/gaurav.sehgal/work/orangedb/data/gist/gt.bin -maskPath /Users/gaurav.sehgal/work/vector_dataset/gist/bench_data/mask_30.bin  -k 100 -M 32 -efSearch 200 -nThreads 32 -readFromDisk 0 -storagePath /Users/gaurav.sehgal/work/orangedb/data/gist/hnsw_index.bin

./build/release/bin/orangedb_main -run benchmarkAcorn -vectorPath /Users/gaurav.sehgal/work/orangedb/data/gist/base.fvecs -queryPath /Users/gaurav.sehgal/work/orangedb/data/gist/query.fvecs -gtPath /Users/gaurav.sehgal/work/orangedb/data/gist/gt.bin -maskPath /Users/gaurav.sehgal/work/vector_dataset/gist/bench_data/mask_30.bin  -k 100 -M 64 -gamma 1 -M_beta 128 -efSearch 200 -nThreads 32 -readFromDisk 0 -storagePath /Users/gaurav.sehgal/work/orangedb/data/gist/acorn_index.bin

./build/release/bin/orangedb_main -run benchmarkAcorn -vectorPath /home/centos/orangedb/data/gist/base.fvecs -queryPath /home/centos/vector_dataset/gist/bench_data/queries.fvecs -gtPath /home/centos/vector_dataset/gist/bench_data/gt.bin -maskPath /home/centos/vector_dataset/gist/bench_data/mask_30.bin  -k 100 -M 32 -gamma 10 -M_beta 64 -efSearch 300 -nThreads 32 -readFromDisk 0 -storagePath /home/centos/orangedb/data/gist/acorn_index.bin

./build/release/bin/orangedb_main -run benchmarkNavix -vectorPath /home/centos/orangedb/data/gist/base.fvecs -queryPath /home/centos/vector_dataset/gist/bench_data/queries.fvecs -gtPath /home/centos/vector_dataset/gist/bench_data/gt.bin -maskPath /home/centos/vector_dataset/gist/bench_data/mask_30.bin  -k 100 -M 32 -efSearch 120 -nThreads 32 -readFromDisk 1 -storagePath /home/centos/orangedb/data/gist/hnsw_index.bin

./build/release/bin/orangedb_main -run benchmarkIRangeGraph -vectorPath /home/centos/orangedb/data/gist/base.fvecs -queryPath /home/centos/vector_dataset/gist/bench_data/queries.fvecs -gtPath /home/centos/vector_dataset/gist/bench_data/gt.bin -maskPath /home/centos/vector_dataset/gist/bench_data/mask_30.bin -k 100 -M 32 -efSearch 120 -efConstruction 200 -nThreads 32 -readFromDisk 0 -storagePath /home/centos/orangedb/data/gist/irangegraph_index.bin -outputPath /home/centos/orangedb/data/gist/irangegraph_search.out

./build/release/bin/orangedb_main -run benchmarkFastReclustering -baseVectorPath /home/centos/vector_dataset/sift10m/base.fvecs -queryVectorPath /home/centos/vector_dataset/sift10m/query.fvecs -groundTruthPath /home/centos/vector_dataset/sift10m/gt.bin -k 100 -numInserts 10 -numVectors 10000000 -numIters 20 -megaCentroidSize 500 -miniCentroidSize 500 -iterations 20 -fast 0 -lambda 0 -nMegaProbes 4 -nMiniProbes 40 -nQueries 50 -readFromDisk 0 -storagePath /Users/gaurav.sehgal/work/orangedb/data/gist/reclustering_fast_index.bin

./build/release/bin/orangedb_main -run generateGT -basePath /home/centos/vector_dataset/sift10m -numVectors 10000000 -k 100 -gtPath /home/centos/vector_dataset/sift10m/gt.bin

./build/release/bin/orangedb_main -run benchmarkFaissClustering -baseVectorPath /home/centos/vector_dataset/sift10m/base.fvecs -queryVectorPath /home/centos/vector_dataset/sift10m/query.fvecss -groundTruthPath /home/centos/vector_dataset/sift10m/gt.bin -k 100 -numVectors 100000000 -nIter 30 -nThreads 32 -numQueries 50 -clusterSize 500 -nProbes 20

