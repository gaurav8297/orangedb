#!/bin/bash

# Function to execute a command with retries
function execute_with_retry {
    local command="$1"
    local max_retries=3
    local retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        $command  # Execute the command

        if [ $? -eq 0 ]; then
            echo "Command executed successfully."
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

echo "Running the baseline with 1.2"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/gist -M 64 -K 100 -efConstruction 200 -efSearch 200 -nThreads 32 -numVectors 1000000 -minAlpha 1.2 -maxAlpha 1.2 -alphaDecay 0.0 >> baseline_1.2.txt"

echo "Running the baseline with 1.15"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/gist -M 64 -K 100 -efConstruction 200 -efSearch 200 -nThreads 32 -numVectors 1000000 -minAlpha 1.15 -maxAlpha 1.15 -alphaDecay 0.0 >> baseline_1.15.txt"

echo "Running the experiment with minAlpha 0.95, maxAlpha 1.2 and alphaDecay 0.02"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/gist -M 64 -K 100 -efConstruction 200 -efSearch 200 -nThreads 32 -numVectors 1000000 -minAlpha 0.95 -maxAlpha 1.2 -alphaDecay 0.02 >> output1.txt"

echo "Running the experiment with minAlpha 0.95, maxAlpha 1.2 and alphaDecay 0.04"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/gist -M 64 -K 100 -efConstruction 200 -efSearch 200 -nThreads 32 -numVectors 1000000 -minAlpha 0.95 -maxAlpha 1.2 -alphaDecay 0.04 >> output2.txt"

echo "Running the experiment with minAlpha 0.95, maxAlpha 1.2 and alphaDecay 0.06"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/gist -M 64 -K 100 -efConstruction 200 -efSearch 200 -nThreads 32 -numVectors 1000000 -minAlpha 0.95 -maxAlpha 1.2 -alphaDecay 0.06 >> output3.txt"

echo "Running the experiment with minAlpha 0.95, maxAlpha 1.2 and alphaDecay 0.08"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/gist -M 64 -K 100 -efConstruction 200 -efSearch 200 -nThreads 32 -numVectors 1000000 -minAlpha 0.95 -maxAlpha 1.2 -alphaDecay 0.08 >> output4.txt"

echo "Running the experiment with minAlpha 0.95, maxAlpha 1.2 and alphaDecay 0.1"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/gist -M 64 -K 100 -efConstruction 200 -efSearch 200 -nThreads 32 -numVectors 1000000 -minAlpha 0.95 -maxAlpha 1.2 -alphaDecay 0.1 >> output5.txt"

echo "Running the experiment with minAlpha 0.95, maxAlpha 1.2 and alphaDecay 0.2"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/gist -M 64 -K 100 -efConstruction 200 -efSearch 200 -nThreads 32 -numVectors 1000000 -minAlpha 0.95 -maxAlpha 1.2 -alphaDecay 0.2 >> output6.txt"
