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
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/openai_dbpedia -M 16 -k 100 -efConstruction 40 -efSearch 300 -nThreads 32 -minAlpha 1.1 -maxAlpha 1.1 -alphaDecay 0" "out_40_1.1.txt"

echo "Running the experiment with minAlpha 0.9, maxAlpha 1.0 and alphaDecay 0.04"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/openai_dbpedia -M 16 -k 100 -efConstruction 40 -efSearch 300 -nThreads 32 -minAlpha 1.2 -maxAlpha 1.2 -alphaDecay 0" "out_40_1.2.txt"

echo "Running the experiment with minAlpha 0.9, maxAlpha 1.2 and alphaDecay 0.06"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/openai_dbpedia -M 16 -k 100 -efConstruction 40 -efSearch 300 -nThreads 32 -minAlpha 0.95 -maxAlpha 1.1 -alphaDecay 0.02" "out_40_0.02.txt"

echo "Running the experiment with minAlpha 0.95, maxAlpha 1.2 and alphaDecay 0.08"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/openai_dbpedia -M 16 -k 100 -efConstruction 40 -efSearch 300 -nThreads 32 -minAlpha 0.95 -maxAlpha 1.1 -alphaDecay 0.04" "out_40_0.04.txt"

echo "Running the experiment with minAlpha 0.95, maxAlpha 1.2 and alphaDecay 0.1"
execute_with_retry "./orangedb_main -basePath /home/g3sehgal/vector_index_exp/openai_dbpedia -M 16 -k 100 -efConstruction 40 -efSearch 300 -nThreads 32 -minAlpha 0.95 -maxAlpha 1.1 -alphaDecay 0.08" "out_40_0.08.txt"
