#!/bin/bash

# Define the arrays for alpha and num_nodes
alphas=(0.05 0.10 0.15 0.20 0.25 0.30)
nodes=(250 500 1000)

# Run the Python script with each combination of alpha and num_nodes
for alpha in "${alphas[@]}"; do
    for num_nodes in "${nodes[@]}"; do
        echo "Running Python script with alpha=$alpha and num_nodes=$num_nodes"
        python gen_steiner_dataset.py --strict_acyclic --num_samples 10 --alpha "$alpha" --num_nodes "$num_nodes"
        if [ $? -ne 0 ]; then
            echo "Error executing Python script with alpha=$alpha and num_nodes=$num_nodes"
            exit 1
        fi
    done
done

echo "Script execution completed."
read -p "Press any key to exit..."
