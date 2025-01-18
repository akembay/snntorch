#!/bin/bash

# Specify the path to your Python interpreter
PYTHON="python"

# Specify the path to your training script
SCRIPT_NAME="delta_mnist_cl_task5.py"

# Replace the path to your Python executable if needed
VIRTUAL_ENV="/home/assel/miniconda3/envs/dvs"

# Activate the virtual environment if using one
if [ -f "$VIRTUAL_ENV" ]; then
    source "$VIRTUAL_ENV"
fi

# Define seed values
SEED_VALUES=(2020)

# Run the script with different seed values
for index in "${!SEED_VALUES[@]}"; do
    seed="${SEED_VALUES[$index]}"
    trial=$((index + 1))  # Adjusting trial to start from 1

    # Create a directory for each trial based on the seed
    output_dir="delta_mnist_OUTPUT/seed_$seed"
    mkdir -p "$output_dir"  # Create the directory if it doesn't exist

    echo "Running $SCRIPT_NAME with seed=$seed and trial=$trial"
    
    # Run the training script without the --seed argument
    $PYTHON $SCRIPT_NAME --target "$output_dir" --seed "$seed" --device 1
done

echo "Script execution completed."
