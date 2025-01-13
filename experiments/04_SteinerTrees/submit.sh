#!/bin/bash

# Job group name
JOB_GROUP="/corneto/pcst"

# Conda environment name
CONDA_ENV="corneto"

# Base directory containing the datasets (d1, d2...)
BASE_DIR="datasets"

# Seed values array
SEED_VALUES=(0)

# Lambda values array
LAMBDAS=(1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0)

# Modes array
MODES=("single" "multi")

# Maximum time for optimization
MAX_TIME=3600

# Create the job group
bgadd $JOB_GROUP

# Iterate over datasets
for DATASET in $BASE_DIR/*; do
  # Extract the dataset name (e.g., "d1")
  DATASET_NAME=$(basename $DATASET)

  # Iterate over each lambda value
  for LAMBDA in "${LAMBDAS[@]}"; do
    # Iterate over each seed value
    for SEED in "${SEED_VALUES[@]}"; do
      # Iterate over each mode
      for MODE in "${MODES[@]}"; do
        # Construct the output and error file names
        OUTPUT_FILE="result_logs/${DATASET_NAME}_lambda${LAMBDA}_seed${SEED}_${MODE}.out"
        ERROR_FILE="results_logs/${DATASET_NAME}_lambda${LAMBDA}_seed${SEED}_${MODE}.err"
        
        # Construct the command to submit the job
        bsub -g $JOB_GROUP -R "rusage[mem=8192]" -M 8192 -o $OUTPUT_FILE -e $ERROR_FILE \
        /bin/bash -c "source ~/.bashrc && conda activate $CONDA_ENV && python script.py --dataset $DATASET --lambd $LAMBDA --mode $MODE --max_time $MAX_TIME --seed $SEED"
      done
    done
  done
done

          
