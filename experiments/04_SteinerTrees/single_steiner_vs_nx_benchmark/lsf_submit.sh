#!/bin/bash

# Job group name
JOB_GROUP="/corneto_st_single"

# Conda environment name
CONDA_ENV="corneto"

# Base directory containing the folders
OUT_DIR="single_st_benchmark_mt300s"

# Define max time
MAX_TIME=600

# Define arrays
#SEED_VALUES=(0 1 2)
#N_NODES=(100 200 300 400 500)
#N_TERMINALS=(5 10 15 20 25)
#N_EDGES=(2)
#STRICT=(0 1)

SEED_VALUES=(0)
N_NODES=(250 500 750 1000 1250 1500)
N_TERMINALS=(10 20 30 40 50)
N_EDGES=(2)
STRICT=(0 1)
EPS=(0.001 0.0001 0.00001)

# Calculate the total number of jobs
TOTAL_JOBS=$(( ${#SEED_VALUES[@]} * ${#N_NODES[@]} * ${#N_TERMINALS[@]} * ${#N_EDGES[@]} * ${#STRICT[@]} * ${#EPS[@]} ))

# Ask user if they want to proceed
echo "This script will generate $TOTAL_JOBS jobs. Do you want to proceed? (y/n)"
read -r PROCEED

if [[ "$PROCEED" != "y" ]]; then
  echo "Exiting script."
  exit 0
fi

# Create the job group
bgadd $JOB_GROUP

for NODE in "${N_NODES[@]}"; do
  for TERMINAL in "${N_TERMINALS[@]}"; do
    for EDGE in "${N_EDGES[@]}"; do
      for STR in "${STRICT[@]}"; do
        for EPS_VAL in "${EPS[@]}"; do
          for SEED in "${SEED_VALUES[@]}"; do
            # Construct the command to submit the job
            bsub -g $JOB_GROUP -R "rusage[mem=4096]" -M 4096 -o /dev/null -e /dev/null \
            /bin/bash -c "source ~/.bashrc && conda activate $CONDA_ENV && python benchmark_single_steiner.py --output_folder ${OUT_DIR} --seed ${SEED} --n_nodes ${NODE} --n_edges ${EDGE} --n_terminals ${TERMINAL} --strict_acyclic ${STR} --max_time ${MAX_TIME} --epsilon ${EPS_VAL}"
          done
        done
      done
    done
  done
done
