#!/usr/bin/env python
"""
submit_jobs.py

Submit performance measurement jobs via SLURM using preprocessed input data.
This script uses the pickle file (saved by generate_ground_truth.py) that contains the
input dictionary and network graph, so that worker jobs do not need to reprocess the raw data.

Before running the Python worker command, the SLURM job will:
  1. Source the user's bash configuration (source $HOME/.bashrc),
  2. Activate a micromamba environment (default: corneto-dev2, configurable via --env),
  3. Optionally load any specified modules (via --modules).

Modes:
  - Submission mode (default): Loops over lambda values and fold indices and submits jobs.
  - Worker mode (--worker): Loads preprocessed data, splits it into folds, runs the analysis,
    computes the error relative to the ground truth, and writes the error to a file.

Usage example (submission mode):
    python submit_jobs.py --pickle input_data_and_graph.pkl --ground_truth ground_truth.csv \
         --n_reps 10 --timelimit 120 --norel 120 --num_folds 5 --seed 42 \
         --partition short --qos cpu_queues --env corneto-dev2 --modules gurobi python/3.8

SLURM will call the same script in worker mode, e.g.:
    sbatch --time=02:00:00 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G \
      --partition=short --qos=cpu_queues \
      --wrap "bash -l -c 'source $HOME/.bashrc && micromamba activate corneto-dev2 && module load gurobi && module load python/3.8 && python submit_jobs.py --worker --lambd 0.1 --fold 2 --pickle input_data_and_graph.pkl ...'"
"""

import argparse
import pickle
import random
import subprocess
import os
import pandas as pd
import numpy as np
from corneto.methods.signalling.carnival import multi_carnival

def split_inputs_outputs_folds(data, folds, seed=None):
    """
    Splits the input data into a list of (train, test) pairs of dictionaries,
    one pair per fold. For each cell (key in data), its 'output' entries are randomly partitioned.
    """
    if seed is not None:
        random.seed(seed)

    folds_list = []
    for _ in range(folds):
        folds_list.append(({}, {}))  # Each element: (train_dict, test_dict)

    for key, value in data.items():
        input_features = value['input']
        outputs = value['output']
        output_keys = list(outputs.keys())
        total_outputs = len(output_keys)

        # Shuffle and partition the outputs
        random.shuffle(output_keys)
        fold_sizes = [total_outputs // folds] * folds
        remainder = total_outputs % folds
        for i in range(remainder):
            fold_sizes[i] += 1

        assigned = []
        start = 0
        for size in fold_sizes:
            assigned.append(output_keys[start:start + size])
            start += size

        for i, (train_dict, test_dict) in enumerate(folds_list):
            test_keys = assigned[i]
            train_keys = [k for k in output_keys if k not in test_keys]
            train_dict[key] = {
                'input': input_features,
                'output': {k: outputs[k] for k in train_keys}
            }
            test_dict[key] = {
                'output': {k: outputs[k] for k in test_keys}
            }
    return folds_list

def run_worker0(args):
    """
    Worker mode:
      - Load the preprocessed input data and network graph from the pickle file.
      - Split the input data into folds.
      - Run multi-carnival on the training data from the specified fold.
      - Compute the error compared to the ground truth and write it to a file.
    """
    print(f"Running worker for lambda = {args.lambd} and fold = {args.fold}.")

    # Load preprocessed objects from pickle
    with open(args.pickle, "rb") as f:
        input_data, G_pkn = pickle.load(f)

    # Split the input_data into training and test folds
    folds_data = split_inputs_outputs_folds(input_data, args.num_folds, seed=args.seed)
    train_data = folds_data[args.fold][0]

    sols_rec = []
    for i in range(args.n_reps):
        P, Gexp, stats = multi_carnival(G_pkn, train_data, lambd=args.lambd)
        P.solve(solver="GUROBI", TimeLimit=args.timelimit, NoRelHeurTime=args.norel,
                Seed=i, verbosity=0)
        sols_rec.append(P.expr.edge_value.value)
        print(f"Worker: Repetition {i} completed.")

    # Compute the average solution (over n_reps)
    avg_solution = pd.DataFrame(np.mean(sols_rec, axis=0), index=Gexp.E)
    abs_avg_solution = pd.DataFrame(np.mean(np.abs(sols_rec), axis=0), index=Gexp.E)

    # Export
    avg_solution.to_csv(f"avg_sol__reps_{args.n_reps}_lambd_{args.lambd}_fold_{args.fold}.csv")
    avg_solution.to_pickle(f"avg_sol__reps_{args.n_reps}_lambd_{args.lambd}_fold_{args.fold}.pkl")

    # Load the ground truth dataset
    if args.ground_truth is not None:
        if args.ground_truth.endswith(".csv"):
            df_gt = pd.read_csv(args.ground_truth, index_col=0)
        elif args.ground_truth.endswith(".pkl"):
            df_gt = pd.read_pickle(args.ground_truth)
        else:
            raise ValueError("Unknown ground truth file format")

        # Compute the error (sum of mean squared differences)
        err = (df_gt - avg_solution).fillna(1).pow(2).mean(axis=0).sum()
        print(f"Computed error for lambda {args.lambd} and fold {args.fold}: {err}")

        # Write the error to a file
        output_filename = f"error_lambda_{args.lambd}_fold_{args.fold}.txt"
        with open(output_filename, "w") as f:
            f.write(str(err))
    print("Worker job completed")



def run_worker(args):
    """
    Worker mode:
      - Load the preprocessed input data and network graph from the pickle file.
      - Split the input data into folds.
      - Run multi-carnival on the training data from the specified fold.
      - Compute the error compared to the ground truth and write it to a file.

    Special edges that appear in Gexp but not in the reference network (G_pkn)
    are ignored. If abs_val is True, the absolute values of the solution are used.
    """
    print(f"Running worker for lambda = {args.lambd} and fold = {args.fold}.")
    print(f"Input data = {args.pickle}, {args.ground_truth}.")

    # Load preprocessed objects from pickle
    with open(args.pickle, "rb") as f:
        input_data, G_pkn = pickle.load(f)

    # Create a mapping from each edge in G_pkn.E to its position (index)
    pos_map = {edge: idx for idx, edge in enumerate(G_pkn.E)}

    # Split the input_data into training and test folds
    folds_data = split_inputs_outputs_folds(input_data, args.num_folds, seed=args.seed)
    train_data = folds_data[args.fold][0]

    sols_rec = []  # List to store per-repetition solution DataFrames.
    for i in range(args.n_reps):
        P, Gexp, stats = multi_carnival(G_pkn, train_data, lambd=args.lambd)
        P.solve(solver="GUROBI", TimeLimit=args.timelimit, NoRelHeurTime=args.norel,
                Seed=i, verbosity=0)

        # Filter: only consider edges in Gexp that are present in G_pkn.
        edge_positions = []
        filtered_values = []
        for edge, val in zip(Gexp.E, P.expr.edge_value.value):
            if edge in pos_map:
                edge_positions.append(pos_map[edge])
                filtered_values.append(val)

        # Build a DataFrame with the filtered values, using positions as the index.
        df_sol = pd.DataFrame(filtered_values, index=edge_positions)

        # Average duplicate indexes, if any.
        if df_sol.index.duplicated().any():
            df_sol = df_sol.groupby(df_sol.index).mean()

        # Use absolute values if requested.
        #if args.abs_val:
        df_sol = df_sol.abs()

        # Reindex to include the full set of edges from G_pkn (fill missing entries with 0).
        df_sol = df_sol.reindex(range(len(G_pkn.E)), fill_value=0)
        sols_rec.append(df_sol)
        print(f"Worker: Repetition {i} completed.")

    # Compute the average solution over all repetitions.
    avg_solution = sum(sols_rec) / args.n_reps

    # For backward compatibility, compute an "absolute average solution".
    # (If abs_val is True, avg_solution is already nonnegative.)
    abs_avg_solution = avg_solution.abs()

    # Convert the numeric index (positions) back to the actual edge representations.
    avg_solution.index = [G_pkn.E[pos] for pos in avg_solution.index]
    abs_avg_solution.index = [G_pkn.E[pos] for pos in abs_avg_solution.index]

    # Export the average solution.
    avg_solution.to_csv(f"avg_sol__reps_{args.n_reps}_lambd_{args.lambd}_fold_{args.fold}.csv")
    avg_solution.to_pickle(f"avg_sol__reps_{args.n_reps}_lambd_{args.lambd}_fold_{args.fold}.pkl")

    # Load the ground truth dataset if provided.
    if args.ground_truth is not None:
        if args.ground_truth.endswith(".csv"):
            df_gt = pd.read_csv(args.ground_truth, index_col=0)
        elif args.ground_truth.endswith(".pkl"):
            df_gt = pd.read_pickle(args.ground_truth)
        else:
            raise ValueError("Unknown ground truth file format")

        # Compute the error (sum of mean squared differences).
        err = (df_gt - avg_solution).fillna(1).pow(2).mean(axis=0).sum()
        print(f"Computed error for lambda {args.lambd} and fold {args.fold}: {err}")

        # Write the error to a file.
        output_filename = f"error_lambda_{args.lambd}_fold_{args.fold}.txt"
        with open(output_filename, "w") as f:
            f.write(str(err))
    print("Worker job completed")



def submit_jobs(args):
    """
    Submission mode:
      Loop over a list of lambda values and fold indices, and submit each as a separate SLURM job.
      Each job is allocated:
         - 2 hours walltime,
         - 1 node,
         - 1 task with 1 CPU,
         - 16 GB of memory.
      The wrapped command will:
         1. Source $HOME/.bashrc,
         2. Activate the specified micromamba environment,
         3. Optionally load additional modules,
         4. Run the Python worker command.
    """
    # Fixed resource settings.
    time_limit = "16:00:00"
    nodes = "1"
    ntasks = "1"
    cpus_per_task = "8"
    memory = "16G"

    # Optionally include account, partition, and qos if provided.
    account_str = f"--account={args.account} " if args.account else ""
    partition_str = f"--partition={args.partition} " if args.partition else ""
    qos_str = f"--qos={args.qos} " if args.qos else ""

    # Base environment activation command.
    env_activation = f"source $HOME/.bashrc && micromamba activate {args.env}"

    lambdas = [0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.5, 2.0]
    for lam in lambdas:
        for fold in range(args.num_folds):
            # Build a command that loads the environment, modules (if any), and then runs the worker.
            if args.modules:
                mod_loads = " && ".join(f"module load {mod}" for mod in args.modules)
                full_prefix = f"{env_activation} && {mod_loads}"
            else:
                full_prefix = env_activation

            wrap_command = (
                f"bash -l -c '{full_prefix} && python {os.path.basename(__file__)} --worker "
                f"--lambd {lam} --fold {fold} --pickle {args.pickle} --ground_truth {args.ground_truth} "
                f"--n_reps {args.n_reps} --timelimit {args.timelimit} --norel {args.norel} "
                f"--num_folds {args.num_folds} --seed {args.seed}'"
            )

            # Build the final sbatch command.
            cmd = (
                f"sbatch --time={time_limit} --ntasks={ntasks} "
                f"--cpus-per-task={cpus_per_task} --mem={memory} "
                f"{account_str}{partition_str}{qos_str}"
                f"--wrap \"{wrap_command}\""
            )
            print("Submitting job with command:")
            print(cmd)
            subprocess.call(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser(
        description="Submit performance measurement jobs using SLURM with preprocessed input data."
    )
    parser.add_argument("--worker", action="store_true",
                        help="Run in worker mode. If not set, the script runs in submission mode.")
    parser.add_argument("--lambd", type=float, default=0,
                        help="Lambda value (worker mode only).")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index (worker mode only).")
    parser.add_argument("--pickle", type=str, default="input_data_and_graph.pkl",
                        help="Path to the pickle file containing the input dictionary and network graph.")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Path to the ground truth network.")
    parser.add_argument("--n_reps", type=int, default=10,
                        help="Number of repetitions for performance measurement.")
    parser.add_argument("--timelimit", type=int, default=1800,
                        help="Time limit for the solver (in seconds).")
    parser.add_argument("--norel", type=int, default=1000,
                        help="NoRelHeurTime parameter for the solver.")
    parser.add_argument("--num_folds", type=int, default=5,
                        help="Number of folds for cross-validation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for fold splitting.")
    # Optionally specify SLURM account, partition, and QoS.
    parser.add_argument("--account", type=str, default="",
                        help="SLURM account name (if required).")
    parser.add_argument("--partition", type=str, default="",
                        help="SLURM partition name (default: short).")
    parser.add_argument("--qos", type=str, default="cpu_queues",
                        help="SLURM QoS name (default: cpu_queues).")
    # Optionally specify modules to load before running the worker.
    parser.add_argument("--modules", nargs="+", default=[],
                        help="List of modules to load (e.g., gurobi python/3.8).")
    # Specify the micromamba environment to activate.
    parser.add_argument("--env", type=str, default="corneto",
                        help="Name of the micromamba environment to activate (default: corneto-dev2).")
    # Although the pickle file includes the preprocessed data, you may include cells for consistency.
    parser.add_argument("--cells", nargs="+",
                        default=["H1793", "LNCAP", "KRJ1", "HCC1143", "EFO21", "PANC1", "HF2597"],
                        help="List of selected cell lines. (Not used if pickle file is provided.)")
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
    else:
        submit_jobs(args)

if __name__ == "__main__":
    main()
