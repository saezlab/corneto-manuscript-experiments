#!/usr/bin/env python
import argparse
import subprocess
import os
import json
import itertools
import numpy as np
import pandas as pd
import corneto as cn
from sklearn.metrics import roc_auc_score

def data_split(df, test_fraction=0.2, random_state=None):
    if not 0 <= test_fraction <= 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    n_rows, n_cols = df.shape

    # Initialize a random number generator.
    rs = np.random.RandomState(random_state)

    # Determine the number of test entries per row.
    n_test = int(round(n_cols * test_fraction))
    if test_fraction > 0 and n_test == 0:
        n_test = 1
    if test_fraction < 1 and n_test >= n_cols:
        n_test = n_cols - 1

    # Build a boolean mask of shape (n_rows, n_cols) indicating test entries.
    test_mask = np.zeros((n_rows, n_cols), dtype=bool)
    for i in range(n_rows):
        # Randomly select columns for test in the i-th row.
        test_cols = rs.choice(n_cols, size=n_test, replace=False)
        test_mask[i, test_cols] = True

    # Create the training and test DataFrames.
    df_train = df.mask(test_mask)
    df_test = df.where(test_mask)

    # Extract the test entries (row, col, value).
    test_row_idx, test_col_idx = np.where(test_mask)
    test_values = df.to_numpy()[test_row_idx, test_col_idx]
    test_entries = pd.DataFrame({
        'row': df.index[test_row_idx],
        'col': df.columns[test_col_idx],
        'value': test_values
    })

    # Extract the training entries.
    train_row_idx, train_col_idx = np.where(~test_mask)
    train_values = df.to_numpy()[train_row_idx, train_col_idx]
    train_entries = pd.DataFrame({
        'row': df.index[train_row_idx],
        'col': df.columns[train_col_idx],
        'value': train_values
    })

    return df_train, df_test, train_entries, test_entries

def corneto_multi_fba(G, nflows=1):
    lb = np.array(G.get_attr_from_edges("default_lb"))
    ub = np.array(G.get_attr_from_edges("default_ub"))
    return cn.opt.Flow(G, lb=lb, ub=ub, values=True, n_flows=nflows)

def run_flow_optimization(
    df_data,
    G,
    test_frac=0.80,
    l2_reg=1e-3,
    lambd=0,
    noise=0.10,
    random_state=42,
    solver="GUROBI",
    mip_focus=1,
    heuristics=0.20,
    verbosity=1,
    mip_gap=0.01,
    time_limit=300,
):
    # Split the data into training and test sets.
    df_train, df_test, _, _ = data_split(df_data, test_fraction=test_frac, random_state=random_state)

    # Use a reproducible random number generator for noise.
    rng = np.random.RandomState(random_state)
    df_train = df_train + rng.normal(scale=noise, size=df_train.shape)

    # Build the optimization model with one flow per training sample.
    P = corneto_multi_fba(G, nflows=df_train.shape[0])
    for i in range(df_train.shape[0]):
        y = df_train.iloc[i].values
        idx = np.flatnonzero(~np.isnan(y))
        print(f"Training data for sample {i}: {len(idx)} entries")
        # Add the data fitting objective for available (non‑NaN) entries.
        P.add_objectives((P.expr.flow[idx, i] - y[idx]).norm(p=2).sum())
        # Add regularization for this sample.
        P.add_objectives(P.expr.flow[:, i].norm(p=2), weights=l2_reg)

    # Add extra constraints/objectives.
    P += cn.opt.Indicator(P.expr.flow)
    P += cn.opt.linear_or(P.expr._flow_i, axis=1, varname="Y")
    P.add_objectives(sum(P.expr.Y), weights=lambd)

    # Solve the optimization problem.
    P.solve(
        solver=solver,
        MIPFocus=mip_focus,
        Heuristics=heuristics,
        verbosity=verbosity,
        MIPGap=mip_gap,
        TimeLimit=time_limit,
    )

    # Construct the prediction DataFrame.
    # (Assuming P.expr.flow.value is (n_fluxes, n_samples); we transpose to (n_samples, n_fluxes))
    df_pred = pd.DataFrame(P.expr.flow.value, index=df_train.columns, columns=df_train.index).T
    return df_pred, P, df_train, df_test

def impute_mean(df):
    return df.fillna(df.mean())


def get_experiment_df():
    noise_values = [0.0, 0.01, 0.10, 0.5, 1.0]
    test_frac_values = [0.20, 0.40, 0.60, 0.80]
    lambd_values = [0, 1e-3, 1e-2, 1e-1, 1.0]
    return pd.DataFrame(
        list(itertools.product(noise_values, test_frac_values, lambd_values)),
        columns=["noise", "test_frac", "lambd"]
    )

def submit_jobs(exp_df, max_tasks=10000):
    for idx, row in exp_df.iterrows():
        if idx >= max_tasks:
            print("Max. num. of tasks reached")
            break
        noise = row["noise"]
        test_frac = row["test_frac"]
        lambd = row["lambd"]
        cmd = [
            "sbatch",
            "--export=ALL,NOISE={noise},TEST_FRAC={test_frac},LAMBD={lambd},TASK_ID={task_id}".format(
                noise=noise, test_frac=test_frac, lambd=lambd, task_id=idx
            ),
            "run_single_experiment.sh"
        ]
        print("Submitting job:", " ".join(cmd))
        try:
            # capture_output=True and text=True allow us to capture stdout and stderr as strings
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Job submitted successfully. Output:", result.stdout)
        except FileNotFoundError as fnf_error:
            # This error is raised if the 'sbatch' command is not found
            print(f"Error: 'sbatch' command not found for job {idx}. Please ensure it is installed and in your PATH.")
            print("Details:", fnf_error)
            continue
        except subprocess.CalledProcessError as cpe:
            # This error is raised if the command returns a non-zero exit status
            print(f"Error: 'sbatch' failed for job {idx}.")
            print("Exit Code:", cpe.returncode)
            print("Error Output:", cpe.stderr)
            continue
        except Exception as e:
            # Catch any other exceptions that might occur
            print(f"An unexpected error occurred for job {idx}: {e}")
            continue

def run_single_experiment(noise, test_frac, lambd, task_id, df_data, G):
    df_pred, P, df_train, df_test = run_flow_optimization(
        df_data,
        G,
        test_frac=test_frac,
        l2_reg=1e-3,
        lambd=lambd,
        noise=noise,
        random_state=42,
        solver="GUROBI",
        mip_focus=1,
        heuristics=0.20,
        verbosity=1,
        mip_gap=0.01,
        time_limit=7200,
    )

    # Compute RMSE only for the test entries (ignoring NaNs).
    rmse = np.sqrt(np.nanmean((df_test - df_pred) ** 2))
    baseline = impute_mean(df_train)
    rmse_impute = np.sqrt(np.nanmean((df_test - baseline) ** 2))

    y_pred_baseline = df_train.abs().mean(axis=0).fillna(0)
    y_pred = df_pred.abs().mean(axis=0)
    # Use the original data as ground truth (convert to binary: nonzero -> 1, zero -> 0).
    y_true_binary = (df_data.abs().mean(axis=0) > 0).astype(int).values
    #auc_roc_baseline = roc_auc_score(y_true_binary, y_pred_baseline)
    #auc_roc = roc_auc_score(y_true_binary, y_pred)

    result = {
        "task_id": task_id,
        "noise": noise,
        "test_frac": test_frac,
        "lambd": lambd,
        "rmse": rmse,
        "rmse_impute": rmse_impute
    }
    os.makedirs("results", exist_ok=True)
    result_path = os.path.join("results", f"result_{task_id}.json")
    with open(result_path, "w") as f:
        json.dump(result, f)
    print(f"Saved result to {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="flux_sample.csv", help="Path to the flux data CSV file")
    parser.add_argument("--exp", type=str, default="experiments.csv", help="Path to the experiments CSV file")
    parser.add_argument("--graph", type=str, default="mitocore.pkl.xz", help="Path to the graph file")
    parser.add_argument("--submit", action="store_true", help="Submit batch jobs")
    parser.add_argument("--noise", type=float, default=0.10, help="Noise level for experiment")
    parser.add_argument("--test_frac", type=float, default=0.20, help="Test fraction for splitting data")
    parser.add_argument("--lambd", type=float, default=0, help="Lambda regularization parameter")
    parser.add_argument("--task_id", type=int, default=0, help="Task ID for the experiment")
    parser.add_argument("--max_tasks", type=int, default=1000, help="Max num of experiments to run")
    args = parser.parse_args()

    print(args)

    G = cn.Graph.load(args.graph)
    df_data = pd.read_csv(args.data, index_col=0)

    if args.submit:
        df_exp = pd.read_csv(args.exp, index_col=0)
        submit_jobs(df_exp, max_tasks=args.max_tasks)
    else:
        if (args.noise is None or args.test_frac is None or
            args.lambd is None or args.task_id is None):
            parser.error("Missing parameters for single experiment run")
        run_single_experiment(args.noise, args.test_frac, args.lambd, args.task_id, df_data, G)
