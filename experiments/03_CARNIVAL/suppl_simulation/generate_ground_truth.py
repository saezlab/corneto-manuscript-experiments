#!/usr/bin/env python
"""
generate_ground_truth.py

Generate the ground truth dataset and save preprocessed data (input dictionary and network Graph).
This script:
  1. Loads and processes the PANACEA data.
  2. Builds the input dictionary (using selected cells and drug) and loads the network graph.
  3. Saves the processed input data and Graph to a pickle file.
  4. Runs the multi-carnival method multiple times to generate and average a ground truth dataset.
  5. Saves the ground truth dataset to a CSV file.

Usage example:
    python generate_ground_truth.py --panacea GSE186341-PANACEA.tsv.xz \
        --network network_collectri.sif --drug PONATINIB \
        --cells H1793 LNCAP KRJ1 HCC1143 EFO21 PANC1 HF2597 \
        --n_reps 10 --timelimit 120 --norel 120 --gt_lambda 0.1 \
        --output ground_truth.csv --pickle input_data_and_graph.pkl
"""

import argparse
import pickle
import corneto as cn
import pandas as pd
import numpy as np
from corneto.methods.signalling.carnival import multi_carnival

def load_panacea_data(panacea_file):
    """
    Load the PANACEA data and perform initial processing.
    """
    df = pd.read_csv(panacea_file, sep='\t')
    df['drug'] = df['obs_id'].str.extract(r'_(.*?)_v')
    df['cell'] = df['obs_id'].str.extract(r'^([^_]*)')
    df['sign'] = np.sign(df['act'])
    return df

def get_measurements(cell, drug, df, resource="dorothea", pipeline="NA+deseq2",
                     statparam="stat", padj=0.05, as_dict=True):
    """
    Extract measurements (e.g., stats) for a given cell and drug.
    """
    df_r = df[
        (df.drug.str.upper() == drug.upper()) &
        (df.cell.str.upper() == cell.upper()) &
        (df.resource == resource) &
        (df.pipeline == pipeline) &
        (df.statparam == statparam) &
        (df.padj <= padj)
    ]
    if as_dict:
        return df_r[["items", "sign"]].set_index("items").to_dict()["sign"]
    return df_r

def build_input_data(selected_cells, selected_drug, df, network_file):
    """
    Build the input data dictionary and load the network graph.
    The input dictionary has one key per cell, with:
      - 'input': pre-defined targets (here, for Ponatinib)
      - 'output': measurements from the PANACEA data.
    """
    input_data = {}
    # Load the network from the SIF file (the column order is assumed to be [source, weight, target])
    G_pkn = cn.Graph.from_sif(network_file, has_header=True, column_order=[0, 2, 1])

    # Define targets for Ponatinib (you may update this list if needed)
    ponatinib_targets = [
        "BCR", "ABL", "VEGFR", "PDGFRA_PDGFRB", "FGFR1", "FGFR2",
        "FGFR3", "FGFR4", "EPH", "SRC", "KIT", "RET", "TIE2", "FLT3"
    ]
    targets = {v: -1 for v in G_pkn.V if v in ponatinib_targets}

    for cell in selected_cells:
        input_data[cell] = {
            "input": targets,
            "output": get_measurements(cell, selected_drug, df)
        }
    return input_data, G_pkn


def generate_ground_truth0(input_data, G_pkn, n_reps, timelimit, norel, gt_lambda, abs=True):
    """
    Run multi-carnival n_reps times to generate ground truth.

    For each repetition:
      - Solve using multi-carnival.
      - Build a DataFrame from P.expr.edge_value.value with index Gexp.E.
      - If duplicate (possibly multi-) indexes exist, average them.
      - If abs=True, take absolute values.
      - Reindex to the full set of edges from G_pkn.E (missing edges get 0).
      - Update a union of edges that appear in Gexp.

    Finally, average the solutions elementwise over all repetitions and
    return only the rows (edges) that appeared in any Gexp, preserving the order
    from G_pkn.E.
    """
    sols = []
    reactions_union = set()

    for i in range(n_reps):
        P, Gexp, stats = multi_carnival(G_pkn, input_data, lambd=gt_lambda)
        P.solve(solver="GUROBI", TimeLimit=timelimit, NoRelHeurTime=norel, Seed=i, verbosity=1)

        # Build DataFrame with experimental edges as index.
        df_sol = pd.DataFrame(P.expr.edge_value.value, index=Gexp.E)
        # Average duplicate indexes if needed.
        if df_sol.index.duplicated().any():
            df_sol = df_sol.groupby(df_sol.index).mean()
        # Use absolute values if requested.
        if abs:
            df_sol = df_sol.abs()

        # Update union of edges seen in Gexp.
        reactions_union.update(df_sol.index)
        # Align the solution to the full set of edges in G_pkn (fill missing with 0).
        df_sol = df_sol.reindex(G_pkn.E, fill_value=0)
        sols.append(df_sol)
        print(f"Repetition {i} completed.")

    # Average over repetitions elementwise.
    df_gt = sum(sols) / n_reps

    # Select only those edges that appeared in any Gexp, preserving order from G_pkn.E.
    final_index = [edge for edge in G_pkn.E if edge in reactions_union]
    return df_gt.loc[final_index]


def generate_ground_truth(input_data, G_pkn, n_reps, timelimit, norel, gt_lambda, abs_val=True):
    """
    Run multi-carnival n_reps times to generate ground truth.

    For each repetition:
      - Solve using multi-carnival.
      - Build a DataFrame from P.expr.edge_value.value with an index given
        by the position of each edge in G_pkn.E.
      - If duplicate (possibly multi-) indexes exist, average them.
      - If abs_val=True, take absolute values.
      - Reindex to the full set of positions (i.e. range(len(G_pkn.E))) from G_pkn.E
        (missing edges get 0).
      - Update a union of positions that appear in the current Gexp.

    Special edges that appear in Gexp but not in G_pkn are ignored.

    Finally, average the solutions elementwise over all repetitions and
    return only the rows (edges) that appeared in any Gexp, preserving the order
    from G_pkn.E.
    """
    sols = []
    union_positions = set()

    # Create a mapping from each edge in G_pkn.E to its position (index).
    pos_map = {edge: idx for idx, edge in enumerate(G_pkn.E)}

    for i in range(n_reps):
        # Run multi-carnival to get a candidate solution.
        P, Gexp, stats = multi_carnival(G_pkn, input_data, lambd=gt_lambda)
        P.solve(solver="GUROBI", TimeLimit=timelimit, NoRelHeurTime=norel, Seed=i, verbosity=1)

        # Filter edges from Gexp: ignore any edge that is not in G_pkn.
        edge_positions = []
        filtered_values = []
        for edge, value in zip(Gexp.E, P.expr.edge_value.value):
            if edge in pos_map:
                edge_positions.append(pos_map[edge])
                filtered_values.append(value)

        # Build DataFrame with positions as the index.
        df_sol = pd.DataFrame(filtered_values, index=edge_positions)

        # If there are duplicate positions, average the values.
        if df_sol.index.duplicated().any():
            df_sol = df_sol.groupby(df_sol.index).mean()

        # Use absolute values if requested.
        if abs_val:
            df_sol = df_sol.abs()

        # Update the union of positions encountered.
        union_positions.update(df_sol.index)

        # Align the solution to the full set of edge positions from G_pkn.E (fill missing with 0).
        df_sol = df_sol.reindex(range(len(G_pkn.E)), fill_value=0)
        sols.append(df_sol)
        print(f"Repetition {i} completed.")

    # Average over all repetitions elementwise.
    df_gt = sum(sols) / n_reps

    # Select only those positions that appeared in any repetition,
    # preserving the original order from G_pkn.E.
    final_positions = [pos for pos in range(len(G_pkn.E)) if pos in union_positions]
    # Convert positions back to the actual edge representations.
    final_edges = [G_pkn.E[pos] for pos in final_positions]

    # Subset the DataFrame to these positions and update the index.
    df_gt = df_gt.loc[final_positions]
    df_gt.index = final_edges
    return df_gt



def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth dataset and save additional processed data (input dictionary and Graph)."
    )
    parser.add_argument("--panacea", type=str, default="GSE186341-PANACEA.tsv.xz",
                        help="Path to the PANACEA data file.")
    parser.add_argument("--network", type=str, default="network_collectri.sif",
                        help="Path to the network SIF file.")
    parser.add_argument("--drug", type=str, default="PONATINIB",
                        help="Selected drug.")
    parser.add_argument("--cells", nargs="+",
                        default=["H1793", "LNCAP", "KRJ1", "HCC1143", "EFO21", "PANC1", "HF2597"],
                        help="List of selected cell lines.")
    parser.add_argument("--n_reps", type=int, default=10,
                        help="Number of repetitions for ground truth generation.")
    parser.add_argument("--timelimit", type=int, default=120,
                        help="Time limit for the solver (in seconds).")
    parser.add_argument("--norel", type=int, default=120,
                        help="NoRelHeurTime parameter for the solver.")
    parser.add_argument("--gt_lambda", type=float, default=0.1,
                        help="Lambda value for ground truth generation.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file for the ground truth dataset.")
    # Set default=None so we can dynamically build the pickle filename
    parser.add_argument("--pickle", type=str, default=None,
                        help="Output pickle file for the processed input data and network Graph.")

    args = parser.parse_args()

    # If user didn't supply a pickle filename, build one based on --n_reps and --gt_lambda
    if not args.pickle:
        args.pickle = f"dataset_{args.n_reps}x_lambd{args.gt_lambda}.pkl"

    if not args.output:
        args.output = f"ground_truth_{args.n_reps}x_lambd{args.gt_lambda}"

    print("Loading PANACEA data...")
    df_panacea = load_panacea_data(args.panacea)

    print("Building input data and loading network graph...")
    input_data, G_pkn = build_input_data(args.cells, args.drug, df_panacea, args.network)

    # Save the processed input data and graph to a pickle file for later use
    with open(args.pickle, "wb") as f:
        pickle.dump((input_data, G_pkn), f)
    print(f"Processed input data and graph saved to '{args.pickle}'.")

    print(f"Generating ground truth dataset (n_reps={args.n_reps}, lambda={args.gt_lambda}, t.limit={args.timelimit})...")
    df_gt = generate_ground_truth(input_data, G_pkn, args.n_reps, args.timelimit, args.norel, args.gt_lambda)

    print("Saving ground truth dataset to", args.output)
    df_gt.to_csv(args.output + ".csv")
    df_gt.to_pickle(args.output + ".pkl")

    print("Ground truth generation completed.")

if __name__ == "__main__":
    main()
