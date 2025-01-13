import argparse
import corneto as cn
import pandas as pd
from corneto.methods.steiner import exact_steiner_tree, create_exact_multi_steiner_tree
import os
import json
import numpy as np

def get_prizes(G):
    return {v: G.get_attr_vertex(v).get("prize", 0) for v in G.V}

def eval_sol(G, s):
    total_prizes = sum(list(get_prizes(G).values()))
    Gs = G.edge_subgraph(np.flatnonzero(s))
    sel_prizes = sum(list(get_prizes(Gs).values()))
    return sel_prizes / total_prizes, Gs.shape

def single_condition(samples, lambd, max_time, seed=0):
    sel = np.zeros(samples[0].shape[1])
    total_rec_prize = 0
    results = []
    problems = []
    for i, G in enumerate(samples):
        prizes = get_prizes(G)
        total_prizes = sum(list(prizes.values()))
        print(i, G.shape, total_prizes)
        P, Gc = exact_steiner_tree(G, prizes, edge_weights=lambd)
        P.solve(solver="GUROBI", IntegralityFocus=1, TimeLimit=max_time, verbosity=0, Seed=seed)
        problems.append(P)
        for n, o in zip(["Edge cost", "Prizes"], P.objectives):
            print(f"- {n}:", o.value)
        total_rec_prize += P.objectives[1].value
        s = (P.expr._flow.value[:G.shape[1]] >= 1e-6).astype(int)
        results.append(eval_sol(G, s))
        sel += s
    return problems, results, sel

def multi_condition(samples, lambd, max_time, seed=0):
    prizes_per_condition = dict()
    for i, G in enumerate(samples):
        prizes_per_condition[i] = {v: G.get_attr_vertex(v).get("prize", 0) for v in G.V}
    P = create_exact_multi_steiner_tree(G, prizes_per_condition, lam=lambd)
    P.solve(solver="GUROBI", verbosity=1, IntegralityFocus=1, TimeLimit=max_time, Seed=seed)
    mc_sel_edges = np.zeros(G.shape[1])
    results = []
    for i, G in enumerate(samples):
        s = (P.expr[f"flow{i}"].value[:G.shape[1]] >= 1e-6).astype(int)
        results.append(eval_sol(G, s))
        mc_sel_edges += s
    
    return P, results, mc_sel_edges

def main(dataset, lambd, mode, max_time, seed):
    with open(os.path.join(dataset, "dataset_config.json"), "r") as f:
        config = json.load(f)
        print(config)

    # Import graphs
    samples = []
    for i in range(config["num_samples"]):
        g = cn.Graph.load(os.path.join(dataset, f"graph_sample_{i}.pkl.xz"))
        samples.append(g)
        print(g.shape)

    if mode == "multi":
        P, result, selected_edges = multi_condition(samples, lambd, max_time, seed=seed)
    else:
        P, result, selected_edges = single_condition(samples, lambd, max_time, seed=seed)

    total_edges_across_samples = sum(selected_edges > 0)
    mean_prop_prizes = np.mean([r[0] for r in result])

    df_result = pd.DataFrame({
        'dataset': [dataset], 
        'num_samples': [config["num_samples"]],
        'num_nodes': [config["num_nodes"]],
        'num_terminals': [config["num_terminals"]],
        'num_common_nodes': config["num_common_nodes"],
        'mode': [mode],
        'lambda': [lambd],
        'max_time': [max_time],
        'mean_prop_score': [mean_prop_prizes],
        'total_edges_across_samples': [total_edges_across_samples]
    })
    
    filename = f"{os.path.basename(dataset.strip('/'))}_lambda{lambd}_time{max_time}_{mode}.csv"
    df_result.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Steiner Tree optimization on given dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--lambd", type=float, default=0.01, help="Lambda value for edge weights")
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="single", help="Mode: single or multi")
    parser.add_argument("--max_time", type=int, default=3600, help="Max time limit for optimization")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")

    args = parser.parse_args()
    main(args.dataset, args.lambd, args.mode, args.max_time, args.seed)
