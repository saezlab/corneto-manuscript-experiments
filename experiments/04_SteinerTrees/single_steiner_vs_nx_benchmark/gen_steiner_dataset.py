import corneto as cn
import networkx as nx
import numpy as np
from networkx.algorithms.approximation import steiner_tree
from corneto.methods.steiner import exact_steiner_tree


def generate_large_network(num_nodes, new_edges_per_node, seed=None):
    return nx.barabasi_albert_graph(num_nodes, new_edges_per_node, seed=seed).to_undirected()

def assign_random_weights(graph, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    for u, v in graph.edges():
        graph[u][v]['weight'] = rng.uniform(1, 10)
    return graph

def generate_steiner_tree_nx(graph, terminals):
    return steiner_tree(graph, terminals, weight='weight')

def generate_steiner_tree_cn(graph, terminals, strict=False, solve=True, **options):
    G = cn.Graph.from_networkx(graph)
    P, Gp = exact_steiner_tree(
        G,
        terminals,
        edge_weights=None, # take from weights
        root=None,
        tolerance=1e-3,
        strict_acyclic=strict
    )
    if solve:
        P.solve(**options)
    return P, G, Gp

def perturb_weights(graph, alpha, variance, base_tree=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    new_weights = {}
    for u, v, data in graph.edges(data=True):
        base_weight = data['weight']
        # Decide if this edge will be mutated or slightly perturbed
        if rng.uniform() <= alpha:
            perturbed_weight = np.abs(rng.normal(rng.uniform(1, 10), variance))
        else:
            perturbed_weight = np.abs(rng.normal(base_weight, variance))
        new_weights[(u, v)] = perturbed_weight
    
    # Assign new weights to the graph
    for (u, v), weight in new_weights.items():
        graph[u][v]['weight'] = weight
    
    return graph

def generate_steiner_trees(graph, terminals, N, alpha, variance, strict_acyclic=False, verbosity=0, rng=None, max_seconds=None):
    if rng is None:
        rng = np.random.default_rng()
    steiner_trees = []
    steiner_trees_nx = []
    pert_graphs = []
    for i in range(N):
        print(f"Generating perturbed graph {i+1}/{N}")
        perturbed_graph = perturb_weights(graph.copy(), alpha, variance, base_tree=None, rng=rng)
        w_g = np.array([data['weight'] for u, v, data in perturbed_graph.edges(data=True)])
        steiner_tree_perturbed = generate_steiner_tree_nx(perturbed_graph, terminals)
        total_weight = sum(data['weight'] for u, v, data in steiner_tree_perturbed.edges(data=True))
        print(f"NX Steiner tree: {steiner_tree_perturbed.number_of_nodes()} nodes, {steiner_tree_perturbed.number_of_edges()} edges, total weight {total_weight}")
        ##print(steiner_tree_perturbed.number_of_nodes(), steiner_tree_perturbed.number_of_edges(), total_weight)
        cfg = {"solver": "GUROBI", "IntegralityFocus": 1, "verbosity": verbosity}
        if max_seconds is not None:
            cfg["TimeLimit"] = max_seconds
        print(cfg)
        print(f"Strict acyclic: {strict_acyclic}")
        P, G_st, G_p = generate_steiner_tree_cn(perturbed_graph, terminals, strict=strict_acyclic, **cfg)
        vec = (np.abs(P.expr.flow.value[:G_st.shape[1]]) >= 1e-6).astype(int)
        G_cn_steiner = G_st.edge_subgraph(np.flatnonzero(vec))
        sel_terminals = set(G_cn_steiner.V).intersection(terminals)
        print(f"N. terminals in cn steiner: {len(sel_terminals)}")
        if len(sel_terminals) != len(terminals):
            print("CN Steiner tree does not contain all terminals")
        cn_cost = w_g.T @ vec
        cn_num_edges = np.sum(vec)
        print(f"Corneto Steiner tree: {cn_num_edges} edges, total weight {cn_cost}")
        steiner_trees_nx.append(steiner_tree_perturbed)
        steiner_trees.append(vec)
        pert_graphs.append(perturbed_graph)
    return steiner_trees_nx, np.array(steiner_trees), pert_graphs


# Generate samples for a given steiner
def noisy_sample(graph, variance, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    new_weights = {}
    for u, v, data in graph.edges(data=True):
        base_weight = data['weight']
        # Decide if this edge will be mutated or slightly perturbed
        perturbed_weight = np.abs(rng.normal(base_weight, variance))
        new_weights[(u, v)] = perturbed_weight

    g = graph.copy()
    # Assign new weights to the graph
    for (u, v), weight in new_weights.items():
        g[u][v]['weight'] = weight
    return g


def main():
    import argparse
    import os
    import pickle
    import json

    parser = argparse.ArgumentParser(description="Generate steiner tree dataset")
    parser.add_argument("--num_nodes", type=int, default=200)
    parser.add_argument("--new_edges_per_node", type=int, default=2)
    parser.add_argument("--num_terminals", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--variance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_seconds", type=int, default=None)
    parser.add_argument("--strict_acyclic", action="store_true")
    parser.add_argument("--verbosity", type=int, default=0)
    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(0, 2**32-1))
    print("Seed: ", args.seed)

    if args.output_dir is None:
        folder_name = f"dataset_nn{args.num_nodes}_ne{args.new_edges_per_node}_nt{args.num_terminals}_ns{args.num_samples}_a{args.alpha}_v{args.variance}_s{args.seed}"
        if os.path.exists(folder_name):
            raise ValueError("Output folder already exists")
        os.makedirs(folder_name)
        args.output_dir = folder_name
    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)


    rng = np.random.default_rng(seed=args.seed)
    graph = generate_large_network(args.num_nodes, args.new_edges_per_node, seed=args.seed)
    graph = assign_random_weights(graph, rng=rng)
    G = cn.Graph.from_networkx(graph)

    G.save(os.path.join(args.output_dir, "cn_graph"))
    nx.write_pajek(graph, os.path.join(args.output_dir, "nx_graph.pajek"))
    
    terminals = rng.choice(list(graph.nodes()), size=args.num_terminals, replace=False).tolist()
    print("Graph generated with {} nodes and {} edges".format(graph.number_of_nodes(), graph.number_of_edges()))
    print("Terminals: ", terminals)

    steiner_trees_nx, steiner_trees, pert_graphs = generate_steiner_trees(graph, terminals, args.num_samples, args.alpha, args.variance, strict_acyclic=args.strict_acyclic, verbosity=args.verbosity, rng=rng)
    for i in range(len(steiner_trees_nx)):
        #nx.write_pajek(steiner_trees_nx[i], os.path.join(args.output_dir, f"nx_steiner_tree_sol_{i}.pajek"))
        nx.write_pajek(pert_graphs[i], os.path.join(args.output_dir, f"weighted_graph_{i}.pajek"))

    # Save numpy steiner_trees array as npz
    np.savez_compressed(os.path.join(args.output_dir, "cn_steiner_trees.npz"), steiner_trees=steiner_trees)

    output_file = os.path.join(args.output_dir, "config.json")

    # Create a dictionary to store the data
    data = {
        "config": {
            "num_nodes": args.num_nodes,
            "new_edges_per_node": args.new_edges_per_node,
            "num_terminals": args.num_terminals,
            "num_samples": args.num_samples,
            "alpha": args.alpha,
            "variance": args.variance,
            "seed": args.seed,
            "output_dir": args.output_dir,
            "terminals": terminals
        },
        "lib_version": {
            "corneto": cn.__version__,
            "networkx": nx.__version__,
            "numpy": np.__version__
        }
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    # Benchmark multi-condition



if __name__ == "__main__":
    main()
