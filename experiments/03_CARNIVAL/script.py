import argparse
import corneto as cn
import pandas as pd 
import numpy as np
import pickle
import time

from corneto.methods.carnival import preprocess_graph, milp_carnival
from corneto.methods.signaling import create_flow_graph, signflow
from corneto.methods.carnival import create_flow_carnival, create_flow_carnival_v2, create_flow_carnival_v3
from corneto.methods import expand_graph_for_flows


def filter_df(
    cells,
    drugs,
    df, 
    resource = "dorothea", 
    pipeline = "NA+deseq2", 
    statparam = "stat", 
    status = "unfiltered", 
    padj = 0.05
):
    c = [c.upper() for c in cells]
    d = [d.upper() for d in drugs]
    dff = df[
        (df.cell.str.upper().isin(c)) & 
        (df.drug.str.upper().isin(d)) & 
        (df.resource == resource) & 
        (df.pipeline == pipeline) &
        (df.statparam == statparam) &
        (df.status == status) &
        (df.padj <= padj)
    ]
    return dff

#   df_conditions = filter_df(selected_cells, [selected_drug], df)

def get_measurements(cell, drug, df, resource = "dorothea", pipeline = "NA+deseq2", statparam = "stat", padj = 0.05, as_dict=True):
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

def postprocess(G, edge_var, inputs, outputs):
    sel_edges = set(np.flatnonzero(np.abs(edge_var) > 0))
    exclude_edges = set()
    G_edges = G.E
    for eidx in sel_edges:
        s, t = G_edges[eidx]
        s = list(s)
        t = list(t)
        if len(s) == 0 or len(t) == 0 or s[0].startswith("_") or t[0].startswith("_"):
            exclude_edges.add(eidx)
    sel_edges = list(sel_edges.difference(exclude_edges))
    G_sol = G.edge_subgraph(sel_edges)
    sel_edges = set(sel_edges)
    G_solp, p_inputs, p_outputs = preprocess_graph(G_sol, inputs, outputs)
    return G_solp
    
def score(G, edge_var, vertex_var, inputs, outputs):
    # Manual score a solution to avoid impl. specific differences
    G_sol = postprocess(G, edge_var, inputs, outputs)
    err_outputs = 0
    Vsol = list(G_sol.V)
    total = 0
    V = list(G.V)
    for k, v in outputs.items():
        if k not in Vsol:
            err_outputs += abs(v)
        else:
            err_outputs += abs(vertex_var[V.index(k)] - v)
        total += abs(v)
    return err_outputs/total, G_sol.shape[1]

def prune(c_data, G_pkn):
    all_inputs, all_outputs = set(), set()
    for k, v in c_data.items():
        for ki, (ti, vi) in v.items():
            if isinstance(ki, str):
                if ti == 'P':
                    all_inputs.add(ki)
                else:
                    all_outputs.add(ki)
            else:
                print(ki, ti, vi)
    
    V = set(G_pkn.vertices)
    c_inputs = V.intersection(all_inputs)
    c_outputs = V.intersection(all_outputs)
    print(f"{len(c_inputs)}/{len(all_inputs)} inputs mapped to the graph")
    print(f"{len(c_outputs)}/{len(all_outputs)} outputs mapped to the graph")
    print(f"Pruning the graph with size: V x E = {G_pkn.shape}...")
    Gp = G_pkn.prune(list(c_inputs), list(c_outputs))
    print(f"Finished. Final size: V x E = {Gp.shape}.")
    return Gp, c_inputs, c_outputs

def convert_input_dict(dataset):
    conditions = dict()
    for k, exp in dataset.items():
        d_k = dict()
        for inp, val in exp["input"].items():
            d_k[inp] = ('P', val)
        for outp, val in exp["output"].items():
            d_k[outp] = ('M', val)
        conditions[k] = d_k
    return conditions

def single_carnival(G, dataset, beta=0.25, max_time=7200, norel=0, seed=0):
    all_edges = np.zeros(G.shape[1])
    selected_edges_per_sample = []
    scores = []
    E = list(G.E)
    problems = []
    graphs = []
    for k, exp in dataset.items():
        sol_edges = np.zeros(G.shape[1])
        exp_inputs, exp_outputs = exp["input"], exp["output"]
        Gp, cp_inputs, cp_outputs = preprocess_graph(G, exp_inputs, exp_outputs)
        print(k, Gp.shape, len(cp_inputs), len(cp_outputs))
        P = milp_carnival(Gp, cp_inputs, cp_outputs, beta_weight=beta)
        P.solve(solver="GUROBI", IntegralityFocus=1, TimeLimit=max_time, NoRelHeurTime=norel, Seed=seed, verbosity=0)
        for o in P.objectives:
            print(o.value)
        s = score(Gp, P.expr.edge_values.value, P.expr.vertex_values.value, cp_inputs, cp_outputs)
        scores.append(s)
        problems.append(P)
        # Select the edges
        E_gp = Gp.E
        sel_edges = np.flatnonzero(np.abs(P.expr.edge_values.value)>0)
        sel_edges = [E_gp[idx] for idx in sel_edges]
        for i, e in enumerate(E):
            if e in sel_edges:
                all_edges[i] += 1
                sol_edges[i] = 1
        selected_edges_per_sample.append(sol_edges)
        graphs.append(Gp)
    return problems, scores, all_edges, selected_edges_per_sample, graphs


def multi_carnival(G, dataset, lambd=0.25, norel=0, max_time=7200, seed=0):
    d = convert_input_dict(dataset)
    G_multi, input_multi, output_multi = prune(d, G)
    print(G_multi.shape, len(input_multi), len(output_multi))
    all_v = input_multi.union(output_multi)
    # Remove non reachable so error is 0 if reachable are fit
    # as in carnival single
    d2 = dict()
    for k, v in d.items():
        filtered_dict = {key: value for key, value in v.items() if key in all_v}
        d2[k] = filtered_dict
    d = d2
    G_multi, input_multi, output_multi = prune(d, G) 
    # Clean non reachable vertices
    G_pkn = create_flow_graph(G_multi, d)
    P = signflow(
        G_pkn,
        d,
        l0_penalty_edges = lambd
    )
    P.solve(solver="GUROBI", IntegralityFocus=1, NoRelHeurTime=norel, Seed=seed, TimeLimit=max_time, verbosity=1)
    valid_edges = set()
    for i, (s, t) in enumerate(G_pkn.E):
        s = list(s)
        t = list(t)
        if len(s)==1 and len(t)==1 and (not s[0].startswith("_")) and (not t[0].startswith("_")):
            valid_edges.add(i)
    all_edges_multi = np.zeros(G_pkn.shape[1])
    sel_edges = np.zeros(G_pkn.shape[1])
    E_multi = list(G_pkn.E)
    scores = []
    for i, (k, v) in enumerate(dataset.items()):
        _, cp_inputs, cp_outputs = preprocess_graph(G_pkn, v["input"], v["output"])
        s = score(G_pkn, P.expr[f"edge_values_{k}"].value, P.expr[f"vertex_values_{k}"].value, cp_inputs, cp_outputs)
        scores.append(s)
        sol_edges = np.flatnonzero(np.abs(P.expr[f"edge_values_{k}"].value) > 0)
        all_edges_multi[sol_edges] += 1
        sel_edges[sol_edges] = 1
    return P, G_pkn, scores, all_edges_multi[list(valid_edges)]


def multi_carnival_flow(G, dataset, acyclic_signal_version=True, lambd=0.25, norel=0, max_time=7200, seed=0):
    d = convert_input_dict(dataset)
    G_multi, input_multi, output_multi = prune(d, G)
    print(G_multi.shape, len(input_multi), len(output_multi))
    all_v = input_multi.union(output_multi)
    exp_list = dict()
    for k, v in dataset.items():
        filtered_in = {key: value for key, value in v["input"].items() if key in all_v}
        filtered_out = {key: value for key, value in v["output"].items() if key in all_v}
        exp_list[k] = {"input": filtered_in, "output": filtered_out}
        print(k, len(filtered_in), len(filtered_out))

    G_exp_e = expand_graph_for_flows(G_multi, exp_list)
    if acyclic_signal_version:
        P = create_flow_carnival_v3(G_exp_e, exp_list, lambd=lambd)
    else:
        P = create_flow_carnival(G_exp_e, exp_list, lambd=lambd)
    P.solve(solver="GUROBI", IntegralityFocus=1, NoRelHeurTime=norel, Seed=seed, TimeLimit=max_time, Threads=4, verbosity=1)
    valid_edges = set()
    for i, (s, t) in enumerate(G_exp_e.E):
        s = list(s)
        t = list(t)
        if len(s)==1 and len(t)==1 and (not s[0].startswith("_")) and (not t[0].startswith("_")):
            valid_edges.add(i)
    all_edges_multi = np.zeros(G_exp_e.shape[1])
    E_multi = list(G_exp_e.E)
    scores = []
    for i, (k, v) in enumerate(dataset.items()):
        _, cp_inputs, cp_outputs = preprocess_graph(G_exp_e, v["input"], v["output"])
        s = score(G_exp_e, P.expr.edge_value.value[:,i], P.expr.vertex_value.value[:,i], cp_inputs, cp_outputs)
        #_, cp_inputs, cp_outputs = preprocess_graph(G_pkn, exp_list[k]["input"], exp_list[k]["output"])
        #s = score(G_pkn, P.expr[f"edge_values_{k}"].value, P.expr[f"vertex_values_{k}"].value, cp_inputs, cp_outputs)
        scores.append(s)
        all_edges_multi[np.flatnonzero(np.abs(P.expr.edge_value.value[:,i]) > 0)] += 1
    return P, G_exp_e, scores, all_edges_multi[list(valid_edges)]


def summary(scores, edge_vector):
    total = 0
    edges = 0
    for err, num_edges in scores:
        total += err
        edges += num_edges
    diff_edges = np.sum(edge_vector > 0)
    return total, diff_edges, diff_edges/edges

def export_result(result_file, problems, cells, graphs, edges=True):
    df_merge_edges = None
    for prob, gprob, scell in zip(problems, graphs, cells):
        # Create DataFrame
        if edges:
            df_result = pd.DataFrame(prob.expr.edge_values.value, index=gprob.E, columns=[scell])
        else:
            df_result = pd.DataFrame(prob.expr.vertex_values.value, index=gprob.V, columns=[scell])
        
        # Handle duplicates by grouping by the index and calculating the mean
        #df_result = df_result.groupby(df_result.index).mean()
        df_result = df_result[~df_result.index.duplicated(keep="first")]
        
        if df_merge_edges is None:
            df_merge_edges = df_result
        else:
            df_merge_edges = pd.concat([df_merge_edges, df_result], axis=1, join="outer")
            
    df_merge_edges = df_merge_edges.fillna(0)
    df_merge_edges.to_csv(result_file)

def main():
    parser = argparse.ArgumentParser(description="Process input parameters.")

    parser.add_argument('graph', type=str, help='PKN file')
    parser.add_argument('dataset', type=str, help='Path to the dataset file (panacea.tsv.xz)')

    parser.add_argument('--beta', type=float, default=0.25, help='Beta weight (default: 0.25)')
    parser.add_argument('--max_time', type=int, default=7200, help='Maximum time (default: 7200)')
    parser.add_argument('--norel_time', type=int, default=0, help='No relaxation time (default: 0)')
    parser.add_argument('--mode', type=str, choices=['single', 'multi'], default='single', help='Mode of operation (single | multi) (default: single)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # add acyclic_flow_version (true, false)
    parser.add_argument('--acyclic_signal_version', action=argparse.BooleanOptionalAction, help='Flow+Acyclic Signal implementation, otherwise is AcyclicFlow (default: False)')
    parser.add_argument('--old', action=argparse.BooleanOptionalAction, help='Old single flow, multi order signal version (default: False)')
    # Boolean option to save the edges and vertex values
    parser.add_argument('--save_values', action=argparse.BooleanOptionalAction, help='Save the values of the edges and vertices (default: False)')
    parser.add_argument('--all_cells', action=argparse.BooleanOptionalAction, help='Use all cells from the dataset instead the default selection (default: False)')


    # Parse arguments
    args = parser.parse_args()

    selected_drug = "PONATINIB"
    selected_cells = ["H1793", "LNCAP", "KRJ1", "HCC1143", "EFO21", "PANC1", "HF2597"]
    ponatinib_targets = ["BCR", "ABL", "VEGFR", "PDGFRA_PDGFRB", "FGFR1", "FGFR2", "FGFR3", "FGFR4", "EPH", "SRC", "KIT", "RET", "TIE2", "FLT3"]

    # Example usage of the arguments
    print(f"Graph: {args.graph}")
    print(f"Dataset: {args.dataset}")
    print(f"Max Time: {args.max_time}")
    print(f"NoRelaxation Time: {args.norel_time}")
    print(f"Mode: {args.mode}")
    print(f"Beta: {args.beta}")
    print(f"Seed: {args.seed}")
    print(f"Flow + Acyclic Signal Version?: {args.acyclic_signal_version}")
    print(f"Old (single flow, multi order signal, dummy nodes) version: {args.old}")

    dataset_name = args.dataset.split('/')[-1].split('.')[0]
    result_name = f"carnival_panacea_{dataset_name}_{args.mode}_b{args.beta}_mt{args.max_time}_norel{args.norel_time}_seed{args.seed}"
    results_file = f"{result_name}.csv"
    print(f"Results file: {results_file}")

    G = cn.Graph.from_sif(args.graph, has_header=True, column_order=[0, 2, 1])
    print(f"Graph: {G.shape}")

    targets = {v: -1 for v in G.V if v in ponatinib_targets}
    print(f"PONATINIB targets (in PKN): {targets}")

    df_panacea = pd.read_csv(args.dataset, sep='\t')
    df_panacea['drug'] = df_panacea['obs_id'].str.extract(r'_(.*?)_v')
    df_panacea['cell'] = df_panacea['obs_id'].str.extract( r'^([^_]*)')
    df_panacea['sign'] = np.sign(df_panacea['act'])

    if args.all_cells:
        selected_cells = None

    if selected_cells is None or len(selected_cells) == 0:
        selected_cells = df_panacea['cell'].unique().tolist()

    input_data = dict()
    for cell in selected_cells:
        d = dict()
        input_data[cell] = d
        d["input"] = targets
        d["output"] = get_measurements(cell, selected_drug, df_panacea)


    print(f"Dataset samples: {input_data.keys()}")
    n_samples = len(input_data)

    t_start = time.time()
    if args.mode == 'single':
        P, scores, edge_vector, list_edges, graphs = single_carnival(G, input_data, max_time=args.max_time, norel=args.norel_time, beta=args.beta, seed=args.seed)
    else:
        if args.old:
            print("Using old version of multi_carnival")
            P, G_mc, scores, edge_vector = multi_carnival(G, input_data, lambd=args.beta, max_time=args.max_time, norel=args.norel_time, seed=args.seed)
        else:
            P, G_mc, scores, edge_vector = multi_carnival_flow(G, input_data, acyclic_signal_version=args.acyclic_signal_version, max_time=args.max_time, norel=args.norel_time, lambd=args.beta, seed=args.seed)
    t_end = time.time()

    summary_score, diff_edges, edge_ratio = summary(scores, edge_vector)
    print(f"Error, num. diff edges, edge ratio: {summary_score}, {diff_edges}, {edge_ratio}")
    print(f"Total time: {t_end - t_start} seconds")

    prop_common_edges = np.sum(edge_vector == n_samples) / np.sum(edge_vector > 0)
    prop_diff_edges = np.sum(edge_vector == 1) / np.sum(edge_vector > 0)
    sparsity = np.sum(edge_vector == 0) / G.shape[1]

    # Save the results with pandas. The results should be a single
    # row with dataset name, num samples, total time, summary score, diff edges, edge ratio
    df = pd.DataFrame({
        'dataset': [args.dataset],
        'num_samples': [len(input_data)],
        'mode': [args.mode],
        'old_multicond': [args.old],
        'beta': [args.beta],
        'max_time': [args.max_time],
        'norel_time': [args.norel_time],
        'seed': [args.seed],
        'acyclic_signal_version': [args.acyclic_signal_version],
        'total_time': [t_end - t_start],
        'total_error': [summary_score],
        'diff_edges': [diff_edges],
        'edge_ratio': [edge_ratio],
        'prop_common_edges': [prop_common_edges],
        'prop_diff_edges': [prop_diff_edges],
        'sparsity': [sparsity]
    })
    # Generate an informative name for the results file based on the input parameters
    # Get the dataset name
    df.to_csv(results_file, index=False)
    
    if args.save_values:
        if args.mode == 'single':
            export_result(f"{result_name}_edges.csv.xz", P, selected_cells, graphs, edges=True)
            export_result(f"{result_name}_vertices.csv.xz", P, selected_cells, graphs, edges=False)
        else:
            if args.old:
                raise NotImplementedError()
            df_edges = pd.DataFrame(P.expr.edge_value.value, index=G_mc.E, columns=selected_cells)
            df_vertices = pd.DataFrame(P.expr.vertex_value.value, index=G_mc.V, columns=selected_cells)
        df_edges.to_csv(f"{result_name}_edges.csv.xz")
        df_vertices.to_csv(f"{result_name}_vertices.csv.xz")

    print("Finished.")

if __name__ == "__main__":
    main()
