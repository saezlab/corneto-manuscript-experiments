#import cobra
import corneto as cn
import miom
import pandas as pd
import numpy as np
from corneto.methods.metabolism.fba import (
    multicondition_imat,
    mimat,
    mimat0
)
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    jaccard_score, hamming_loss
)

n_splits = 5
gaps = [0.10]
gap = 0.10
eps = 1e-3
threads = 16
alphas = [0.0, 0.1, 1.0, 1.5, 2.0, 2.5, 3.0]
norel = 1800
maxtime = norel*3
round_marginal = False # old test, not used
maxtime_single = 600
seed=0 # default seed, we generated results varying the seed from 0 to 9

def combine(df_data, df_fold):
    dfc = pd.concat([df_data, df_fold])
    dfc = dfc[~dfc.index.duplicated(keep='last')]
    return dfc.loc[df_data.index]

def compute_confusion_matrix(y_true, y_pred):
    TP = TN = FP = FN = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1

    return TP, TN, FP, FN


def create_folds(df, n_splits=n_splits, seed=0):
    # Get the sub-df with at least some values
    folds = []
    df_sub = df[df.abs().sum(axis=1) > 0]
    # Stack the df and split with KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df_sub_stacked = df_sub.stack()
    for idx_train, idx_val in kf.split(df_sub_stacked):
        df_train = df_sub_stacked.iloc[idx_train].unstack()
        df_train = combine(df, df_train)
        df_val = df_sub_stacked.iloc[idx_val].unstack()
        df_val = combine(df, df_val)
        folds.append((df_train, df_val))
    return folds


def run_imat(df_data, verbose=1, threads=threads, eps=eps, maxtime=maxtime):
    imat_sols = []
    for col in df_data.columns:
        # Count total number of nans in column:
        n_val = df_data[col].isna().sum()
        print(f"Column {col}, {n_val} nans")
        w = df_data[col].fillna(0).values
        print("RH:", np.sum(w>0), "RL:", np.sum(w<0), "Unknown:", np.sum(w==0))
        P = mimat(G, w, eps=eps)
        if threads is None:
            P.solve(solver="GUROBI", verbosity=verbose, IntegralityFocus=1, TimeLimit=maxtime, MIPGap=gap)
        else:
            P.solve(solver="GUROBI", verbosity=verbose, IntegralityFocus=1, TimeLimit=maxtime, Threads=threads, MIPGap=gap)
        cum_err = 0
        for i, o in enumerate(P.objectives):
            print(f"Obj{i}: {o.value}")
            cum_err += o.value
            print(f"Acum. error = {cum_err}")
        wp = (np.abs(P.expr.flow.value) >= eps*(1-eps)).astype(int)
        imat_sols.append(wp)
    df_imat_sol = pd.DataFrame(np.array(imat_sols).T, index=df_data.index, columns=df_data.columns)
    return df_imat_sol


df_yeastn = pd.read_csv("yeast-GEM.txt", sep="\t")
model = miom.load_gem("yeast-v8.5.0.miom")
G = cn.Graph.from_miom_model(model)
df_model = df_yeastn.set_index("Rxn name").loc[model.R['id']]
df_data = pd.read_csv('data/input_data.csv').set_index("Rxn name")

df_true = df_data.copy()
folds = create_folds(df_true, n_splits=n_splits, seed=seed)

print("Running multi...")
results = []
for g in gaps:
    for a in alphas:
        print(f"Running multi-cond for gap {g}...")
        for i, (f_train, f_val) in enumerate(folds):
            print(f"Fold {i}, gap={g}, alpha={a}, norel={norel}, eps={eps}, round={round_marginal}")
            w = f_train.fillna(0).values
            print("RH:", np.sum(w>0), "RL:", np.sum(w<0), "Unknown:", np.sum(w==0))
            # NOTE: alpha is "lambda"
            P = multicondition_imat(G, w, eps=eps, alpha=a)
            if threads is None:
                P.solve(solver="GUROBI", verbosity=1, Seed=seed, IntegralityFocus=1, NoRelHeurTime=norel, TimeLimit=maxtime, MIPGap=g)
            else:
                P.solve(solver="GUROBI", verbosity=1, Seed=seed, IntegralityFocus=1, NoRelHeurTime=norel, TimeLimit=maxtime, Threads=threads, MIPGap=g)
            cum_err = 0
            errors = []
            for j, o in enumerate(P.objectives):
                cum_err += o.value
                errors.append(o.value)
            print(f"Total training error = {cum_err}")
            sol=pd.DataFrame((np.abs(P.expr.flow.value) >= eps*(1-eps)).astype(int), index=df_true.index, columns=df_true.columns)
            selected_reactions = (sol.sum(axis=1) > 0).sum()

            val_pos = (df_true[f_train.isna()] > 0).sum()
            val_neg = (df_true[f_train.isna()] < 0).sum()
            print("Validation positive:")
            print(val_pos)
            print("Validation negative:")
            print(val_neg)

            # Compute metrics
            df_data_true = pd.DataFrame(df_true[f_train.isna()].clip(0, 1).stack(), columns=["y_true"])
            df_pred_multi = pd.DataFrame(sol[f_train.isna()].stack(), columns=["y_pred"])
            df_data_pred_multi = df_data_true.join(df_pred_multi)
            # Compute accuracy, f1 score and roc auc score
            acc = accuracy_score(df_data_pred_multi.y_true, df_data_pred_multi.y_pred)
            roc_auc = roc_auc_score(df_data_pred_multi.y_true, df_data_pred_multi.y_pred)
            TP, TN, FP, FN = compute_confusion_matrix(df_data_pred_multi.y_true, df_data_pred_multi.y_pred)
            TPR = TP / (TP + FN) # recall
            PPV = TP / (TP + FP) # precision
            F1 = 2 * ((PPV * TPR) / (PPV + TPR))
            result = {
                'fold': i,
                'val_n_pos': val_pos.sum(),
                'val_n_neg': val_neg.sum(),
                'mip_gap': g,
                'lambda': a, #lambda
                'norel': norel,
                'eps': eps,
                'round': round_marginal,
                'n_rxns': selected_reactions,
                'total_error': cum_err,
                'abs_val_err': np.sum(np.abs(df_data_pred_multi.y_true - df_data_pred_multi.y_pred)),
                'TP': np.round(TP, 3),
                'TN': np.round(TN, 3),
                'FP': np.round(FP, 3),
                'FN': np.round(FN, 3),
                'F1': np.round(F1, 3),
                'TPR_RECALL': np.round(TPR, 3),
                'PPV_PRECISION': np.round(PPV, 3),
                'seed': seed
                #'train_errors': errors,
            }
            print(result)
            results.append(result)
            pd.DataFrame(results).to_csv(f"multi_imat_seed{seed}.csv")
