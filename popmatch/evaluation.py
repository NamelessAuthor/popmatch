from .experiment import dict_wrapper

import math
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, ttest_rel


def smd(df, formula):
    f = Formula(formula)
    dm = f.get_matrix_model(df).values
    ms = ModelSpec.from_spec(dm)
    group = str(f.lhs)
    assert(np.unique(df[group]) == np.array([0, 1]))

    smds = []
    for feature in ms.rhs.term_indices:
        if feature == 1:
            continue

        indices = ms.rhs.term_indices[feature]

        if len(indices) == 1:
            # Continuous
            v = df[:, indices[0]]
            v0, v1 = v[df[group] == 0], v[df[group] == 1]
            rn0, rn1 = v0.shape[0] - 1, v1.shape[0] - 1

            denominator = math.sqrt((rn1 * v1.std() ** 2 + rn0 * v0.std() ** 2) / (rn1 + rn0))
            denominator = max(1e-6, denominator)
            smd = abs((v1.mean() - v0.mean()) / denominator)
            v = v1.std() / v0.std()

        else:
            # Categorical
            v = df[:, indices].values
            v0 = v[df[group] == 0]
            v1 = v[df[group] == 1]

            contingency_table = np.array([v0.sum(axis=0), v1.sum(axis=0)])
            x2 = stats.chi2_contingency(contingency_table, correction=False)[0]

            smd = np.sqrt((x2 / v.shape[0]) / (len(indices)))
            v = 0
        smds.append((feature, smd, v))
    return smds


@dict_wrapper('{splitid}_{matching}_smds')
def compute_smd(input_df, data_continuous, data_categorical, data_ordinal, splitid_matching_groups):
    smd = []
    gdf_0 = input_df[splitid_matching_groups == 0]
    gdf_1 = input_df[splitid_matching_groups == 1]
    for c in data_continuous:
        X_0 = gdf_0[c].values
        X_1 = gdf_1[c].values
        n_0 = X_0.shape[0]
        n_1 = X_1.shape[0]
        if n_0 == 0 or n_1 == 0:
            smd.append((c, -1, -1))
            continue 
        pooled_std = np.sqrt(((n_0 - 1) * X_0.std() ** 2 + (n_1 - 1) * X_1.std() ** 2) / (n_0 + n_1 - 2))
        smd.append((c, (np.abs(X_0.mean() - X_1.mean()) / pooled_std), X_1.std() / X_0.std()))

    for c in data_categorical + data_ordinal:
        ct = pd.crosstab(splitid_matching_groups[splitid_matching_groups >= 0], input_df[splitid_matching_groups >= 0][c])
        x2 = chi2_contingency(ct, correction=False)[0]
        smd.append((c, np.sqrt(x2 / (input_df.shape[0] * (min(ct.shape[0], ct.shape[1]) - 1))), 0.))

    smds = pd.DataFrame(smd, columns=['feature', 'smd', 'variance_ratio'])
    return smds


@dict_wrapper('{splitid}_{matching}_n0', '{splitid}_{matching}_n1', '{splitid}_{matching}_target_diff', '{splitid}_{matching}_ind_pvalue', '{splitid}_{matching}_rel_pvalue')
def compute_target_mean_difference(input_y, splitid_matching_groups, splitid_matching_map):
    n0 = (splitid_matching_groups == 0).sum()
    n1 = (splitid_matching_groups == 1).sum()
    if n0 == 0 or n1 == 0:
        return n0, n1, -1
    diff = input_y[splitid_matching_groups == 1].mean() - input_y[splitid_matching_groups == 0].mean()
    ind_res = ttest_ind(input_y[splitid_matching_map['index_pop_0']], input_y[splitid_matching_map['index_pop_1']])
    rel_res = ttest_rel(input_y[splitid_matching_map['index_pop_0']], input_y[splitid_matching_map['index_pop_1']])
    return n0, n1, diff, ind_res.pvalue, rel_res.pvalue

@dict_wrapper('{splitid}_{matching}_ate_diff', '{splitid}_{matching}_outcome_diff', '{splitid}_{matching}_ite_diff')
def compute_synth_metrics(data_true_outcome, data_true_ite, splitid_matching_groups):
    mask_1 = (splitid_matching_groups == 1)
    mask_0 = (splitid_matching_groups == 0)
    ite_diff = data_true_ite[mask_1].mean() - data_true_ite[mask_0].mean()
    outcome_diff = data_true_outcome[mask_1].mean() - data_true_outcome[mask_0].mean()
    ate = data_true_ite + data_true_outcome
    ate_diff = ate[mask_1].mean() - ate[mask_0].mean()
    return ate_diff, outcome_diff, ite_diff


@dict_wrapper('{splitid}_{matching}_n0, {splitid}_{matching}_n1, {splitid}_{matching}_target_diff')
def compute_km_log_rank(input_y, splitid_matching_groups):
    n0 = (splitid_matching_groups == 0).sum()
    n1 = (splitid_matching_groups == 1).sum()
    diff = np.abs(input_y[splitid_matching_groups == 0].mean() - input_y[splitid_matching_groups == 1].mean())
    return n0, n1, diff


@dict_wrapper('input_simulated_split_population_ratio', 'input_simulated_split_target_difference',
              'input_simulated_split_smd', 'input_simulated_split_smd_weight')
def compute_simulation_params(data_df, data_target, data_continuous, data_categorical, data_ordinal, data_population):
    _, n = np.unique(data_population, return_counts=True)
    n_0, n_1 = (n[0] / (n[0] + n[1]), n[1] / (n[0] + n[1]))
    t_0 = data_df.loc[data_population == 0, data_target].mean()
    t_1 = data_df.loc[data_population == 1, data_target].mean()
    smds = compute_smd(data_df, data_continuous, data_categorical, data_ordinal, data_population)
    smd  = max(smds.smd.mean(), 0.1)
    
    if n_0 > n_1:
        n_0, n_1 = n_1, n_0
        t_0, t_1 = t_1, t_0
    diff = t_1 - t_0

    print('Target diff:', diff, 'Target SMD:', smd)
    return [n_0, n_1], diff / 2., smd / 2., diff * 10 / smd

