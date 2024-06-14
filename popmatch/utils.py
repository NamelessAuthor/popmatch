from itertools import islice
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler



def cumulate(l):
    r = []
    for i in l:
        r.append(i)
        yield r


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def get_best_group_from_autorank_results(result):
    rankdf = result.rankdf

    # Find the best method
    best = rankdf.meanrank.argsort().iloc[0]
    best_median =  rankdf['median'].iloc[best]
    methods = rankdf[rankdf.ci_upper >= best_median].index

    return methods.values


def get_best_group_from_dbscan(df, top_col, threshold_col, threshold, is_2d):
    valid = (df[threshold_col] <= threshold)
    values = df[valid][[top_col, threshold_col]].copy()
    values[top_col] = values[top_col].abs()
    if not is_2d:
        values = values[[top_col]]
    i_best = values[top_col].abs().argmin()
    X = values.values
    X = StandardScaler().fit_transform(X)
    
    db = DBSCAN(eps=0.3, min_samples=2).fit(X)
    labels = db.labels_
    print(labels)
    if labels[i_best] == -1:
        return np.array([np.arange(df.shape[0])[valid][i_best]])
    return np.arange(df.shape[0])[valid][np.where(labels == labels[i_best])[0]]

def get_best_A2A(df, top_col, threshold_mask=None):
    values = df[top_col].copy()
    if threshold_mask is not None:
        values[~threshold_mask] += np.inf
    i_best = values.abs().argmin()
    return np.array([i_best])

def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def get_best_group_from_pareto(df, top_col, threshold_col, threshold):
    valid = (df[threshold_col] <= threshold)
    values = df[valid][[top_col, threshold_col]].copy()
    values[top_col] = values[top_col].abs()
    is_best = is_pareto_efficient_simple(values.values)
    return np.arange(df.shape[0])[valid][is_best]