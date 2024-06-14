from .evaluation import compute_smd
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from .experiment import dict_wrapper
from .cluster import GeneralPurposeClustering
from scipy.sparse import csc_array
from psmpy import PsmPy
from sklearn.metrics import pairwise_distances
import warnings


try:    
    from rpy2.robjects.packages import importr
    import rpy2.robjects as robjects
    import rpy2.robjects.pandas2ri as pandas2ri
    from rpy2.robjects import Formula
    from rpy2.rinterface_lib.na_values import NA_Integer, NA_Character
except:
    import warnings
    warnings.warn('Could not load rpy2, MatchIt methods will not be available.')


@dict_wrapper('{splitid}_population')
def split_populations_with_error(input_df, data_target,
                                 data_continuous_std, data_categorical, data_ordinal,
                                 input_simulated_split_population_ratio,
                                 input_simulated_split_target_difference,
                                 input_simulated_split_smd,
                                 input_simulated_split_smd_weight,
                                 input_random_state=None,
                                 ):

    def loss(df, cluster_ids):
        y_0 = input_df[data_target][cluster_ids == 0].mean()
        y_1 = input_df[data_target][cluster_ids == 1].mean()
        dy = (input_simulated_split_target_difference - (y_1 - y_0)) ** 2

        smds = compute_smd(df, data_continuous_std, data_categorical, data_ordinal, cluster_ids)
        smd = smds.smd.mean()
        dsmd = (input_simulated_split_smd - smd) ** 2
        
        return dy + input_simulated_split_smd_weight * dsmd
    
    def early_stopping(prev_loss, curr_loss):
        return (curr_loss < 10e-4)

    gps = GeneralPurposeClustering(input_simulated_split_population_ratio, loss,
                                   verbose=1, random_state=input_random_state,
                                   early_stopping=early_stopping)

    gps.fit(input_df)
    return gps.cluster_id_


@dict_wrapper()
def split_stats(input_df, data_target,
                data_continuous_std, data_categorical, data_ordinal,
                splitid_population,
                input_simulated_split_target_difference):

    y_0 = input_df[data_target][splitid_population == 0].mean()
    y_1 = input_df[data_target][splitid_population == 1].mean()
    print('Target', input_simulated_split_target_difference, 'Obtained', (y_1 - y_0))

    smds = compute_smd(input_df, data_continuous_std, data_categorical, data_ordinal, splitid_population)
    print('SMD', smds.smd.mean())



@dict_wrapper('input_df', 'input_X', 'input_y', 'input_reversed')
def load_biggest_population(data_df, data_X, data_y, data_population=None):
    reversed = False

    # If data_population is not None, it means that it is a real case. In this situation, we pick
    # the biggest population available to maximize chances to have an interesting problem
    if data_population is not None:
        _, (n_0, n_1) = np.unique(data_population, return_counts=True)
        print(n_0, n_1)
        largest_population = np.argmax([n_0, n_1])
        mask = (data_population == largest_population)
        data_df = data_df[mask].reset_index(drop=True)
        data_X = data_X[mask].reset_index(drop=True)
        data_y = data_y[mask]
        reversed = (largest_population == 1)
    return data_df, data_X, data_y, reversed



@dict_wrapper('input_df', 'input_X', 'input_y', '{splitid}_population', 'input_reversed')
def load_real_problem(data_df, data_X, data_y, data_population=None):
    assert(data_population is not None)

    # We put the smallest population as 0 because this is how matching algo works afterward.
    _, (n_0, n_1) = np.unique(data_population, return_counts=True)
    reversed = False
    if n_1 < n_0:
        data_population = 1 - data_population
        reversed = True

    if hasattr(data_population, 'values'):
        data_population = data_population.values

    return data_df, data_X, data_y, data_population, reversed


@dict_wrapper('{splitid}_psmpy_groups', '{splitid}_psmpy_map') 
def psmpy_match_cv(data_X, splitid_population, splitid_propensity_score, input_random_state):
    pass

@dict_wrapper('{splitid}_psmpy_groups', '{splitid}_psmpy_map') 
def psmpy_match(input_X, splitid_population, splitid_propensity_score, input_random_state):
    df = input_X.copy()
    df['groups'] = splitid_population
    df['index'] = np.arange(input_X.shape[0])

    psm = PsmPy(df, treatment='groups', indx='index', exclude = [], seed=input_random_state)
    psm.logistic_ps(balance=True)
    
    # We give psmpy the score we want and override its interals
    psm.predicted_data['propensity_score'] = splitid_propensity_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        psm.knn_matched(matcher='propensity_score', replacement=False, drop_unmatched=True)

    groups = pd.DataFrame(-np.ones(df.shape[0]), index=df.index)
    groups.loc[psm.matched_ids["index"]] = 0
    groups.loc[psm.matched_ids["matched_ID"]] = 1
    matchmap = pd.DataFrame.from_dict({'index_pop_0': psm.matched_ids["index"],
                                       'index_pop_1': psm.matched_ids["matched_ID"],
                                       'distance': np.zeros(psm.matched_ids["index"].shape[0])})


    return groups.values[:, 0], matchmap


@dict_wrapper('{splitid}_bipartify_groups', '{splitid}_bipartify_map') 
def bipartify(input_df, splitid_population, splitid_propensity_score,
              data_categorical, data_continuous_std,
              n_match=1, feature_weight=0.1, verbose=0):
    """Perform population matching.
    """


    # Categories must be ordered by importance

    # The principle of the algorithm is the following:
    # - We split by cateogrical variables
    # - If the size of the group is reasonable, we perform bipartite match
    # - Else, we continue to break it down using ordinal variables
    # - If we run out of categorical and we are still not good, then do nn matching.
    
    continuous_features = input_df[data_continuous_std]

    match_pop_0 = []
    match_pop_1 = []
    match_dists = []

    # For convenience it is easier to have propensity in the data
    input_df['_propensity'] = splitid_propensity_score
    input_df['_population'] = splitid_population

    if len(data_categorical) > 0:
        iter_categorical = input_df.groupby(data_categorical)
    else:
        iter_categorical = [[None, input_df]]

    for _, gdf in iter_categorical:

        pop = gdf['_population']
        gdf_0, gdf_1 = gdf[pop == 0], gdf[pop == 1]

        if min(gdf_0.shape[0], gdf_1.shape[0]) == 0:
            continue

        if n_match > 1:
            gdf_0 = pd.concat([gdf_0] * n_match)

        if gdf_0.shape[0] < 100 and gdf_0.shape[1] < 100:
            # Work on dense data, it's easier
            ps_dis = pairwise_distances(gdf_0[['_propensity']], gdf_1[['_propensity']])
            fe_dis = pairwise_distances(continuous_features.loc[gdf_0.index].values, continuous_features.loc[gdf_1.index].values)
            dis = ps_dis + feature_weight * fe_dis
            pop_0_idx, pop_1_idx = linear_sum_assignment(dis)
            match_pop_0.append(gdf_0.index[pop_0_idx].values)
            match_pop_1.append(gdf_1.index[pop_1_idx].values)
            match_dists.append(dis[pop_0_idx, pop_1_idx])
            continue

        for i in range(1, 10):
            try:
                # print(i, '/10', gdf_0.shape, gdf_1.shape)
                ps_dis = NearestNeighbors(n_neighbors=5 * i, radius=i)
                ps_dis.fit(gdf_0[['_propensity']])
                ps_dis = ps_dis.radius_neighbors_graph(gdf_1[['_propensity']], mode='distance').T
                ps_dis_coo = ps_dis.tocoo()
                col, row = ps_dis_coo.col, ps_dis_coo.row
                if col.shape[0] == 0:
                    continue
                if (#indices.size == 0 or
                    np.unique(row).shape[0] < ps_dis.shape[0] or
                    np.unique(col).shape[0] < ps_dis.shape[1]):

                    continue

                fe_dis = continuous_features.loc[gdf_0.index[row]].values - continuous_features.loc[gdf_1.index[col]].values
                fe_dis = np.sqrt((fe_dis ** 2).sum(axis=1))
                fe_dis = csc_array((fe_dis, (row, col)))
                dis = ps_dis + feature_weight * fe_dis
                # If all distances are defined, bipartite match stalls. We use hungarian in this case
                #print(dis.count_nonzero(), np.multiply(*dis.shape))
                print(dis.shape, dis.nnz)
                if dis.count_nonzero() == np.multiply(*dis.shape):
                    pop_0_idx, pop_1_idx = linear_sum_assignment(dis.todense())
                    assert(len(pop_0_idx.shape) == len(pop_1_idx.shape))
                else:
                    pop_0_idx, pop_1_idx = min_weight_full_bipartite_matching(dis)
                    assert(len(pop_0_idx.shape) == len(pop_1_idx.shape))

                match_pop_0.append(gdf_0.index[pop_0_idx].values)
                match_pop_1.append(gdf_1.index[pop_1_idx].values)
                match_dists.append(dis[pop_0_idx, pop_1_idx].A1)
                break

            except ValueError as e:
                if str(e) != "no full matching exists":
                    raise
                pass

    # We create a group indicator and return it.
    groups = pd.DataFrame(-np.ones(input_df.shape[0]), index=input_df.index)
    match_pop_0 = np.hstack(match_pop_0)
    match_pop_1 = np.hstack(match_pop_1)
    match_dists = np.hstack(match_dists)
    groups.loc[match_pop_0] = 0
    groups.loc[match_pop_1] = 1

    matchmap = pd.DataFrame.from_dict({'index_pop_0': match_pop_0, 'index_pop_1': match_pop_1, 'distance': match_dists})

    return groups.values[:, 0], matchmap


def bipartify_cv(df, population, propensity,
                 categorical, ordinal, continuous,
                 n_match=1, feature_weight=0.1, verbose=1):
    
    best_groups = None
    best_matchmap = None
    best_smd = None
    best_idx = None

    # Ordinal features must be ordered by decreasing importance.
    for i in range(len(ordinal) + 1):
        groups, matchmap = bipartify(df, population, propensity,
                                            categorical + ordinal[:i], continuous + ordinal[i:],
                                            n_match=n_match, feature_weight=feature_weight,
                                            verbose=verbose)
        mask = (groups >= 0).values
        smd = compute_smd(df[mask], groups.values[mask], continuous, categorical + ordinal)
        smd = smd.smd.mean()
        if best_smd is None or smd < best_smd:
            best_smd = smd
            best_groups = groups
            best_matchmap = matchmap
            best_idx = i

    return best_groups, best_matchmap, best_smd, best_idx


###############################################################################
# MatchIt                                                                     #
###############################################################################

def import_matchit():
    from rpy2.robjects.packages import importr
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects.vectors import StrVector
    import rpy2.robjects as robjects

    package_names = ["MatchIt"]

    names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        robjects.r.options(download_file_method='curl')
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=0)
        utils.chooseCRANmirror(ind=0)
        utils.install_packages(StrVector(names_to_install))

    return importr("MatchIt")


def matchit_match_cv():
    distances = ['glm', 'gam', 'gbm', 'elasticnet', 'rpart',
                 'randomforest', 'nnet', 'cbps', 'bart',
                 'scaled_euclidean', 'robust_mahalanobis',]
    methods = ['nearest', 'optimal', 'full', 'genetic', 'cem',
               'exact', 'subclass']  # 'cardinality',

@dict_wrapper('matchit{splitid}_{distance}_{method}_groups', 'matchit{splitid}_{distance}_{method}_map') 
def matchit_match(input_df, data_continuous, data_categorical,
                  data_ordinal,
                  splitid_population, input_random_state,
                  distance=None, method=None):

    import_matchit()
    matchit = robjects.r['matchit']
    input_df['pop'] = splitid_population

    # convert the data frame from Pandas to R
    with robjects.conversion.localconverter(
        robjects.default_converter + pandas2ri.converter):
        rdf = robjects.conversion.py2rpy(input_df)

    feats = data_continuous + [f'factor({c})' for c in data_categorical + data_ordinal]
    formula = 'pop ~ ' + ' + '.join(feats)

    res = matchit(formula=Formula(formula),
                  data=rdf, method=method,
                  distance=distance)

    rmatch = np.array(res[0])
    # For some reason, the cem method returns a full array with negative values at population
    if method == 'cem':
        rmatch = rmatch[splitid_population == 1]
    idx = np.where(splitid_population == 1)[0]
    assert(rmatch.shape[0] == idx.shape[0])
    if np.issubdtype(rmatch.dtype, np.integer):
        match_pop_1 = idx[rmatch != NA_Integer]
        match_pop_0 = rmatch[rmatch != NA_Integer]
    else:
        match_pop_1 = idx[rmatch != NA_Character]
        match_pop_0 = rmatch[rmatch != NA_Character].astype(int)

    assert(match_pop_0.shape[0] == match_pop_1.shape[0])
    assert(np.unique(match_pop_0).shape[0] == np.unique(match_pop_1).shape[0])

    # We create a group indicator and return it.
    groups = pd.DataFrame(-np.ones(input_df.shape[0]), index=input_df.index)
    groups.loc[match_pop_0] = 0
    groups.loc[match_pop_1] = 1

    matchmap = pd.DataFrame.from_dict({'index_pop_0': match_pop_0, 'index_pop_1': match_pop_1,
                                       'distance': np.zeros(match_pop_0.shape[0])})


    input_df.drop('pop', axis=1, inplace=True)
    return groups.values[:, 0], matchmap