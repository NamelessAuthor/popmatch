import numpy as np
import pandas as pd
import os
from autorank import autorank, plot_stats, create_report
import sys
from scipy.stats import kendalltau
from matplotlib import pyplot as plt

sys.path.append('../')
from popmatch.utils import get_best_group_from_autorank_results, get_best_group_from_dbscan, get_best_A2A, get_best_group_from_pareto


datasets = [
    'groupon',
    'nhanes',
    'horse',
    'synthetic_7_0',
    'synthetic_7_1',
    'synthetic_7_2',
    'synthetic_7_3',
    'synthetic_7_4',
    'synthetic_7_5',
    'synthetic_7_6',
    'synthetic_7_7',
    'synthetic_7_8',
    'synthetic_7_9',
    'synthetic_7_10',
]

results = []
artificials = []

for dataset in datasets:
    filename = f"{dataset}/results.parquet"
    ar_filename = f"{dataset}/artificial.parquet"
    if not os.path.exists(filename) or not os.path.exists(ar_filename):
        print(f'{dataset} not loaded')
        continue
    result = pd.read_parquet(filename)
    result['dataset'] = dataset
    results.append(result)
    artificial = pd.read_parquet(ar_filename)
    artificial['dataset'] = dataset
    artificials.append(artificial)

results = pd.concat(results)
artificials = pd.concat(artificials)
artificials['split'] = artificials['split_id'].str.extract(r'split(\d+)$')


def _add_bounds(df, values, errors, prefix):
    mins = None
    maxs = None
    gaps = None
    avg_error = None
    if values.shape[0] > 0:     
        mins = values.min()
        maxs = values.max()
        gaps = np.abs(maxs - mins)
        if errors is not None:
            avg_error = np.mean(np.abs(errors))
    df[prefix + 'target_min'] = mins
    df[prefix + 'target_max'] = maxs
    df[prefix + 'target_gap'] = gaps
    df[prefix + 'avg_error'] = avg_error


class Bounds:
    def __init__(self, artificial_df, value_col, error_col,
                 threshold_col, top_col,
                 top_rank='auto', threshold=0.1):
        self.artificial_df = artificial_df.set_index(['dataset', 'split', 'matching']).unstack(level=-1).target.abs()
        self.value_col = value_col
        self.error_col = error_col
        self.threshold_col = threshold_col
        self.threshold = threshold
        self.top_col = top_col
        self.top_rank = top_rank

    def __call__(self, df):

        artificial_df = self.artificial_df.loc[df.group_dataset.iloc[0], :]

        # SMD
        filtered_df = df[df[self.threshold_col] < self.threshold]
        _add_bounds(df, filtered_df[self.value_col].values, filtered_df[self.error_col].values, self.threshold_col)

        # A2A
        # ar_result = autorank(artificial_df[df.matching.unique()].abs(), alpha=0.05, force_mode='nonparametric')
        # ar_best_methods = get_best_group_from_autorank_results(ar_result)
        db_mask = get_best_group_from_dbscan(df, self.top_col, self.threshold_col, 2., True)
        ar_best_methods = df['matching'].iloc[db_mask].values
        filtered_df = df[df.matching.isin(ar_best_methods)]
        filtered_df = filtered_df[filtered_df[self.threshold_col] < self.threshold]
        _add_bounds(df, filtered_df[self.value_col].values, filtered_df[self.error_col].values, f'{self.threshold_col}+{self.top_col}')

        # SMD-A2A
        db_mask = get_best_group_from_dbscan(df, self.top_col, self.threshold_col, self.threshold, False)
        ar_best_methods = df['matching'].iloc[db_mask].values
        filtered_df = df[df.matching.isin(ar_best_methods)]
        filtered_df = filtered_df[filtered_df[self.threshold_col] < self.threshold]
        _add_bounds(df, filtered_df[self.value_col].values, filtered_df[self.error_col].values, f'{self.threshold_col}-{self.top_col}')

        # SMD+A2A
        db_mask = get_best_group_from_dbscan(df, self.top_col, self.threshold_col, self.threshold, True)
        ar_best_methods = df['matching'].iloc[db_mask].values
        filtered_df = df[df.matching.isin(ar_best_methods)]
        filtered_df = filtered_df[filtered_df[self.threshold_col] < self.threshold]
        _add_bounds(df, filtered_df[self.value_col].values, filtered_df[self.error_col].values, f'{self.threshold_col}+{self.top_col}')

        # SMDOptimA2A
        db_mask = get_best_A2A(df[['diff', 'smd']], self.top_col, threshold_mask=df[self.threshold_col] < self.threshold)
        ar_best_methods = df['matching'].iloc[db_mask].values
        filtered_df = df[df.matching.isin(ar_best_methods)]
        filtered_df = filtered_df[filtered_df[self.threshold_col] < self.threshold]
        _add_bounds(df, filtered_df[self.value_col].values, filtered_df[self.error_col].values, f'min{self.top_col}')


        # SMDOptim
        db_mask = np.array([np.argmin(df[self.threshold_col].values)])
        ar_best_methods = df['matching'].iloc[db_mask].values
        filtered_df = df[df.matching.isin(ar_best_methods)]
        filtered_df = filtered_df[filtered_df[self.threshold_col] < self.threshold]
        _add_bounds(df, filtered_df[self.value_col].values, filtered_df[self.error_col].values, f'min{self.threshold_col}')


        # Pareto
        db_mask = get_best_group_from_pareto(df, self.top_col, self.threshold_col, self.threshold)
        ar_best_methods = df['matching'].iloc[db_mask].values
        filtered_df = df[df.matching.isin(ar_best_methods)]
        filtered_df = filtered_df[filtered_df[self.threshold_col] < self.threshold]
        _add_bounds(df, filtered_df[self.value_col].values, filtered_df[self.error_col].values, f'pareto')

        
        return df

results = results[~results['method'].str.contains('logit')]
artificials = artificials[~artificials['method'].str.contains('logit')]

matching_map = {
    'bart_nearest': 'Bart nearest',
    'bart_optimal': 'Bart optimal',
    'bipartify_identity_logistic-regression': 'Bipartify LR',
    'bipartify_identity_psmpy': 'Bipartify PsmPy',
    'bipartify_identity_random-forest': 'Bipartify RF',
    'cbps_nearest': 'CBPS nearest',
    'cbps_optimal': 'CBPS optimal',
    'elasticnet_nearest': 'ElasticNet nearest',
    'elasticnet_optimal': 'ElasticNet optimal',
    'gam_nearest': 'GAM nearest',
    'gam_optimal': 'GAM optimal',
    'glm_nearest': 'GLM nearest',
    'glm_optimal': 'GLM optimal',
    'psmpy_identity_logistic-regression': 'PsmPy LR',
    'psmpy_identity_psmpy': 'PsmPy PsmPy',
    'psmpy_identity_random-forest': 'PsmPy RF',
    'rpart_nearest': 'Rpart nearest',
    'rpart_optimal': 'Rpart optimal',
}
results.matching.replace(matching_map, inplace=True)
artificials.matching.replace(matching_map, inplace=True)

results['absate'] = results.ate.abs()
r = results[~results.matching.str.contains('nearest')]
r = r.pivot(index='matching', columns='dataset', values='absate')
r = r[[f'synthetic_7_{i}' for i in range(11)]]
print(r)
ar = autorank(-r.T)
plt.figure(figsize=(16, 4))
plot_stats(ar, ax=plt.gca())
print(create_report(ar))
plt.tight_layout()
plt.savefig('rank.pdf')
plt.savefig('rank.png')


results['absdiff'] = results['diff'].abs()

r = results[results.dataset.isin(['groupon', 'nhanes', 'horse'])]
r = r[~r.matching.str.contains('nearest')]
r = r.pivot(index='matching', columns='dataset', values=['smd', 'absdiff'])
print(r.to_latex(float_format="%.3f"))

results['group_dataset'] = results['dataset']  # Ugly hack because I need to know the dataset in the grouping...
agg = Bounds(artificials, 'target', 'ate', 'smd', 'diff')
results = results.groupby('dataset').apply(agg).reset_index(drop=True)


results = results.fillna(-99999).groupby('dataset').mean()
# results = results[['smdtarget_gap', 'difftarget_gap', 'smd-difftarget_gap', 'diff-smdtarget_gap', 'smd+difftarget_gap',
#                    'smdavg_error', 'diffavg_error', 'smd-diffavg_error', 'diff-smdavg_error', 'smd+diffavg_error',]]

#results = results[['smdtarget_gap', 'difftarget_gap', 'smd-difftarget_gap', 'smd+difftarget_gap', 'smd#difftarget_gap',
#                   'smdavg_error', 'diffavg_error', 'smd-diffavg_error', 'smd+diffavg_error', 'smd#diffavg_error']]


results = results[['smdtarget_gap', 'smd-difftarget_gap', 'paretotarget_gap',
    'smdavg_error', 'smd-diffavg_error', 'paretoavg_error', 'mindiffavg_error', 'minsmdavg_error']]


#print(results)
print(results.to_latex(float_format="%.3f"))