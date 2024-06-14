import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr, kendalltau


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

for dataset in datasets:
    filename = f"{dataset}/artificial.parquet"
    if not os.path.exists(filename):
        continue
    result = pd.read_parquet(filename)
    result['dataset'] = dataset
    results.append(result)

results = pd.concat(results)


# Note to self: Spearman gives almost identical results.
class Correlation:

    def __init__(self, true_values, pred_metric, metric=kendalltau):
        self.true_values = true_values
        self.pred_metric = pred_metric
        self.metric = metric

    def __call__(self, df):
        corr, pvalue = self.metric(df[self.true_values].values, df[self.pred_metric].values)

        return pd.DataFrame.from_dict({'corr': [corr], 'pvalue': [pvalue]})


# Inspired from https://github.com/lucky7323/nDCG/blob/master/ndcg.py
# Note to self: I gave up NDCG because I did not find a good way to define relevance. Measures
#               are either all identical, or complete nonsense.
class NDCG:

    def __init__(self, true_values, pred_metric):
        self.true_values = true_values
        self.pred_metric = pred_metric

    def __call__(self, df):
        rel_true, rel_pred_metric = df[self.true_values], df[self.pred_metric]
        idx_true = np.argsort(rel_true)[::-1]
        rel_true, rel_pred_metric = rel_true.iloc[idx_true], rel_pred_metric.iloc[idx_true]
        rel_pred = rel_true.iloc[np.argsort(rel_pred_metric)[::-1]]
        # print(rel_true, rel_pred)

        p = min(rel_true.shape[0], rel_pred.shape[0])
        discount = 1 / (np.log2(np.arange(p) + 2))

        idcg = np.sum(rel_true.iloc[:p] * discount)
        dcg = np.sum(rel_pred.iloc[:p] * discount)

        return dcg / idcg

results = results[~results['method'].str.contains('logit')]
np.random.seed(0)
results['random'] = np.random.random(results.shape[0])
results['abstarget'] = results['target'].abs()
results['invtarget'] = 1 / results['target'].abs()
results['invsmd'] = 1 / results['smd']
results['split'] = results['split_id'].str.extract(r'split(\d+)$')


def format_mean_std(row):
    return f"{row['corr_mean']:.2f} ± {row['corr_std']:.2f}"

def extract_corr(df, agg):
    df = df.groupby(['dataset', 'split']).apply(agg).reset_index().drop(['split', 'level_2'], axis=1)
    df = df.groupby('dataset').agg(['mean', 'std'])
    df.columns = df.columns.map('_'.join)
    df['corr'] = df.apply(format_mean_std, axis=1)
    df = df[['corr', 'pvalue_mean']]
    return df

agg = Correlation('target', 'smd')
results_smd = extract_corr(results, agg)
results_smd.columns = pd.MultiIndex.from_product([['Target'], results_smd.columns])

agg = Correlation('abstarget', 'smd')
results_smd_abs = extract_corr(results, agg)
results_smd_abs.columns = pd.MultiIndex.from_product([['Absolute target'], results_smd_abs.columns])

agg = Correlation('abstarget', 'random')
results_rnd = extract_corr(results, agg)
results_rnd.columns = pd.MultiIndex.from_product([['Random'], results_rnd.columns])

results = pd.concat([results_smd, results_smd_abs, results_rnd], axis=1)

print(results)
# print(results.to_latex(float_format="%.2f"))

results = pd.concat([results_smd, results_smd_abs, results_rnd], axis=1)

# print(results)
print(results.to_latex(float_format="%.2f"))
