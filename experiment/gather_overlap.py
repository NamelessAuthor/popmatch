import numpy as np
import pandas as pd
import os


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
    filename = f"{dataset}/overlaps.parquet"
    if not os.path.exists(filename):
        continue
    result = pd.read_parquet(filename)
    result['dataset'] = dataset
    results.append(result)

results = pd.concat(results)
results.model.replace({'logistic-regression': 'LR', 'random-forest': 'RF', 'psmpy': 'PsmPy'}, inplace=True)

results = results.set_index(['dataset', 'transform', 'model']).unstack([-2, -1])
print(results.to_latex(float_format="%.2f"))