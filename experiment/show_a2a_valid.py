from glob import glob
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

ds = ['horse', 'nhanes', 'groupon']


for d in ds:
    print(d)
    df = pd.read_parquet(f'{d}/results.parquet')
    df = df[~df.method.str.contains('nearest')]
    df = df[~df.method.str.contains('logit')]
    methods = df['matching'].values
    X = df['diff'].values[:, None]
    argmin = np.argmin(np.abs(X))
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.3, min_samples=2).fit(X)
    labels = db.labels_
    min_label = labels[argmin]
    valid_methods = methods[labels == min_label]
    print(valid_methods)

