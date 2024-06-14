import sys
import os
sys.path.append('../../')
import pandas as pd
from psmpy import PsmPy
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
from popmatch.data import load_data, standardize_continuous_features

from statsmodels.stats.meta_analysis import effectsize_smd
from formulaic import ModelSpec, Formula
from formulaic.parser.types.factor import Factor
from popmatch.match import bipartify, split_populations_with_error, split_stats, psmpy_match, matchit_match, load_biggest_population, load_real_problem
from popmatch.evaluation import compute_smd, compute_target_mean_difference, compute_simulation_params, compute_synth_metrics
from popmatch.preprocess import preprocess
from popmatch.propensity import propensity_score, compute_propensity_overlap
from popmatch.experiment import dict_cache
from popmatch.plot import plot_smds
from adjustText import adjust_text
import itertools
import tqdm


dirname = os.path.split(os.path.dirname(__file__))[1]



experiment = {
    'input': {
        'dataset': dirname,
        #'propensity_model': 'logistic_regression',
        #'propensity_model': 'random_forest',
        #'propensity_model': 'psmpy',
        'propensity_transform': 'identity',
        # 'propensity_transform': 'logit',
        'clip_score': 0.05,
        'calibrated': True,
        'random_state': 12,
        #'simulated_split_population_ratio': [0.3, 0.7],
        #'simulated_split_target_difference': 0.2,
        #'simulated_split_smd_weight': 10,
    },
}

for not_already in dict_cache(experiment, 'data', cache_path='./data_cache'):
    load_data(experiment)
standardize_continuous_features(experiment)
preprocess(experiment)
compute_simulation_params(experiment)


df = pd.read_parquet('artificial.parquet')

