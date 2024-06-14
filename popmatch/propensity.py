from .experiment import dict_router, dict_wrapper

import numpy as np
import pandas as pd
from psmpy import PsmPy
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import seaborn as sns
from matplotlib import pyplot as plt


class ValidAccuracy():
    def __init__(self, clip_score):
        self.clip_score = clip_score

    def __call__(self, y, y_pred, **kwargs):
        accuracy = accuracy_score(y, y_pred)
        mean_valid_entries = ((y_pred >= self.clip_score) & (y_pred <= (1 - self.clip_score))).mean()
        overlap = compute_propensity_overlap(
            np.hstack([np.zeros(y.shape[0]), np.ones(y_pred.shape[0])]), np.hstack([y, y_pred]))
        return accuracy + mean_valid_entries + int(overlap > .5)


@dict_wrapper('{output}_population', '{output}_propensity_score', '{output}_ordinal_ordered')
def propensity_logistic_regression(input_X, data_transformer, data_ordinal,
                                   input_propensity_transform, splitid_population, input_random_state,
                                   input_calibrated=True, input_clip_score=0.001):
    
    scoring = make_scorer(ValidAccuracy(input_clip_score), needs_proba=True)
    n_0 = (splitid_population == 0).sum()
    n_1 = (splitid_population == 1).sum()
    n = n_0 + n_1
    sample_weight = None
    if input_calibrated:
        sample_weight = splitid_population.copy().astype(float)
        sample_weight[splitid_population == 0] = n_1 / n
        sample_weight[splitid_population == 1] = n_0 / n
    clf = LogisticRegressionCV(random_state=input_random_state, max_iter=10000, scoring=scoring)
    clf.fit(input_X, splitid_population, sample_weight=sample_weight)
    propensity_score = clf.predict_proba(input_X)[:, 1]
    
    feature_ordered = np.argsort(clf.coef_[0])[::-1]
    # Keep only the ordinal feature, unique names because they may be OHE
    feature_ordinal_ordered = []
    for f in feature_ordered:
        f = data_transformer.get_feature_name_from_index(f)
        if not f in data_ordinal or f in feature_ordinal_ordered:
            continue
        feature_ordinal_ordered.append(f)

    if input_clip_score is not None:
        propensity_score = np.clip(propensity_score, input_clip_score, 1 - input_clip_score)

    if input_propensity_transform == 'logit':
        propensity_score = np.log(propensity_score / (1 - propensity_score))
    
    return splitid_population, propensity_score, feature_ordinal_ordered


@dict_wrapper('{output}_population', '{output}_propensity_score', '{output}_ordinal_ordered')
def propensity_psmpy(input_X, input_y, input_propensity_transform, splitid_population, input_random_state,
                                   input_calibrated=True, input_clip_score=0.001):
    
    df = input_X.copy()
    df['groups'] = splitid_population
    df['index'] = np.arange(input_X.shape[0])

    psm = PsmPy(df, treatment='groups', indx='index', exclude = [], seed=input_random_state)
    psm.logistic_ps(balance=input_calibrated)

    if input_propensity_transform == 'identity':
        propensity_score = psm.predicted_data['propensity_score']
    elif input_propensity_transform == 'logit':
        propensity_score = psm.predicted_data['propensity_logit']
    else:
        raise ValueError()

    return splitid_population, propensity_score, None


@dict_wrapper('{output}_population', '{output}_propensity_score', '{output}_ordinal_ordered')
def propensity_random_forest(input_X, data_transformer, data_ordinal,
                             input_propensity_transform, splitid_population,
                             input_calibrated=True, input_clip_score=0.001):
    

    scoring = make_scorer(ValidAccuracy(input_clip_score), needs_proba=True)
    clf = RandomForestClassifier(min_weight_fraction_leaf=0.01)
    if input_calibrated:
        clf.fit(input_X, splitid_population)
        clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid', n_jobs=-1)
        param_grid = {
            'base_estimator__n_estimators': [5, 10, 50, 100],
            'base_estimator__max_depth': [2, 5, 8, 15]
        }
        clf = GridSearchCV(clf, param_grid, scoring=scoring, error_score='raise', n_jobs=-1)
        clf.fit(input_X, splitid_population)
        feature_importances = clf.best_estimator_.calibrated_classifiers_[0].base_estimator.feature_importances_
    else:
        param_grid = {
            'n_estimators': [5],#, 10, 50, 100],
            'max_depth': [2, 5,],# 8, None]
        }
        clf = GridSearchCV(clf, param_grid, scoring=scoring, n_jobs=-1)
        clf.fit(input_X, splitid_population)
        feature_importances = clf.best_estimator_.feature_importances_

    propensity_score = clf.predict_proba(input_X)[:, 1]

    feature_ordered = np.argsort(feature_importances)[::-1]
    # Keep only the ordinal feature, unique names because they may be OHE
    feature_ordinal_ordered = []
    for f in feature_ordered:
        f = data_transformer.get_feature_name_from_index(f)
        if not f in data_ordinal or f in feature_ordinal_ordered:
            continue
        feature_ordinal_ordered.append(f)

    if input_clip_score is not None:
        propensity_score = np.clip(propensity_score, input_clip_score, 1 - input_clip_score)
    
    if input_propensity_transform == 'logit':
        propensity_score = np.log(propensity_score / (1 - propensity_score))

    return splitid_population, propensity_score, feature_ordinal_ordered


@dict_router
def propensity_score(input_propensity_model=None):

    if input_propensity_model == 'logistic-regression':
        return propensity_logistic_regression
    elif input_propensity_model == 'random-forest':
        return propensity_random_forest
    elif input_propensity_model == 'psmpy':
        return propensity_psmpy
    
def compute_propensity_overlap(splitid_population, splitid_propensity_score, save_plot=None):
    dist_0 = splitid_propensity_score[splitid_population == 0]
    dist_1 = splitid_propensity_score[splitid_population == 1]
    if save_plot is not None:
        plt.figure(figsize=(8, 6))
        sns.kdeplot(dist_0, label="Control", shade=True)  # Plot the first distribution
        sns.kdeplot(dist_1, label="Treated", shade=True)  # Plot the second distribution
        plt.xlabel("Propensity")
        plt.ylabel("Density")
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(save_plot)
    bins = np.arange(-0.05, 1.06, 0.10)
    dist_0 = np.histogram(dist_0, bins=bins)[0] / dist_0.shape[0]
    dist_1 = np.histogram(dist_1, bins=bins)[0] / dist_1.shape[0]
    overlap = np.min([dist_0, dist_1], axis=0)

    assert(overlap[1:-1].sum() <= 1.)
    return overlap[1:-1].sum()
    