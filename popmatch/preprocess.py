from .experiment import dict_wrapper
import pandas as pd
import re

from formulaic import Formula, ModelSpec
from sklearn.base import BaseEstimator, TransformerMixin


class FormulaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, formula):
        self.formula = Formula(formula)
    
    def fit(self, X, y=None):
        self.model_spec_ = ModelSpec.from_spec(self.formula.get_model_matrix(X))
        return self
    
    def transform(self, X):
        # We remove the intercept because models already handle it
        return self.model_spec_.get_model_matrix(X)
    
    def get_feature_name_from_index(self, index):
        # Note: this does not handle a lot of custom transform and such
        name = self.model_spec_.column_names[index]
        if not name.startswith('C('):
            return name
        groups = re.search('\(([^)]*)\)', name).groups()
        return groups[0]
    

@dict_wrapper('data_X', 'data_y', 'data_formula', 'data_transformer')
def preprocess(data_df, data_target, data_continuous_std, data_categorical, data_ordinal):
    terms = data_continuous_std + ['C({})'.format(c) for c in data_categorical + data_ordinal]
    data_formula = ' + '.join(terms)
    data_transformer = FormulaTransformer(data_formula)
    data_X = data_transformer.fit_transform(data_df)
    data_X = pd.DataFrame(data_X)
    data_y = data_df[data_target].values
    assert(data_df.shape[0] == data_X.shape[0])
    
    return data_X, data_y, data_formula, data_transformer
