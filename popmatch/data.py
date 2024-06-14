import numpy as np
from scipy.io import arff
import os
from sklearn.datasets import make_classification, make_regression
from sklearn.utils import check_random_state
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
from .experiment import dict_wrapper, dict_router
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# from .regression import synthetic_data
from .utils import batched


def simulate_with_confounders(n_features, n_confounders, n=1000, sigma=0.1, adj=0.0):
    """Synthetic data with a different amount of confounders
    Args:
        r (int): number of groups of 3 confounders
        n (int, optional): number of observations
        p (int optional): number of groups of 3 covariates
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    assert((n_features - n_confounders) % 2 == 0)
    assert(n_features % 3 == 0)
    # Features are splitted as follows:
    # A B C D E F G H I J
    # -----------         = propensity features
    #         ---         = confounders
    #         ----------- = outcome features

    p = (n_features) * 3
    
    X = np.random.normal(size=(n, p * 3))
    
    tau = []
    for triplet in batched(range(p), n=3):
        cols = X[:, slice(triplet[0], triplet[-1] + 1)].copy()
        cols[1] += cols[0]
        cols = cols[:, 1:]
        tau.append(np.clip(np.max(cols, axis=1), 0, None))
    tau = sum(tau)
    if r > 0:
        tau = tau / (((r - 1) // 3) + 1)
    b = np.zeros(n)
    e = 1 / (1 + sum([np.exp(X[:, i]) for i in range(lf)]) / (lf + 1) * 2)
    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e



@dict_wrapper('data_df', 'data_target', 'data_continuous', 'data_ordinal', 'data_categorical')
def load_heart(input_dataset):
    assert(input_dataset == 'heart')

    df = pd.read_csv(os.path.dirname(__file__) + '/../datasets/heart.csv')

    target = 'target'
    continuous = ['age', 'trestbps', 'chol', 'thalach',  'oldpeak']
    ordinal = ['restecg', 'ca']
    categorical = ['sex', 'cp', 'fbs',  'exang', 'slope', 'thal']

    return df, target, continuous, ordinal, categorical



@dict_wrapper('data_df', 'data_target', 'data_continuous', 'data_ordinal', 'data_categorical')
def load_mental_health(input_dataset):
    assert(input_dataset == 'mental_health')

    df = pd.read_csv(os.path.dirname(__file__) + '/../datasets/mental_health.csv')

    raise NotImplementedError()
    return df, target, continuous, ordinal, categorical


@dict_wrapper('data_df', 'data_user_df', 'data_population', 'data_time', 'data_target', 'data_continuous', 'data_ordinal', 'data_categorical')
def load_pbcseq(input_dataset):
    assert(input_dataset == 'pbcseq')

    data = arff.loadarff(os.path.dirname(__file__) + '/../datasets/pbcseq.arff')
    df = pd.DataFrame(data[0])
    df.status = df.status.replace({1: 0}).replace({2: 1})

    continuous = ['age']
    ordinal = []
    categorical = ['sex']

    return df, 'status', continuous, ordinal, categorical, 'drug'


@dict_wrapper('data_df', 'data_target', 'data_continuous', 'data_ordinal', 'data_categorical', 'data_population')
def load_groupon(input_dataset):
    assert(input_dataset == 'groupon')

    df = pd.read_csv(os.path.dirname(__file__) + '/../datasets/groupon.csv')

    target = 'revenue'
    continuous = ['prom_length', 'price', 'discount_pct', 'coupon_duration']
    ordinal = []
    categorical = ['featured', 'limited_supply']
    population = df['treatment']

    return df, target, continuous, ordinal, categorical, population


@dict_wrapper('data_df', 'data_target', 'data_continuous', 'data_ordinal', 'data_categorical', 'data_population')
def load_horse(input_dataset):
    assert(input_dataset == 'horse')

    df = pd.read_csv(os.path.dirname(__file__) + '/../datasets/horse.csv')

    target = 'outcome'
    df.outcome.replace({'lived': 0, 'died': 1, 'euthanized': 1}, inplace=True)
    df.age.replace({'young': 0, 'adult': 1}, inplace=True)

    df.temp_of_extremities.replace({'cold': 0, 'cool': 1, 'normal': 2, 'warm': 3}, inplace=True)
    df.peripheral_pulse.replace({'absent': 0, 'reduced': 1, 'normal': 2, 'increased': 3}, inplace=True)
    df.capillary_refill_time.replace({'less_3_sec': 0, '3': 1, 'more_3_sec': 2}, inplace=True)
    df.peristalsis.replace({'absent': 0, 'hypomotile': 1, 'normal': 2, 'hypermotile': 3}, inplace=True)
    df.abdominal_distention.replace({'none': 0, 'slight': 1, 'moderate': 2, 'severe': 3}, inplace=True)
    df.nasogastric_tube.replace({'none': 0, 'slight': 1, 'significant': 2}, inplace=True)
    df.nasogastric_reflux.replace({'none': 0, 'less_1_liter': 1, 'more_1_liter': 2}, inplace=True)
    df.rectal_exam_feces.replace({'absent': 0, 'decreased': 1, 'normal': 2, 'increased': 3}, inplace=True)
    df.abdomo_appearance.replace({'clear': 0, 'cloudy': 1, 'serosanguious': 2}, inplace=True)
    continuous = ['rectal_temp', 'pulse', 'respiratory_rate', 'nasogastric_reflux_ph',
                  'packed_cell_volume', 'total_protein', 'abdomo_protein']
    ordinal = ['age', 'temp_of_extremities', 'peripheral_pulse', 'capillary_refill_time', 'peristalsis',
               'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces',
               'abdomo_appearance']
    categorical = ['mucous_membrane', 'pain', 'abdomen']
    population = df['surgery'].replace({"no": 0, "yes": 1})

    mean_imputer = SimpleImputer()
    for column in continuous:
        df[column] = mean_imputer.fit_transform(df[[column]].values)[:, 0]
    most_fqt_imputer = SimpleImputer(strategy='most_frequent')
    for column in ordinal + categorical:
        df[column] = most_fqt_imputer.fit_transform(df[[column]].values)[:, 0]

    return df, target, continuous, ordinal, categorical, population


@dict_wrapper('data_df', 'data_target', 'data_continuous', 'data_ordinal', 'data_categorical', 'data_population')
def load_nhanes(input_dataset):
    assert(input_dataset == 'nhanes')

    df = pd.read_csv(os.path.dirname(__file__) + '/../datasets/nhanes.csv')
    df.rename(columns=lambda x: x.replace('.', '_'), inplace=True)
    df['arthritis_type'].replace({'Non-arthritis': 0, "Rheumatoid arthritis": 1}, inplace=True)

    # Map ordinal to ints
    df['strata'] = df['strata'] - df['strata'].min()
    df['education'].replace({'School': 0, "High.School": 1, 'College': 2}, inplace=True)
    df['annualincome'].replace({'Non-arthritis': 0, "Rheumatoid arthritis": 1}, inplace=True)
    df['physical_activity'].replace({'Non-arthritis': 0, "Rheumatoid arthritis": 1}, inplace=True)
    df['healthy_diet'].replace({'Non-arthritis': 0, "Rheumatoid arthritis": 1}, inplace=True)
    df['heart_attack'].replace({'No': 0, "Yes": 1}, inplace=True)
    
    target = 'heart_attack'
    continuous = ['interview_weight', 'MEC_weight', 'bmi', 'age']
    ordinal = ['strata', 'education', 'annualincome', 'physical_activity', 'healthy_diet']
    categorical = ['gender', 'PSU', 'diabetes', 'smoke', 'race', 'born', 'marriage', 'medical_access', 'blood_pressure', 'covered_health']
    population = 'arthritis_type'

    return df, target, continuous, ordinal, categorical, df[population]


@dict_wrapper('data_df', 'data_target', 'data_continuous', 'data_ordinal', 'data_categorical', 'data_population',
              'data_true_propensity', 'data_true_outcome', 'data_true_ite')
def load_synthetic(input_dataset):
    args = input_dataset.split('_')
    assert(args[0] == 'synthetic')

    mode = int(args[1])
    if mode <= 5:
        y, X, treatment, tau, b, e = synthetic_data(mode=int(mode))
    elif mode == 6:
        common_features = int(args[2])
        y, X, treatment, tau, b, e = simulate_with_common_features(common_features)
    elif mode == 7:
        common_features = int(args[2])
        y, X, treatment, tau, b, e = simulate_with_common_features(common_features, n=3000, p=10)

    target = 'target'
    continuous = ['feat_' + str(i) for i in range(X.shape[1])]
    ordinal = []
    categorical = []
    df = pd.DataFrame(X, columns=continuous)
    df['target'] = y

    return df, target, continuous, ordinal, categorical, treatment, e, b, tau

@dict_router
def load_data(input_dataset):

    if input_dataset == 'heart':
        return load_heart
    elif input_dataset == 'mental_health':
        return load_mental_health
    elif input_dataset == 'pbcseq':
        return load_pbcseq   
    elif input_dataset == 'groupon':
        return load_groupon
    elif input_dataset == 'nhanes':
        return load_nhanes
    elif input_dataset == 'horse':
        return load_horse
    elif input_dataset.startswith('synthetic'):
        return load_synthetic

@dict_wrapper('data_df', 'data_continuous_std')
def standardize_continuous_features(data_df, data_continuous):
    continuous_std = []
    for c in data_continuous:
        data_df[c + '_std'] = StandardScaler().fit_transform(data_df[[c]])
        continuous_std.append(c + '_std')
    return data_df, continuous_std


def perform_1_to_1_matching(X_source, X_target):
    target_mask = np.ones(X_target.shape[0], dtype=bool)
    target_idx = np.arange(X_target.shape[0])
    nn = []
    nbrs = None
    for sample in X_source:
        i = None
        if nbrs:
            i = nbrs.kneighbors([sample])[1] 
            i = target_idx[target_mask][i]
        
        if i is None or i in nn:
            target_mask[np.asarray(nn)] = False
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_target[target_mask])
            i = nbrs.kneighbors([sample])[1] 
            i = target_idx[target_mask][i]

        nn.append(i)
    return np.asarray(nn)


def generate_dataset(n_controls, n_treated, continuous_features, categorical_features,
                     n_informative_propensity, n_informative_outcome,
                     propensity_factor=.5, random_state=None):
    
    n_samples = n_controls + n_treated
    n_features = continuous_features + len(categorical_features)
    n_informative = n_informative_outcome + n_informative_propensity

    # In order to simulate a bias in propensity, we generate a classification problem
    # with errors equal to the propensity factor.

    X_p, y_p = make_classification(
        n_samples=n_samples,
        n_features=n_informative_propensity,
        n_informative=n_informative_propensity,
        n_redundant=0,
        weights=[n_controls / n_samples, n_treated / n_samples],
        flip_y = propensity_factor,
        random_state=random_state,
    )
    
    # Now we generate a regression problem which will be our outcome, but we ignore the
    # labels because we will tweak it afterward.

    X, y, coefs = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative_outcome,
        coef=True,
        random_state=random_state,
    )

    rng = check_random_state(random_state)

    # We replace part of X with data from the classification problem
    X[:, n_informative_outcome:n_informative_outcome + n_informative_propensity] = X_p

    # We add coefs to these features, but with a lesser weight (by default the weight is 100)
    #coefs[n_informative_outcome:n_informative_outcome + n_informative_propensity] = \
    #    30 * rng.uniform(size=(n_informative_propensity))

    # Now we create two coefs table: one for control, one for treated
    coefs_control = coefs.copy()
    coefs_treated = coefs.copy()
    coefs_treated[:n_informative] += 10 * rng.uniform(size=(n_informative   ))

    y = np.zeros(n_samples)
    y[y_p == 0] = np.dot(X[y_p == 0], coefs_control)
    y[y_p == 1] = np.dot(X[y_p == 1], coefs_treated)

    # Turn some features into categorical ones
    idx_cat = rng.choice(n_informative, size=len(categorical_features), replace=False)
    for i, n in zip(idx_cat, categorical_features):
        X[:, i] = np.digitize(X[:, i], np.quantile(X[:, i], (np.arange(n - 1) + 1) / n))

    return {'X': X, 'y': y, 'groups': y_p,
             'idx_cat': idx_cat, 'n_samples': n_samples,
             'n_informative': n_informative}


def generate_pairing_with_errors(dataset, levels, random_state=None):
    X = dataset['X']
    groups = dataset['groups']
    n_informative = dataset['n_informative']
    
    # First, we take each percent of population in order to create the confidence
    # levels.
    percents = [i[0] for i in levels]
    assert(abs(sum(percents) - 1.0) < 0.001)

    # Then, we arbitrarily cluster the samples in order to apply the given error
    # rate in the matching
    rng = check_random_state(random_state)
    rand = rng.uniform((groups == 1).sum())
    true_confidence = groups.copy()
    true_confidence[true_confidence == 1] = np.digitize(rand, np.cumsum(percents)[:-1]) + 1
    fake_confidence = true_confidence.copy()
    corrupted_pairing_candidates = np.arange(X.shape[0])
    corrupted_pairing_candidates[groups == 1] = -1

    for level, (_, error_rate, masked_feature) in zip(range(len(levels), 0, -1), levels):

        # Select a subset of patients corresponding to this level
        to_corrupt = np.where(true_confidence == level)[0]
        to_corrupt = rng.choice(to_corrupt, size=int(error_rate * to_corrupt.shape[0]), replace=False)
        fake_confidence[to_corrupt] = 0  # We turn real paired data into control

        # Randomly drop informative features from data
        X_ = np.delete(X, rng.choice(n_informative, size=masked_feature, replace=False), axis=1)

        # Now we use NN to create fake matches with non paired data
        indices = perform_1_to_1_matching(X_[to_corrupt], X_[corrupted_pairing_candidates != -1])
        fake_confidence[corrupted_pairing_candidates[indices]] = level
        corrupted_pairing_candidates[indices] = -1

    assert((fake_confidence > 0).sum() == groups.sum())

    return fake_confidence        


