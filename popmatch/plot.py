from matplotlib import pyplot as plt
import pandas as pd
import itertools
import seaborn as sns



def plot_smds(experiment, split_ids, matchings, filename):
    smds = []
    for splitid, matching in itertools.product(split_ids, matchings):
        smd = experiment[splitid][f"{matching}_smds"]
        smd['split_id'] = splitid
        smd['matching'] = matching
        smds.append(smd)
    smds = pd.concat(smds)
    sns.catplot(
        y="feature",       # x variable name
        x="smd",       # y variable name
        hue="matching",  # group variable name
        data=smds,     # dataframe to plot
        kind="bar",
        orient='h',
    )
    plt.savefig(filename)

def plot_table(experiment, split_ids, matchings):
    pass