# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: 'Python 3.9.12 (''.venv'': poetry)'
#     language: python
#     name: python3
# ---

# %%
import os
from pathlib import Path

import pandas as pd
import networkx as nx
import scipy.sparse as sp
import yaml

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

import mscproject.dataprep as dp
import mscproject.features as feat
import mscproject.simulate as sim


# %%
# %load_ext autoreload
# %autoreload 2

# %%
conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

# Load features
companies_features = pd.read_parquet(conf_dict["companies_features"])
persons_features = pd.read_parquet(conf_dict["persons_features"])
edges_features = pd.read_parquet(conf_dict["edges_features"])

# %%
companies_features.info()

# %%
persons_features.info()


# %%
def rename_columns(df):
    df = df.copy()
    column_names = df.columns.tolist()
    first = column_names.index("indegree")
    last = column_names.index("neighbourhood_0")
    graph_feats = column_names[first:last]
    keep_feats = column_names[0:last] + [f"neighbourhood_{x}" for x in graph_feats]
    df.columns = keep_feats
    return df


# %%
# Rename columns for each dataframe
companies_features = rename_columns(companies_features)
persons_features = rename_columns(persons_features)
# edges_features = rename_columns(edges_features)

# %%
# Overwrite old features with new features
companies_features.to_parquet(conf_dict["companies_features"])
persons_features.to_parquet(conf_dict["persons_features"])
