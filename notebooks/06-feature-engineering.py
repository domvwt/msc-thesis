# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: 'Python 3.9.12 (''.venv'': poetry)'
#     language: python
#     name: python3
# ---

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
import yaml

import networkx as nx

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

import mscproject.features as ft

plt.style.use("seaborn-white")
conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

# %%
# %load_ext autoreload
# %autoreload 2

# %%
companies_df = pd.read_parquet(conf_dict["companies_nodes"])
persons_df = pd.read_parquet(conf_dict["persons_nodes"])
edges_df = pd.read_parquet(conf_dict["edges"])

# %%
graph = ft.make_graph(edges=edges_df)

# %%
node_features = ft.generate_node_features(graph=graph)

# %%
node_features.describe()

# %%
list(nx.generators.ego_graph(graph, "2356236782051912119", 1, undirected=True).nodes)

# %%
list(nx.all_neighbors(graph, "2356236782051912119"))

# %%
neighbourhood_features = ft.get_local_neighbourhood_features(graph, node_features)

# %%
neighbourhood_features.sort_values(by="neighbourhood_closeness", ascending=False).head(
    10
)

# %%
