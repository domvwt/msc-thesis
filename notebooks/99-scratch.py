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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import torch
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
companies_path = "data/graph-features/companies.parquet"
persons_path = "data/graph-features/persons.parquet"
edges_path = "data/graph-features/edges.parquet"

# %%
companies_df = pd.read_parquet(companies_path)
persons_df = pd.read_parquet(persons_path)
edges_df = pd.read_parquet(edges_path)

# %%
print(companies_df.columns)
print(persons_df.columns)
print(edges_df.columns)

# %%
companies_out = companies_df.loc[:, :"is_anomalous"]
persons_out = persons_df.loc[:, :"is_anomalous"]
edges_out = edges_df.loc[:, :"is_anomalous"]

# %%
print(companies_out.columns)
print(persons_out.columns)
print(edges_out.columns)

# %%
companies_output_path = "data/graph-anomalies/companies.parquet"
persons_output_path = "data/graph-anomalies/persons.parquet"
edges_output_path = "data/graph-anomalies/edges.parquet"

# %%
companies_out.to_parquet(companies_output_path)
persons_out.to_parquet(persons_output_path)
edges_out.to_parquet(edges_output_path)
