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
data_dir = "data/pyg/processed/components"

# %%
data_files = [f for f in Path(data_dir).iterdir() if f.is_file()]

# %%
df01 = torch.load(data_files[0])

# %%
df01.edge_stores

# %%
data_dict = {}

for i, df in enumerate(data_files):
    df = torch.load(df)
    data_dict.update({i: (df.num_nodes, df.num_edges)})

# %%
df = pd.DataFrame.from_dict(data_dict, orient="index")

# %%
df.describe()

# %%
df.sort_values(by=0, ascending=False)
