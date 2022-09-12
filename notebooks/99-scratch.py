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
import torch
from mscproject.datasets import CompanyBeneficialOwners

dataset_path = "data/pyg/"

# Set the device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset.
dataset = CompanyBeneficialOwners(dataset_path, to_undirected=True)
dataset = dataset.data.to(device)

# %%
from torch_geometric.loader import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
for batch in loader:
    print(batch.num_graphs)
    break
