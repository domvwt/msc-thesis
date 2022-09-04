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
rng = np.random.default_rng()

# %%
ints1 = rng.integers(4, size=100)
ints2 = rng.integers(100, size=100)

# %%
df = pd.DataFrame({"a": ints1, "b": ints2})

# %%
df.groupby("a").sample(n=1, random_state=42)

# %%
