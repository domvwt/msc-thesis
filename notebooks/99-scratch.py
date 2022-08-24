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
for data_split in ["train", "valid", "test"]:
    for feature_type in ["companies", "persons", "edges"]:
        feature_path = (
            Path(conf_dict["preprocessed_features_path"])
            / data_split
            / f"{feature_type}.parquet"
        )
        print(f"Loading {feature_path}")
        df = pd.read_parquet(feature_path)
        # print(df.head())
        # print first 20 components
        print(df["component"].unique()[:20])
        print()

# %%
