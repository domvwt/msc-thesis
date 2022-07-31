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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import pdcast as pdc

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

import pycaret.classification as clf
import sklearn.preprocessing as pre

from sklearn.metrics import average_precision_score


# %%
turbo_mode = False

# %%
# Read config.
conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

persons_df = pd.read_parquet(conf_dict["persons_nodes"])
companies_df = pd.read_parquet(conf_dict["companies_nodes"])
edges_df = pd.read_parquet(conf_dict["edges"])

features_path = conf_dict["preprocessed_features_path"]


# %%
# Load features for data split.
def load_features(path_root, split):
    features_dir = Path(path_root) / split
    companies_df = pd.read_parquet(features_dir / "companies.parquet").dropna()
    persons_df = pd.read_parquet(features_dir / "persons.parquet").dropna()
    return companies_df, persons_df


# %%
companies_train_df, persons_train_df = load_features(features_path, "train")

# %%
select_cols = set(companies_train_df.columns) & set(persons_train_df.columns)
processed_feature_cols = [x for x in select_cols if x.endswith("__processed")]
raw_feature_cols = [x.split("__processed")[0] for x in processed_feature_cols]
target = "is_anomalous"

entities_df = pd.concat([companies_train_df, persons_train_df], axis=0)[select_cols]
# Sample entities_df so that half of the entities are anomalous.
def balanced_sample(entities_df) -> pd.DataFrame:
    n_entities = len(entities_df)
    n_anomalous = len(entities_df[entities_df[target] == True])
    n_normal = n_entities - n_anomalous
    n_sample = min(n_anomalous, n_normal)
    anomalous_df = entities_df.query("is_anomalous == False").sample(n_sample)
    normal_df = entities_df.query("is_anomalous == True").sample(n_sample)
    return pd.concat([anomalous_df, normal_df], axis=0)


drop_cols = ["id", "name", "component"]

entities_features_df = entities_df.drop(drop_cols, axis=1)
balanced_sample_df = balanced_sample(entities_features_df)


# %%
# Perform model selection with PyCaret.

categorical_features = ["isCompany"]

s = clf.setup(
    entities_features_df,
    target="is_anomalous",
    categorical_features=categorical_features,
    silent=True,
    feature_selection=True,
    feature_selection_method="boruta",
)

clf.add_metric("apc", "APC", average_precision_score, target="pred_proba")


# %%
best_model_imbalanced = clf.compare_models(sort="APC", turbo=turbo_mode)
