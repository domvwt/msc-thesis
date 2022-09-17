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
#     display_name: pycaret
#     language: python
#     name: pycaret
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

features_path = conf_dict["features_path"]


# %%
# Load features for data split.
def load_features(path_root):
    companies_df = pd.read_parquet(conf_dict["companies_features"])
    persons_df = pd.read_parquet(conf_dict["persons_features"])
    return companies_df, persons_df


# %%
companies_df, persons_df = load_features(features_path)

# %%
companies_df.columns

# %%
common_cols = set(companies_df.columns) & set(persons_df.columns)
drop_cols = ["id", "name", "component"]
select_cols = sorted(common_cols.difference(drop_cols))

target = "is_anomalous"

entities_df = pd.concat([companies_df, persons_df], axis=0)[
    list(common_cols)
].reset_index(drop=True)


# %%
entities_df[select_cols].info()

# %%
entities_df["indegree"] = entities_df["indegree"].astype(float)
entities_df["outdegree"] = entities_df["outdegree"].astype(float)


# %%
def get_data_split_masks(df: pd.DataFrame):
    component_mod = df["component"].to_numpy() % 10
    train_mask = component_mod <= 7
    val_mask = component_mod == 8
    test_mask = component_mod == 9
    assert not np.any(train_mask & val_mask)
    assert not np.any(train_mask & test_mask)
    assert not np.any(val_mask & test_mask)
    return train_mask, val_mask, test_mask


# %%
train_mask, val_mask, test_mask = get_data_split_masks(entities_df)

# %%
train_df = entities_df.loc[train_mask].drop(drop_cols, axis=1)
val_df = entities_df.loc[val_mask].drop(drop_cols, axis=1)
test_df = entities_df.loc[test_mask].drop(drop_cols, axis=1)

print("Anomaly proportions")
train_proportion = train_df["is_anomalous"].sum() / len(train_df)
print(f"Train: {train_proportion:.3f}")
val_proportion = val_df["is_anomalous"].sum() / len(val_df)
print(f"Val: {val_proportion:.3f}")
test_proportion = test_df["is_anomalous"].sum() / len(test_df)
print(f"Test: {test_proportion:.3f}")

# %%
# Get indexes in both train and test.
train_idx = train_df.index
test_idx = test_df.index
train_test_idx = train_idx.intersection(test_idx)
assert len(train_test_idx) == 0

# %%
from imblearn.under_sampling import RandomUnderSampler

# %%
# Join val to train.
train_df = pd.concat([train_df, val_df], axis=0)

# %%
# Perform model selection with PyCaret.

categorical_features = ["isCompany"]

s = clf.setup(
    train_df,
    test_data=test_df,
    target="is_anomalous",
    categorical_features=categorical_features,
    silent=True,
    normalize=True,
    # feature_selection=True,
    # feature_selection_method="boruta",
    fix_imbalance=True,  # Uses SMOTE.
    fix_imbalance_method=RandomUnderSampler(sampling_strategy="auto"),
)

clf.add_metric("aprc", "Avg. Prec.", average_precision_score, target="pred_proba")


# %%
exclude = [
    "rbfsvm",
    "mlp",
    "gbc",
    "gpc",
]

# %%
best_model = clf.compare_models(sort="aprc", turbo=turbo_mode, fold=5, exclude=exclude)

# %%
Path("reports").mkdir(exist_ok=True)
model_selection_grid = clf.pull()
model_selection_grid.to_csv("reports/pycaret-model-selection.csv", index=False)


# %%
# Highlight highest value in each column
def highlight_max(s):
    is_max = s == s.max()
    return ["font-weight: bold" if v else "" for v in is_max]


model_selection_grid.style.apply(
    highlight_max, axis=0, subset=model_selection_grid.columns[1:]
).format("{:.3f}", subset=model_selection_grid.columns[1:]).to_html(
    "reports/pycaret-model-selection.html"
)

# %%
tuned_model = clf.tune_model(
    best_model,
    optimize="aprc",
    fold=3,
    search_library="optuna",
    n_iter=20,
    choose_better=True,
)

# %%
predictions = clf.predict_model(tuned_model, data=test_df, raw_score=True)
predictions["pred_proba"] = predictions["Score_True"]
predictions["actual"] = predictions["is_anomalous"].astype(int)
predictions

# %%
best_model_class = model_selection_grid.iloc[0]["Model"].replace(" ", "")
best_model_class

# %%
output_path = Path(f"data/predictions/{best_model_class}.csv")
output_path.parent.mkdir(exist_ok=True)
predictions.to_csv(output_path, index=False)

# %%
