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
import matplotlib.pyplot as plt
import yaml

import pdcast as pdc

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

import mscproject.features as feats
import mscproject.preprocess as pre
import mscproject.pygloaders as pgl
from mscproject.metrics import EvalMetrics
import catboost as cb

from imblearn.under_sampling import RandomUnderSampler


# %%
OPTUNA_DB = "sqlite:///data/optuna-03.db"
MODEL_DIR = Path("data/models")
PREDICTIONS_DIR = Path("data/predictions")

# %% [markdown]
# ### CatBoost Paper: https://arxiv.org/pdf/1706.09516.pdf

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
companies_df.describe().T

# %%
common_cols = set(companies_df.columns) & set(persons_df.columns)
drop_cols = ["id", "name", "component"]
select_cols = sorted(common_cols.difference(drop_cols))

target = "is_anomalous"

entities_df = pd.concat([companies_df, persons_df], axis=0)[list(common_cols)]


# %%
masks = pgl.get_data_split_masks(entities_df)

# %%
train_df = entities_df.loc[list(masks.train.numpy())].drop(drop_cols, axis=1)
valid_df = entities_df.loc[list(masks.val.numpy())].drop(drop_cols, axis=1)
test_df = entities_df.loc[list(masks.test.numpy())].drop(drop_cols, axis=1)

# %%
train_df

# %%
train_df.query("isCompany == False and indegree > 0")

# %%
# Create train, test, and valid pools for CatBoost.

cat_features = ["isCompany"]

X_train = train_df.drop(target, axis=1)
y_train = train_df[target].astype(np.int8)

X_val = valid_df.drop(target, axis=1)
y_val = valid_df[target].astype(np.int8)

X_test = test_df.drop(target, axis=1)
y_test = test_df[target].astype(np.int8)

# %%
# Use imblearn to undersample the majority class.
# rus = RandomUnderSampler(random_state=42)
# X_train, y_train = rus.fit_resample(X_train, y_train)

# %%
train_pool = cb.Pool(
    X_train.to_numpy(),
    y_train.to_numpy(),
    cat_features=cat_features,
    feature_names=X_train.columns.to_list(),
)

val_pool = cb.Pool(
    X_val.to_numpy(),
    y_val.to_numpy(),
    cat_features=cat_features,
    feature_names=X_val.columns.to_list(),
)

trail_val_pool = cb.Pool(
    pd.concat([X_train, X_val]).to_numpy(),
    pd.concat([y_train, y_val]).to_numpy(),
    cat_features=cat_features,
    feature_names=X_test.columns.to_list(),
)

test_pool = cb.Pool(
    X_test.to_numpy(),
    y_test.to_numpy(),
    cat_features=cat_features,
    feature_names=X_test.columns.to_list(),
)

# %%
class_weights = {0: 1, 1: 10}

# %%
# Tune CatBoost model with Optuna.
import optuna


def objective(trial: optuna.Trial):

    params = {
        "objective": "Logloss",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["Ordered", "Plain"]
        ),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "6gb",
        "class_weights": class_weights,
        "eval_metric": "PRAUC",
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0, 10
        )
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    clf = cb.CatBoostClassifier(**params)

    clf.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose=False,
    )

    best_iter = clf.get_best_iteration()
    if best_iter < 1:
        # Failed to train.
        raise optuna.TrialPruned()

    trial.set_user_attr("best_iter", best_iter)

    y_val_pred = clf.predict_proba(val_pool)[:, 1]
    eval_metrics = EvalMetrics.from_numpy(y_val_pred, y_val.to_numpy(), 10)

    return eval_metrics.average_precision


# STUDY_NAME = "CATBOOST-RUS"
STUDY_NAME = "CATBOOST_WEIGHTED"

delete = False
if delete:
    optuna.delete_study(study_name=STUDY_NAME, storage=OPTUNA_DB)

# delete study
study = optuna.create_study(
    direction="maximize", study_name=STUDY_NAME, storage=OPTUNA_DB, load_if_exists=True
)

n_trials = 100

completed_trials = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
remaining_trials = n_trials - completed_trials

study.optimize(objective, n_trials=remaining_trials)

# %%
# Get best parameters.
best_params = study.best_params
user_attrs = study.best_trial.user_attrs

# Create CatBoost model.
clf = cb.CatBoostClassifier(
    iterations=1000,
    **best_params,
    class_weights=class_weights,
    random_seed=42,
    eval_metric="PRAUC",
)

# Fit model.
clf.fit(
    trail_val_pool,
    verbose=200,
)

# Print confusion matrix.
y_pred = clf.predict(test_pool)
print(pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]))

# Evaluate model.
y_test_pred_proba = np.array([x[1] for x in clf.predict_proba(test_pool)])
eval_metrics = EvalMetrics.from_numpy(y_test_pred_proba, y_test.to_numpy())
print(eval_metrics)

# Store predictions.
predictions = pd.DataFrame(
    {
        "id": test_df.index,
        "pred_proba": y_test_pred_proba,
        "actual": y_test.to_numpy(),
    }
)

# %%
print(user_attrs)

# %%
print(best_params)

# %%
predictions.to_csv(PREDICTIONS_DIR / "CatBoost.csv", index=False)

# %%
# Get Shapley values.
shap_values = clf.get_feature_importance(test_pool, type="ShapValues")[:, 1:]


import shap

# %%
shap.summary_plot(shap_values, X_test)

# %%
# Plot partial dependence plots.
shap.summary_plot(shap_values, X_test, plot_type="bar")

# %%
for i in range(X_test.shape[1]):
    shap.dependence_plot(f"rank({i})", shap_values, X_test)
    plt.show()

# %%

# %%

# %%

# %%
