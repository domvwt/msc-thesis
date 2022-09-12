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
import optuna
import pandas as pd
import torch
from tqdm import tqdm

from mscproject.datasets import CompanyBeneficialOwners
from mscproject.transforms import RemoveSelfLoops
import mscproject.models as mod
import mscproject.experiment as exp

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

# %%
study_names = (
    # "pyg_model_selection_ALL",
    "pyg_model_selection_GCN",
    "pyg_model_selection_GraphSAGE",
    "pyg_model_selection_GAT",
    "pyg_model_selection_HGT",
    "pyg_model_selection_HAN",
)

# %%
from IPython.display import display

# %%
OPTUNA_STORAGE = "sqlite:///data/optuna.db"
MODEL_DIR = "data/models/pyg"

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

trials_dfs = [
    study.trials_dataframe().assign(study_name=study.study_name)
    for study in (
        optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE)
        for study_name in study_names
    )
]
eval_df = pd.concat(trials_dfs, join="inner", axis=0)


# %%
# for study in (
#     optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE)
#     for study_name in study_names
# ):
#     print(study.study_name)
#     # Plot the results.
#     mean_top_10_loss = study.trials_dataframe()["value"].sort_values().head(10).mean()
#     print("Mean of top 10 best loss:", mean_top_10_loss)
#     optuna.visualization.plot_optimization_history(study).show()
#     # optuna.visualization.plot_contour(study).show()
#     optuna.visualization.plot_slice(study).show()
#     optuna.visualization.plot_param_importances(study).show()
#     print()
#     print()

# %%
def mark_outliers(df):
    df = df.copy()
    upper_threshold = df.value.mean() + 3 * df.value.std()
    df["outlier"] = df.value > upper_threshold
    return df


# %%
metric = "user_attrs_aprc"
first = True

best_trials = {}

for study_name, df in zip(study_names, trials_dfs):
    if study_name.endswith("_ALL"):
        continue
    print(study_name)
    top = df.sort_values(metric, ascending=False)[:30]
    param_columns = [x for x in top.columns if x.startswith("params")]
    top = mark_outliers(top)
    display(top[["outlier", "value", metric, *param_columns]][:10])

    model_type = study_name.split("_")[-1]
    best_trials[model_type] = top.query("not outlier").iloc[0].to_dict()
    best_trials[model_type]["model_type"] = model_type

    print()

# %%
import mscproject.experiment as exp


# %%
# remove prefix from string
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


# %%
def build_experiment_from_trial_params(trial_params, dataset, verbose=False):
    param_dict = {
        remove_prefix(k, "params_"): v
        for k, v in trial_params.items()
        if k.startswith("params")
    }
    # Rename key from "n_layers" to "num_layers"
    if "n_layers" in param_dict:
        param_dict["num_layers"] = param_dict.pop("n_layers")
    param_dict["in_channels"] = -1
    param_dict["out_channels"] = 1
    param_dict["act_first"] = True
    param_dict["add_self_loops"] = False
    param_dict["model_type"] = mod.get_model(trial_params["model_type"])
    param_dict["v2"] = True
    lr = trial_params["user_attrs_learning_rate"]
    param_dict["jk"] = None if param_dict["jk"] == "none" else param_dict["jk"]
    if verbose:
        print(param_dict)
    return exp.get_model_and_optimiser(param_dict, dataset, lr)


# %%
dataset_path = "data/pyg/"

# Set the device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {str(device).upper()}")

# Load the dataset.
dataset = CompanyBeneficialOwners(dataset_path, to_undirected=True)
dataset = dataset.data.to(device)

model_metrics = {}

models_dir = Path("models/pyg")
models_dir.mkdir(parents=True, exist_ok=True)

for model_name in best_trials.keys():
    print("Training model:", model_name)
    trial_dict = best_trials[model_name]

    if trial_dict["params_add_self_loops"]:
        dataset = AddSelfLoops(fill_value=1.0)(dataset)  # type: ignore
    else:
        dataset = RemoveSelfLoops()(dataset)

    model, optimiser = build_experiment_from_trial_params(
        trial_dict, dataset, verbose=True
    )

    # Train and evaluate the model.
    best_epoch = int(trial_dict["user_attrs_best_epoch"])

    progress = tqdm(range(best_epoch))

    for epoch in progress:
        loss = exp.train(model, dataset, optimiser, on_val=True)
        progress.set_description(f"Train loss: {loss:.4f}")

    eval_metrics = exp.evaluate(
        model, dataset, on_train=False, on_val=False, on_test=True
    )

    model_metrics[model_name] = eval_metrics.test
    print(eval_metrics.test)

    # Save the trained model.
    torch.save(model.state_dict(), MODEL_DIR + f"{model_name}.pt")
    print()

# %%
from torch_geometric.transforms import AddSelfLoops
from mscproject.transforms import RemoveSelfLoops

# Load and evaluate models
for model_name in best_trials.keys():
    trial_params = best_trials[model_name]

    dataset = CompanyBeneficialOwners(dataset_path, to_undirected=True)
    dataset = dataset.data.to(device)

    model, _ = build_experiment_from_trial_params(trial_params, dataset)
    model.load_state_dict(torch.load(MODEL_DIR + f"{model_name}.pt"))

    model.to(device)

    if trial_params["params_add_self_loops"]:
        dataset = AddSelfLoops(fill_value=1.0)(dataset)
    else:
        dataset = RemoveSelfLoops()(dataset)

    eval_metrics = exp.evaluate(
        model, dataset, on_train=False, on_val=False, on_test=True
    )

    model_metrics[model_name] = eval_metrics.test
    print(eval_metrics.test)

# %%
model_metrics

# %%
import dataclasses as dc

# %%
pd.DataFrame.from_dict(model_metrics, orient="index")

# %%
