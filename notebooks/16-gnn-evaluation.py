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
import optuna
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.transforms import AddSelfLoops
from mscproject.transforms import RemoveSelfLoops

from mscproject.datasets import CompanyBeneficialOwners
from mscproject.transforms import RemoveSelfLoops
import mscproject.models as mod
import mscproject.experiment as exp

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

# %%
MODEL_DIR = Path("data/models/pyg/weights-unregularised/")
OPTUNA_DB = Path("data/optuna-06.db")
DATASET_PATH = Path("data/pyg")
PREDICTION_DIR = Path("data/predictions")

model_names = [x.stem for x in MODEL_DIR.iterdir()]
model_names

# %%
dataset = CompanyBeneficialOwners(DATASET_PATH, to_undirected=True)
dataset = dataset.data.to("cpu")


# %%
def get_best_trial(model_name):
    study = optuna.load_study(
        study_name=f"pyg_model_selection_{model_name}_ARCHITECTURE",
        storage=f"sqlite:///{OPTUNA_DB}",
    )
    model_params = study.best_params
    user_attrs = study.best_trial.user_attrs
    return model_params, user_attrs


# %%
model_name = "GraphSAGE"
study = optuna.load_study(
    study_name=f"pyg_model_selection_{model_name}_ARCHITECTURE",
    storage=f"sqlite:///{OPTUNA_DB}",
)

# %%
study.trials_dataframe().sort_values("value", ascending=False)[:10].T

# %%
get_best_trial("GraphSAGE")

# %%
get_best_trial("KGNN")

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

model_metrics = {}

# Load and evaluate models
for model_name in model_names:

    print("Evaluating model:", model_name)

    model_path = MODEL_DIR / f"{model_name}.pt"

    if not model_path.exists():
        print(f"Model {model_name} does not exist, skipping")
        continue
    else:
        print(f"Loading model from: {model_path}")

    model_params, user_attrs = get_best_trial(model_name)
    user_attrs["model_type"] = model_name
    del user_attrs["aprc_history"]

    print(f"Using model params: {model_params}")
    print(f"Using user attrs: {user_attrs}")

    dataset = CompanyBeneficialOwners(DATASET_PATH, to_undirected=True)
    dataset = dataset.data.to(device)

    model, optimiser, _ = exp.build_experiment_from_trial_params(
        model_params, user_attrs, dataset
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    model.to(device)

    if model_params["add_self_loops"]:
        dataset = AddSelfLoops(fill_value=1.0)(dataset)
    else:
        dataset = RemoveSelfLoops()(dataset)

    eval_metrics = exp.evaluate(
        model, dataset, on_train=False, on_val=False, on_test=True
    )

    model_metrics[model_name] = eval_metrics.test
    print(model_name, eval_metrics.test)

    print("Making predictions...")
    prediction_dict = model(dataset.x_dict, dataset.edge_index_dict)
    prediction_df_list = []

    print("Saving predictions...")
    for node_type in dataset.node_types:
        prediction = (
            prediction_dict[node_type][dataset[node_type].test_mask]
            .cpu()
            .detach()
            .numpy()
            .flatten()
        )
        actual = (
            dataset.y_dict[node_type][dataset[node_type].test_mask]
            .cpu()
            .detach()
            .numpy()
            .flatten()
        )
        df = pd.DataFrame({"pred_proba": prediction, "actual": actual})
        prediction_df_list.append(df)

    prediction_df = pd.concat(prediction_df_list)
    prediction_df.to_csv(PREDICTION_DIR / f"{model_name}.csv", index=False)


# %%
performance_comparison = pd.DataFrame.from_dict(model_metrics, orient="index")
performance_comparison.to_csv("reports/test-performance-pyg.csv", index_label="model")

# %%
