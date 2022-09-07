import argparse
import functools as ft
import math
import time
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

import mscproject.models as mod
from mscproject.datasets import CompanyBeneficialOwners
from mscproject.metrics import EvalMetrics


# Create parser for command line arguments.
def get_parser():
    parser = argparse.ArgumentParser(description="Optimise MSC model.")
    parser.add_argument(
        "-m",
        "--model-type-name",
        type=str,
        help="Model type to optimise. One of ['GCN', 'GraphSAGE', 'GAT', 'HAN', 'HGT'] or 'ALL'",
        required=True,
    )
    return parser


# Early Stopping Callback
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.epoch = 0
        self.counter = 0
        self.best_score = None
        self.best_loss = None
        self.best_epoch = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):

        self.epoch += 1
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.best_epoch = self.epoch
            self.counter = 0


def train(model, input_data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(input_data.x_dict, input_data.edge_index_dict)

    company_train_mask = input_data["company"].train_mask
    person_train_mask = input_data["person"].train_mask

    companies_out = out["company"][company_train_mask]
    persons_out = out["person"][person_train_mask]
    out_tensor = torch.cat((companies_out, persons_out), dim=0).float().squeeze()

    companies_y = input_data.y_dict["company"][company_train_mask]
    persons_y = input_data.y_dict["person"][person_train_mask]

    y_tensor = torch.cat((companies_y, persons_y), dim=0).float().squeeze()

    # Multiply importance of anomalous data by 10.
    importance = (y_tensor * 9) + 1

    loss = F.binary_cross_entropy(out_tensor, y_tensor, weight=importance)
    loss.backward()
    optimizer.step()

    return float(loss)


class EvalResult(NamedTuple):
    train: EvalMetrics
    val: EvalMetrics


@torch.no_grad()
def test(model, dataset) -> EvalResult:
    model.eval()

    prediction_dict = model(dataset.x_dict, dataset.edge_index_dict)

    eval_metrics_list = []

    for split in ["train_mask", "val_mask"]:

        masks = []
        actuals = []
        predictions = []

        for node_type in ["company", "person"]:
            mask = dataset[node_type][split]
            actual = dataset.y_dict[node_type][mask]
            prediction = prediction_dict[node_type][mask]

            masks.append(mask)
            predictions.append(prediction)
            actuals.append(actual)

        combined_predictions = torch.cat(predictions, dim=0).squeeze()
        combined_actuals = torch.cat(actuals, dim=0).squeeze()

        eval_metrics_list.append(
            EvalMetrics.from_tensors(
                combined_predictions, combined_actuals, pos_weight_multiplier=10
            )
        )

    return EvalResult(*eval_metrics_list)


def get_model_and_optimiser(
    param_dict: dict, dataset: HeteroData, learning_rate: float
):
    model_type = param_dict.pop("model_type")
    edge_aggregator = param_dict.pop("edge_aggr")
    weight_decay = param_dict.pop("weight_decay")

    hidden_channels = 2 ** param_dict.pop("hidden_channels_log2")
    param_dict["hidden_channels"] = hidden_channels

    # Instantiate the model.
    if model_type.is_heterogeneous:
        model = model_type(**param_dict, metadata=dataset.metadata())
    else:
        model = model_type(**param_dict)
        model = to_hetero(model, metadata=dataset.metadata(), aggr=edge_aggregator)

    # Initialise the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialise lazy modules.
    with torch.no_grad():
        _ = model(dataset.x_dict, dataset.edge_index_dict)

    # Initialise the optimiser.
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return model, optimiser


def optimise_model(trial: optuna.Trial, dataset: HeteroData, model_type_name: str):

    # Set the hyperparameters of the model.
    param_dict = {}

    # Select the model.
    if model_type_name == "ALL":
        model_type = mod.get_model(
            trial.suggest_categorical(
                "model", ["GCN", "GraphSAGE", "GAT", "HAN", "HGT"]
            )
        )
    else:
        model_type = mod.get_model(model_type_name)
    param_dict["model_type"] = model_type

    # NOTE: Jumping Knowledge restricted to 'last' due to difficulty of implementation
    #   for heterogeneous models. Other modes will *not* work.
    jk_choices = {
        "none": None,
        "last": "last",
        # "cat": "cat",
        # "max": "max",
    }
    jk_choice = jk_choices.get(
        str(trial.suggest_categorical("jk", sorted(jk_choices.keys())))
    )

    aggr_choices = ["sum", "mean", "min", "max"]
    heads_choices = [1, 2, 4, 8, 16]
    hidden_channels_min = 1

    if model_type.__name__ == "GCN":
        param_dict["aggr"] = trial.suggest_categorical("gcn_aggr", aggr_choices)
        param_dict["bias"] = trial.suggest_categorical("bias", [True, False])
    elif model_type.__name__ == "GAT":
        param_dict["v2"] = True
        param_dict["heads"] = trial.suggest_categorical("heads", heads_choices)
        param_dict["concat"] = trial.suggest_categorical("concat", [True, False])
        if param_dict["concat"]:
            hidden_channels_min = int(math.log2(param_dict["heads"]))
    elif model_type.__name__ == "HAN":
        param_dict["heads"] = trial.suggest_categorical("heads", heads_choices)
        param_dict["negative_slope"] = trial.suggest_float("negative_slope", 0.0, 1.0)
        param_dict["dropout"] = trial.suggest_float("han_dropout", 0, 1)
        hidden_channels_min = int(math.log2(param_dict["heads"]))
    elif model_type.__name__ == "HGT":
        param_dict["heads"] = trial.suggest_categorical("heads", heads_choices)
        param_dict["group"] = trial.suggest_categorical("group", aggr_choices)
        hidden_channels_min = int(math.log2(param_dict["heads"]))

    hidden_channels_log2 = trial.suggest_int(
        "hidden_channels_log2", hidden_channels_min, 8
    )
    param_dict["hidden_channels_log2"] = hidden_channels_log2
    hidden_channels = 2**hidden_channels_log2
    trial.set_user_attr("n_hidden", hidden_channels)

    param_dict.update(
        dict(
            in_channels=-1,
            hidden_channels=hidden_channels,
            num_layers=trial.suggest_int("n_layers", 1, 4),
            out_channels=1,
            dropout=trial.suggest_float("dropout", 0, 1),
            act=trial.suggest_categorical("act", ["relu", "gelu"]),
            # NOTE: act_first only has an effect when normalisation is used.
            act_first=True,
            # NOTE: normalisation is not used as data is not batched.
            norm=None,
            jk=jk_choice,
            add_self_loops=False,
        )
    )

    param_dict["edge_aggr"] = trial.suggest_categorical("edge_aggr", aggr_choices)
    param_dict["weight_decay"] = trial.suggest_float("weight_decay", 0.0, 0.2)

    # Set the learning rate.
    learning_rate = 0.01
    trial.set_user_attr("learning_rate", learning_rate)

    # Get the model and optimiser.
    model, optimiser = get_model_and_optimiser(
        param_dict, dataset, learning_rate=learning_rate
    )

    # Initialise the early stopping callback.
    early_stopping = EarlyStopping(patience=10, verbose=False)

    # Train and evaluate the model.
    max_epochs = 300
    val_loss = np.inf

    best_loss = np.inf
    best_eval_metrics = None

    while not early_stopping.early_stop and early_stopping.epoch < max_epochs:
        _ = train(model, dataset, optimiser)
        eval_metrics = test(model, dataset)
        val_loss = eval_metrics.val.loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_eval_metrics = eval_metrics
        early_stopping(val_loss)

    # Log the number of epochs.
    trial.set_user_attr("total_epochs", early_stopping.epoch)
    trial.set_user_attr("best_epoch", early_stopping.best_epoch)

    if best_eval_metrics:
        trial.set_user_attr("acc", best_eval_metrics.val.accuracy)
        trial.set_user_attr("precision", best_eval_metrics.val.precision)
        trial.set_user_attr("recall", best_eval_metrics.val.recall)
        trial.set_user_attr("f1", best_eval_metrics.val.f1)
        trial.set_user_attr("auc", best_eval_metrics.val.auroc)
        trial.set_user_attr("aprc", best_eval_metrics.val.average_precision)

    return early_stopping.best_loss


def main():
    parser = get_parser()
    args = parser.parse_args()
    assert args.model_type_name in {
        "GCN",
        "GraphSAGE",
        "GAT",
        "HAN",
        "HGT",
        "ALL",
    }, "Model type must be one of: 'GCN', 'GraphSAGE', 'GAT', 'HAN', 'HGT', 'ALL'"

    dataset_path = "data/pyg/"

    # Set the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset.
    dataset = CompanyBeneficialOwners(dataset_path, to_undirected=True)
    dataset = dataset.data.to(device)

    study_name = f"pyg_model_selection_{args.model_type_name}"

    # Delete study if it already exists.
    # optuna.delete_study(study_name, storage="sqlite:///optuna.db")

    # Optimize the model.
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )

    total_num_trials = 200
    current_trials = len(study.trials)
    remaining_trials = total_num_trials - current_trials
    optimise_model_partial = ft.partial(
        optimise_model, dataset=dataset, model_type_name=args.model_type_name
    )
    study.optimize(optimise_model_partial, n_trials=remaining_trials)

    # Print top models.
    print()
    print("Top Models:")
    drop_cols = ["datetime_start", "datetime_complete", "duration", "state"]
    trials_df: pd.DataFrame = study.trials_dataframe()
    trials_df = trials_df.sort_values("value").head(10)
    trials_df = trials_df.drop(drop_cols, axis=1)
    print(trials_df.T)

    # Plot the results.
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_contour(study).show()
    optuna.visualization.plot_slice(study).show()
    optuna.visualization.plot_param_importances(study).show()


if __name__ == "__main__":
    main()
