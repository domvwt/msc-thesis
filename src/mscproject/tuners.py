import functools as ft
import math
from typing import NamedTuple

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

import mscproject.models as mod
from mscproject.datasets import CompanyBeneficialOwners
from mscproject.metrics import EvalMetrics


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


def optimise_model(trial: optuna.Trial, dataset: HeteroData, device: torch.device):

    # Select the model.
    model_type = mod.get_model(
        trial.suggest_categorical("model", ["GCN", "GraphSAGE", "GAT"])
    )

    norm_choices = {"none": None, "BatchNorm": "BatchNorm"}
    norm_choice = norm_choices.get(
        str(trial.suggest_categorical("norm", ["none", "BatchNorm"])), None
    )

    jk_choices = {
        "none": None,
        "last": "last",
        "cat": "cat",
        "max": "max",
    }
    jk_choice = jk_choices.get(
        str(trial.suggest_categorical("jk", sorted(jk_choices.keys())))
    )

    # Set the hyperparameters of the model.
    param_dict = {}

    aggr_choices = ["sum", "mean", "min", "max"]
    hidden_channels_min = 0

    if model_type.__name__ == "GCN":
        param_dict["aggr"] = trial.suggest_categorical("gcn_aggr", aggr_choices)
        param_dict["bias"] = trial.suggest_categorical("bias", [True, False])
    elif model_type.__name__ == "GAT":
        param_dict["v2"] = True
        param_dict["heads"] = trial.suggest_categorical("heads", [1, 2, 4, 8, 16])
        param_dict["concat"] = trial.suggest_categorical("concat", [True, False])
        if param_dict["concat"]:
            hidden_channels_min = int(math.log2(param_dict["heads"]))

    hidden_channels_pow = trial.suggest_int(
        "hidden_channels_pow", hidden_channels_min, 7
    )
    hidden_channels = 2**hidden_channels_pow

    trial.set_user_attr("n_hidden", hidden_channels)

    param_dict.update(
        dict(
            in_channels=-1,
            hidden_channels=hidden_channels,
            num_layers=trial.suggest_int("n_layers", 1, 4),
            out_channels=1,
            dropout=trial.suggest_float("dropout", 0, 0.5),
            act=trial.suggest_categorical("act", ["relu", "gelu"]),
            act_first=trial.suggest_categorical("act_first", [True, False]),
            norm=norm_choice,
            jk=jk_choice,
            add_self_loops=False,
        )
    )

    # Instantiate the model.
    model = model_type(**param_dict)
    model = to_hetero(
        model,
        metadata=dataset.metadata(),
        aggr=str(trial.suggest_categorical("edge_aggr", aggr_choices)),
    )

    # Initialize the model.
    model.to(device)

    # Initialize lazy modules.
    with torch.no_grad():  # Initialize lazy modules.
        _ = model(dataset.x_dict, dataset.edge_index_dict)

    # Set the learning rate.
    trial.set_user_attr("learning_rate", 0.01)

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trial.user_attrs["learning_rate"],
        weight_decay=trial.suggest_float("weight_decay", 0, 0.1),
    )

    # Initialize the early stopping callback.
    early_stopping = EarlyStopping(patience=10, verbose=False)

    # Train and evaluate the model.
    max_epochs = 200
    val_loss = np.inf

    while not early_stopping.early_stop and early_stopping.epoch < max_epochs:
        _ = train(model, dataset, optimizer)
        eval_metrics = test(model, dataset)
        val_loss = eval_metrics.val.loss
        early_stopping(val_loss)

    # Log the number of epochs.
    trial.set_user_attr("total_epochs", early_stopping.epoch)
    trial.set_user_attr("best_epoch", early_stopping.best_epoch)

    return early_stopping.best_loss


def main():
    dataset_path = "data/pyg/"

    # Set the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset.
    dataset = CompanyBeneficialOwners(dataset_path, to_undirected=True)
    dataset = dataset.data.to(device)

    # Optimize the model.
    study = optuna.create_study(
        direction="minimize",
        study_name="pyg_model_selection",
        storage="sqlite:///pyg_model_selection.db",
        load_if_exists=True,
    )

    optimise_model_partial = ft.partial(optimise_model, dataset=dataset, device=device)
    study.optimize(optimise_model_partial, n_trials=200)

    # Print the best model.
    print(study.best_params)
    print(study.best_value)


if __name__ == "__main__":
    main()
