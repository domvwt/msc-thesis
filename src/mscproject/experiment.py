import argparse
import dataclasses as dc
import functools as ft
import time
from pathlib import Path
from pprint import pformat
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from torch_geometric.transforms import AddSelfLoops

import mscproject.models as mod
from mscproject.datasets import CompanyBeneficialOwners
from mscproject.metrics import EvalMetrics
from mscproject.transforms import RemoveSelfLoops

MODEL_DIR = Path("data/models/pyg")


# Create parser for command line arguments.
def get_parser():
    parser = argparse.ArgumentParser(description="Optimise GNN model.")
    parser.add_argument(
        "-m",
        "--model-type-name",
        type=str,
        choices=["KGNN", "GraphSAGE", "GAT", "HAN", "HGT", "ALL"],
        required=True,
        help="Model type to optimise.",
    )
    parser.add_argument(
        "-s",
        "--study-type",
        type=str,
        choices=["ARCHITECTURE", "REGULARISATION", "WEIGHTS"],
        required=True,
        help="Experiment type.",
    )
    parser.add_argument(
        "-b",
        "--base-study-type",
        type=str,
        required=False,
        choices=["ARCHITECTURE", "REGULARISATION"],
        default="",
        help=(
            "Best parameters from this study will be used as the starting point for the optimisation. "
        ),
    )
    parser.add_argument(
        "-n",
        "--n-trials",
        type=int,
        default=100,
        help="Total number of trials to run. Default 100.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        required=False,
        help="Overwrite the study if it exists.",
    )
    parser.add_argument(
        "--db", type=str, required=True, help="Database path for Optuna"
    )
    return parser


# Early Stopping Callback
class EarlyStopping:
    """Early stops the training if score doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0.0):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation score improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.epoch = 0
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, score):

        self.epoch += 1

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = self.epoch
            self.counter = 0


def train(model, input_data, optimizer, on_train=True, on_val=False, on_test=False):
    model.train()
    optimizer.zero_grad()
    out = model(input_data.x_dict, input_data.edge_index_dict)

    company_mask = torch.empty_like(input_data["company"].train_mask).to(torch.bool)
    person_mask = torch.empty_like(input_data["person"].train_mask).to(torch.bool)

    if on_train:
        company_mask += input_data["company"].train_mask
        person_mask += input_data["person"].train_mask

    if on_val:
        company_mask += input_data["company"].val_mask
        person_mask += input_data["person"].val_mask

    if on_test:
        company_mask += input_data["company"].test_mask
        person_mask += input_data["person"].test_mask

    companies_out = out["company"][company_mask]
    persons_out = out["person"][person_mask]
    out_tensor = torch.cat((companies_out, persons_out), dim=0).float().squeeze()

    companies_y = input_data.y_dict["company"][company_mask]
    persons_y = input_data.y_dict["person"][person_mask]

    y_tensor = torch.cat((companies_y, persons_y), dim=0).float().squeeze()

    # Multiply importance of anomalous data by 10.
    importance = (y_tensor * 9) + 1
    loss = F.binary_cross_entropy(out_tensor, y_tensor, weight=importance)

    loss.backward()
    optimizer.step()

    return float(loss)


@dc.dataclass
class EvalResult:
    train: Optional[EvalMetrics] = None
    val: Optional[EvalMetrics] = None
    test: Optional[EvalMetrics] = None


@torch.no_grad()
def evaluate(
    model, dataset: HeteroData, on_train=True, on_val=True, on_test=False
) -> EvalResult:
    model.eval()

    prediction_dict = model(dataset.x_dict, dataset.edge_index_dict)

    eval_split_mask = [on_train, on_val, on_test]
    eval_split_names = np.array(["train", "val", "test"])
    eval_splits = eval_split_names[eval_split_mask]

    eval_result = EvalResult()

    for split in eval_splits:

        actuals = []
        predictions = []

        for node_type in dataset.node_types:
            mask_name = f"{split}_mask"
            mask = dataset[node_type][mask_name]
            actual = dataset.y_dict[node_type][mask]
            prediction = prediction_dict[node_type][mask]

            predictions.append(prediction)
            actuals.append(actual)

        combined_predictions = torch.cat(predictions, dim=0).squeeze()
        combined_actuals = torch.cat(actuals, dim=0).squeeze()

        result = EvalMetrics.from_tensors(
            combined_predictions, combined_actuals, pos_weight_multiplier=10
        )

        setattr(eval_result, split, result)

    return eval_result


def get_model_and_optimiser(
    param_dict: dict, dataset: HeteroData, learning_rate: float
):
    param_dict = param_dict.copy()

    model_type = param_dict.pop("model_type")
    edge_aggregator = param_dict.pop("edge_aggr")
    weight_decay = param_dict.pop("weight_decay", 0)

    if "hidden_channels_log2" in param_dict:
        param_dict["hidden_channels"] = 2 ** param_dict.pop("hidden_channels_log2")

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


def train_optuna(
    trial: optuna.Trial,
    param_dict,
    dataset,
    model,
    optimiser,
    save_best=False,
    model_path=None,
):
    # Train and evaluate the model.
    max_epochs = 2000
    val_aprc = -np.inf

    best_aprc = -np.inf
    best_eval_metrics = None
    aprc_history = []

    # Initialise the early stopping callback.
    early_stopping = EarlyStopping(patience=200, delta=0.01, verbose=False)

    print("Training model:", flush=True)
    print(pformat(param_dict), flush=True)
    start_time = time.time()
    while not early_stopping.early_stop and early_stopping.epoch < max_epochs:
        _ = train(model, dataset, optimiser)
        eval_metrics = evaluate(model, dataset)
        val_aprc = eval_metrics.val.average_precision
        aprc_history.append(val_aprc)

        early_stopping(eval_metrics.val.average_precision)
        time_per_epoch = (time.time() - start_time) / (early_stopping.epoch + 1)

        if save_best and (
            (trial.number == 0 and val_aprc > best_aprc)
            or (trial.number > 0 and val_aprc > best_aprc > trial.study.best_value)
        ):
            print()
            print("Saving best model of study...", flush=True)
            assert model_path is not None, "Model path must be specified."
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)

        if val_aprc > best_aprc:
            best_aprc = val_aprc
            best_eval_metrics = eval_metrics

        print(
            "; ".join(
                [
                    f"Epoch: {early_stopping.epoch}",
                    f"Time per epoch: {time_per_epoch:.2f}s",
                    f"Val loss: {eval_metrics.val.loss:.4f}",
                    f"Val APRC: {val_aprc:.4f}",
                    f"Best Val APRC: {best_aprc:.4f}",
                ]
            ),
            flush=True,
            end="\r",
        )

    print()

    # Training time in HH:MM:SS.
    training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"Training time: {training_time}", flush=True)

    # Log the number of epochs.
    best_epoch = early_stopping.best_epoch or early_stopping.epoch
    trial.set_user_attr("total_epochs", early_stopping.epoch)
    trial.set_user_attr("best_epoch", best_epoch)

    # Log the aprc history.
    trial.set_user_attr("aprc_history", aprc_history)

    if best_eval_metrics:
        trial.set_user_attr("loss", best_eval_metrics.val.loss)
        trial.set_user_attr("acc", best_eval_metrics.val.accuracy)
        trial.set_user_attr("precision", best_eval_metrics.val.precision)
        trial.set_user_attr("recall", best_eval_metrics.val.recall)
        trial.set_user_attr("f1", best_eval_metrics.val.f1)
        trial.set_user_attr("auc", best_eval_metrics.val.auroc)
        trial.set_user_attr("aprc", best_eval_metrics.val.average_precision)

    print(f"Best epoch: {best_epoch}")
    print(best_eval_metrics.val, flush=True)

    return best_eval_metrics.val.average_precision


def optimise_architecture(
    trial: optuna.Trial, dataset: HeteroData, model_type_name: str
):
    # Clear the CUDA cache if applicable.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    # Set the hyperparameters of the model.
    param_dict = {}

    # Select the model.
    if model_type_name == "ALL":
        model_type = mod.get_model(
            trial.suggest_categorical(
                "model", ["KGNN", "GraphSAGE", "GAT", "HAN", "HGT"]
            )
        )
    else:
        model_type = mod.get_model(model_type_name)

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

    aggr_choices = {"sum", "mean", "min", "max", "lstm"}
    heads_min = 0
    heads_max = 4
    hidden_channels_min = 1
    hidden_channels_max = 8
    num_layers_max = 8

    add_self_loops = trial.suggest_categorical("add_self_loops", [True, False])

    if add_self_loops:
        dataset: HeteroData = AddSelfLoops(fill_value=1.0)(dataset)  # type: ignore
    else:
        dataset: HeteroData = RemoveSelfLoops()(dataset)

    if model_type.__name__ == "KGNN":
        param_dict["aggr"] = trial.suggest_categorical("gnn_aggr", aggr_choices)
        param_dict["bias"] = trial.suggest_categorical("bias", [True, False])
    elif model_type.__name__ == "GraphSAGE":
        param_dict["aggr"] = trial.suggest_categorical("gnn_aggr", aggr_choices)
        param_dict["bias"] = trial.suggest_categorical("bias", [True, False])
    elif model_type.__name__ == "GAT":
        param_dict["v2"] = trial.suggest_categorical("v2", [True])
        param_dict["heads_log2"] = trial.suggest_int("heads_log2", heads_min, heads_max)
        param_dict["concat"] = trial.suggest_categorical("concat", [True, False])
        if param_dict["concat"]:
            hidden_channels_min = int(param_dict["heads_log2"])
    elif model_type.__name__ == "HAN":
        param_dict["heads_log2"] = trial.suggest_int("heads_log2", heads_min, heads_max)
        param_dict["negative_slope"] = trial.suggest_float("negative_slope", 0.0, 1.0)
        param_dict["han_dropout"] = trial.suggest_float("han_dropout", 0, 1)
        hidden_channels_min = int(param_dict["heads_log2"])
    elif model_type.__name__ == "HGT":
        param_dict["heads_log2"] = trial.suggest_int("heads_log2", heads_min, heads_max)
        param_dict["group"] = trial.suggest_categorical("group", aggr_choices)
        hidden_channels_min = int(param_dict["heads_log2"])

    hidden_channels_log2 = trial.suggest_int(
        "hidden_channels_log2", hidden_channels_min, hidden_channels_max
    )
    hidden_channels = 2**hidden_channels_log2

    num_layers = trial.suggest_int("num_layers", 1, num_layers_max)
    trial.set_user_attr("n_hidden", hidden_channels)

    if "heads_log2" in param_dict:
        heads = 2 ** param_dict["heads_log2"]
        param_dict["heads"] = heads
        trial.set_user_attr("n_heads", heads)

    # Not to be confused with KGNN edge aggregation - this is used by the
    # `to_hetero` function.
    edge_aggr = trial.suggest_categorical("edge_aggr", aggr_choices)
    param_dict["weight_decay"] = 0

    param_dict.update(
        dict(
            model_type=model_type,
            in_channels=-1,
            hidden_channels=hidden_channels,
            out_channels=1,
            num_layers=num_layers,
            dropout=0,
            act=trial.suggest_categorical("act", ["leaky_relu", "relu", "gelu"]),
            # NOTE: act_first only has an effect when normalisation is used.
            act_first=True,
            edge_aggr=edge_aggr,
            # NOTE: normalisation is not used as data is not batched.
            norm=None,
            jk=jk_choice,
            # NOTE: add_self_loops is directly applied to the dataset.
            add_self_loops=False,
        )
    )

    # Set the learning rate.
    learning_rate = 0.01
    trial.set_user_attr("learning_rate", learning_rate)

    # Get the model and optimiser.
    model, optimiser = get_model_and_optimiser(
        param_dict, dataset, learning_rate=learning_rate
    )
    model_path = MODEL_DIR / "unregularised" / f"{model_type_name}.pt"
    return train_optuna(
        trial,
        param_dict,
        dataset,
        model,
        optimiser,
        save_best=True,
        model_path=model_path,
    )


def build_experiment_from_trial_params(
    model_params, user_attrs, dataset, verbose=False
):
    # Rename key from "n_layers" to "num_layers"
    model_params["in_channels"] = -1
    model_params["out_channels"] = 1
    model_params["act_first"] = True
    model_params["add_self_loops"] = False
    model_params["model_type"] = mod.get_model(user_attrs["model_type"])
    model_params["v2"] = True
    lr = user_attrs["learning_rate"]
    model_params["jk"] = None if model_params["jk"] == "none" else model_params["jk"]
    model, optimiser = get_model_and_optimiser(model_params, dataset, lr)
    return model, optimiser, model_params


def optimise_regularisation(
    trial: optuna.Trial, dataset: HeteroData, model_params: dict, user_attrs: dict
):
    # Clear the CUDA cache if applicable.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    if model_params["add_self_loops"]:
        dataset = AddSelfLoops(fill_value=1.0)(dataset)  # type: ignore
    else:
        dataset = RemoveSelfLoops()(dataset)

    model_params["weight_decay"] = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    model_params["dropout"] = trial.suggest_uniform("dropout", 0.0, 0.5)

    model, optimiser, param_dict = build_experiment_from_trial_params(
        model_params, user_attrs, dataset
    )
    model_path = MODEL_DIR / "regularised" / f"{user_attrs['model_type']}.pt"
    return train_optuna(
        trial,
        param_dict,
        dataset,
        model,
        optimiser,
        save_best=True,
        model_path=model_path,
    )


def optimise_weights(
    trial: optuna.Trial,
    dataset: HeteroData,
    model_params: dict,
    user_attrs: dict,
    model_path: Path,
):
    # Clear the CUDA cache if applicable.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    if model_params["add_self_loops"]:
        dataset = AddSelfLoops(fill_value=1.0)(dataset)  # type: ignore
    else:
        dataset = RemoveSelfLoops()(dataset)

    model_params["weight_decay"] = 0
    model_params["dropout"] = 0

    model, optimiser, param_dict = build_experiment_from_trial_params(
        model_params, user_attrs, dataset
    )
    return train_optuna(
        trial,
        param_dict,
        dataset,
        model,
        optimiser,
        save_best=True,
        model_path=model_path,
    )


def main():
    parser = get_parser()
    args = parser.parse_args()

    study_name = f"pyg_model_selection_{args.model_type_name}_{args.experiment_type}"
    base_study_name = (
        f"pyg_model_selection_{args.model_type_name}_{args.base_study_type}"
    )

    if args.overwrite:
        # Delete study if it already exists.
        optuna.delete_study(study_name, storage=args.db)

    # Create study
    study = optuna.create_study(
        direction=optuna.study.StudyDirection.MAXIMIZE,
        study_name=study_name,
        storage=args.db,
        load_if_exists=True,
    )

    completed_trials = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    remaining_trials = args.n_trials - completed_trials

    print(f"Completed trials: {completed_trials}")
    print(f"Remaining trials: {remaining_trials}")

    if remaining_trials <= 0:
        print(f"Study already contains {completed_trials} trials.")
    else:
        # Set the device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {str(device).upper()}")

        # Load the dataset.
        dataset_path = "data/pyg/"
        dataset = CompanyBeneficialOwners(dataset_path, to_undirected=True)
        dataset = dataset.data.to(device)

        if args.experiment_type == "ARCHITECTURE":
            trial_function = ft.partial(
                optimise_architecture,
                dataset=dataset,
                model_type_name=args.model_type_name,
            )
        elif args.experiment_type == "REGULARISATION":
            base_study = optuna.load_study(
                study_name=base_study_name,
                storage=args.db,
            )
            model_params = base_study.best_params
            user_attrs = base_study.best_trial.user_attrs
            user_attrs["model_type"] = args.model_type_name
            print("Using model params:\n", pformat(model_params))
            print("Using user attrs:\n", pformat(user_attrs))
            trial_function = ft.partial(
                optimise_regularisation,
                dataset=dataset,
                model_params=model_params,
                user_attrs=user_attrs,
            )
        elif args.experiment_type == "WEIGHTS":
            base_study = optuna.load_study(
                study_name=base_study_name,
                storage=args.db,
            )
            model_params = base_study.best_params
            user_attrs = base_study.best_trial.user_attrs
            user_attrs["model_type"] = args.model_type_name
            if args.base_study_type == "REGULARISATION":
                base_model_type = "regularised"
            else:
                base_model_type = "unregularised"
            model_path = MODEL_DIR / base_model_type / f"{user_attrs['model_type']}.pt"
            print("Using model params:\n", pformat(model_params))
            print("Using user attrs:\n", pformat(user_attrs))
            trial_function = ft.partial(
                optimise_weights,
                dataset=dataset,
                model_params=model_params,
                user_attrs=user_attrs,
                model_path=model_path,
            )
        else:
            raise ValueError(f"Unknown experiment type: {args.experiment_type}")

        study.optimize(trial_function, n_trials=remaining_trials)

    # Print top models.
    print()
    print("Top Models:")
    drop_cols = ["datetime_start", "datetime_complete", "duration", "state"]
    trials_df: pd.DataFrame = study.trials_dataframe()
    trials_df = trials_df.sort_values("value", ascending=False).head(10)
    trials_df = trials_df.drop(drop_cols, axis=1)
    print(trials_df.T)

    # Plot the results.
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_contour(study).show()
    optuna.visualization.plot_slice(study).show()
    optuna.visualization.plot_param_importances(study).show()


if __name__ == "__main__":
    main()
