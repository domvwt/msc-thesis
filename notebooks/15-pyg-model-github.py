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
import yaml

from pathlib import Path

import torch
import torch.nn.functional as F


from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_hetero


from mscproject.metrics import EvalMetrics
from mscproject import models
from mscproject.datasets import CompanyBeneficialOwners

# TODO: regularisation like https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch
# TODO: follow this example https://github.com/pyg-team/pytorch_geometric/issues/3958

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

# %%
conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())
dataset_path = "data/pyg/"

dataset = CompanyBeneficialOwners(dataset_path, to_undirected=True)

input_data = dataset[0]  # type: ignore
input_metadata = dataset.metadata()

model = models.GAT(
    in_channels=-1,
    hidden_channels=16,
    num_layers=3,
    out_channels=1,
    jk="last",
    # heads=1,
    # concat=True,
    v2=True,
    add_self_loops=False,
)

model = to_hetero(model, metadata=input_metadata, aggr="sum")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset, model = dataset.data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(dataset.x_dict, dataset.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0)


# %%
def train():
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


# %%
@torch.no_grad()
def test():
    model.eval()

    prediction_dict = model(input_data.x_dict, input_data.edge_index_dict)

    metrics_dict = {}

    for split in ["train_mask", "val_mask"]:

        masks = []
        actuals = []
        predictions = []

        for node_type in ["company", "person"]:
            mask = input_data[node_type][split]
            actual = input_data.y_dict[node_type][mask]
            prediction = prediction_dict[node_type][mask]

            masks.append(mask)
            predictions.append(prediction)
            actuals.append(actual)

        combined_predictions = torch.cat(predictions, dim=0).squeeze()
        combined_actuals = torch.cat(actuals, dim=0).squeeze()

        metrics_dict[split] = EvalMetrics.from_tensors(
            combined_predictions, combined_actuals, pos_weight_multiplier=10
        )

    return metrics_dict


metrics_history = []

for epoch in range(1, 101):
    # # ! Use average precision score to evaluate model.
    loss = train()
    metrics_dict = test()
    metrics_history.append(metrics_dict)
    print(f"Epoch: {epoch:03d}")
    print(f"Train: {metrics_dict['train_mask']}")
    print(f"Valid: {metrics_dict['val_mask']}")
    print("-" * 79)

# %%
dataset["company"].feature_names
