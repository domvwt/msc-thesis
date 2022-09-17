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
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.transforms import AddSelfLoops
from torch_geometric.nn import to_hetero
from mscproject.transforms import RemoveSelfLoops

from mscproject.datasets import CompanyBeneficialOwners
from mscproject.transforms import RemoveSelfLoops
import mscproject.models as mod
import mscproject.experiment as exp

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {str(device).upper()}")

dataset_path = "data/pyg/"

dataset = CompanyBeneficialOwners(dataset_path, to_undirected=True)
dataset = dataset.data.to(device)


# %% [markdown]
# ## GCN

# %% [markdown]
# ### TODO
#
# - Focus on GCN and GraphSAGE
# - Choose correct depth and layer size
# - Choose best optimizer
# - Choose best aggregation functions
# - Leave out weight decay and dropout for now

# %%
def get_model_and_optimiser():
    model = mod.GCN(
        in_channels=-1,
        out_channels=1,
        hidden_channels=2**5,
        num_layers=5,
        dropout=0.0,
        metadata=dataset.metadata,
        act="leaky_relu",
        aggr="sum",
    )

    model = to_hetero(model, dataset.metadata(), aggr="sum")

    # Initialise lazy modules.
    with torch.no_grad():
        _ = model(dataset.x_dict, dataset.edge_index_dict)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    # optimiser = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0)
    # optimiser = torch.optim.RMSprop(model.parameters(), lr=0.005, weight_decay=0)

    return model, optimiser


# %%
train_losses = []
val_losses = []
val_aprcs = []

model, optimiser = get_model_and_optimiser()

# %%
for i in tqdm(range(500)):
    # model.reset_parameters()
    loss = exp.train(model, dataset, optimiser)
    eval_result = exp.evaluate(model, dataset)
    train_losses.append(eval_result.train.loss)
    val_losses.append(eval_result.val.loss)
    val_aprcs.append(eval_result.val.average_precision)

# %%
eval_result.train.loss

# %%
loss

# %%
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
# plt.ylim(0, 2)

# %%
plt.plot(val_aprcs)

# %%
np.array(val_aprcs).max()

# %%
