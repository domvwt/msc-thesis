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
from ast import Call
from importlib.metadata import metadata
import os
from platform import release
import yaml
import dataclasses as dc

from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

import numpy as np
from torch import Tensor
from torch.nn import BCELoss
from torch_geometric.transforms.to_undirected import ToUndirected
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.data import HeteroData
from torch_geometric.datasets import DBLP
from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_hetero, GraphConv
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.models import MLP, GCN, GraphSAGE, PNA, GIN, GAT, Node2Vec
from torch.nn import Linear, ReLU, Sigmoid, Dropout
from torch_geometric.typing import Adj, OptTensor

from mscproject.metrics import EvalMetrics

# TODO: regularisation like https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch
# TODO: follow this example https://github.com/pyg-team/pytorch_geometric/issues/3958


while not Path("data") in Path(".").iterdir():
    os.chdir("..")


class MyDataset(InMemoryDataset):
    """
    https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html?highlight=inmemorydataset#creating-in-memory-datasets
    """

    data: HeteroData

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = type(self).__name__
        super().__init__(root, transform, pre_transform)
        loaded_data = torch.load(self.processed_paths[0])
        loaded_data = ToUndirected(merge=False)(loaded_data)
        self.data, self.slices = self.collate([loaded_data])  # type: ignore

    @property
    def processed_dir(self):
        assert self.root, "Please specify a root directory"
        return str(Path(self.root) / "processed")

    @property
    def processed_file_names(self):
        assert self.processed_dir is not None, "Please specify `processed_dir`"
        return "data.pt"

    def metadata(self):
        return self.data.metadata()


# * Heterogeneous Graph Samplers
# ? https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html#heterogeneous-graph-samplers
# * PyG provides various functionalities for sampling heterogeneous graphs, i.e. in the standard torch_geometric.loader.NeighborLoader
# * class or in dedicated heterogeneous graph samplers such as torch_geometric.loader.HGTLoader. This is especially useful for efficient
# * representation learning on large heterogeneous graphs, where processing the full number of neighbors is too computationally expensive.
# * Heterogeneous graph support for other samplers such as torch_geometric.loader.ClusterLoader or torch_geometric.loader.GraphSAINTLoader
# * will be added soon. Overall, all heterogeneous graph loaders will produce a HeteroData object as output, holding a subset of the original
# * data, and mainly differ in the way their sampling procedures works. As such, only minimal code changes are required to convert the
# * training procedure from full-batch training to mini-batch training.

# %%

conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())
data_path = "data/pyg/processed/data.pt"
dataset_path = "data/pyg/"

data = torch.load(data_path)
# data = ToUndirected(merge=False)(data)
print(data)

dataset = MyDataset(dataset_path)

USE_DATASET = True

if USE_DATASET:
    input_data: HeteroData = dataset[0]  # type: ignore
    input_metadata = dataset.metadata()
else:
    input_data: HeteroData = data  # type: ignore
    input_metadata = input_data.metadata()


class BinaryClassifierGAT(GAT):
    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        *,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
    ) -> Tensor:
        x = super().forward(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
        return F.sigmoid(x)


model = BinaryClassifierGAT(
    in_channels=-1,
    hidden_channels=16,
    num_layers=2,
    out_channels=1,
    heads=1,
    concat=True,
    v2=True,
    add_self_loops=False,
)
model = to_hetero(model, metadata=input_metadata, aggr="sum")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset, model = dataset.data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(dataset.x_dict, dataset.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0)


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


for epoch in range(1, 201):
    # ! Use average precision score to evaluate model.
    loss = train()
    metrics_dict = test()
    print(f"Epoch: {epoch:03d}")
    print(f"Train: {metrics_dict['train_mask']}")
    print(f"Valid: {metrics_dict['val_mask']}")
    print("-" * 79)

# %%
