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
import os
import yaml

from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from torch_geometric.transforms.to_undirected import ToUndirected
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.data import HeteroData
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HeteroConv, Linear, SAGEConv, BatchNorm, to_hetero

# TODO: regularisation like https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch
# TODO: follow this example https://github.com/pyg-team/pytorch_geometric/issues/3958


while not Path("data") in Path(".").iterdir():
    os.chdir("..")


conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())
path = conf_dict["pyg_data"]
data = torch.load(path)
data = ToUndirected(merge=False)(data)
print(data)

# %%
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        # Sigmoid to get 0-1 values.
        return F.sigmoid(x)

# %%

model = GNN(hidden_channels=64, out_channels=1, num_layers=2)
model = to_hetero(model, metadata=data.metadata(), aggr="sum")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)

    companies_out = out["company"]
    persons_out = out["person"]
    out_tensor = torch.cat((companies_out, persons_out), dim=0).float().squeeze()

    companies_y = data.y_dict["company"]
    persons_y = data.y_dict["person"]

    y_tensor = torch.cat((companies_y, persons_y), dim=0).float().squeeze()

    loss = F.binary_cross_entropy(out_tensor, y_tensor)
    loss.backward()
    optimizer.step()

    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['author'][split]
        acc = (pred[mask] == data['author'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


for epoch in range(1, 101):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

# %%
