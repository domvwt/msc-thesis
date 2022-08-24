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

from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HeteroConv, Linear, SAGEConv, BatchNorm

# TODO: regularisation like https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch
# TODO: follow this example https://github.com/pyg-team/pytorch_geometric/issues/3958


while not Path("data") in Path(".").iterdir():
    os.chdir("..")

class MyDataset(InMemoryDataset):
    """
    https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html?highlight=inmemorydataset#creating-in-memory-datasets
    """
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = type(self).__name__
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate([torch.load(self.processed_paths[0])])


    @property
    def processed_dir(self):
        assert self.root, "Please specify a root directory"
        return str(Path(self.root) / "processed")

    @property
    def processed_file_names(self):
        assert self.processed_dir is not None, "Please specify `processed_dir`"
        return "data.pt"


# path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
path = Path("data/pyg/MyDataset")
path.mkdir(parents=True, exist_ok=True)
dataset = MyDataset(str(path))
data = dataset[0]
print(data)

# %%

# We initialize conference node features with a single feature.
# data['conference'].x = torch.ones(data['conference'].num_nodes, 1)

# class HeteroGNN(torch.nn.Module):
#     def __init__(self, metadata, hidden_channels, out_channels, num_layers):
#         super().__init__()

#         self.convs = torch.nn.ModuleList()
#         for _ in range(num_layers):
#             conv = HeteroConv({
#                 edge_type: SAGEConv((-1, -1), hidden_channels)
#                 for edge_type in metadata[1]
#             })
#             self.convs.append(conv)

#         self.lin = Linear(hidden_channels, out_channels)

#     def forward(self, x_dict, edge_index_dict):
#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)
#             x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
#         return self.lin(x_dict["author"])


# %%
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            if _ == num_layers - 1:
                # Overwrite number of channels in last layer
                hidden_channels = out_channels
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), out_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        self.batchnorm_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.batchnorm_dict[node_type] = BatchNorm(hidden_channels)

        self.convs.append(Linear(hidden_channels, out_channels))


    def forward(self, x_dict, edge_index_dict, p_dropout=0.0):

        # Dropout
        x_dict = {key: F.dropout(x, p=p_dropout, training=self.training) for key, x in x_dict.items()}
        for i in range(len(self.convs)-1):
            x_dict.update(self.convs[i](x_dict, edge_index_dict))
            # Batch normalisation
            # ? FIXME: this is not working as expected
            # x_dict.update({key: self.batchnorm_dict[key](x) for key, x in x_dict.items()})
            # Activation function
            x_dict.update({key: F.leaky_relu(x) for key, x in x_dict.items()})
            # Dropout
            x_dict.update({key: F.dropout(x, p=p_dropout, training=self.training) for key, x in x_dict.items()})
        return self.convs[-1](x_dict, edge_index_dict)   

# %%


model = HeteroGNN(data.metadata(), hidden_channels=64, out_channels=1, num_layers=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    # out_keys = out.keys()
    # assert len(out_keys) == 1, "Only one output expected"
    # node_type = tuple(out_keys)[0]

    companies_in_output = torch.tensor(out["company"]).numel() > 0
    persons_in_output = torch.tensor(out["person"]).numel() > 0

    companies_mask = data["company"].train_mask
    persons_mask = data["person"].train_mask
    companies_out = out["company"][companies_mask] if companies_in_output else torch.tensor([])
    persons_out = out["person"][persons_mask] if persons_in_output else torch.tensor([])
    out_tensor = torch.cat((companies_out, persons_out), dim=1).squeeze().float()
    companies_y = data["company"].y[companies_mask] if companies_in_output else torch.tensor([])
    persons_y = data["person"].y[persons_mask] if persons_in_output else torch.tensor([])
    y_tensor = torch.cat((companies_y, persons_y), dim=0).float()
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
