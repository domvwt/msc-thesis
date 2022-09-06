from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor, sigmoid
from torch.nn import ModuleDict
from torch_geometric.nn import Linear
from torch_geometric.nn import models as pygmodels
from torch_geometric.nn.conv import (
    GraphConv,
    HANConv,
    HEATConv,
    HGTConv,
    MessagePassing,
)
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor


class SigmoidMixin(BasicGNN):
    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        *,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
    ) -> Tensor:
        x = super().forward(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
        return sigmoid(x)


class GCN(SigmoidMixin):
    """
    https://github.com/pyg-team/pytorch_geometric/discussions/3479
    """

    supports_edge_weight = True
    supports_edge_attr = False
    is_heterogeneous = False

    def init_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> MessagePassing:
        return GraphConv(in_channels, out_channels, **kwargs)


class GraphSAGE(pygmodels.GraphSAGE, SigmoidMixin):

    is_heterogeneous = False

    pass


class GAT(pygmodels.GAT, SigmoidMixin):

    is_heterogeneous = False

    pass


class BasicHeteroGNN(BasicGNN):

    is_heterogeneous = True

    output_act = torch.sigmoid

    def __init__(self, *args, metadata: Metadata, **kwargs):
        super().__init__(*args, metadata=metadata, **kwargs)

        # Replace homogeneous linear layer with heterogeneous.
        if hasattr(self, "lin"):
            del self.lin
            self.lin = ModuleDict()
            for node_type in metadata[0]:
                self.lin[str(node_type)] = Linear(-1, self.out_channels)

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        *,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
    ) -> dict:
        """"""
        # ! NOTE - Jumping Knowledge only works in mode 'last' !

        xs: Dict[str, List[Tensor]] = defaultdict(list)
        for i in range(self.num_layers):

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x_dict = self.convs[i](
                    x_dict,
                    edge_index_dict,
                    edge_weight=edge_weight,
                    edge_attr=edge_attr,
                )
            elif self.supports_edge_weight:
                x_dict = self.convs[i](x_dict, edge_index_dict, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x_dict = self.convs[i](x_dict, edge_index_dict, edge_attr=edge_attr)
            else:
                x_dict = self.convs[i](x_dict, edge_index_dict)
            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act is not None and self.act_first:
                x_dict = {key: self.act(x) for key, x in x_dict.items()}
            if self.norms is not None:
                x_dict = {key: self.norms[i](x) for x, key in x_dict.items()}
            if self.act is not None and not self.act_first:
                x_dict = {key: self.act(x) for key, x in x_dict.items()}
            x_dict = {
                key: F.dropout(x, p=self.dropout, training=self.training)
                for key, x in x_dict.items()
            }
            if hasattr(self, "jk"):
                for node_type, x in x_dict.items():
                    xs[str(node_type)].append(x)

        x_dict = (
            {key: self.jk(xs[key]) for key in x_dict.keys()}
            if hasattr(self, "jk")
            else x_dict
        )

        if hasattr(self, "lin"):
            for node_type, x in x_dict.items():
                x_dict[node_type] = self.lin[str(node_type)](x)

        if any(len(x.shape) != 1 for x in x_dict.values()):
            x_dict = {key: x.mean(dim=1) for key, x in x_dict.items()}

        # Check final output is a single dimensional tensor
        for node_type, x in x_dict.items():
            assert len(x.shape) == 1, f"{node_type} is not a single dimensional tensor"

        x_dict = {key: self.output_act(x) for key, x in x_dict.items()}

        return x_dict


class HAN(BasicHeteroGNN):

    supports_edge_weight = False
    supports_edge_attr = False
    is_heterogeneous = True

    def init_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> MessagePassing:
        if out_channels < kwargs["heads"]:
            out_channels = kwargs["heads"]
        return HANConv(in_channels, out_channels, **kwargs)


class HGT(BasicHeteroGNN):

    supports_edge_weight = False
    supports_edge_attr = False
    is_heterogeneous = True

    def init_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> MessagePassing:
        if out_channels < kwargs["heads"]:
            out_channels = kwargs["heads"]
        return HGTConv(in_channels, out_channels, **kwargs)


GNN = TypeVar("GNN", GCN, GraphSAGE, GAT, HAN, HGT)


def get_model(model_name) -> GNN:
    models = {
        "GCN": GCN,
        "GraphSAGE": GraphSAGE,
        "GAT": GAT,
        "HAN": HAN,
        "HGT": HGT,
    }
    return models[model_name]
