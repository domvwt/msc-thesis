from enum import Enum
from typing import TypeVar

from torch import Tensor, sigmoid
from torch_geometric.nn import models as pygmodels
from torch_geometric.nn.conv import GraphConv, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.typing import Adj, OptTensor


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

    def init_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> MessagePassing:
        return GraphConv(in_channels, out_channels, **kwargs)


class GraphSAGE(pygmodels.GraphSAGE, SigmoidMixin):
    pass


class GAT(pygmodels.GAT, SigmoidMixin):
    pass


GNN = TypeVar("GNN", GCN, GraphSAGE, GAT)


class ModelType(Enum):
    GCN = "GCN"
    GraphSAGE = "GraphSAGE"
    GAT = "GAT"


def get_model(model_name) -> GNN:
    models = {
        "GCN": GCN,
        "GraphSAGE": GraphSAGE,
        "GAT": GAT,
    }
    return models[model_name]
