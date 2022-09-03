import dataclasses as dc

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

from torchmetrics.functional import (
    f1_score,
    average_precision,
    auroc,
    precision,
    recall,
    precision_recall,
    precision_recall_curve,
    accuracy,
)


@dc.dataclass
class EvalMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    average_precision: float

    @classmethod
    def from_tensors(
        cls,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        pos_weight_multiplier: float = 1.0,
    ):
        weights = torch.ones_like(y_true)
        weights[y_true == 1] = pos_weight_multiplier
        return EvalMetrics(
            loss=float(
                F.binary_cross_entropy(y_pred.float(), y_true.float(), weight=weights)
            ),
            accuracy=float(accuracy(y_pred.int(), y_true.int())),
            precision=float(precision(y_pred, y_true)),
            recall=float(recall(y_pred, y_true)),
            f1=float(f1_score(y_pred, y_true)),
            auroc=float(auroc(y_pred, y_true)),
            average_precision=float(average_precision(y_pred, y_true, pos_label=1).item()),
        )

    @classmethod
    def from_numpy(cls, y_pred: np.ndarray, y_true: np.ndarray, pos_weight_multiplier: float = 1.0):
        y_pred_tensor = torch.from_numpy(y_pred)
        y_true_tensor = torch.from_numpy(y_true)
        return cls.from_tensors(y_pred_tensor, y_true_tensor, pos_weight_multiplier)

    def __str__(self):
        return (
            f"loss: {self.loss:.4f}, "
            f"accuracy: {self.accuracy:.4f}, "
            f"precision: {self.precision:.4f}, "
            f"recall: {self.recall:.4f}, "
            f"f1: {self.f1:.4f}, "
            f"auroc: {self.auroc:.4f}, "
            f"average_precision: {self.average_precision:.4f}"
        )