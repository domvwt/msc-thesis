import dataclasses as dc
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import (
    accuracy,
    auroc,
    average_precision,
    f1_score,
    precision,
    recall,
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
            average_precision=float(
                average_precision(y_pred, y_true, pos_label=1).item()
            ),
        )

    @classmethod
    def from_numpy(
        cls, y_pred: np.ndarray, y_true: np.ndarray, pos_weight_multiplier: float = 1.0
    ):
        y_pred_tensor = torch.from_numpy(y_pred)
        y_true_tensor = torch.from_numpy(y_true)
        return cls.from_tensors(y_pred_tensor, y_true_tensor, pos_weight_multiplier)

    def __str__(self):
        return (
            f"loss: {self.loss:.3f}, "
            f"acc: {self.accuracy:.3f}, "
            f"prc: {self.precision:.3f}, "
            f"rec: {self.recall:.3f}, "
            f"f1: {self.f1:.3f}, "
            f"auc: {self.auroc:.3f}, "
            f"aprc: {self.average_precision:.3f}"
        )


@dc.dataclass(frozen=True)
class EvalMetricsTuple:
    train: Optional[EvalMetrics] = None
    test: Optional[EvalMetrics] = None

    def __str__(self):
        return f"train: {self.train}, " f"test: {self.test}"
