from copy import deepcopy
from typing import Any, override
import typing
import numpy as np

from .base import Model
from ..utils import sparsity

class CompressiveLinearModel(Model):
    def __init__(self, w: np.typing.NDArray[np.float64]):
        self.w: np.typing.NDArray[np.float64] = deepcopy(w)

    @override
    def __call__(
        self, X: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        return X @ self.w

    @override
    def compute_metrics(
        self,
        X: np.typing.NDArray[np.float64],
        y: np.typing.NDArray[np.float64],
        metrics_dict: dict[
            str,
            typing.Callable[
                [np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]],
                float,
            ],
        ],
    ) -> dict[str, Any]:
        metrics = super().compute_metrics(X, y, metrics_dict)

        metrics["sparsity"] = sparsity(self.w)

        metrics["path"] = deepcopy(self.w)

        return metrics