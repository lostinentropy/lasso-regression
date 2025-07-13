from typing import Any, override
import numpy as np

from .models.compressive import CompressiveLinearModel
from .utils import prox

class Optimizer:
    def step(self, _model: Any, _params: Any): ...


class SubGradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.01, lam: float = 0.01):
        self.learning_rate: float = learning_rate
        self.lam: float = lam

    @override
    def step(
        self,
        model: CompressiveLinearModel,
        params: dict[str, np.typing.NDArray[np.float64]],
    ):
        X = params["X"]
        y = params["y"]
        residual = model(X) - y
        grad = X.T @ residual / X.shape[0] + self.lam * np.sign(model.w)
        model.w -= self.learning_rate * grad


class ISTA(Optimizer):
    def __init__(self, learning_rate: float = 0.01, lam: float = 0.1) -> None:
        self.learning_rate: float = learning_rate
        self.lam: float = lam

    @override
    def step(
        self,
        model: CompressiveLinearModel,
        params: dict[str, np.typing.NDArray[np.float64]],
    ):
        X = params["X"]
        y = params["y"]
        grad = X.T @ (X @ model.w - y)

        model.w = prox(
            model.w - self.learning_rate * grad, self.learning_rate * self.lam
        )


class CoordinateDescent(Optimizer):
    def __init__(self, lam: float = 0.1) -> None:
        self.lam: float = lam

    @override
    def step(
        self,
        model: CompressiveLinearModel,
        params: dict[str, np.typing.NDArray[np.float64]],
    ):
        pass
