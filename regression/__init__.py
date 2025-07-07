from copy import deepcopy
from typing import Any, override
import typing
import numpy as np


def sparsity(vector: np.typing.NDArray[np.float64], epsilon: float = 0.001) -> float:
    return float(np.less(vector, epsilon).sum())


class Model:
    def __call__(self, *args: Any, **kwds: Any) -> Any: ...

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
    ) -> dict[str, float]:
        metrics = dict[str, float]()
        for metric_name in metrics_dict:
            metrics[metric_name] = metrics_dict[metric_name](self(X), y)
        return metrics


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
    ) -> dict[str, float]:
        metrics = super().compute_metrics(X, y, metrics_dict)

        metrics["sparsity"] = sparsity(self.w)

        return metrics


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


def fit(
    model: Model,
    optimizer: Optimizer,
    X: np.typing.NDArray[np.float64],
    y: np.typing.NDArray[np.float64],
    metrics_dict: dict[
        str,
        typing.Callable[
            [np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]],
            float,
        ],
    ],
    num_epochs: int = 1000,
    batch_size: int = 100,
    seed: int = 42,
) -> dict[str, list[float]]:
    generator = np.random.default_rng(seed=seed)

    metrics = model.compute_metrics(X, y, metrics_dict)
    metrics_history: dict[str, list[float]] = {name: [] for name in metrics}
    for name in metrics_dict:
        metrics_history[name].append(metrics[name])

    for _ in range(num_epochs):
        indices = generator.permutation(len(X))
        batches = np.array_split(indices, np.ceil(len(X) / batch_size))

        for _ in batches:
            optimizer.step(model, {"X": X, "y": y})

            metrics = model.compute_metrics(X, y, metrics_dict)
            for name in metrics_history:
                metrics_history[name].append(metrics[name])

    return metrics_history
