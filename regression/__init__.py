import typing
import numpy as np

from .models.base import Model
from .optimizer import Optimizer


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
