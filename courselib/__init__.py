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
    generator: np.random.Generator | None = None,
) -> dict[str, list[float]]:
    """
    Updates a model using the given optimizer and dataset. Additionally tracks
    metrics during training.

    Parameters
    ----------
    model : Model
        The model to train
    optimizer : Optimizer
        The optimizer to train the model with
    X : NDArray
        Array containing the training samples
    y : NDArray
        Array containing the corresponding values
    metrics_dict : dict
        A dictionary holding additional metric to track
    num_epochs : int, optional
        The number of training epochs
    batch_size : int
        The batch size to use during training
    generator : optional
        Optional random number generator generator
    """
    if generator is None:
        generator = np.random.default_rng()

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
