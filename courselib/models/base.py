from typing import Any
import typing
import numpy as np


class Model:
    """
    Base class for trainable models
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Applies the model to a given input
        """
        ...

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
        """
        Computes metrics of the model
        """
        metrics = dict[str, float]()
        for metric_name in metrics_dict:
            metrics[metric_name] = metrics_dict[metric_name](self(X), y)
        return metrics
