from typing import Any
import typing
import numpy as np

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
    ) -> dict[str, Any]:
        metrics = dict[str, float]()
        for metric_name in metrics_dict:
            metrics[metric_name] = metrics_dict[metric_name](self(X), y)
        return metrics