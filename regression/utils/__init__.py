import numpy as np


def sparsity(vector: np.typing.NDArray[np.float64], epsilon: float = 0.0001) -> float:
    return float(np.less(vector, epsilon).sum())


def prox(z: np.typing.NDArray[np.float64], threshold: float = 0.1):
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)
