import numpy as np


def sparsity(vector: np.typing.NDArray[np.float64], epsilon: float = 0.0001) -> float:
    """
    Computes the approximate sparsity of a vector

    Parameters
    ----------
    vector :
        The vector to consider
    epsilon : float, optional
        The constant below which entries are classified as sparse
    """
    return float(np.less(vector, epsilon).sum())


def prox(z: np.typing.NDArray[np.float64], threshold: float = 0.1):
    """
    The proximal mapping operator

    Parameters
    ----------
    z :
        The vector to consider
    threshold : float, optional
        The constant below which entries are classified as sparse
    """
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)
