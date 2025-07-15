import numpy as np


def random_sparse(
    dim: int,
    sparsity: int,
    loc: float = 0.0,
    scale: float = 1.0,
    generator: np.random.Generator | None = None,
):
    """
    Create a random sparse vector

    Parameters
    ----------
    dim : int
        The number of dimensions in the vector.
    sparsity : int
        The number on nonzero entries.
    loc : float, optional
        Mean (“centre”) of the distribution.
    scale : float, optional
        Standard deviation (spread or “width”) of the distribution. Must be
        non-negative.
    """
    if generator is None:
        generator = np.random.default_rng()

    w = generator.normal(loc, scale, size=[dim])
    selected_indices = generator.choice(dim, dim - sparsity, replace=False)
    np.put(w, selected_indices, 0.0)

    return w


def generate_dataset(
    dim: int,
    number_of_items: int,
    sparsity: int,
    noise_scale: float = 1.0,
    generator: np.random.Generator | None = None,
    w: np.typing.NDArray[np.float64] | None = None,
) -> tuple[
    np.typing.NDArray[np.float64],
    np.typing.NDArray[np.float64],
    np.typing.NDArray[np.float64],
]:
    if generator is None:
        generator = np.random.default_rng()

    X = generator.normal(size=[number_of_items, dim])
    if w is None:
        w = random_sparse(dim, sparsity, generator=generator)
    epsilon = generator.normal(scale=noise_scale, size=[number_of_items])

    y = X @ w + epsilon

    return (X, y, w)
