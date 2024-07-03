import numpy as np

def rosenbrock(x):
    """Rosenbrock function

    Args:
        x (np.ndarray): The input vector

    Returns:
        float: The value of the Rosenbrock function
    """
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def rosen_boundaries(dims):
    """The boundaries for the Rosenbrock function

    Args:
        dims (int): The number of dimensions
    
    Returns:
        List[Tuple[float, float]]: The boundaries for the Rosenbrock function with 'dims' dimensions
    """
    return [(-2.048, 2.048)] * dims
