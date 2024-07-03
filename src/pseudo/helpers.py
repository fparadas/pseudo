import numpy as np

def denorm(x, bounds):
    """Denormalize the population `x` given the boundaries `bounds`

    Args:
        x (np.array): An array within the range of 0 and 1
        bounds (List[Float]): The boundaries of the problem

    Returns:
        np.array: The denormalized array
    """
    min_boundaries, max_boundaries = np.asarray(bounds).T
    diff = np.fabs(min_boundaries - max_boundaries)

    return (x * diff) + min_boundaries

def init(fobj, bounds, shape):
    """Initialize a population of shape `shape` with random uniform values between 0 and 1

    Args:
        fobj (function): A vectorized function
        bounds (list): The boundaries of our problem
        shape (tuple): A tuple containing 

    Returns:
        np.array: The population
        np.array: The fitness of the population
        np.array: The velocities of the population
    """

    population = np.random.uniform(0, 1, shape)

    return population, fobj(denorm(population, bounds)), np.zeros_like(population)