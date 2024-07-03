import numpy as np

def cross(Cr, mutants, population):
    """Cross the population with the mutants given the crossover rate `Cr`

    Args:
        Cr (float): The crossover rate
        mutants (np.array): The mutants 
        population (np.array): The population

    Returns:
        np.array: The new population after crossing with mutants
    """
    NP, DIM = population.shape

    cross_points = np.random.rand(NP, DIM) < Cr

    if not np.any(cross_points):
        cross_points[np.random.randint(0, NP)] = True

    return np.where(cross_points, mutants, population)