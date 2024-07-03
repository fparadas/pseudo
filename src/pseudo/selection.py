import numpy as np

def improve(f_trial, trial, fitness, population):
    """Selects the best individuals from the trial and the population.

    Args:
        f_trial (np.ndarray): Fitness of the trial population.
        trial (np.ndarray): Trial population.
        fitness (np.ndarray): Fitness of the population.
        population (np.ndarray): Population.

    Returns:
        np.ndarray: The updated fitness of the population.
        np.ndarray: The updated population
    """
    improved = f_trial < fitness
    population[improved] = trial[improved]
    fitness[improved] = f_trial[improved]

    return fitness, population