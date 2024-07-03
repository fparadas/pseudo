import numpy as np

from pseudo.helpers import denorm, init
from pseudo.mutation import mutate
from pseudo.crossover import cross
from pseudo.selection import improve


def minimize(fobj, boundaries, NP, F, Cr, y, max_iter=100):
    """Minimize the objective function `fobj`

    Args:
        fobj (function): The objective function to minimize
        boundaries (List[Float]): The boundaries of the problem
        NP (int): The number of individuals in the population
        F (np.ndarray): The mutation rates
        Cr (float): The crossover rate
        y (int): The number of random samples to use in mutation
        max_iter (int, optional): The max number of generations. Defaults to 100.

    Yields:
        np.ndarray: The best individual found
        float: The fitness of the best individual
    """
    # step 0: setup utility functions
    fobj_vectorized = np.vectorize(fobj, signature="(n)->()")
    DIM = len(boundaries)

    # step 1: init population
    population, fitness, velocity = init(fobj_vectorized, boundaries, (NP, DIM))
    best = np.argmin(fitness)

    for _ in range(max_iter):
        # step 2: mutation
        mutants = mutate(F, y, best, velocity, population, fitness)

        # step 3: crossover
        trial = cross(Cr, population, mutants)

        # step 4: selection
        f_trial = fobj_vectorized(denorm(trial, boundaries))
        fitness, population = improve(f_trial, trial, fitness, population)

        # step 5: yield result
        best = np.argmin(fitness)

        yield denorm(population[best], boundaries), fitness[best]
