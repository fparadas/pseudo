import numpy as np
import pytest
from pseudo.mutation import mutate

def test_mutate_valid_inputs():
    w = np.array([0.5, 0.3, 0.2])
    y = 2
    best = 0
    population = np.array([[0.2, 0.3], [0.4, 0.5], [-0.3, -0.2]])
    fitness = np.array([0.1, 0.2, 0.05])
    velocity = np.random.rand(*population.shape)
    expected_shape = population.shape
    result = mutate(w, y, best, velocity, population, fitness)
    assert result.shape == expected_shape, "Output shape does not match expected shape."
    assert np.all(result >= 0) and np.all(result <= 1), "Output values are not within the expected range."

def test_mutate_edge_cases():
    w = np.array([100, 100, 100])  # Extreme values for w
    y = 1  # Minimum possible value for y
    best = 0
    population = np.array([[0.01, 0.01], [0.02, 0.02]])  # Small population
    velocity = np.ones_like(population) * np.array([1000, -1000])  # Extreme velocity values
    fitness = np.array([0.1, 0.2])
    result = mutate(w, y, best, velocity, population, fitness)
    assert np.all(result >= -1) and np.all(result <= 1), "Output values are not within the expected range with extreme inputs."

def test_mutate_output_type():
    w = np.array([0.5, 0.3, 0.2])
    y = 2
    best = 0
    velocity = np.array([0.1, -0.1])
    population = np.array([[0.2, 0.3], [0.4, 0.5], [-0.3, -0.2]])
    fitness = np.array([0.1, 0.2, 0.05])
    result = mutate(w, y, best, velocity, population, fitness)
    assert isinstance(result, np.ndarray), "Output is not a numpy array."