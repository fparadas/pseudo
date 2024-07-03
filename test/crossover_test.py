import numpy as np
import pytest
from pseudo.crossover import cross

def test_no_crossover_points():
    np.random.seed(1)  # Setting seed for predictable randomness
    Cr = 0.0  # No crossover should happen, but function forces at least one
    mutants = np.array([[1, 1, 1], [1, 1, 1]])
    population = np.array([[0, 0, 0], [0, 0, 0]])
    result = cross(Cr, mutants, population)
    assert np.any(result != population)  # At least one element should differ

def test_all_crossover_points():
    np.random.seed(2)
    Cr = 1.0  # All elements should crossover
    mutants = np.array([[1, 1], [1, 1]])
    population = np.array([[0, 0], [0, 0]])
    expected = mutants
    result = cross(Cr, mutants, population)
    np.testing.assert_array_equal(result, expected)

def test_shape_preservation():
    np.random.seed(3)
    Cr = 0.5
    mutants = np.array([[1, 1, 1], [1, 1, 1]])
    population = np.array([[0, 0, 0], [0, 0, 0]])
    result = cross(Cr, mutants, population)
    assert result.shape == population.shape
