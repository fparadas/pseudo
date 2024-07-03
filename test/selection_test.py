import numpy as np
import pytest
from pseudo.selection import improve

def test_improvement():
    population = np.array([[1, 2], [3, 4]])
    trial = np.array([[0, 1], [2, 3]])
    fitness = np.array([5, 7])
    f_trial = np.array([1, 5])

    expected_population = trial
    expected_fitness = np.array([1, 5])

    new_fitness, new_population = improve(f_trial, trial, fitness, population)

    assert np.array_equal(new_population, expected_population)
    assert np.array_equal(new_fitness, expected_fitness)

def test_no_improvement():
    population = np.array([[1, 2], [3, 4]])
    trial = np.array([[2, 3], [4, 5]])
    fitness = np.array([5, 7])
    f_trial = np.array([6, 8])

    expected_population = population.copy()
    expected_fitness = fitness.copy()

    new_fitness, new_population = improve(f_trial, trial, fitness, population)

    assert np.array_equal(new_population, expected_population)
    assert np.array_equal(new_fitness, expected_fitness)

def test_partial_improvement():
    population = np.array([[1, 2], [3, 4], [5, 6]])
    trial = np.array([[0, 1], [4, 5], [4, 5]])
    fitness = np.array([5, 7, 11])
    f_trial = np.array([1, 8, 10])

    expected_population = np.array([[0, 1], [3, 4], [4, 5]])
    expected_fitness = np.array([1, 7, 10])

    new_fitness, new_population = improve(f_trial, trial, fitness, population)

    assert np.array_equal(new_population, expected_population)
    assert np.array_equal(new_fitness, expected_fitness)

def test_equality():
    population = np.array([[1, 2], [3, 4]])
    trial = np.array([[1, 2], [3, 4]])
    fitness = np.array([5, 7])
    f_trial = np.array([5, 7])

    expected_population = population.copy()
    expected_fitness = fitness.copy()

    new_fitness, new_population = improve(f_trial, trial, fitness, population)

    assert np.array_equal(new_population, expected_population)
    assert np.array_equal(new_fitness, expected_fitness)