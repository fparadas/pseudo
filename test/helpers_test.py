import numpy as np
import pytest
from pseudo.helpers import denorm, init

def test_denorm_single_boundary():
    x = np.array([0, 0.5, 1])
    bounds = [(0, 10)]
    expected = np.array([0, 5, 10])
    print(denorm(x, bounds))
    np.testing.assert_array_almost_equal(denorm(x, bounds), expected)

def test_denorm_multiple_boundaries():
    x = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    bounds = [(0, 10), (100, 200)]
    expected = np.array([[0, 100], [5, 150], [10, 200]])
    np.testing.assert_array_almost_equal(denorm(x, bounds), expected)

def test_denorm_zero_range_boundaries():
    x = np.array([0, 0.5, 1])
    bounds = [(5, 5)]
    expected = np.array([5, 5, 5])
    np.testing.assert_array_almost_equal(denorm(x, bounds), expected)

def test_denorm_negative_positive_boundaries():
    x = np.array([0, 0.5, 1])
    bounds = [(-10, 10)]
    expected = np.array([-10, 0, 10])
    np.testing.assert_array_almost_equal(denorm(x, bounds), expected)

def test_denorm_empty_array():
    x = np.array([])
    bounds = [(0, 10)]
    expected = np.array([])
    np.testing.assert_array_almost_equal(denorm(x, bounds), expected)

def test_init_with_valid_inputs():
    fobj = lambda x: np.sum(x, axis=1)
    bounds = [(-10, 10), (-20, 20)]
    shape = (3, 2)  # 3 particles, 2 dimensions
    population, fitness, velocities = init(fobj, bounds, shape)
    
    # Test shapes
    assert population.shape == shape
    assert fitness.shape == (3,)  # Assuming fobj returns a single fitness value per particle
    assert velocities.shape == shape
    
    # Test population range
    assert np.all(population >= 0) and np.all(population <= 1)
    
    # Test velocities initialization
    assert np.all(velocities == 0)