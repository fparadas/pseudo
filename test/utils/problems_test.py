import pytest
import numpy as np
from pseudo.utils.problems import rosenbrock, rosen_boundaries

def test_rosenbrock_known_input():
    x = np.array([1, 2, 3])
    expected = 201.0
    assert rosenbrock(x) == expected

def test_rosenbrock_zeros():
    x = np.zeros(5)
    assert rosenbrock(x) > 0

def test_rosenbrock_minimum_point():
    x = np.ones(5)
    assert rosenbrock(x) == 0

def test_rosen_boundaries_single_dimension():
    boundaries = rosen_boundaries(1)
    assert len(boundaries) == 1
    assert boundaries[0] == (-2.048, 2.048)

def test_rosen_boundaries_multiple_dimensions():
    dims = 5
    boundaries = rosen_boundaries(dims)
    assert len(boundaries) == dims
    for boundary in boundaries:
        assert boundary == (-2.048, 2.048)

def test_rosen_boundaries_zero_dimensions():
    boundaries = rosen_boundaries(0)
    assert len(boundaries) == 0