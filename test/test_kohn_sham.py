import numpy as np
from kohn_sham import kinetic_energy_operator

def test_kinetic_energy_operator():
    # Test with G=0
    g_squared_zero = np.array([0.0])
    T_zero = kinetic_energy_operator(g_squared_zero)
    assert np.isclose(T_zero[0], 0.0)

    # Test with a positive G^2
    g_squared_positive = np.array([4.0]) # Corresponds to G=2
    T_positive = kinetic_energy_operator(g_squared_positive)
    assert np.isclose(T_positive[0], 0.5 * 4.0)
    assert np.isclose(T_positive[0], 2.0)

    # Test with multiple G^2 values
    g_squared_multiple = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    T_multiple = kinetic_energy_operator(g_squared_multiple)
    expected_T_multiple = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    assert np.allclose(T_multiple, expected_T_multiple)

    # Test with an empty array
    g_squared_empty = np.array([])
    T_empty = kinetic_energy_operator(g_squared_empty)
    assert T_empty.shape == (0,)

