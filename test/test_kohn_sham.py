import numpy as np
import pytest
from kohn_sham import kinetic_energy_operator, external_potential
from grid import setup_real_space_grid # Import setup_real_space_grid for testing V_ext

def test_kinetic_energy_operator():
    # Test with a simple G-squared array
    g_squared = np.array([0.0, 1.0, 4.0, 9.0])
    expected_T = np.array([0.0, 0.5, 2.0, 4.5])
    
    T_operator = kinetic_energy_operator(g_squared)
    
    np.testing.assert_allclose(T_operator, expected_T)
    assert T_operator.shape == g_squared.shape
    assert T_operator.dtype == np.float64

    # Test with an empty array
    g_squared_empty = np.array([])
    T_operator_empty = kinetic_energy_operator(g_squared_empty)
    assert T_operator_empty.shape == (0,)

    # Test with a larger array
    g_squared_large = np.linspace(0, 100, 1000)
    T_operator_large = kinetic_energy_operator(g_squared_large)
    np.testing.assert_allclose(T_operator_large, 0.5 * g_squared_large)

def test_external_potential():
    # Define a small grid for testing
    L = 2.0
    N = 2
    r_coords, dr = setup_real_space_grid(L, N) # r_coords will be (2,2,2,3)

    # Expected coordinates for a 2x2x2 grid (first point is (0,0,0), second is (1,0,0), etc.)
    # r_coords[0,0,0] = [0,0,0]
    # r_coords[1,0,0] = [1,0,0] (since L=2, N=2, dr=1)

    # Calculate the potential
    Z = 1.0
    V_ext = external_potential(r_coords, Z=Z)

    # Assert shape and dtype
    assert V_ext.shape == (N, N, N)
    assert V_ext.dtype == np.float64

    # Test value at the origin (r=0)
    # Due to the 1e-10 handling, it should be -Z / 1e-10 = -1e10
    expected_V_origin = -Z / 1e-10
    np.testing.assert_allclose(V_ext[0, 0, 0], expected_V_origin)

    # Test value at a non-origin point, e.g., r=(1,0,0)
    # The magnitude is 1.0. So V_ext = -Z / 1.0 = -1.0
    # r_coords[1,0,0] corresponds to (1.0, 0.0, 0.0)
    expected_V_at_1_0_0 = -Z / np.linalg.norm(r_coords[1,0,0])
    np.testing.assert_allclose(V_ext[1, 0, 0], expected_V_at_1_0_0)
    assert V_ext[1, 0, 0] == -1.0 # Specifically check for -1.0

    # Test with a different Z
    Z_test = 2.0
    V_ext_Z2 = external_potential(r_coords, Z=Z_test)
    np.testing.assert_allclose(V_ext_Z2[0, 0, 0], -Z_test / 1e-10)
    np.testing.assert_allclose(V_ext_Z2[1, 0, 0], -Z_test / np.linalg.norm(r_coords[1,0,0]))
    assert V_ext_Z2[1, 0, 0] == -2.0

    # Test that values far from origin are closer to zero
    # The point (1,1,1) has magnitude sqrt(3)
    # r_coords[1,1,1] corresponds to (1.0, 1.0, 1.0)
    expected_V_at_1_1_1 = -Z / np.linalg.norm(r_coords[1,1,1])
    np.testing.assert_allclose(V_ext[1, 1, 1], expected_V_at_1_1_1)
    assert V_ext[1, 1, 1] == -1.0 / np.sqrt(3)

