import numpy as np
from grid import setup_real_space_grid, setup_reciprocal_space_grid

def test_setup_real_space_grid():
    L = 10.0
    N = 4
    r_coords, dr = setup_real_space_grid(L, N)

    # Check shape
    assert r_coords.shape == (N, N, N, 3)

    # Check grid spacing
    expected_dr = L / N
    assert np.isclose(dr, expected_dr)

    # Check first point (should be (0,0,0))
    assert np.allclose(r_coords[0, 0, 0], [0.0, 0.0, 0.0])

    # Check last point (should be (L-dr, L-dr, L-dr))
    assert np.allclose(r_coords[-1, -1, -1], [L - dr, L - dr, L - dr])

    # Check a specific point
    assert np.allclose(r_coords[1, 0, 0], [dr, 0.0, 0.0])

def test_setup_reciprocal_space_grid_no_cutoff():
    L = 2 * np.pi # Makes G-vectors simple integers
    N = 4
    ecut = 100.0 # High cutoff to include all G-vectors

    g_vectors, g_squared, g_indices = setup_reciprocal_space_grid(L, N, ecut)

    # For N=4, the frequencies are [-2, -1, 0, 1] * (2*pi/L)
    # With L=2*pi, these become [-2, -1, 0, 1]
    expected_freqs = np.array([0.0, 1.0, -2.0, -1.0]) # numpy.fft.fftfreq order

    # Check number of G-vectors (should be N*N*N)
    assert len(g_vectors) == N**3

    # Check that G=0 is present
    assert any(np.allclose(g, [0, 0, 0]) for g in g_vectors)
    assert any(np.isclose(gs, 0.0) for gs in g_squared)

    # Check the range of G-vectors
    assert np.all(g_vectors >= -2.0)
    assert np.all(g_vectors <= 1.0)

    # Check that g_squared corresponds to g_vectors
    for i in range(len(g_vectors)):
        assert np.isclose(g_squared[i], np.sum(g_vectors[i]**2))

def test_setup_reciprocal_space_grid_with_cutoff():
    L = 2 * np.pi
    N = 4
    ecut = 0.6 # 0.5 * G^2 <= 0.6 => G^2 <= 1.2

    g_vectors, g_squared, g_indices = setup_reciprocal_space_grid(L, N, ecut)

    # Expected G-vectors (magnitudes squared):
    # G=0: 0
    # G=1 (e.g., (1,0,0)): 1
    # G=sqrt(2) (e.g., (1,1,0)): 2
    # G=sqrt(3) (e.g., (1,1,1)): 3
    # G=2 (e.g., (2,0,0)): 4
    # G=sqrt(5) (e.g., (2,1,0)): 5
    # G=sqrt(6) (e.g., (2,1,1)): 6
    # G=3 (e.g., (2,2,1)): 9

    # Only G^2 <= 1.2 should be included.
    # This means only G=0 and G=1 (and its permutations/signs) should be present.
    # G=0: (0,0,0) -> 1 vector
    # G=1: (+-1,0,0), (0,+-1,0), (0,0,+-1) -> 6 vectors
    # Total expected: 1 + 6 = 7 vectors

    assert len(g_vectors) == 7
    assert len(g_squared) == 7
    assert len(g_indices) == 7

    # Check that all included G-vectors satisfy the cutoff
    assert np.all(0.5 * g_squared <= ecut + 1e-9) # Add tolerance for float comparison

    # Check that G=0 is always included
    assert any(np.allclose(g, [0, 0, 0]) for g in g_vectors)
    assert any(np.isclose(gs, 0.0) for gs in g_squared)

    # Check that no G-vectors with G^2 > 1.2 are present
    assert np.all(g_squared <= 1.2 + 1e-9)

