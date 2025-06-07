import numpy as np

def kinetic_energy_operator(g_squared):
    """
    Calculates the diagonal kinetic energy operator in reciprocal space.

    Args:
        g_squared (np.ndarray): Array of squared G-vector magnitudes.

    Returns:
        np.ndarray: Diagonal kinetic energy operator (1D array).
                    T_G = 0.5 * G^2 in atomic units.
    """
    # In atomic units, T = -0.5 * nabla^2.
    # For a plane wave e^(iG.r), nabla^2 e^(iG.r) = -G^2 e^(iG.r).
    # So, T_G = 0.5 * G^2.
    return 0.5 * g_squared

def external_potential(r_coords, Z=1.0):
    """
    Calculates the external potential V_ext(r) for a hydrogen atom (Coulomb potential).

    Args:
        r_coords (np.ndarray): Array of real-space coordinates (N, N, N, 3).
        Z (float): Nuclear charge. Default is 1.0 for hydrogen.

    Returns:
        np.ndarray: External potential in real space (N, N, N).
                    V_ext(r) = -Z / r in atomic units.
    """
    # Calculate the magnitude of each position vector
    r_magnitudes = np.linalg.norm(r_coords, axis=-1)

    # Handle the singularity at r=0 by replacing 0 with a small epsilon.
    # The point (0,0,0) is included in the grid due to np.linspace(0, L, N, endpoint=False).
    # For r=0, the potential is infinite. We replace r=0 with a small non-zero value
    # to avoid division by zero and provide a large negative value at the origin.
    # A more sophisticated approach in real-world DFT codes often involves pseudopotentials
    # or Fourier space methods for the Coulomb potential.
    r_magnitudes_safe = np.where(r_magnitudes == 0, 1e-10, r_magnitudes)

    V_ext = -Z / r_magnitudes_safe
    return V_ext

