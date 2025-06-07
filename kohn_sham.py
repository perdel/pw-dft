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

