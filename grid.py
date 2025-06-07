import numpy as np

def setup_real_space_grid(L, N):
    """
    Sets up a 3D real-space grid.

    Args:
        L (float): Length of the cubic simulation box in Bohr.
        N (int): Number of grid points along each dimension.

    Returns:
        tuple: A tuple containing:
            - r_coords (np.ndarray): Array of real-space coordinates (N, N, N, 3).
            - dr (float): Grid spacing.
    """
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    z = np.linspace(0, L, N, endpoint=False)
    
    # Create a 3D meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    r_coords = np.stack((X, Y, Z), axis=-1)
    
    dr = L / N
    return r_coords, dr

def setup_reciprocal_space_grid(L, N, ecut):
    """
    Sets up a 3D reciprocal-space grid (G-vectors) and filters them
    based on a kinetic energy cutoff.

    Args:
        L (float): Length of the cubic simulation box in Bohr.
        N (int): Number of grid points along each dimension.
        ecut (float): Kinetic energy cutoff in Hartree.

    Returns:
        tuple: A tuple containing:
            - g_vectors (np.ndarray): Array of G-vectors (num_g_vectors, 3).
            - g_squared (np.ndarray): Array of squared G-vector magnitudes (num_g_vectors,).
            - g_indices (np.ndarray): Array of original grid indices for the selected G-vectors.
    """
    # Reciprocal lattice vectors for a cubic box
    # G_x = 2*pi*n_x / L
    # G_y = 2*pi*n_y / L
    # G_z = 2*pi*n_z / L
    
    # Frequencies for FFT (numpy.fft.fftfreq)
    # These are scaled by 2*pi/L to get G-vectors
    freq_x = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    freq_y = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    freq_z = np.fft.fftfreq(N, d=L/N) * 2 * np.pi

    # Create a 3D meshgrid for G-vectors
    GX, GY, GZ = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
    
    # Flatten the G-vectors and calculate their squared magnitudes
    g_vectors_full = np.stack((GX.flatten(), GY.flatten(), GZ.flatten()), axis=-1)
    g_squared_full = np.sum(g_vectors_full**2, axis=1)

    # Apply kinetic energy cutoff: T = 0.5 * G^2 <= E_cut
    # Note: G=0 term is always included
    cutoff_mask = (0.5 * g_squared_full <= ecut)
    
    g_vectors = g_vectors_full[cutoff_mask]
    g_squared = g_squared_full[cutoff_mask]
    
    # Store the original indices to map back to the full grid if needed for FFTs
    g_indices = np.where(cutoff_mask)[0]

    return g_vectors, g_squared, g_indices

