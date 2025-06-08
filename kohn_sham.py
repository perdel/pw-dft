import numpy as np
import numpy.fft as fft # Import fft explicitly for clarity

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

def hartree_potential(density_r, L, N):
    """
    Calculates the Hartree potential V_H(r) from the electron density n(r)
    using Fourier transforms.

    V_H(G) = (4 * pi / G^2) * n(G) for G != 0
    V_H(G=0) = 0 (convention for periodic systems, setting average potential to zero)

    Args:
        density_r (np.ndarray): Electron density in real space (N, N, N).
        L (float): Length of the cubic simulation box in Bohr.
        N (int): Number of grid points along each dimension.

    Returns:
        np.ndarray: Hartree potential in real space (N, N, N).
    """
    # 1. Fourier Transform the real-space density to reciprocal space
    density_g = fft.fftn(density_r)

    # 2. Reconstruct the full G-vector grid for the Coulomb kernel
    # These are the frequencies for FFT, scaled by 2*pi/L to get G-vectors
    freq_x = fft.fftfreq(N, d=L/N) * 2 * np.pi
    freq_y = fft.fftfreq(N, d=L/N) * 2 * np.pi
    freq_z = fft.fftfreq(N, d=L/N) * 2 * np.pi

    GX, GY, GZ = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
    g_squared_full = GX**2 + GY**2 + GZ**2

    # 3. Calculate the Coulomb kernel in reciprocal space: 4*pi / G^2
    # Initialize with zeros. The G=0 term will remain zero.
    coulomb_kernel_g = np.zeros_like(g_squared_full)
    
    # Apply 4*pi / G^2 for all non-zero G-vectors.
    # This correctly handles the G=0 term by leaving it as 0.
    non_zero_g_mask = g_squared_full != 0
    coulomb_kernel_g[non_zero_g_mask] = 4 * np.pi / g_squared_full[non_zero_g_mask]

    # 4. Multiply density_g by the Coulomb kernel to get V_H(G)
    V_H_g = coulomb_kernel_g * density_g

    # 5. Inverse Fourier Transform back to real space
    V_H_r = fft.ifftn(V_H_g)

    # The Hartree potential must be real. Return the real part.
    return V_H_r.real

def exchange_correlation_potential_lda(density_r):
    """
    Calculates the exchange-correlation potential V_xc(r) using the Local Density Approximation (LDA).
    For simplicity, this implementation uses the exchange-only part for a uniform electron gas.

    V_x(r) = - (3/pi)^(1/3) * n(r)^(1/3)

    Args:
        density_r (np.ndarray): Electron density in real space (N, N, N).

    Returns:
        np.ndarray: Exchange-correlation potential in real space (N, N, N).
    """
    # Ensure density is non-negative and add a small epsilon to avoid issues with 0^(1/3)
    # For physical densities, n(r) >= 0.
    density_safe = np.maximum(density_r, 1e-15) # Small positive value

    # Calculate the exchange potential V_x(r)
    # Constant factor: (3/pi)^(1/3)
    constant_factor = (3.0 / np.pi)**(1.0/3.0)
    
    V_xc = -constant_factor * (density_safe**(1.0/3.0))
    
    return V_xc
