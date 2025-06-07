import numpy as np
from grid import setup_real_space_grid, setup_reciprocal_space_grid
from kohn_sham import kinetic_energy_operator, external_potential, hartree_potential # Import hartree_potential

def main():
    # --- Simulation Parameters ---
    L = 10.0  # Length of the cubic simulation box in Bohr
    N = 32    # Number of grid points along each dimension (N x N x N grid)
    ecut = 20.0 # Kinetic energy cutoff in Hartree
    Z_nucleus = 1.0 # Nuclear charge for hydrogen atom

    print(f"Setting up simulation box: L={L} Bohr, N={N} grid points per dimension.")
    print(f"Kinetic energy cutoff: {ecut} Hartree.")
    print(f"Nuclear charge (Z): {Z_nucleus}")

    # --- 1. Setup Real-Space Grid ---
    r_coords, dr = setup_real_space_grid(L, N)
    print(f"\nReal-space grid setup complete. Grid spacing dr = {dr:.4f} Bohr.")
    # print(f"Shape of r_coords: {r_coords.shape}") # (N, N, N, 3)

    # --- 2. Setup Reciprocal-Space Grid (G-vectors) ---
    g_vectors, g_squared, g_indices = setup_reciprocal_space_grid(L, N, ecut)
    print(f"\nReciprocal-space grid setup complete.")
    print(f"Number of G-vectors after cutoff: {len(g_vectors)}")
    # print(f"Shape of g_vectors: {g_vectors.shape}") # (num_g_vectors, 3)
    # print(f"Shape of g_squared: {g_squared.shape}") # (num_g_vectors,)

    # --- 3. Calculate Kinetic Energy Operator ---
    T_operator = kinetic_energy_operator(g_squared)
    print(f"\nKinetic energy operator calculated.")
    print(f"Min kinetic energy: {np.min(T_operator):.4f} Hartree (should be 0 for G=0).")
    print(f"Max kinetic energy: {np.max(T_operator):.4f} Hartree (should be close to ecut).")

    # --- 4. Calculate External Potential (for Hydrogen Atom) ---
    V_ext = external_potential(r_coords, Z=Z_nucleus)
    print(f"\nExternal potential calculated.")
    print(f"Min external potential: {np.min(V_ext):.4f} Hartree (at origin, should be very negative).")
    print(f"Max external potential: {np.max(V_ext):.4f} Hartree (at furthest point, should be close to 0).")
    # print(f"Shape of V_ext: {V_ext.shape}") # (N, N, N)

    # --- 5. Calculate Hartree Potential ---
    # For demonstration, use the analytical ground state density of the hydrogen atom.
    # n(r) = (1/pi) * exp(-2r)
    r_magnitudes = np.linalg.norm(r_coords, axis=-1)
    
    # Calculate the unnormalized density
    # Using a small epsilon for r=0 to avoid issues, though exp(-2r) is not singular at r=0.
    r_magnitudes_safe_for_density = np.where(r_magnitudes == 0, 1e-10, r_magnitudes)
    dummy_density_r = (1/np.pi) * np.exp(-2 * r_magnitudes_safe_for_density)
    
    # Normalize the dummy density to the number of electrons (Z_nucleus for a neutral atom)
    # This is crucial for the Hartree potential calculation, as n(G=0) corresponds to total electrons.
    total_electrons_in_box = np.sum(dummy_density_r) * dr**3
    dummy_density_r_normalized = dummy_density_r / total_electrons_in_box * Z_nucleus

    V_H = hartree_potential(dummy_density_r_normalized, L, N)
    print(f"\nHartree potential calculated using a dummy density (normalized to {Z_nucleus} electron).")
    print(f"Min Hartree potential: {np.min(V_H):.4f} Hartree.")
    print(f"Max Hartree potential: {np.max(V_H):.4f} Hartree.")
    print(f"Average Hartree potential: {np.mean(V_H):.4f} Hartree (should be close to 0 due to G=0 handling).")
    # print(f"Shape of V_H: {V_H.shape}") # (N, N, N)

    # --- Next steps will involve setting up potentials and the self-consistency loop ---
    print("\nInitial setup complete. Ready for potential implementations and Kohn-Sham solver.")

if __name__ == '__main__':
    main()

