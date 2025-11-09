# =====================================================
# position_calculator.py
# Calculate exact positions of metal atoms and intersections
# =====================================================

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from interstitial_engine import (
    LatticeParams,
    Sublattice,
    lattice_vectors,
    bravais_basis,
    frac_to_cart,
    max_multiplicity_for_scale,
    build_geo,
    _make_key,
)


# -----------------
# Coordinate conversion utilities
# -----------------

def cart_to_frac(cart: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to fractional coordinates.
    
    Args:
        cart: Cartesian coordinates (3,) or (N, 3)
        a, b, c: Lattice vectors
    
    Returns:
        Fractional coordinates in same shape as input
    """
    M = np.column_stack([a, b, c])
    if cart.ndim == 1:
        return np.linalg.solve(M.T, cart)
    else:
        return np.linalg.solve(M.T, cart.T).T


def wrap_to_unit_cell(frac: np.ndarray) -> np.ndarray:
    """
    Wrap fractional coordinates to [0, 1).
    
    Args:
        frac: Fractional coordinates (3,) or (N, 3)
    
    Returns:
        Wrapped coordinates in [0, 1)
    """
    return frac - np.floor(frac)


def is_in_unit_cell(frac: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Check if fractional coordinates are within unit cell [0, 1).
    
    Args:
        frac: Fractional coordinates (N, 3)
        tol: Tolerance for boundary checks
    
    Returns:
        Boolean array (N,)
    """
    return np.all((frac >= -tol) & (frac < 1.0 - tol), axis=1)


# -----------------
# Metal atom position generation
# -----------------

@dataclass
class MetalAtomData:
    """Complete information about metal atoms in the structure"""
    fractional: np.ndarray  # (N, 3) fractional coordinates
    cartesian: np.ndarray   # (N, 3) Cartesian coordinates
    sublattice_id: np.ndarray  # (N,) which sublattice each atom belongs to
    sublattice_name: List[str]  # Names of sublattices
    radius: np.ndarray      # (N,) actual radius in Angstroms
    alpha_ratio: np.ndarray # (N,) alpha ratio for each atom


def generate_metal_positions(
    sublattices: List[Sublattice],
    p: LatticeParams,
    scale_s: float,
    supercell: Tuple[int, int, int] = (1, 1, 1)
) -> MetalAtomData:
    """
    Generate complete metal atom positions.
    
    Args:
        sublattices: List of sublattice definitions
        p: Lattice parameters
        scale_s: Current s value (sphere radius parameter)
        supercell: Number of unit cells to generate (nx, ny, nz)
    
    Returns:
        MetalAtomData with all metal atom information
    """
    a_vec, b_vec, c_vec = lattice_vectors(p)
    
    all_frac = []
    all_cart = []
    all_sublattice_id = []
    all_sublattice_names = []
    all_radius = []
    all_alpha = []
    
    nx, ny, nz = supercell
    
    for sub_idx, sub in enumerate(sublattices):
        if not sub.visible:
            continue
            
        # Get basis positions for this Bravais lattice type
        basis = bravais_basis(sub.bravais)
        offset = np.array(sub.offset_frac, dtype=float)
        
        # Generate positions for the supercell
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    cell_offset = np.array([i, j, k], dtype=float)
                    
                    for basis_pos in basis:
                        # Fractional position
                        frac = np.array(basis_pos, dtype=float) + offset + cell_offset
                        
                        # Cartesian position
                        cart = frac_to_cart(frac, a_vec, b_vec, c_vec)
                        
                        all_frac.append(frac)
                        all_cart.append(cart)
                        all_sublattice_id.append(sub_idx)
                        all_sublattice_names.append(sub.name)
                        all_radius.append(sub.alpha_ratio * scale_s * p.a)
                        all_alpha.append(sub.alpha_ratio)
    
    if not all_frac:
        # Return empty structure
        return MetalAtomData(
            fractional=np.empty((0, 3)),
            cartesian=np.empty((0, 3)),
            sublattice_id=np.empty((0,), dtype=int),
            sublattice_name=[],
            radius=np.empty((0,)),
            alpha_ratio=np.empty((0,))
        )
    
    return MetalAtomData(
        fractional=np.array(all_frac),
        cartesian=np.array(all_cart),
        sublattice_id=np.array(all_sublattice_id, dtype=int),
        sublattice_name=all_sublattice_names,
        radius=np.array(all_radius),
        alpha_ratio=np.array(all_alpha)
    )


# -----------------
# Intersection position calculation
# -----------------

@dataclass
class IntersectionData:
    """Complete information about intersection sites"""
    fractional: np.ndarray  # (N, 3) fractional coordinates
    cartesian: np.ndarray   # (N, 3) Cartesian coordinates
    multiplicity: np.ndarray  # (N,) number of spheres intersecting
    contributing_atoms: List[List[int]]  # Indices of atoms forming each intersection


def compute_periodic_distances(
    points: np.ndarray,
    centers: np.ndarray,
    shifts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute minimum distances from points to centers under PBC.
    
    Args:
        points: Query points (M, 3)
        centers: Center positions (N, 3)
        shifts: Lattice translation vectors (27, 3)
    
    Returns:
        distances: (M, N) minimum distance from each point to each center
        shift_indices: (M, N) which shift gave minimum distance
    """
    M = len(points)
    N = len(centers)
    
    min_distances = np.full((M, N), np.inf)
    shift_indices = np.zeros((M, N), dtype=int)
    
    for s_idx, shift in enumerate(shifts):
        # Shift centers
        shifted_centers = centers + shift  # (N, 3)
        
        # Compute distances (M, N)
        diff = points[:, None, :] - shifted_centers[None, :, :]  # (M, N, 3)
        distances = np.linalg.norm(diff, axis=2)  # (M, N)
        
        # Update minimum
        mask = distances < min_distances
        min_distances[mask] = distances[mask]
        shift_indices[mask] = s_idx
    
    return min_distances, shift_indices


def identify_contributing_atoms(
    intersection_points: np.ndarray,
    metal_atoms: MetalAtomData,
    shifts: np.ndarray,
    tolerance: float = 1e-3
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Identify which metal atoms contribute to each intersection.
    
    Args:
        intersection_points: Cartesian coordinates of intersections (M, 3)
        metal_atoms: Metal atom data
        shifts: Lattice translation vectors (27, 3)
        tolerance: Distance tolerance for "on sphere surface"
    
    Returns:
        multiplicities: (M,) number of contributing atoms
        contributing_lists: List of lists of atom indices
    """
    if len(intersection_points) == 0:
        return np.empty((0,), dtype=int), []
    
    # Compute distances
    distances, _ = compute_periodic_distances(
        intersection_points,
        metal_atoms.cartesian,
        shifts
    )
    
    # Check which atoms are within tolerance of their sphere radius
    # distances[i, j] should be ≈ metal_atoms.radius[j]
    on_surface = np.abs(distances - metal_atoms.radius[None, :]) < tolerance
    
    multiplicities = np.sum(on_surface, axis=1)
    contributing_lists = []
    
    for i in range(len(intersection_points)):
        contributing = np.where(on_surface[i])[0].tolist()
        contributing_lists.append(contributing)
    
    return multiplicities, contributing_lists


def cluster_intersections_pbc(
    positions_frac: np.ndarray,
    multiplicities: np.ndarray,
    contributing: List[List[int]],
    eps_frac: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """
    Cluster intersection positions accounting for PBC.
    
    Args:
        positions_frac: Fractional coordinates (N, 3)
        multiplicities: Multiplicity values (N,)
        contributing: Contributing atom indices for each position
        eps_frac: Clustering threshold in fractional coordinates
    
    Returns:
        unique_positions: Fractional coordinates of cluster centers
        unique_multiplicities: Maximum multiplicity in each cluster
        unique_contributing: Contributing atoms for unique positions
    """
    if len(positions_frac) == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=int), []
    
    # Wrap to unit cell
    wrapped = wrap_to_unit_cell(positions_frac)
    
    used = np.zeros(len(wrapped), dtype=bool)
    unique_pos = []
    unique_mult = []
    unique_contrib = []
    
    eps2 = eps_frac ** 2
    
    for i in range(len(wrapped)):
        if used[i]:
            continue
        
        # Find all positions within eps of this one (considering PBC)
        cluster = [i]
        for j in range(i + 1, len(wrapped)):
            if used[j]:
                continue
            
            # Compute distance considering PBC wrapping
            diff = wrapped[j] - wrapped[i]
            # Handle periodic wrapping: if diff > 0.5, subtract 1; if < -0.5, add 1
            diff = diff - np.round(diff)
            dist2 = np.sum(diff ** 2)
            
            if dist2 < eps2:
                cluster.append(j)
        
        # Mark as used
        used[cluster] = True
        
        # Compute cluster representative
        cluster_positions = wrapped[cluster]
        
        # Handle PBC in averaging: unwrap relative to first point
        ref = cluster_positions[0]
        unwrapped = cluster_positions.copy()
        for k in range(1, len(unwrapped)):
            diff = unwrapped[k] - ref
            diff = diff - np.round(diff)  # Unwrap
            unwrapped[k] = ref + diff
        
        # Average and wrap back
        mean_pos = np.mean(unwrapped, axis=0)
        mean_pos = wrap_to_unit_cell(mean_pos)
        
        # Maximum multiplicity in cluster
        max_mult = int(np.max(multiplicities[cluster]))
        
        # Merge contributing atoms (unique)
        all_contrib = set()
        for idx in cluster:
            all_contrib.update(contributing[idx])
        
        unique_pos.append(mean_pos)
        unique_mult.append(max_mult)
        unique_contrib.append(sorted(list(all_contrib)))
    
    return (
        np.array(unique_pos),
        np.array(unique_mult, dtype=int),
        unique_contrib
    )


def calculate_intersections_detailed(
    sublattices: List[Sublattice],
    p: LatticeParams,
    scale_s: float,
    target_N: Optional[int] = None,
    k_samples: int = 16,
    tol_inside: float = 1e-3,
    cluster_eps_frac: float = 0.01,
    unit_cell_only: bool = True
) -> IntersectionData:
    """
    Calculate detailed intersection positions for a given configuration.
    
    Args:
        sublattices: List of sublattice definitions
        p: Lattice parameters
        scale_s: Current s value
        target_N: Optional target multiplicity (if None, returns all intersections)
        k_samples: Number of samples per pair circle
        tol_inside: Tolerance for "inside sphere"
        cluster_eps_frac: Clustering epsilon in fractional coordinates
        unit_cell_only: If True, only return intersections in [0,1) unit cell
    
    Returns:
        IntersectionData with all intersection information
    """
    # Use existing engine to get raw intersection samples
    max_mult, sample_positions, sample_counts = max_multiplicity_for_scale(
        sublattices=sublattices,
        p=p,
        repeat_ignored=1,
        scale_s=scale_s,
        k_samples=k_samples,
        tol_inside=tol_inside,
        cluster_eps=None,  # We'll do our own clustering
        early_stop_at=None
    )
    
    if len(sample_positions) == 0:
        return IntersectionData(
            fractional=np.empty((0, 3)),
            cartesian=np.empty((0, 3)),
            multiplicity=np.empty((0,), dtype=int),
            contributing_atoms=[]
        )
    
    # Get lattice vectors and geometry
    a_vec, b_vec, c_vec = lattice_vectors(p)
    key = _make_key(sublattices, p)
    centers, alphas, shifts, _ = build_geo(key)
    
    # Generate metal atom data for the central cell
    metal_atoms = generate_metal_positions(sublattices, p, scale_s, supercell=(1, 1, 1))
    
    # Identify contributing atoms for each sample point
    multiplicities, contributing = identify_contributing_atoms(
        sample_positions,
        metal_atoms,
        shifts,
        tolerance=tol_inside * 2  # Slightly looser for identification
    )
    
    # Convert to fractional coordinates
    frac_positions = cart_to_frac(sample_positions, a_vec, b_vec, c_vec)
    
    # Cluster in fractional space with PBC
    unique_frac, unique_mult, unique_contrib = cluster_intersections_pbc(
        frac_positions,
        multiplicities,
        contributing,
        eps_frac=cluster_eps_frac
    )
    
    # Filter by target multiplicity if specified
    if target_N is not None:
        mask = unique_mult >= target_N
        unique_frac = unique_frac[mask]
        unique_mult = unique_mult[mask]
        unique_contrib = [unique_contrib[i] for i in range(len(mask)) if mask[i]]
    
    # Filter to unit cell if requested
    if unit_cell_only:
        mask = is_in_unit_cell(unique_frac)
        unique_frac = unique_frac[mask]
        unique_mult = unique_mult[mask]
        unique_contrib = [unique_contrib[i] for i in range(len(mask)) if mask[i]]
    
    # Convert back to Cartesian
    if len(unique_frac) > 0:
        unique_cart = np.array([
            frac_to_cart(frac, a_vec, b_vec, c_vec)
            for frac in unique_frac
        ])
    else:
        unique_cart = np.empty((0, 3))
    
    return IntersectionData(
        fractional=unique_frac,
        cartesian=unique_cart,
        multiplicity=unique_mult,
        contributing_atoms=unique_contrib
    )


# -----------------
# Complete position output
# -----------------

@dataclass
class CompleteStructureData:
    """Complete structure with metal atoms and intersections"""
    metal_atoms: MetalAtomData
    intersections: IntersectionData
    lattice_params: LatticeParams
    scale_s: float
    lattice_vectors: Tuple[np.ndarray, np.ndarray, np.ndarray]


def calculate_complete_structure(
    sublattices: List[Sublattice],
    p: LatticeParams,
    scale_s: float,
    target_N: Optional[int] = None,
    supercell_metals: Tuple[int, int, int] = (1, 1, 1),
    k_samples: int = 16,
    unit_cell_only: bool = True
) -> CompleteStructureData:
    """
    Calculate complete structure: metal atoms + intersections.
    
    Args:
        sublattices: List of sublattice definitions
        p: Lattice parameters
        scale_s: Current s value
        target_N: Optional target multiplicity for intersections
        supercell_metals: Supercell size for metal atoms (for visualization)
        k_samples: Sampling density for intersections
        unit_cell_only: Return only intersections in unit cell
    
    Returns:
        CompleteStructureData with all structural information
    """
    # Generate metal positions
    metal_atoms = generate_metal_positions(
        sublattices=sublattices,
        p=p,
        scale_s=scale_s,
        supercell=supercell_metals
    )
    
    # Calculate intersections
    intersections = calculate_intersections_detailed(
        sublattices=sublattices,
        p=p,
        scale_s=scale_s,
        target_N=target_N,
        k_samples=k_samples,
        unit_cell_only=unit_cell_only
    )
    
    # Get lattice vectors
    vecs = lattice_vectors(p)
    
    return CompleteStructureData(
        metal_atoms=metal_atoms,
        intersections=intersections,
        lattice_params=p,
        scale_s=scale_s,
        lattice_vectors=vecs
    )


# -----------------
# Export utilities
# -----------------

def format_position_dict(data: CompleteStructureData) -> Dict:
    """
    Format structure data as dictionary for JSON export.
    
    Args:
        data: Complete structure data
    
    Returns:
        Dictionary representation
    """
    p = data.lattice_params
    a_vec, b_vec, c_vec = data.lattice_vectors
    
    result = {
        'lattice_parameters': {
            'a': float(p.a),
            'b': float(p.a * p.b_ratio),
            'c': float(p.a * p.c_ratio),
            'alpha': float(p.alpha),
            'beta': float(p.beta),
            'gamma': float(p.gamma),
            'b_ratio': float(p.b_ratio),
            'c_ratio': float(p.c_ratio),
        },
        'lattice_vectors': {
            'a': a_vec.tolist(),
            'b': b_vec.tolist(),
            'c': c_vec.tolist(),
        },
        'scale_s': float(data.scale_s),
        'metal_atoms': {
            'count': len(data.metal_atoms.fractional),
            'positions': {
                'fractional': data.metal_atoms.fractional.tolist(),
                'cartesian': data.metal_atoms.cartesian.tolist(),
            },
            'sublattice_id': data.metal_atoms.sublattice_id.tolist(),
            'sublattice_name': data.metal_atoms.sublattice_name,
            'radius_angstrom': data.metal_atoms.radius.tolist(),
            'alpha_ratio': data.metal_atoms.alpha_ratio.tolist(),
        },
        'intersections': {
            'count': len(data.intersections.fractional),
            'positions': {
                'fractional': data.intersections.fractional.tolist(),
                'cartesian': data.intersections.cartesian.tolist(),
            },
            'multiplicity': data.intersections.multiplicity.tolist(),
            'contributing_atoms': data.intersections.contributing_atoms,
        }
    }
    
    return result


def format_metal_atoms_csv(data: CompleteStructureData) -> str:
    """
    Format metal atoms as CSV string.
    
    Args:
        data: Complete structure data
    
    Returns:
        CSV string
    """
    lines = []
    lines.append("atom_index,sublattice_name,sublattice_id,frac_x,frac_y,frac_z,cart_x,cart_y,cart_z,radius_angstrom,alpha_ratio")
    
    for i in range(len(data.metal_atoms.fractional)):
        frac = data.metal_atoms.fractional[i]
        cart = data.metal_atoms.cartesian[i]
        lines.append(
            f"{i},{data.metal_atoms.sublattice_name[i]},{data.metal_atoms.sublattice_id[i]},"
            f"{frac[0]:.6f},{frac[1]:.6f},{frac[2]:.6f},"
            f"{cart[0]:.6f},{cart[1]:.6f},{cart[2]:.6f},"
            f"{data.metal_atoms.radius[i]:.6f},{data.metal_atoms.alpha_ratio[i]:.6f}"
        )
    
    return "\n".join(lines)


def format_intersections_csv(data: CompleteStructureData) -> str:
    """
    Format intersections as CSV string.
    
    Args:
        data: Complete structure data
    
    Returns:
        CSV string
    """
    lines = []
    lines.append("intersection_index,multiplicity,frac_x,frac_y,frac_z,cart_x,cart_y,cart_z,contributing_atom_indices")
    
    for i in range(len(data.intersections.fractional)):
        frac = data.intersections.fractional[i]
        cart = data.intersections.cartesian[i]
        mult = data.intersections.multiplicity[i]
        contrib = ";".join(map(str, data.intersections.contributing_atoms[i]))
        
        lines.append(
            f"{i},{mult},"
            f"{frac[0]:.6f},{frac[1]:.6f},{frac[2]:.6f},"
            f"{cart[0]:.6f},{cart[1]:.6f},{cart[2]:.6f},"
            f"{contrib}"
        )
    
    return "\n".join(lines)


def format_xyz(data: CompleteStructureData, include_intersections: bool = True) -> str:
    """
    Format structure as XYZ file for visualization.
    
    Args:
        data: Complete structure data
        include_intersections: Include intersection sites as dummy atoms
    
    Returns:
        XYZ format string
    """
    lines = []
    
    # Count atoms
    n_atoms = len(data.metal_atoms.cartesian)
    n_intersections = len(data.intersections.cartesian) if include_intersections else 0
    total = n_atoms + n_intersections
    
    lines.append(str(total))
    lines.append(f"s={data.scale_s:.6f} a={data.lattice_params.a:.6f}")
    
    # Metal atoms
    for i in range(len(data.metal_atoms.cartesian)):
        cart = data.metal_atoms.cartesian[i]
        name = data.metal_atoms.sublattice_name[i]
        lines.append(f"{name}  {cart[0]:.6f}  {cart[1]:.6f}  {cart[2]:.6f}")
    
    # Intersections (as X with multiplicity in comment)
    if include_intersections:
        for i in range(len(data.intersections.cartesian)):
            cart = data.intersections.cartesian[i]
            mult = data.intersections.multiplicity[i]
            lines.append(f"X{mult}  {cart[0]:.6f}  {cart[1]:.6f}  {cart[2]:.6f}")
    
    return "\n".join(lines)
