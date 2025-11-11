# =====================================================
# interstitial_engine_improved.py (ENHANCED ACCURACY)
# Improved numerical accuracy through:
# - Adaptive sampling density
# - Weighted clustering
# - Iterative refinement
# - Better periodic handling
# =====================================================

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple, Optional, Dict, List
import numpy as np
from math import sqrt
try:
    from scipy.spatial import cKDTree as KDTree
    _HAVE_SCIPY = True
except Exception:
    KDTree = None
    _HAVE_SCIPY = False


# -----------------
# Data structures
# -----------------
@dataclass
class LatticeParams:
    a: float = 1.0
    b_ratio: float = 1.0
    c_ratio: float = 1.0
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

@dataclass
class Sublattice:
    name: str
    bravais: str
    offset_frac: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha_ratio: float = 1.0
    visible: bool = True


# -----------------
# Lattices & vectors
# -----------------
def bravais_basis(bravais: str) -> np.ndarray:
    B = {
        "cubic_P":        [(0,0,0)],
        "cubic_I":        [(0,0,0),(0.5,0.5,0.5)],
        "cubic_F":        [(0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
        "cubic_Diamond":  [
            (0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5),
            (0.25,0.25,0.25),(0.75,0.75,0.25),(0.75,0.25,0.75),(0.25,0.75,0.75)
        ],
        "cubic_Pyrochlore":[
            (0.5,0.5,0.5),(0,0,0.5),(0,0.5,0),(0.5,0,0),
            (0.25,0.75,0),(0.75,0.25,0),(0.75,0.75,0.5),(0.25,0.25,0.5),
            (0.75,0,0.25),(0.25,0.5,0.25),(0.25,0,0.75),(0.75,0.5,0.75),
            (0,0.25,0.75),(0.5,0.75,0.75),(0.5,0.25,0.25),(0,0.75,0.25)
        ],
        "tetragonal_P":   [(0,0,0)],
        "tetragonal_I":   [(0,0,0),(0.5,0.5,0.5)],
        "orthorhombic_P": [(0,0,0)],
        "orthorhombic_C": [(0,0,0),(0.5,0.5,0)],
        "orthorhombic_I": [(0,0,0),(0.5,0.5,0.5)],
        "orthorhombic_F": [(0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
        "hexagonal_P":    [(0,0,0)],
        "hexagonal_HCP":  [(0,0,0),(1/3,2/3,0.5)],
        "rhombohedral_R": [(0,0,0)],
        "monoclinic_P":   [(0,0,0)],
        "monoclinic_C":   [(0,0,0),(0.5,0.5,0)],
        "triclinic_P":    [(0,0,0)],
    }
    if bravais not in B:
        raise ValueError(f"Unknown bravais type: {bravais}")
    return np.asarray(B[bravais], dtype=float)

def lattice_vectors(p: LatticeParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = p.a
    b = p.a * p.b_ratio
    c = p.a * p.c_ratio
    if abs(p.alpha-90)<1e-9 and abs(p.beta-90)<1e-9 and abs(p.gamma-90)<1e-9:
        return np.array([a,0,0.0]), np.array([0.0,b,0.0]), np.array([0.0,0.0,c])

    alp = np.deg2rad(p.alpha); bet = np.deg2rad(p.beta); gam = np.deg2rad(p.gamma)
    avec = np.array([a,0,0], float)
    bvec = np.array([b*np.cos(gam), b*np.sin(gam), 0.0], float)
    cx  = c*np.cos(bet)
    cy  = c*(np.cos(alp) - np.cos(bet)*np.cos(gam))/max(1e-12, np.sin(gam))
    cz2 = max(0.0, c*c - cx*cx - cy*cy)
    cvec= np.array([cx, cy, sqrt(cz2)], float)
    return avec, bvec, cvec

def frac_to_cart(frac: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return frac[0]*a + frac[1]*b + frac[2]*c


# -----------------
# Minimal-image centers (no supercell)
# -----------------
def generate_centers_minimal(sub: Sublattice, p: LatticeParams) -> np.ndarray:
    a,b,c = lattice_vectors(p)
    basis = bravais_basis(sub.bravais)
    off = np.array(sub.offset_frac, float)
    pts = [frac_to_cart(np.asarray(b0)+off, a,b,c) for b0 in basis]
    return np.asarray(pts, float)

@lru_cache(maxsize=256)
def centers_alphas_and_shifts(subs_key, p_key):
    """Cache central-cell centers, per-center alpha, and 27 neighbor lattice shifts."""
    p = LatticeParams(*p_key)
    a,b,c = lattice_vectors(p)
    shifts = []
    for i in (-1,0,1):
        for j in (-1,0,1):
            for k in (-1,0,1):
                shifts.append(frac_to_cart(np.array([i,j,k], float), a,b,c))
    shifts = np.asarray(shifts, float)

    centers = []
    alphas = []
    for (bravais, ox,oy,oz, alpha_ratio, visible) in subs_key:
        if not visible: 
            continue
        sub = Sublattice("x", bravais, (ox,oy,oz), alpha_ratio, True)
        pts = generate_centers_minimal(sub, p)
        centers.append(pts)
        alphas.append(np.full(len(pts), alpha_ratio, float))
    if centers:
        centers = np.vstack(centers)
        alphas  = np.concatenate(alphas)
    else:
        centers = np.empty((0,3)); alphas = np.empty((0,))
    return centers, alphas, shifts


# -----------------
# Pair list under PBC (reference→neighbors)
# -----------------
def periodic_candidate_pairs(centers: np.ndarray, shifts: np.ndarray, cutoff: float) -> List[Tuple[int,int,int]]:
    """
    Find all center pairs within cutoff distance under PBC.
    Includes:
    - Cross-sublattice pairs (different centers)
    - Within-sublattice interactions with periodic images (same center, different cells)
    """
    if len(centers) == 0:
        return []
    pairs: List[Tuple[int,int,int]] = []
    cutoff2 = cutoff*cutoff
    n = len(centers)
    
    for s_idx, S in enumerate(shifts):
        shifted = centers + S
        B = 512
        for start in range(0, n, B):
            end = min(n, start+B)
            block = centers[start:end]
            d2 = np.sum((block[:,None,:] - shifted[None,:,:])**2, axis=2)
            bi, bj = np.where(d2 < cutoff2)
            for ii, jj in zip(bi, bj):
                gi = start + ii
                
                # Central cell (s_idx == 13): avoid exact self-pairs (gi == jj)
                # but allow gi < jj for cross-pairs
                if s_idx == 13:
                    if gi < jj:
                        pairs.append((gi, jj, s_idx))
                # Non-central cells: include all pairs
                # This finds within-sublattice interactions across periodic images
                else:
                    pairs.append((gi, jj, s_idx))
    
    return pairs


# -----------------
# Enhanced pair-circle sampling with adaptive density
# -----------------
def pair_circle_samples_adaptive(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float, 
                                 k_base: int = 16, expected_mult: Optional[int] = None) -> np.ndarray:
    """
    Enhanced sampling with adaptive density based on intersection geometry.
    Uses more samples for configurations likely to have higher multiplicity.
    """
    v = c2 - c1
    d2 = float(np.dot(v,v))
    if d2 <= 1e-24:
        return np.empty((0,3))
    d = sqrt(d2)
    
    # Allow tangent spheres with small tolerance
    tol = 1e-10
    if not (abs(r1 - r2) - tol <= d <= (r1 + r2) + tol):
        return np.empty((0,3))
    
    n = v / d
    t = (r1*r1 - r2*r2 + d*d) / (2*d*d)
    center = c1 + t*v
    
    # Build orthonormal basis
    tmp = np.array([1.0,0.0,0.0]) if abs(n[0]) <= 0.9 else np.array([0.0,1.0,0.0])
    u = np.cross(tmp, n)
    nu = float(np.linalg.norm(u))
    if nu < 1e-12:
        tmp = np.array([0.0,0.0,1.0])
        u = np.cross(tmp, n)
        nu = float(np.linalg.norm(u))
        if nu < 1e-12:
            return np.empty((0,3))
    u /= nu
    w = np.cross(n,u)
    
    # Calculate circle radius
    off = center - c1
    h2 = r1*r1 - float(np.dot(off,off))
    if h2 < -1e-10:
        return np.empty((0,3))
    h = sqrt(max(0.0, h2))
    
    # Single point for tangent spheres
    if h < 1e-12:
        return center.reshape(1, 3)
    
    # Adaptive sampling: more points for smaller circles or higher expected multiplicities
    # Small circles are more likely to have all samples at high multiplicity
    circle_fraction = h / max(r1, r2)
    
    # Scale k based on geometry
    if expected_mult and expected_mult >= 4:
        # For high multiplicity, use more samples
        k = max(k_base * 2, 32)
    elif circle_fraction < 0.3:
        # Very small circles: likely high symmetry
        k = max(k_base * 2, 24)
    elif circle_fraction > 0.8:
        # Large circles: can use fewer samples
        k = k_base
    else:
        # Medium circles: use base or slightly more
        k = int(k_base * (1.5 - 0.5 * circle_fraction))
    
    # Generate samples
    ang = np.linspace(0.0, 2.0*np.pi, k, endpoint=False)
    return center + (h*np.cos(ang))[:,None]*u + (h*np.sin(ang))[:,None]*w


# -----------------
# Weighted clustering that preserves high-multiplicity centers
# -----------------
def _cluster_weighted(points: np.ndarray, counts: np.ndarray, eps: float, 
                     weight_by_multiplicity: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced clustering that better preserves high-multiplicity intersection positions.
    Uses weighted averaging based on multiplicity.
    """
    if len(points) == 0:
        return points, counts
    
    used = np.zeros(len(points), bool)
    reps = []
    repc = []
    eps2 = eps*eps
    
    # Sort by multiplicity (descending) to process high-mult points first
    if weight_by_multiplicity:
        sort_idx = np.argsort(-counts)
    else:
        sort_idx = np.arange(len(points))
    
    for idx in sort_idx:
        if used[idx]: 
            continue
        
        # Find all points within eps of this one
        cluster_indices = [idx]
        for j in range(len(points)):
            if j != idx and not used[j]:
                d2 = float(np.dot(points[idx]-points[j], points[idx]-points[j]))
                if d2 < eps2:
                    cluster_indices.append(j)
        
        used[cluster_indices] = True
        
        # Weighted average based on multiplicity
        if weight_by_multiplicity and len(cluster_indices) > 1:
            cluster_points = points[cluster_indices]
            cluster_counts = counts[cluster_indices].astype(float)
            
            # Use multiplicity as weight, with higher multiplicities having more influence
            weights = cluster_counts ** 2  # Square to emphasize high-mult points
            weights /= np.sum(weights)
            
            rep_point = np.sum(cluster_points * weights[:, None], axis=0)
        else:
            # Simple average
            rep_point = np.mean(points[cluster_indices], axis=0)
        
        # Max multiplicity in cluster
        max_count = int(np.max(counts[cluster_indices]))
        
        reps.append(rep_point)
        repc.append(max_count)
    
    return np.asarray(reps), np.asarray(repc, int)


# -----------------
# Iterative refinement to converge to true intersection points
# -----------------
def refine_intersection_position(initial_pos: np.ndarray, 
                                centers: np.ndarray, 
                                radii: np.ndarray,
                                shifts: np.ndarray,
                                max_iter: int = 10,
                                tol: float = 1e-6) -> np.ndarray:
    """
    Iteratively refine an intersection position to minimize distance variance
    from the spheres it should lie on.
    """
    pos = initial_pos.copy()
    
    for iteration in range(max_iter):
        # Find which spheres this point should lie on
        contributing = []
        target_radii = []
        
        for s_idx, S in enumerate(shifts):
            shifted_centers = centers + S
            distances = np.linalg.norm(shifted_centers - pos, axis=1)
            
            # Check which spheres the point is close to
            for i, (d, r) in enumerate(zip(distances, radii)):
                if abs(d - r) < 0.1:  # Within 10% of sphere surface
                    contributing.append(shifted_centers[i])
                    target_radii.append(r)
        
        if len(contributing) < 2:
            break
        
        contributing = np.array(contributing)
        target_radii = np.array(target_radii)
        
        # Newton-style update: move point to better satisfy all sphere constraints
        gradients = []
        residuals = []
        
        for center, radius in zip(contributing, target_radii):
            vec = pos - center
            dist = np.linalg.norm(vec)
            if dist > 1e-12:
                # Gradient of (dist - radius)^2
                grad = 2.0 * (dist - radius) * vec / dist
                gradients.append(grad)
                residuals.append(dist - radius)
        
        if not gradients:
            break
        
        # Compute update
        avg_gradient = np.mean(gradients, axis=0)
        avg_residual = np.mean(np.abs(residuals))
        
        # Adaptive step size
        step_size = min(0.5, 0.1 * avg_residual)
        update = -step_size * avg_gradient
        
        # Apply update
        new_pos = pos + update
        
        # Check convergence
        if np.linalg.norm(update) < tol:
            break
        
        pos = new_pos
    
    return pos


# -----------------
# Enhanced multiplicity counting with better accuracy
# -----------------
def _count_multiplicity_kdtree(sample_pts: np.ndarray, tree: KDTree, radii: np.ndarray, 
                               tol_inside: float) -> np.ndarray:
    """Enhanced multiplicity counting with adaptive tolerance"""
    rmax = float(np.max(radii))
    idxs = tree.query_ball_point(sample_pts, r=rmax + tol_inside)
    out = np.zeros(len(sample_pts), dtype=int)
    
    for i, neigh in enumerate(idxs):
        if not neigh:
            continue
        d = np.linalg.norm(tree.data[neigh] - sample_pts[i], axis=1)
        # Use tighter tolerance for high-precision mode
        effective_tol = min(tol_inside, 0.001 * np.min(radii[neigh]))
        out[i] = int(np.sum(d <= (radii[neigh] + effective_tol)))
    
    return out


# -----------------
# Public API with enhanced accuracy
# -----------------
def max_multiplicity_for_scale(
    sublattices: List[Sublattice],
    p: LatticeParams,
    repeat_ignored: int,
    scale_s: float,
    k_samples: int = 16,
    tol_inside: float = 1e-3,
    cluster_eps: Optional[float] = None,
    early_stop_at: Optional[int] = None,
    high_accuracy: bool = False,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Enhanced version with improved numerical accuracy.
    
    Args:
        high_accuracy: If True, use enhanced sampling, weighted clustering, and refinement
    """
    key = _make_key(sublattices, p)
    centers, alphas, shifts, zero_idx = build_geo(key)
    if centers.size == 0:
        return 0, np.empty((0,3)), np.empty((0,))
    
    radii = (alphas * scale_s * p.a).astype(float)
    rmax = float(np.max(radii))
    
    # Improved cutoff calculation
    a_vec, b_vec, c_vec = lattice_vectors(p)
    a_mag = float(np.linalg.norm(a_vec))
    cutoff = max(2.0*rmax + a_mag, 3.0*rmax)
    
    pairs = _build_pairs_for_cutoff(centers, shifts, cutoff=cutoff)
    if not pairs:
        return 0, np.empty((0,3)), np.empty((0,))
    
    if _HAVE_SCIPY:
        tree = KDTree(centers.copy())
    else:
        tree = None
    
    samples: List[np.ndarray] = []
    counts: List[int] = []
    
    # Adjust sampling based on accuracy mode
    if high_accuracy:
        k_effective = max(k_samples * 2, 32)
    else:
        k_effective = k_samples
    
    for (i, j, s_idx) in pairs:
        c1 = centers[i]
        c2 = centers[j] + shifts[s_idx]
        r1 = radii[i]
        r2 = radii[j]
        
        # Use adaptive sampling
        pts = pair_circle_samples_adaptive(
            c1, r1, c2, r2, 
            k_base=k_effective,
            expected_mult=early_stop_at
        )
        
        if pts.size == 0:
            continue
        
        # Count multiplicities
        if _HAVE_SCIPY:
            total = np.zeros(len(pts), dtype=int)
            for S in shifts:
                P = pts - S
                c = _count_multiplicity_kdtree(P, tree, radii, tol_inside)
                total += c
            c = total
        else:
            all_centers = np.vstack([centers + S for S in shifts])
            all_radii   = np.tile(radii, len(shifts))
            d = np.linalg.norm(all_centers[None,:,:] - pts[:,None,:], axis=2)
            c = np.sum(d <= (all_radii[None,:] + tol_inside), axis=1)
        
        # Refine high-multiplicity points if in high accuracy mode
        if high_accuracy and np.any(c >= 4):
            high_mult_mask = c >= 4
            high_mult_pts = pts[high_mult_mask]
            
            # Refine each high-multiplicity point
            refined_pts = []
            for pt in high_mult_pts:
                refined = refine_intersection_position(
                    pt, centers, radii, shifts,
                    max_iter=10, tol=1e-6
                )
                refined_pts.append(refined)
            
            if refined_pts:
                pts[high_mult_mask] = np.array(refined_pts)
        
        if early_stop_at is not None and np.any(c >= early_stop_at):
            keep = np.where(c >= 2)[0][:10]  # Keep more samples for better accuracy
            samples.extend(pts[keep])
            counts.extend([int(x) for x in c[keep]])
            m = int(np.max(c))
            if cluster_eps is None:
                cluster_eps = 0.01 * p.a if high_accuracy else 0.1 * p.a
            reps, repc = _cluster_weighted(
                np.asarray(samples), 
                np.asarray(counts, int), 
                eps=cluster_eps,
                weight_by_multiplicity=high_accuracy
            )
            return m, reps, repc
        
        good = np.where(c >= 2)[0]
        if good.size:
            samples.extend(pts[good])
            counts.extend([int(x) for x in c[good]])
    
    if not samples:
        return 0, np.empty((0,3)), np.empty((0,))
    
    if cluster_eps is None:
        cluster_eps = 0.01 * p.a if high_accuracy else 0.1 * p.a
    
    # Use weighted clustering for better accuracy
    reps, repc = _cluster_weighted(
        np.asarray(samples), 
        np.asarray(counts, int), 
        eps=cluster_eps,
        weight_by_multiplicity=high_accuracy
    )
    
    # Final refinement pass for high-accuracy mode
    if high_accuracy and len(reps) > 0:
        refined_reps = []
        for rep, count in zip(reps, repc):
            if count >= 4:  # Only refine high-multiplicity points
                refined = refine_intersection_position(
                    rep, centers, radii, shifts,
                    max_iter=15, tol=1e-7
                )
                refined_reps.append(refined)
            else:
                refined_reps.append(rep)
        reps = np.array(refined_reps)
    
    mmax = int(repc.max()) if len(repc) else 0
    return mmax, reps, repc


def find_threshold_s_for_N(
    N_target: int,
    sublattices: List[Sublattice],
    p: LatticeParams,
    repeat: int = 1,
    s_min: float = 0.01,
    s_max: float = 0.9,
    k_samples_coarse: int = 8,
    k_samples_fine: int = 16,
    tol_inside: float = 1e-3,
    cluster_eps: Optional[float] = None,
    max_iter: int = 20,
    high_accuracy: bool = False,
) -> Tuple[Optional[float], Dict[int, float]]:
    """
    Enhanced version with high_accuracy option
    """
    milestones: Dict[int, float] = {}
    
    # Adjust sampling for high accuracy
    if high_accuracy:
        k_samples_coarse = max(k_samples_coarse * 2, 16)
        k_samples_fine = max(k_samples_fine * 2, 32)
    
    # coarse sweep
    for s in np.linspace(s_min, s_max, 12):
        m,_,_ = max_multiplicity_for_scale(
            sublattices, p, 1, s,
            k_samples=k_samples_coarse, tol_inside=tol_inside,
            cluster_eps=cluster_eps, early_stop_at=N_target,
            high_accuracy=high_accuracy
        )
        milestones.setdefault(int(m), float(s))
        if m >= N_target:
            s_hi = s
            break
    else:
        return None, milestones
    
    # refine low bound
    s_lo = s_min
    for s2 in np.linspace(s_min, s_hi, 12):
        m,_,_ = max_multiplicity_for_scale(
            sublattices, p, 1, s2,
            k_samples=k_samples_coarse, tol_inside=tol_inside,
            cluster_eps=cluster_eps, early_stop_at=N_target,
            high_accuracy=high_accuracy
        )
        if m < N_target: s_lo = s2
        else:
            s_hi = s2; break
    
    # bisection with adaptive sampling
    for it in range(max_iter):
        mid = 0.5*(s_lo+s_hi)
        if (s_hi - s_lo) < 1e-5:
            break
        ks = k_samples_coarse if (s_hi - s_lo) > 0.02 else k_samples_fine
        m,_,_ = max_multiplicity_for_scale(
            sublattices, p, 1, mid,
            k_samples=ks, tol_inside=tol_inside,
            cluster_eps=cluster_eps, early_stop_at=N_target,
            high_accuracy=high_accuracy
        )
        if m >= N_target: s_hi = mid
        else: s_lo = mid
    
    milestones[N_target] = s_hi
    return s_hi, milestones


# -----------------
# Cached geometry bundle (minimal image)
# -----------------
@dataclass(frozen=True)
class GeoKey:
    p: Tuple[float,float,float,float,float,float]
    subs: Tuple[Tuple[str,float,float,float,float,bool], ...]

def _make_key(sublattices: List[Sublattice], p: LatticeParams) -> GeoKey:
    subs_key = tuple((s.bravais, float(s.offset_frac[0]), float(s.offset_frac[1]), float(s.offset_frac[2]),
                      float(s.alpha_ratio), bool(s.visible)) for s in sublattices)
    p_key = (float(p.a), float(p.b_ratio), float(p.c_ratio), float(p.alpha), float(p.beta), float(p.gamma))
    return GeoKey(p=p_key, subs=subs_key)

@lru_cache(maxsize=256)
def build_geo(key: GeoKey):
    centers, alphas, shifts = centers_alphas_and_shifts(key.subs, key.p)
    zero_shift_idx = 13
    return centers, alphas, shifts, zero_shift_idx


def _build_pairs_for_cutoff(centers: np.ndarray, shifts: np.ndarray, cutoff: float) -> List[Tuple[int,int,int]]:
    return periodic_candidate_pairs(centers, shifts, cutoff)
