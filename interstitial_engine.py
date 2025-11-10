# =====================================================
# interstitial_engine.py (TURBO ULTRA)
# Minimal-image + KDTree + pruned pairs + early-stop
# OPTIMIZED: vectorized caching, reduced allocations
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
# Pair-circle sampling
# -----------------
def pair_circle_samples(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float, k: int = 8) -> np.ndarray:
    v = c2 - c1
    d2 = float(np.dot(v,v))
    if d2 <= 1e-24:
        return np.empty((0,3))
    d = sqrt(d2)
    if not (abs(r1 - r2) < d < (r1 + r2)):
        return np.empty((0,3))
    n = v / d
    t = (r1*r1 - r2*r2 + d*d) / (2*d*d)
    center = c1 + t*v
    tmp = np.array([1.0,0.0,0.0]) if abs(n[0]) <= 0.9 else np.array([0.0,1.0,0.0])
    u = np.cross(tmp, n); nu = float(np.linalg.norm(u))
    if nu < 1e-12:
        tmp = np.array([0.0,0.0,1.0])
        u = np.cross(tmp, n); nu = float(np.linalg.norm(u))
        if nu < 1e-12:
            return np.empty((0,3))
    u /= nu
    w = np.cross(n,u)
    off = center - c1
    h2 = r1*r1 - float(np.dot(off,off))
    if h2 <= 0.0:
        return np.empty((0,3))
    h = sqrt(h2)
    ang = np.linspace(0.0, 2.0*np.pi, k, endpoint=False)
    return center + (h*np.cos(ang))[:,None]*u + (h*np.sin(ang))[:,None]*w


def _cluster(points: np.ndarray, counts: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, counts
    used = np.zeros(len(points), bool)
    reps = []
    repc = []
    eps2 = eps*eps
    for i in range(len(points)):
        if used[i]: 
            continue
        acc = [i]
        for j in range(i+1, len(points)):
            if used[j]: 
                continue
            d2 = float(np.dot(points[i]-points[j], points[i]-points[j]))
            if d2 < eps2:
                acc.append(j)
        used[acc] = True
        reps.append(np.mean(points[acc], axis=0))
        repc.append(int(np.max(counts[acc])))
    return np.asarray(reps), np.asarray(repc, int)


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


# -----------------
# Public API
# -----------------
def _build_pairs_for_cutoff(centers: np.ndarray, shifts: np.ndarray, cutoff: float) -> List[Tuple[int,int,int]]:
    return periodic_candidate_pairs(centers, shifts, cutoff)

def _count_multiplicity_kdtree(sample_pts: np.ndarray, tree: KDTree, radii: np.ndarray, tol_inside: float) -> np.ndarray:
    rmax = float(np.max(radii))
    idxs = tree.query_ball_point(sample_pts, r=rmax + tol_inside)
    out = np.zeros(len(sample_pts), dtype=int)
    for i, neigh in enumerate(idxs):
        if not neigh:
            continue
        d = np.linalg.norm(tree.data[neigh] - sample_pts[i], axis=1)
        out[i] = int(np.sum(d <= (radii[neigh] + tol_inside)))
    return out

def max_multiplicity_for_scale(
    sublattices: List[Sublattice],
    p: LatticeParams,
    repeat_ignored: int,
    scale_s: float,
    k_samples: int = 8,
    tol_inside: float = 1e-3,
    cluster_eps: Optional[float] = None,
    early_stop_at: Optional[int] = None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    key = _make_key(sublattices, p)
    centers, alphas, shifts, zero_idx = build_geo(key)
    if centers.size == 0:
        return 0, np.empty((0,3)), np.empty((0,))
    radii = (alphas * scale_s * p.a).astype(float)
    rmax = float(np.max(radii))

    pairs = _build_pairs_for_cutoff(centers, shifts, cutoff=2.0*rmax)
    if not pairs:
        return 0, np.empty((0,3)), np.empty((0,))

    if _HAVE_SCIPY:
        tree = KDTree(centers.copy())
    else:
        tree = None

    samples: List[np.ndarray] = []
    counts: List[int] = []

    for (i, j, s_idx) in pairs:
        c1 = centers[i]
        c2 = centers[j] + shifts[s_idx]
        r1 = radii[i]; r2 = radii[j]
        pts = pair_circle_samples(c1, r1, c2, r2, k=k_samples)
        if pts.size == 0:
            continue

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

        if early_stop_at is not None and np.any(c >= early_stop_at):
            keep = np.where(c >= 2)[0][:3]
            samples.extend(pts[keep])
            counts.extend([int(x) for x in c[keep]])
            m = int(np.max(c))
            if cluster_eps is None:
                cluster_eps = 0.1 * p.a
            reps, repc = _cluster(np.asarray(samples), np.asarray(counts, int), eps=cluster_eps)
            return m, reps, repc

        good = np.where(c >= 2)[0]
        if good.size:
            samples.extend(pts[good])
            counts.extend([int(x) for x in c[good]])

    if not samples:
        return 0, np.empty((0,3)), np.empty((0,))

    if cluster_eps is None:
        cluster_eps = 0.1 * p.a
    reps, repc = _cluster(np.asarray(samples), np.asarray(counts, int), eps=cluster_eps)
    mmax = int(repc.max()) if len(repc) else 0
    return mmax, reps, repc


def find_threshold_s_for_N(
    N_target: int,
    sublattices: List[Sublattice],
    p: LatticeParams,
    repeat: int = 1,
    s_min: float = 0.01,
    s_max: float = 0.9,
    k_samples_coarse: int = 4,
    k_samples_fine: int = 8,
    tol_inside: float = 1e-3,
    cluster_eps: Optional[float] = None,
    max_iter: int = 20,
) -> Tuple[Optional[float], Dict[int, float]]:
    """
    Aggressive speed optimizations:
    - Fewer coarse samples (k=4 default)
    - Fewer fine samples (k=8 default)
    - Shorter bisection (20 vs 28 iterations)
    - Earlier convergence check
    """
    milestones: Dict[int, float] = {}
    # coarse sweep: only 12 points instead of 16
    for s in np.linspace(s_min, s_max, 12):
        m,_,_ = max_multiplicity_for_scale(
            sublattices, p, 1, s,
            k_samples=k_samples_coarse, tol_inside=tol_inside,
            cluster_eps=cluster_eps, early_stop_at=N_target
        )
        milestones.setdefault(int(m), float(s))
        if m >= N_target:
            s_hi = s
            break
    else:
        return None, milestones

    # refine low bound: only 12 points
    s_lo = s_min
    for s2 in np.linspace(s_min, s_hi, 12):
        m,_,_ = max_multiplicity_for_scale(
            sublattices, p, 1, s2,
            k_samples=k_samples_coarse, tol_inside=tol_inside,
            cluster_eps=cluster_eps, early_stop_at=N_target
        )
        if m < N_target: s_lo = s2
        else:
            s_hi = s2; break

    # bisection: adaptive max_iter with early exit if range is tiny
    for it in range(max_iter):
        mid = 0.5*(s_lo+s_hi)
        if (s_hi - s_lo) < 1e-5:
            break
        ks = k_samples_coarse if (s_hi - s_lo) > 0.02 else k_samples_fine
        m,_,_ = max_multiplicity_for_scale(
            sublattices, p, 1, mid,
            k_samples=ks, tol_inside=tol_inside,
            cluster_eps=cluster_eps, early_stop_at=N_target
        )
        if m >= N_target: s_hi = mid
        else: s_lo = mid

    milestones[N_target] = s_hi
    return s_hi, milestones
