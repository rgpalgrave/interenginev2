# =====================================================================
# integrated_streamlit_app.py
# Combined interstitial-site finder + coordination calculator
# Features: 1D/2D scanning + unit cell visualization with metal atoms
# =====================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import csv
import sys
from pathlib import Path

# Import coordination calculator modules
# Adjust paths if needed for your deployment
try:
    from interstitial_engine import (
        LatticeParams,
        Sublattice,
        max_multiplicity_for_scale,
        find_threshold_s_for_N,
        lattice_vectors,
        bravais_basis,
        frac_to_cart,
    )
    from position_calculator import (
        generate_metal_positions,
        generate_intersection_positions,
        cart_to_frac,
        wrap_to_unit_cell,
        is_in_unit_cell,
    )
except ImportError:
    st.error("Could not import coordination calculator modules. Ensure interstitial_engine.py and position_calculator.py are in the Python path.")
    st.stop()

# =====================================================================
# Configuration
# =====================================================================

st.set_page_config(layout="wide", page_title="Crystallography Analysis Suite")

# Chemistry radii database (Å)
ANION_RADII: Dict[str, float] = {
    "O": 1.38, "S": 1.84, "Se": 1.98, "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20,
}
METAL_RADII: Dict[str, Dict[int, float]] = {
    "Li": {1: 0.76}, "Na": {1: 1.02}, "K": {1: 1.38}, "Rb": {1: 1.52}, "Cs": {1: 1.67},
    "Be": {2: 0.59}, "Mg": {2: 0.72}, "Ca": {2: 1.00}, "Sr": {2: 1.18}, "Ba": {2: 1.35},
    "Al": {3: 0.535}, "Ga": {3: 0.62}, "In": {3: 0.80}, "Tl": {1: 1.59, 3: 0.885},
    "Si": {4: 0.40}, "Ge": {4: 0.53}, "Sn": {2: 1.18, 4: 0.69}, "Pb": {2: 1.19, 4: 0.775},
    "Ti": {3: 0.67, 4: 0.605}, "Zr": {4: 0.72}, "Hf": {4: 0.71},
    "V": {2: 0.79, 3: 0.64, 4: 0.58, 5: 0.54},
    "Nb": {5: 0.64}, "Ta": {5: 0.64},
    "Mo": {4: 0.65, 5: 0.61, 6: 0.59}, "W": {4: 0.66, 5: 0.62, 6: 0.60},
    "Sc": {3: 0.745}, "Y": {3: 0.90},
    "Cr": {2: 0.80, 3: 0.615, 6: 0.52},
    "Mn": {2: 0.83, 3: 0.645, 4: 0.67},
    "Fe": {2: 0.78, 3: 0.645},
    "Co": {2: 0.745, 3: 0.61},
    "Ni": {2: 0.69, 3: 0.56},
    "Cu": {1: 0.91, 2: 0.73}, "Zn": {2: 0.74},
    "Cd": {2: 0.95}, "Hg": {2: 1.16},
    "Ru": {3: 0.68, 4: 0.62}, "Rh": {3: 0.665, 4: 0.60}, "Pd": {2: 0.86, 4: 0.615},
    "Ag": {1: 1.15}, "Au": {1: 1.37, 3: 0.85},
    "Pt": {2: 0.80, 4: 0.625},
    "La": {3: 1.032}, "Ce": {3: 1.01, 4: 0.87}, "Pr": {3: 0.99}, "Nd": {3: 0.983},
    "Sm": {3: 0.958}, "Eu": {2: 1.25, 3: 0.947}, "Gd": {3: 0.938}, "Tb": {3: 0.923},
    "Dy": {3: 0.912}, "Ho": {3: 0.901}, "Er": {3: 0.89}, "Tm": {3: 0.88},
    "Yb": {2: 1.16, 3: 0.868}, "Lu": {3: 0.861},
    "B": {3: 0.27}, "P": {5: 0.52}, "As": {5: 0.60}, "Sb": {5: 0.74}, "Bi": {3: 1.03, 5: 0.76},
}

# Wyckoff presets
Wyck = {
    "cubic_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "3c (0,1/2,1/2)":      {"type": "fixed", "xyz": (0.0, 0.5, 0.5)},
        "3d (1/2,0,0)":        {"type": "fixed", "xyz": (0.5, 0.0, 0.0)},
        "6e (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "cubic_F": {
        "4a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "4b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "8c (1/4,1/4,1/4)":    {"type": "fixed", "xyz": (0.25,0.25,0.25)},
        "24d (0,1/4,1/4)":     {"type": "fixed", "xyz": (0.0, 0.25,0.25)},
        "24e (x,0,0)":         {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "32f (x,x,x)":         {"type": "free",  "xyz": ("x","x","x")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "cubic_I": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "6c (0,1/2,0)":        {"type": "fixed", "xyz": (0.0, 0.5, 0.0)},
        "6d (1/2,0,0)":        {"type": "fixed", "xyz": (0.5, 0.0, 0.0)},
        "12e (x,0,0)":         {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "tetragonal_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "1c (1/2,1/2,0)":      {"type": "fixed", "xyz": (0.5, 0.5, 0.0)},
        "1d (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "2e (0,1/2,0)":        {"type": "fixed", "xyz": (0.0, 0.5, 0.0)},
        "2f (0,1/2,1/2)":      {"type": "fixed", "xyz": (0.0, 0.5, 0.5)},
        "2g (1/2,0,0)":        {"type": "fixed", "xyz": (0.5, 0.0, 0.0)},
        "2h (1/2,0,1/2)":      {"type": "fixed", "xyz": (0.5, 0.0, 0.5)},
        "2i (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "2j (x,x,0)":          {"type": "free",  "xyz": ("x","x",0.0)},
        "2k (x,x,z)":          {"type": "free",  "xyz": ("x","x","z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "tetragonal_I": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "4d (0,1/2,1/4)":      {"type": "fixed", "xyz": (0.0, 0.5, 0.25)},
        "4e (0,0,z)":          {"type": "free",  "xyz": (0.0, 0.0,"z")},
        "8g (0,1/2,z)":        {"type": "free",  "xyz": (0.0, 0.5,"z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "orthorhombic_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "1c (0,1/2,0)":        {"type": "fixed", "xyz": (0.0, 0.5, 0.0)},
        "1d (1/2,0,0)":        {"type": "fixed", "xyz": (0.5, 0.0, 0.0)},
        "2e (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "2f (0,y,0)":          {"type": "free",  "xyz": (0.0,"y", 0.0)},
        "2g (0,0,z)":          {"type": "free",  "xyz": (0.0, 0.0,"z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "orthorhombic_C": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (0,1/2,0)":        {"type": "fixed", "xyz": (0.0, 0.5, 0.0)},
        "4g (0,y,0)":          {"type": "free",  "xyz": (0.0,"y", 0.0)},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "orthorhombic_I": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "4e (0,0,z)":          {"type": "free",  "xyz": (0.0, 0.0,"z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "orthorhombic_F": {
        "4a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "4b (1/2,1/2,1/2)":    {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "8f (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "hexagonal_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "2c (1/3,2/3,0)":      {"type": "fixed", "xyz": (1/3, 2/3, 0.0)},
        "2d (1/3,2/3,1/2)":    {"type": "fixed", "xyz": (1/3, 2/3, 0.5)},
        "2e (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "2f (x,x,0)":          {"type": "free",  "xyz": ("x","x", 0.0)},
        "2g (x,x,z)":          {"type": "free",  "xyz": ("x","x","z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "hexagonal_HCP": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (0,0,1/4)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.25)},
        "2c (1/3,2/3,1/4)":    {"type": "fixed", "xyz": (1/3, 2/3, 0.25)},
        "2d (1/3,2/3,3/4)":    {"type": "fixed", "xyz": (1/3, 2/3, 0.75)},
        "4f (1/3,2/3,z)":      {"type": "free",  "xyz": (1/3, 2/3, "z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "rhombohedral_R": {
        "3a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "3b (0,0,1/2)":        {"type": "fixed", "xyz": (0.0, 0.0, 0.5)},
        "6c (0,0,z)":          {"type": "free",  "xyz": (0.0, 0.0, "z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "monoclinic_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2e (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "monoclinic_C": {
        "2a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "4i (x,0,0)":          {"type": "free",  "xyz": ("x", 0.0, 0.0)},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "triclinic_P": {
        "1a (0,0,0)":          {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (x,y,z)":          {"type": "free",  "xyz": ("x","y","z")},
        "Free placement":      {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
}

def norm_el(sym: str) -> str:
    """Normalize element symbol."""
    sym = sym.strip()
    return sym[0].upper() + sym[1:].lower() if sym else sym


# =====================================================================
# STREAMLIT SESSION STATE MANAGEMENT
# =====================================================================

if "subs" not in st.session_state:
    st.session_state.subs = []
if "p" not in st.session_state:
    st.session_state.p = LatticeParams(a=4.0)
if "repeat" not in st.session_state:
    st.session_state.repeat = 1
if "a" not in st.session_state:
    st.session_state.a = 4.0
if "tol_inside" not in st.session_state:
    st.session_state.tol_inside = 0.01
if "cluster_eps" not in st.session_state:
    st.session_state.cluster_eps = 0.4
if "data_1d" not in st.session_state:
    st.session_state.data_1d = None
if "clicked_point" not in st.session_state:
    st.session_state.clicked_point = None


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def _cell_corners_and_edges(a_vec: np.ndarray, b_vec: np.ndarray, c_vec: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    """Generate unit cell corners and edges for visualization."""
    fracs = np.array([
        [0,0,0],[1,0,0],[0,1,0],[0,0,1],
        [1,1,0],[1,0,1],[0,1,1],[1,1,1]
    ], float)
    M = np.vstack([a_vec, b_vec, c_vec]).T
    corners = (fracs @ M.T)
    edges = [
        (0,1),(0,2),(0,3),
        (1,4),(1,5),
        (2,4),(2,6),
        (3,5),(3,6),
        (4,7),(5,7),(6,7)
    ]
    return corners, edges

def _points_for_sublattice(sub: Sublattice, p: LatticeParams) -> np.ndarray:
    """Generate all points in a sublattice for the unit cell."""
    a_vec, b_vec, c_vec = lattice_vectors(p)
    basis = bravais_basis(sub.bravais)
    off = np.array(sub.offset_frac, float)
    pts = []
    for b in basis:
        frac = np.asarray(b) + off
        f_mod = np.mod(frac, 1.0)
        cart = frac_to_cart(f_mod, a_vec, b_vec, c_vec)
        pts.append(cart)
    return np.asarray(pts)

def _cart_to_frac(cart: np.ndarray, a_vec: np.ndarray, b_vec: np.ndarray, c_vec: np.ndarray) -> np.ndarray:
    """Convert Cartesian to fractional coordinates."""
    M = np.vstack([a_vec, b_vec, c_vec]).T
    return np.linalg.solve(M, cart.T).T


# =====================================================================
# TITLE AND NAVIGATION
# =====================================================================

st.title("🔬 Crystallography Analysis Suite")
st.markdown("*Integrated interstitial-site finder + coordination calculator*")

mode = st.radio("Select mode:", ["1D Parameter Scanning", "Coordination Calculator"])

# =====================================================================
# SHARED CONFIGURATION SECTION
# =====================================================================

st.divider()
st.subheader("⚙️ Configuration")

config_col1, config_col2 = st.columns(2)

with config_col1:
    st.write("**Lattice Parameters**")
    st.session_state.a = st.number_input("a (Å)", 1.0, 20.0, st.session_state.a, 0.1)
    b_ratio = st.number_input("b/a", 0.5, 3.0, st.session_state.p.b_ratio, 0.1)
    c_ratio = st.number_input("c/a", 0.5, 3.0, st.session_state.p.c_ratio, 0.1)

with config_col2:
    st.write("**Angles**")
    alpha = st.number_input("α (°)", 50.0, 130.0, st.session_state.p.alpha, 1.0)
    beta = st.number_input("β (°)", 50.0, 130.0, st.session_state.p.beta, 1.0)
    gamma = st.number_input("γ (°)", 50.0, 130.0, st.session_state.p.gamma, 1.0)

st.session_state.p = LatticeParams(
    a=st.session_state.a,
    b_ratio=b_ratio,
    c_ratio=c_ratio,
    alpha=alpha,
    beta=beta,
    gamma=gamma
)

# =====================================================================
# SUBLATTICE MANAGEMENT
# =====================================================================

st.divider()
st.subheader("🧬 Sublattices")

# Add sublattice
sub_col1, sub_col2, sub_col3 = st.columns(3)

with sub_col1:
    sub_name = st.text_input("Sublattice name", "Metal1", key="sub_name_input")

with sub_col2:
    bravais_options = [
        "cubic_P", "cubic_I", "cubic_F", "cubic_Diamond", "cubic_Pyrochlore",
        "tetragonal_P", "tetragonal_I",
        "orthorhombic_P", "orthorhombic_C", "orthorhombic_I", "orthorhombic_F",
        "hexagonal_P", "hexagonal_HCP",
        "rhombohedral_R",
        "monoclinic_P", "monoclinic_C",
        "triclinic_P"
    ]
    bravais = st.selectbox("Bravais lattice", bravais_options, key="bravais_select")

with sub_col3:
    if st.button("➕ Add Sublattice"):
        new_sub = Sublattice(
            name=sub_name,
            bravais=bravais,
            offset_frac=(0.0, 0.0, 0.0),
            alpha_ratio=1.0,
            visible=True
        )
        st.session_state.subs.append(new_sub)
        st.rerun()

# Display and edit sublattices
if st.session_state.subs:
    st.write("**Current Sublattices:**")
    for idx, sub in enumerate(st.session_state.subs):
        with st.expander(f"🔹 {sub.name} ({sub.bravais})"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                offset_x = st.number_input(f"Offset x ({sub.name})", -1.0, 1.0, sub.offset_frac[0], 0.01, key=f"ox_{idx}")
            with col2:
                offset_y = st.number_input(f"Offset y ({sub.name})", -1.0, 1.0, sub.offset_frac[1], 0.01, key=f"oy_{idx}")
            with col3:
                offset_z = st.number_input(f"Offset z ({sub.name})", -1.0, 1.0, sub.offset_frac[2], 0.01, key=f"oz_{idx}")
            with col4:
                alpha_ratio = st.number_input(f"α ratio ({sub.name})", 0.1, 3.0, sub.alpha_ratio, 0.1, key=f"alpha_{idx}")
            
            visible = st.checkbox(f"Visible", sub.visible, key=f"vis_{idx}")
            
            st.session_state.subs[idx] = Sublattice(
                name=sub.name,
                bravais=sub.bravais,
                offset_frac=(offset_x, offset_y, offset_z),
                alpha_ratio=alpha_ratio,
                visible=visible
            )
            
            if st.button(f"🗑️ Remove {sub.name}", key=f"del_{idx}"):
                st.session_state.subs.pop(idx)
                st.rerun()
else:
    st.info("No sublattices configured yet. Add one above.")

# =====================================================================
# TOLERANCE AND CLUSTERING PARAMETERS
# =====================================================================

st.divider()
st.write("**Calculation Parameters**")
param_col1, param_col2 = st.columns(2)

with param_col1:
    st.session_state.tol_inside = st.number_input(
        "Tolerance (Ångströms)", 0.001, 0.1, st.session_state.tol_inside, 0.001
    )

with param_col2:
    st.session_state.cluster_eps = st.number_input(
        "Clustering epsilon (fraction of a)", 0.1, 1.0, st.session_state.cluster_eps, 0.1
    )

# =====================================================================
# MODE 1: 1D PARAMETER SCANNING
# =====================================================================

if mode == "1D Parameter Scanning":
    st.divider()
    st.subheader("📊 1D Parameter Scan")
    
    if not st.session_state.subs:
        st.warning("Please configure at least one sublattice first.")
    else:
        scan_col1, scan_col2, scan_col3 = st.columns(3)
        
        with scan_col1:
            scan_param = st.selectbox(
                "Parameter to scan",
                ["a", "b/a", "c/a", "α", "β", "γ"],
                key="scan_param"
            )
        
        with scan_col2:
            param_min = st.number_input("Min value", -5.0, 5.0, 0.5, 0.1, key="param_min")
            param_max = st.number_input("Max value", -5.0, 5.0, 2.0, 0.1, key="param_max")
        
        with scan_col3:
            n_points = st.number_input("Number of points", 5, 100, 20, 1, key="n_points_1d")
            k_target = st.number_input("Target multiplicity (N)", 2, 24, 4, 1, key="k_target_1d")
        
        run_scan = st.button("🚀 Run 1D Scan", key="run_1d_scan")
        
        if run_scan:
            with st.spinner("Scanning parameter space..."):
                xs = np.linspace(param_min, param_max, n_points)
                ys = []
                
                def compute_multiplicity(param_val):
                    p_test = LatticeParams(
                        a=st.session_state.p.a if scan_param != "a" else param_val,
                        b_ratio=st.session_state.p.b_ratio if scan_param != "b/a" else param_val,
                        c_ratio=st.session_state.p.c_ratio if scan_param != "c/a" else param_val,
                        alpha=st.session_state.p.alpha if scan_param != "α" else param_val,
                        beta=st.session_state.p.beta if scan_param != "β" else param_val,
                        gamma=st.session_state.p.gamma if scan_param != "γ" else param_val,
                    )
                    m, _, _ = max_multiplicity_for_scale(
                        st.session_state.subs, p_test, 1, 0.35,
                        k_samples=4, tol_inside=st.session_state.tol_inside,
                        cluster_eps=st.session_state.cluster_eps * st.session_state.a
                    )
                    return float(m)
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    ys = list(executor.map(compute_multiplicity, xs))
                
                st.session_state.data_1d = {
                    "xs": xs,
                    "ys": np.array(ys),
                    "param": scan_param,
                    "k_target": k_target
                }
        
        # Display 1D scan results
        if st.session_state.data_1d is not None:
            data = st.session_state.data_1d
            
            # Interactive Plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data["xs"],
                y=data["ys"],
                mode="markers+lines",
                name="Maximum Multiplicity",
                marker=dict(size=8, color=data["ys"], colorscale="Viridis", showscale=True),
                line=dict(width=2)
            ))
            fig.add_hline(y=data["k_target"], line_dash="dash", line_color="red", annotation_text=f"Target: N={data['k_target']}")
            fig.update_layout(
                title=f"1D Scan: {data['param']} vs Maximum Multiplicity",
                xaxis_title=data["param"],
                yaxis_title="Max Multiplicity",
                height=500,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Click interface for passing to coordinator
            st.info("💡 Click on a point in the plot below to pass those parameters to the Coordination Calculator")
            
            click_col1, click_col2 = st.columns(2)
            with click_col1:
                selected_idx = st.number_input(
                    "Point index to inspect", 0, len(data["xs"])-1, 0, 1,
                    help="Select a point from the 1D scan to visualize in the coordination calculator"
                )
            
            with click_col2:
                if st.button("📌 Use this point in Coordination Calculator"):
                    param_val = data["xs"][selected_idx]
                    st.session_state.clicked_point = {
                        "param_name": data["param"],
                        "param_value": param_val,
                        "multiplicity": float(data["ys"][selected_idx])
                    }
                    st.success(f"✅ Loaded {data['param']} = {param_val:.4f}")
            
            # Export options
            st.divider()
            st.write("**Export Results**")
            
            exp_format = st.selectbox("Export format", ["CSV", "NumPy NPZ", "JSON"], key="exp_1d_format")
            
            if exp_format == "CSV":
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow([data["param"], "Max Multiplicity"])
                for x, y in zip(data["xs"], data["ys"]):
                    writer.writerow([x, y])
                
                st.download_button(
                    label="📥 Download CSV",
                    data=output.getvalue(),
                    file_name=f"scan_1d_{data['param']}.csv",
                    mime="text/plain",
                    key="download_1d_csv"
                )
            
            elif exp_format == "NumPy NPZ":
                buf = io.BytesIO()
                np.savez(
                    buf,
                    x_values=data["xs"],
                    y_values=data["ys"],
                    x_label=np.string_(data["param"]),
                    k_target=data["k_target"]
                )
                buf.seek(0)
                
                st.download_button(
                    label="📥 Download NPZ",
                    data=buf.getvalue(),
                    file_name=f"scan_1d_{data['param']}.npz",
                    mime="application/octet-stream",
                    key="download_1d_npz"
                )
            
            elif exp_format == "JSON":
                import json
                json_data = {
                    "parameter": data["param"],
                    "k_target": data["k_target"],
                    "x_values": data["xs"].tolist(),
                    "y_values": data["ys"].tolist(),
                }
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"scan_1d_{data['param']}.json",
                    mime="application/json",
                    key="download_1d_json"
                )

# =====================================================================
# MODE 2: COORDINATION CALCULATOR
# =====================================================================

elif mode == "Coordination Calculator":
    st.divider()
    st.subheader("🎯 Coordination Calculator")
    
    if not st.session_state.subs:
        st.warning("Please configure at least one sublattice first.")
    else:
        # Update parameters from 1D scan if clicked
        if st.session_state.clicked_point is not None:
            pt = st.session_state.clicked_point
            st.info(f"📍 Using parameters from 1D scan: {pt['param_name']} = {pt['param_value']:.4f}")
            
            # Update the appropriate parameter
            if pt["param_name"] == "a":
                st.session_state.p.a = pt["param_value"]
            elif pt["param_name"] == "b/a":
                st.session_state.p.b_ratio = pt["param_value"]
            elif pt["param_name"] == "c/a":
                st.session_state.p.c_ratio = pt["param_value"]
            elif pt["param_name"] == "α":
                st.session_state.p.alpha = pt["param_value"]
            elif pt["param_name"] == "β":
                st.session_state.p.beta = pt["param_value"]
            elif pt["param_name"] == "γ":
                st.session_state.p.gamma = pt["param_value"]
        
        calc_col1, calc_col2, calc_col3, calc_col4 = st.columns(4)
        
        with calc_col1:
            vis_s = st.number_input("Sphere radius scale (s)", 0.0, 5.0, 0.35, 0.01, key="vis_s")
        
        with calc_col2:
            show_mult = st.number_input("Show intersections with N =", 2, 24, 4, 1, key="show_mult")
        
        with calc_col3:
            marker_size = st.slider("Marker size (Å)", 0.05, 0.5, 0.15, 0.01)
        
        with calc_col4:
            draw_btn = st.button("🔄 Render Unit Cell")
        
        if draw_btn:
            a_vec, b_vec, c_vec = lattice_vectors(st.session_state.p)
            sub_pts = []
            for i, sub in enumerate(st.session_state.subs):
                if not sub.visible:
                    continue
                pts = _points_for_sublattice(sub, st.session_state.p)
                sub_pts.append((sub.name, pts))
            
            # Get intersection points
            m_all, reps, repc = max_multiplicity_for_scale(
                st.session_state.subs, st.session_state.p, 1, vis_s,
                k_samples=8, tol_inside=st.session_state.tol_inside,
                cluster_eps=st.session_state.cluster_eps * st.session_state.a, 
                early_stop_at=None
            )
            
            # Filter for selected multiplicity
            tol_frac = 1e-6
            rep_list = []
            for pt, cnt in zip(reps, repc):
                if int(cnt) != int(show_mult):
                    continue
                f = _cart_to_frac(np.asarray(pt), a_vec, b_vec, c_vec)
                if np.all(f >= -tol_frac) and np.all(f <= 1.0 + tol_frac):
                    rep_list.append(pt)
            rep_arr = np.asarray(rep_list) if rep_list else np.empty((0,3))
            
            # Create 3D visualization
            fig = go.Figure()
            
            # Unit cell edges
            corners, edges = _cell_corners_and_edges(a_vec, b_vec, c_vec)
            for i, j in edges:
                fig.add_trace(go.Scatter3d(
                    x=[corners[i,0], corners[j,0]],
                    y=[corners[i,1], corners[j,1]],
                    z=[corners[i,2], corners[j,2]],
                    mode="lines",
                    line=dict(width=4, color="gray"),
                    showlegend=False,
                    hoverinfo="skip"
                ))
            
            # Metal atoms from each sublattice
            palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
            for idx, (name, pts) in enumerate(sub_pts):
                if len(pts) == 0:
                    continue
                fig.add_trace(go.Scatter3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2],
                    mode="markers",
                    marker=dict(size=6, opacity=0.9, color=palette[idx % len(palette)]),
                    name=f"{name} sites",
                    marker_symbol="circle",
                ))
            
            # Intersection points
            if rep_arr.size:
                fig.add_trace(go.Scatter3d(
                    x=rep_arr[:,0], y=rep_arr[:,1], z=rep_arr[:,2],
                    mode="markers",
                    marker=dict(
                        size=max(2, int(marker_size / max(1e-9, st.session_state.a) * 30)),
                        opacity=0.95,
                        color="red"
                    ),
                    name=f"Intersections N={show_mult}",
                    marker_symbol="diamond",
                ))
            else:
                st.warning("No intersections of the selected multiplicity found in the central unit cell at this s value.")
            
            fig.update_scenes(aspectmode="data")
            fig.update_layout(
                scene=dict(
                    xaxis_title="x (Å)",
                    yaxis_title="y (Å)",
                    zaxis_title="z (Å)",
                    xaxis=dict(showbackground=False),
                    yaxis=dict(showbackground=False),
                    zaxis=dict(showbackground=False),
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=0, b=0),
                height=700,
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Max Multiplicity", int(m_all))
            with col2:
                st.metric("Intersections with N =", len(rep_arr) if rep_arr.size else 0)
            with col3:
                st.metric("Sphere scale (s)", f"{vis_s:.3f}")
            with col4:
                a_len = np.linalg.norm(a_vec)
                st.metric("Lattice constant (a)", f"{a_len:.4f} Å")

st.divider()
st.markdown("*Built with ❤️ using Streamlit and coordination geometry calculations*")
