# =====================================================================
# integrated_streamlit_app_improved.py
# Crystallography Analysis Suite - Unified Mode
# Combines best features from earlier UI with position calculator
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
import json
import pandas as pd
from pathlib import Path

# Import coordination calculator modules
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

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
        calculate_complete_structure,
        format_position_dict,
        format_metal_atoms_csv,
        format_intersections_csv,
        format_xyz,
    )
except ImportError as e:
    st.error(f"❌ Could not import coordination calculator modules.")
    st.error(f"Error: {str(e)}")
    st.info(f"Debug info - Current directory: {current_dir}")
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

# Wyckoff presets - abbreviated for clarity
Wyck = {
    "cubic_P": {
        "1a (0,0,0)": {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "1b (1/2,1/2,1/2)": {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "Free placement": {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "cubic_F": {
        "4a (0,0,0)": {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "4b (1/2,1/2,1/2)": {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "Free placement": {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
    "cubic_I": {
        "2a (0,0,0)": {"type": "fixed", "xyz": (0.0, 0.0, 0.0)},
        "2b (1/2,1/2,1/2)": {"type": "fixed", "xyz": (0.5, 0.5, 0.5)},
        "Free placement": {"type": "free3d", "xyz": (0.0, 0.0, 0.0)},
    },
}

def norm_el(sym: str) -> str:
    """Normalize element symbol."""
    sym = sym.strip()
    return sym[0].upper() + sym[1:].lower() if sym else sym


# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================

if "subs" not in st.session_state:
    st.session_state.subs = []
if "p" not in st.session_state:
    st.session_state.p = LatticeParams(a=4.0)
if "a" not in st.session_state:
    st.session_state.a = 4.0
if "tol_inside" not in st.session_state:
    st.session_state.tol_inside = 0.01
if "cluster_eps" not in st.session_state:
    st.session_state.cluster_eps = 0.4
if "structure" not in st.session_state:
    st.session_state.structure = None
if "calc_s" not in st.session_state:
    st.session_state.calc_s = 0.35
if "calc_N" not in st.session_state:
    st.session_state.calc_N = 4


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def _cell_corners_and_edges(a_vec: np.ndarray, b_vec: np.ndarray, c_vec: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Generate unit cell corners and edges for visualization."""
    fracs = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], float)
    M = np.vstack([a_vec, b_vec, c_vec]).T
    corners = (fracs @ M.T)
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7)
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


# =====================================================================
# TITLE AND HEADER
# =====================================================================

st.title("🔬 Crystallography Analysis Suite")
st.markdown("*Advanced coordination environment analysis and visualization*")

# =====================================================================
# CONFIGURATION SECTION
# =====================================================================

st.divider()
st.subheader("⚙️ Configuration")

config_col1, config_col2, config_col3 = st.columns(3)

with config_col1:
    st.write("**Lattice Parameters**")
    st.session_state.a = st.number_input("a (Å)", 1.0, 20.0, st.session_state.a, 0.1)
    b_ratio = st.number_input("b/a", 0.5, 3.0, st.session_state.p.b_ratio, 0.1)
    c_ratio = st.number_input("c/a", 0.5, 3.0, st.session_state.p.c_ratio, 0.1)

with config_col2:
    st.write("**Angles (°)**")
    alpha = st.number_input("α", 50.0, 130.0, st.session_state.p.alpha, 1.0)
    beta = st.number_input("β", 50.0, 130.0, st.session_state.p.beta, 1.0)
    gamma = st.number_input("γ", 50.0, 130.0, st.session_state.p.gamma, 1.0)

with config_col3:
    st.write("**Calculation Parameters**")
    st.session_state.tol_inside = st.number_input(
        "Tolerance (Å)", 0.001, 0.1, st.session_state.tol_inside, 0.001,
        help="Tolerance for point-in-sphere checks"
    )
    st.session_state.cluster_eps = st.number_input(
        "Clustering ε (fraction of a)", 0.1, 1.0, st.session_state.cluster_eps, 0.1,
        help="Distance threshold for clustering intersections"
    )

# Update lattice parameters
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
    if st.button("➕ Add Sublattice", use_container_width=True):
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
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                offset_x = st.number_input(f"Offset x", -1.0, 1.0, sub.offset_frac[0], 0.01, key=f"ox_{idx}")
            with col2:
                offset_y = st.number_input(f"Offset y", -1.0, 1.0, sub.offset_frac[1], 0.01, key=f"oy_{idx}")
            with col3:
                offset_z = st.number_input(f"Offset z", -1.0, 1.0, sub.offset_frac[2], 0.01, key=f"oz_{idx}")
            with col4:
                alpha_ratio = st.number_input(f"α ratio", 0.1, 3.0, sub.alpha_ratio, 0.1, key=f"alpha_{idx}")
            with col5:
                visible = st.checkbox("Visible", sub.visible, key=f"vis_{idx}")
            
            st.session_state.subs[idx] = Sublattice(
                name=sub.name,
                bravais=sub.bravais,
                offset_frac=(offset_x, offset_y, offset_z),
                alpha_ratio=alpha_ratio,
                visible=visible
            )
            
            if st.button(f"🗑️ Remove", key=f"del_{idx}", use_container_width=True):
                st.session_state.subs.pop(idx)
                st.rerun()
else:
    st.info("No sublattices configured yet. Add one above to begin.")

# =====================================================================
# CALCULATION MODES
# =====================================================================

st.divider()
st.subheader("🎯 Analysis")

if not st.session_state.subs:
    st.warning("⚠️ Please configure at least one sublattice to proceed.")
else:
    mode = st.radio(
        "Select analysis type:",
        ["Direct Calculation", "Find Threshold for N", "1D Parameter Scan"],
        horizontal=True
    )
    
    # =====================================================================
    # MODE 1: DIRECT CALCULATION
    # =====================================================================
    
    if mode == "Direct Calculation":
        st.write("**Calculate structure for specific s and N values**")
        
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        
        with calc_col1:
            st.session_state.calc_s = st.number_input(
                "Sphere radius scale (s)", 0.01, 5.0, st.session_state.calc_s, 0.01
            )
        
        with calc_col2:
            st.session_state.calc_N = st.number_input(
                "Target multiplicity (N)", 2, 24, st.session_state.calc_N, 1
            )
        
        with calc_col3:
            if st.button("🚀 Calculate", use_container_width=True, key="calc_direct"):
                with st.spinner("Calculating structure..."):
                    try:
                        structure = calculate_complete_structure(
                            sublattices=st.session_state.subs,
                            p=st.session_state.p,
                            scale_s=st.session_state.calc_s,
                            target_N=st.session_state.calc_N,
                            supercell_metals=(1, 1, 1),
                            k_samples=16,
                            unit_cell_only=True
                        )
                        st.session_state.structure = structure
                        st.session_state.structure.target_N = st.session_state.calc_N
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Calculation failed: {e}")
    
    # =====================================================================
    # MODE 2: FIND THRESHOLD
    # =====================================================================
    
    elif mode == "Find Threshold for N":
        st.write("**Find optimal sphere radius (s) for target multiplicity (N)**")
        
        thresh_col1, thresh_col2, thresh_col3, thresh_col4 = st.columns(4)
        
        with thresh_col1:
            target_N_thresh = st.number_input(
                "Target N", 2, 24, 4, 1, key="target_N_thresh"
            )
        
        with thresh_col2:
            s_min = st.number_input("s min", 0.01, 1.0, 0.1, 0.01, key="s_min_thresh")
        
        with thresh_col3:
            s_max = st.number_input("s max", 0.5, 5.0, 0.9, 0.1, key="s_max_thresh")
        
        with thresh_col4:
            if st.button("🔍 Find Threshold", use_container_width=True, key="find_thresh"):
                with st.spinner(f"Finding s for N={target_N_thresh}..."):
                    try:
                        s_opt, milestones = find_threshold_s_for_N(
                            N_target=target_N_thresh,
                            sublattices=st.session_state.subs,
                            p=st.session_state.p,
                            repeat=1,
                            s_min=s_min,
                            s_max=s_max,
                            k_samples_coarse=4,
                            k_samples_fine=8,
                            tol_inside=st.session_state.tol_inside,
                            cluster_eps=st.session_state.cluster_eps * st.session_state.a,
                            max_iter=20
                        )
                        
                        if s_opt is not None:
                            st.success(f"✅ Optimal s = {s_opt:.6f} for N ≥ {target_N_thresh}")
                            st.session_state.calc_s = s_opt
                            st.session_state.calc_N = target_N_thresh
                            
                            # Show milestones
                            st.write("**Multiplicity Milestones:**")
                            milestone_data = {
                                'N': sorted(milestones.keys()),
                                's value': [milestones[n] for n in sorted(milestones.keys())]
                            }
                            st.dataframe(pd.DataFrame(milestone_data), use_container_width=True, hide_index=True)
                            
                            # Auto-calculate with optimal s
                            with st.spinner(f"Calculating structure with optimal s..."):
                                structure = calculate_complete_structure(
                                    sublattices=st.session_state.subs,
                                    p=st.session_state.p,
                                    scale_s=s_opt,
                                    target_N=target_N_thresh,
                                    supercell_metals=(1, 1, 1),
                                    k_samples=16,
                                    unit_cell_only=True
                                )
                                st.session_state.structure = structure
                                st.session_state.structure.target_N = target_N_thresh
                                st.rerun()
                        else:
                            st.error(f"❌ Could not find s for N={target_N_thresh} in range [{s_min}, {s_max}]")
                    except Exception as e:
                        st.error(f"❌ Threshold search failed: {e}")
    
    # =====================================================================
    # MODE 3: 1D PARAMETER SCAN
    # =====================================================================
    
    elif mode == "1D Parameter Scan":
        st.write("**Scan lattice parameter and find multiplicity profile**")
        
        scan_col1, scan_col2, scan_col3, scan_col4 = st.columns(4)
        
        with scan_col1:
            scan_param = st.selectbox(
                "Parameter",
                ["a", "b/a", "c/a", "α", "β", "γ"],
                key="scan_param"
            )
        
        with scan_col2:
            param_min = st.number_input("Min", -5.0, 5.0, 0.5, 0.1, key="param_min_scan")
        
        with scan_col3:
            param_max = st.number_input("Max", -5.0, 5.0, 2.0, 0.1, key="param_max_scan")
        
        with scan_col4:
            n_points = st.number_input("Points", 5, 100, 20, 1, key="n_points_scan")
        
        if st.button("🚀 Run Scan", use_container_width=True, key="run_scan"):
            with st.spinner("Scanning parameter space..."):
                captured_p = st.session_state.p
                captured_subs = st.session_state.subs
                captured_tol_inside = st.session_state.tol_inside
                captured_cluster_eps = st.session_state.cluster_eps
                captured_a = st.session_state.a
                
                xs = np.linspace(param_min, param_max, n_points)
                ys = []
                
                def compute_multiplicity(param_val):
                    p_test = LatticeParams(
                        a=captured_p.a if scan_param != "a" else param_val,
                        b_ratio=captured_p.b_ratio if scan_param != "b/a" else param_val,
                        c_ratio=captured_p.c_ratio if scan_param != "c/a" else param_val,
                        alpha=captured_p.alpha if scan_param != "α" else param_val,
                        beta=captured_p.beta if scan_param != "β" else param_val,
                        gamma=captured_p.gamma if scan_param != "γ" else param_val,
                    )
                    m, _, _ = max_multiplicity_for_scale(
                        captured_subs, p_test, 1, 0.35,
                        k_samples=4, tol_inside=captured_tol_inside,
                        cluster_eps=captured_cluster_eps * captured_a
                    )
                    return float(m)
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    ys = list(executor.map(compute_multiplicity, xs))
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="markers+lines",
                    name="Max Multiplicity",
                    marker=dict(size=8, color=ys, colorscale="Viridis", showscale=True),
                ))
                fig.update_layout(
                    title=f"1D Scan: {scan_param} vs Max Multiplicity",
                    xaxis_title=scan_param,
                    yaxis_title="Max Multiplicity",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Export option
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow([scan_param, "Max Multiplicity"])
                for x, y in zip(xs, ys):
                    writer.writerow([x, y])
                
                st.download_button(
                    label="📥 Download CSV",
                    data=output.getvalue(),
                    file_name=f"scan_1d_{scan_param}.csv",
                    mime="text/plain",
                )

# =====================================================================
# RESULTS DISPLAY
# =====================================================================

if st.session_state.structure is not None:
    structure = st.session_state.structure
    
    st.divider()
    st.subheader("📊 Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Metal Atoms", len(structure.metal_atoms.fractional))
    col2.metric("Intersections", len(structure.intersections.fractional))
    col3.metric("s value", f"{structure.scale_s:.4f}")
    col4.metric("Target N", structure.target_N)
    
    # Tabs for detailed views
    tab1, tab2, tab3, tab4 = st.tabs(["Metal Atoms", "Intersections", "3D Visualization", "Export"])
    
    with tab1:
        st.subheader("Metal Atom Positions")
        if len(structure.metal_atoms.fractional) > 0:
            df_metals = pd.DataFrame({
                'Index': range(len(structure.metal_atoms.fractional)),
                'Sublattice': structure.metal_atoms.sublattice_name,
                'Frac X': np.round(structure.metal_atoms.fractional[:, 0], 6),
                'Frac Y': np.round(structure.metal_atoms.fractional[:, 1], 6),
                'Frac Z': np.round(structure.metal_atoms.fractional[:, 2], 6),
                'Cart X': np.round(structure.metal_atoms.cartesian[:, 0], 4),
                'Cart Y': np.round(structure.metal_atoms.cartesian[:, 1], 4),
                'Cart Z': np.round(structure.metal_atoms.cartesian[:, 2], 4),
                'Radius (Å)': np.round(structure.metal_atoms.radius, 4),
            })
            st.dataframe(df_metals, use_container_width=True, hide_index=True)
        else:
            st.info("No metal atoms found.")
    
    with tab2:
        st.subheader("Intersection Positions")
        if len(structure.intersections.fractional) > 0:
            df_intersections = pd.DataFrame({
                'Index': range(len(structure.intersections.fractional)),
                'N': structure.intersections.multiplicity,
                'Frac X': np.round(structure.intersections.fractional[:, 0], 6),
                'Frac Y': np.round(structure.intersections.fractional[:, 1], 6),
                'Frac Z': np.round(structure.intersections.fractional[:, 2], 6),
                'Cart X': np.round(structure.intersections.cartesian[:, 0], 4),
                'Cart Y': np.round(structure.intersections.cartesian[:, 1], 4),
                'Cart Z': np.round(structure.intersections.cartesian[:, 2], 4),
                'Contributing Atoms': [','.join(map(str, atoms)) for atoms in structure.intersections.contributing_atoms],
            })
            st.dataframe(df_intersections, use_container_width=True, hide_index=True)
        else:
            st.info("No intersections found for current parameters.")
    
    with tab3:
        st.subheader("3D Structure Visualization")
        
        a_vec, b_vec, c_vec = lattice_vectors(st.session_state.p)
        fig_3d = go.Figure()
        
        # Unit cell edges
        corners, edges = _cell_corners_and_edges(a_vec, b_vec, c_vec)
        for i, j in edges:
            fig_3d.add_trace(go.Scatter3d(
                x=[corners[i, 0], corners[j, 0]],
                y=[corners[i, 1], corners[j, 1]],
                z=[corners[i, 2], corners[j, 2]],
                mode="lines",
                line=dict(width=2, color="gray"),
                showlegend=False,
                hoverinfo="skip"
            ))
        
        # Metal atoms
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        unique_sublattices = np.unique(structure.metal_atoms.sublattice_name)
        for idx, sub_name in enumerate(unique_sublattices):
            mask = np.array(structure.metal_atoms.sublattice_name) == sub_name
            pts_cart = structure.metal_atoms.cartesian[mask]
            fig_3d.add_trace(go.Scatter3d(
                x=pts_cart[:, 0], y=pts_cart[:, 1], z=pts_cart[:, 2],
                mode="markers",
                marker=dict(size=5, opacity=0.8, color=palette[idx % len(palette)]),
                name=f"Metal: {sub_name}",
            ))
        
        # Intersections
        if len(structure.intersections.fractional) > 0:
            cart_pos = structure.intersections.cartesian
            mults = structure.intersections.multiplicity
            fig_3d.add_trace(go.Scatter3d(
                x=cart_pos[:, 0], y=cart_pos[:, 1], z=cart_pos[:, 2],
                mode="markers",
                marker=dict(
                    size=6,
                    color=mults,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Multiplicity"),
                    symbol="diamond",
                    line=dict(color="black", width=0.5)
                ),
                name="Intersections",
                text=[f"N={m}" for m in mults],
                hovertemplate='<b>Intersection</b><br>Pos: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>%{text}<extra></extra>'
            ))
        
        fig_3d.update_scenes(aspectmode="data")
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)",
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False),
            ),
            height=700,
            showlegend=True,
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with tab4:
        st.subheader("Export Data")
        
        col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
        
        with col_exp1:
            json_data = json.dumps(format_position_dict(structure), indent=2)
            st.download_button(
                label="📥 JSON",
                data=json_data,
                file_name=f"structure_s{structure.scale_s:.4f}_N{structure.target_N}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_exp2:
            csv_metals = format_metal_atoms_csv(structure)
            st.download_button(
                label="📥 Metals CSV",
                data=csv_metals,
                file_name=f"metal_atoms_s{structure.scale_s:.4f}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_exp3:
            csv_intersections = format_intersections_csv(structure)
            st.download_button(
                label="📥 Intersections CSV",
                data=csv_intersections,
                file_name=f"intersections_s{structure.scale_s:.4f}_N{structure.target_N}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_exp4:
            xyz_data = format_xyz(structure, include_intersections=True)
            st.download_button(
                label="📥 XYZ",
                data=xyz_data,
                file_name=f"structure_s{structure.scale_s:.4f}_N{structure.target_N}.xyz",
                mime="text/plain",
                use_container_width=True
            )

st.divider()
st.markdown("*Built with ❤️ using Streamlit and coordination geometry calculations*")
