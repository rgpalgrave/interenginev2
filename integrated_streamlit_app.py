# =====================================================================
# crystallography_app_v2.py
# Redesigned Streamlit UI for Crystallography Analysis Suite
# Single page layout with global lattice and multiple sublattices
# =====================================================================

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import pandas as pd
import json
import io
import sys
from pathlib import Path

# Import modules
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from interstitial_engine import (
        LatticeParams,
        Sublattice,
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
    st.error(f"❌ Could not import modules: {e}")
    st.stop()

# =====================================================================
# Page configuration
# =====================================================================

st.set_page_config(
    page_title="Crystallography Analysis Suite",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🔬 Crystallography Analysis Suite")
st.markdown("*Calculate metal atom and intersection site positions*")

# =====================================================================
# Session state initialization
# =====================================================================

if "sublattices" not in st.session_state:
    st.session_state.sublattices = []

if "lattice_params" not in st.session_state:
    st.session_state.lattice_params = {
        "bravais": "cubic_P",
        "a": 4.0,
        "b_ratio": 1.0,
        "c_ratio": 1.0,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    }

if "calculation_result" not in st.session_state:
    st.session_state.calculation_result = None


# =====================================================================
# Bravais lattice options
# =====================================================================

BRAVAIS_OPTIONS = [
    "cubic_P", "cubic_I", "cubic_F", "cubic_Diamond", "cubic_Pyrochlore",
    "tetragonal_P", "tetragonal_I",
    "orthorhombic_P", "orthorhombic_C", "orthorhombic_I", "orthorhombic_F",
    "hexagonal_P", "hexagonal_HCP",
    "rhombohedral_R",
    "monoclinic_P", "monoclinic_C",
    "triclinic_P"
]


# =====================================================================
# SECTION 1: Global Lattice Parameters
# =====================================================================

st.divider()
st.subheader("📐 Global Lattice Parameters")
st.markdown("*All sublattices inherit these parameters*")

col1, col2, col3, col4 = st.columns(4)

with col1:
    lattice_type = st.selectbox(
        "Bravais lattice type",
        BRAVAIS_OPTIONS,
        index=BRAVAIS_OPTIONS.index(st.session_state.lattice_params["bravais"]),
        key="lattice_type"
    )
    st.session_state.lattice_params["bravais"] = lattice_type

with col2:
    a_val = st.number_input(
        "a (Å)",
        min_value=1.0,
        max_value=20.0,
        value=st.session_state.lattice_params["a"],
        step=0.1,
        key="a_input"
    )
    st.session_state.lattice_params["a"] = a_val

with col3:
    b_ratio = st.number_input(
        "b/a (ratio)",
        min_value=0.5,
        max_value=3.0,
        value=st.session_state.lattice_params["b_ratio"],
        step=0.1,
        key="b_ratio_input"
    )
    st.session_state.lattice_params["b_ratio"] = b_ratio

with col4:
    c_ratio = st.number_input(
        "c/a (ratio)",
        min_value=0.5,
        max_value=3.0,
        value=st.session_state.lattice_params["c_ratio"],
        step=0.1,
        key="c_ratio_input"
    )
    st.session_state.lattice_params["c_ratio"] = c_ratio

col5, col6, col7 = st.columns(3)

with col5:
    alpha = st.number_input(
        "α (degrees)",
        min_value=50.0,
        max_value=130.0,
        value=st.session_state.lattice_params["alpha"],
        step=1.0,
        key="alpha_input"
    )
    st.session_state.lattice_params["alpha"] = alpha

with col6:
    beta = st.number_input(
        "β (degrees)",
        min_value=50.0,
        max_value=130.0,
        value=st.session_state.lattice_params["beta"],
        step=1.0,
        key="beta_input"
    )
    st.session_state.lattice_params["beta"] = beta

with col7:
    gamma = st.number_input(
        "γ (degrees)",
        min_value=50.0,
        max_value=130.0,
        value=st.session_state.lattice_params["gamma"],
        step=1.0,
        key="gamma_input"
    )
    st.session_state.lattice_params["gamma"] = gamma

# Create LatticeParams object
current_lattice = LatticeParams(
    a=st.session_state.lattice_params["a"],
    b_ratio=st.session_state.lattice_params["b_ratio"],
    c_ratio=st.session_state.lattice_params["c_ratio"],
    alpha=st.session_state.lattice_params["alpha"],
    beta=st.session_state.lattice_params["beta"],
    gamma=st.session_state.lattice_params["gamma"],
)

# Display computed lattice parameters
st.info(
    f"**Computed lattice:** "
    f"a={st.session_state.lattice_params['a']:.3f} Å, "
    f"b={st.session_state.lattice_params['a'] * st.session_state.lattice_params['b_ratio']:.3f} Å, "
    f"c={st.session_state.lattice_params['a'] * st.session_state.lattice_params['c_ratio']:.3f} Å"
)


# =====================================================================
# SECTION 2: Sublattice Management
# =====================================================================

st.divider()
st.subheader("🧬 Sublattices")
st.markdown("*Define metal atom sublattices - all use the global lattice type*")

# Add new sublattice form
st.write("**Add new sublattice:**")
col_add1, col_add2, col_add3, col_add4, col_add5, col_add6 = st.columns(6)

with col_add1:
    new_sub_name = st.text_input("Name", value="Metal1", key="new_sub_name")

with col_add2:
    new_offset_x = st.number_input("Offset x (frac)", -1.0, 1.0, 0.0, 0.01, key="new_offset_x")

with col_add3:
    new_offset_y = st.number_input("Offset y (frac)", -1.0, 1.0, 0.0, 0.01, key="new_offset_y")

with col_add4:
    new_offset_z = st.number_input("Offset z (frac)", -1.0, 1.0, 0.0, 0.01, key="new_offset_z")

with col_add5:
    new_sphere_size = st.number_input(
        "Sphere size (×a)",
        min_value=0.01,
        max_value=5.0,
        value=0.35,
        step=0.01,
        key="new_sphere_size"
    )

with col_add6:
    if st.button("➕ Add", use_container_width=True, key="add_sublattice_btn"):
        new_sub = {
            "name": new_sub_name,
            "offset": (new_offset_x, new_offset_y, new_offset_z),
            "sphere_size": new_sphere_size,
        }
        st.session_state.sublattices.append(new_sub)
        st.rerun()

# Display and manage existing sublattices
if st.session_state.sublattices:
    st.write("**Current sublattices:**")
    
    for idx, sub in enumerate(st.session_state.sublattices):
        col_disp1, col_disp2, col_disp3, col_disp4, col_disp5, col_del = st.columns([1, 1.2, 1.2, 1.2, 1.2, 0.8])
        
        with col_disp1:
            st.write(f"**{sub['name']}**")
        
        with col_disp2:
            st.write(f"Offset: ({sub['offset'][0]:.3f}, {sub['offset'][1]:.3f}, {sub['offset'][2]:.3f})")
        
        with col_disp3:
            st.write(f"Sphere: {sub['sphere_size']:.4f}×a = {sub['sphere_size']*st.session_state.lattice_params['a']:.4f} Å")
        
        with col_disp4:
            st.write(f"Bravais: {st.session_state.lattice_params['bravais']}")
        
        with col_disp5:
            pass
        
        with col_del:
            if st.button("🗑️", key=f"del_sub_{idx}", use_container_width=True):
                st.session_state.sublattices.pop(idx)
                st.rerun()
else:
    st.info("No sublattices added yet. Add one above to begin.")


# =====================================================================
# SECTION 3: Calculation Parameters
# =====================================================================

st.divider()
st.subheader("⚙️ Calculation Parameters")

col_calc1, col_calc2, col_calc3 = st.columns(3)

with col_calc1:
    scale_s = st.number_input(
        "Scale factor (s)",
        min_value=0.01,
        max_value=2.0,
        value=0.35,
        step=0.01,
        help="Multiplier for sphere radius: actual_radius = s × sphere_size × a"
    )

with col_calc2:
    target_N = st.number_input(
        "Target intersection order (N)",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help="Minimum multiplicity for intersection sites"
    )

with col_calc3:
    k_samples = st.number_input(
        "Sampling density (k)",
        min_value=4,
        max_value=32,
        value=16,
        step=1,
        help="Number of samples per pair circle (higher = better but slower)"
    )

col_calc4, col_calc5 = st.columns(2)

with col_calc4:
    tol_inside = st.number_input(
        "Tolerance (Å)",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        help="Tolerance for point-in-sphere tests"
    )

with col_calc5:
    cluster_eps_frac = st.number_input(
        "Clustering tolerance (frac units)",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        help="Distance threshold for clustering nearby intersections"
    )


# =====================================================================
# SECTION 4: Calculation & Results
# =====================================================================

st.divider()
st.subheader("🚀 Calculate")

if not st.session_state.sublattices:
    st.warning("⚠️ Please add at least one sublattice to proceed.")
else:
    col_btn1, col_btn2 = st.columns([3, 1])
    
    with col_btn1:
        run_calc = st.button(
            "🔧 Calculate Structure",
            use_container_width=True,
            key="run_calc",
            type="primary"
        )
    
    if run_calc:
        with st.spinner("Calculating metal atom and intersection positions..."):
            try:
                # Build sublattice objects
                sublattice_objs = []
                for sub_dict in st.session_state.sublattices:
                    sub_obj = Sublattice(
                        name=sub_dict["name"],
                        bravais=st.session_state.lattice_params["bravais"],
                        offset_frac=sub_dict["offset"],
                        alpha_ratio=sub_dict["sphere_size"],
                        visible=True
                    )
                    sublattice_objs.append(sub_obj)
                
                # Calculate
                structure = calculate_complete_structure(
                    sublattices=sublattice_objs,
                    p=current_lattice,
                    scale_s=scale_s,
                    target_N=target_N,
                    supercell_metals=(1, 1, 1),
                    k_samples=k_samples,
                    unit_cell_only=True
                )
                
                # Store result and add metadata
                st.session_state.calculation_result = structure
                st.session_state.calculation_result.target_N = target_N
                st.session_state.calculation_result.scale_s = scale_s
                
                st.success("✅ Calculation complete!")
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Calculation failed: {str(e)}")
                import traceback
                st.write(traceback.format_exc())


# =====================================================================
# SECTION 5: Results Display
# =====================================================================

if st.session_state.calculation_result is not None:
    result = st.session_state.calculation_result
    
    st.divider()
    st.subheader("📊 Results")
    
    # Metrics row
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    
    with metric_col1:
        st.metric("Metal Atoms", len(result.metal_atoms.fractional))
    
    with metric_col2:
        st.metric("Intersections", len(result.intersections.fractional))
    
    with metric_col3:
        st.metric("Scale (s)", f"{result.scale_s:.4f}")
    
    with metric_col4:
        st.metric("Target N", result.target_N)
    
    with metric_col5:
        max_mult = int(np.max(result.intersections.multiplicity)) if len(result.intersections.multiplicity) > 0 else 0
        st.metric("Max Multiplicity", max_mult)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["3D Structure", "Metal Atoms", "Intersections", "Export"])
    
    # =====================================================================
    # TAB 1: 3D Visualization
    # =====================================================================
    
    with tab1:
        st.subheader("3D Unit Cell Visualization")
        st.markdown("*Rotate, zoom, and pan the 3D view*")
        
        # Get lattice vectors
        a_vec, b_vec, c_vec = lattice_vectors(current_lattice)
        
        # Create figure
        fig_3d = go.Figure()
        
        # Draw unit cell edges
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], dtype=float)
        
        # Convert fractional corners to Cartesian
        corners_cart = np.array([
            frac_to_cart(np.array(c), a_vec, b_vec, c_vec)
            for c in corners
        ])
        
        # Define edges
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5),
            (2, 4), (2, 6),
            (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 7)
        ]
        
        # Draw edges
        for i, j in edges:
            fig_3d.add_trace(go.Scatter3d(
                x=[corners_cart[i, 0], corners_cart[j, 0]],
                y=[corners_cart[i, 1], corners_cart[j, 1]],
                z=[corners_cart[i, 2], corners_cart[j, 2]],
                mode="lines",
                line=dict(color="gray", width=3),
                showlegend=False,
                hoverinfo="skip"
            ))
        
        # Color palette for sublattices
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        
        # Draw metal atoms
        unique_sublattices = sorted(set(result.metal_atoms.sublattice_name))
        for sub_idx, sub_name in enumerate(unique_sublattices):
            mask = np.array(result.metal_atoms.sublattice_name) == sub_name
            pts_cart = result.metal_atoms.cartesian[mask]
            
            fig_3d.add_trace(go.Scatter3d(
                x=pts_cart[:, 0],
                y=pts_cart[:, 1],
                z=pts_cart[:, 2],
                mode="markers",
                marker=dict(
                    size=8,
                    color=colors[sub_idx % len(colors)],
                    opacity=0.9,
                    symbol="circle"
                ),
                name=f"Metal: {sub_name}",
                text=[f"{sub_name}" for _ in range(len(pts_cart))],
                hovertemplate="<b>%{text}</b><br>Pos: (%{x:.4f}, %{y:.4f}, %{z:.4f})<extra></extra>"
            ))
        
        # Draw intersections
        if len(result.intersections.fractional) > 0:
            cart_pos = result.intersections.cartesian
            mults = result.intersections.multiplicity
            
            fig_3d.add_trace(go.Scatter3d(
                x=cart_pos[:, 0],
                y=cart_pos[:, 1],
                z=cart_pos[:, 2],
                mode="markers",
                marker=dict(
                    size=7,
                    color=mults,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Multiplicity", x=1.15),
                    symbol="diamond",
                    line=dict(color="white", width=1),
                    opacity=0.8
                ),
                name="Intersections",
                text=[f"N={m}" for m in mults],
                hovertemplate="<b>Intersection</b><br>N: %{text}<br>Pos: (%{x:.4f}, %{y:.4f}, %{z:.4f})<extra></extra>"
            ))
        
        # Update layout
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="x (Å)",
                yaxis_title="y (Å)",
                zaxis_title="z (Å)",
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False),
                aspectmode="data",
            ),
            height=700,
            showlegend=True,
            hovermode="closest",
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # =====================================================================
    # TAB 2: Metal Atoms Data
    # =====================================================================
    
    with tab2:
        st.subheader("Metal Atom Positions (Unit Cell)")
        
        if len(result.metal_atoms.fractional) > 0:
            df_metals = pd.DataFrame({
                'Index': range(len(result.metal_atoms.fractional)),
                'Sublattice': result.metal_atoms.sublattice_name,
                'Frac X': np.round(result.metal_atoms.fractional[:, 0], 6),
                'Frac Y': np.round(result.metal_atoms.fractional[:, 1], 6),
                'Frac Z': np.round(result.metal_atoms.fractional[:, 2], 6),
                'Cart X (Å)': np.round(result.metal_atoms.cartesian[:, 0], 6),
                'Cart Y (Å)': np.round(result.metal_atoms.cartesian[:, 1], 6),
                'Cart Z (Å)': np.round(result.metal_atoms.cartesian[:, 2], 6),
                'Radius (Å)': np.round(result.metal_atoms.radius, 6),
            })
            st.dataframe(df_metals, use_container_width=True, hide_index=True)
        else:
            st.info("No metal atoms in unit cell.")
    
    # =====================================================================
    # TAB 3: Intersections Data
    # =====================================================================
    
    with tab3:
        st.subheader("Intersection Site Positions (Unit Cell)")
        
        if len(result.intersections.fractional) > 0:
            df_intersections = pd.DataFrame({
                'Index': range(len(result.intersections.fractional)),
                'Multiplicity': result.intersections.multiplicity,
                'Frac X': np.round(result.intersections.fractional[:, 0], 6),
                'Frac Y': np.round(result.intersections.fractional[:, 1], 6),
                'Frac Z': np.round(result.intersections.fractional[:, 2], 6),
                'Cart X (Å)': np.round(result.intersections.cartesian[:, 0], 6),
                'Cart Y (Å)': np.round(result.intersections.cartesian[:, 1], 6),
                'Cart Z (Å)': np.round(result.intersections.cartesian[:, 2], 6),
            })
            st.dataframe(df_intersections, use_container_width=True, hide_index=True)
            
            # Multiplicity distribution
            mult_dist = pd.Series(result.intersections.multiplicity).value_counts().sort_index()
            st.write("**Multiplicity Distribution:**")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.bar_chart(mult_dist)
            with col_m2:
                st.write(mult_dist.to_frame("Count"))
        else:
            st.info("No intersections found for current parameters.")
    
    # =====================================================================
    # TAB 4: Export Data
    # =====================================================================
    
    with tab4:
        st.subheader("Export Structure Data")
        
        col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
        
        # JSON export
        with col_exp1:
            json_data = json.dumps(format_position_dict(result), indent=2)
            st.download_button(
                label="📥 JSON",
                data=json_data,
                file_name=f"structure_s{result.scale_s:.4f}_N{result.target_N}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Metal atoms CSV
        with col_exp2:
            csv_metals = format_metal_atoms_csv(result)
            st.download_button(
                label="📥 Metal Atoms CSV",
                data=csv_metals,
                file_name=f"metal_atoms_s{result.scale_s:.4f}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Intersections CSV
        with col_exp3:
            csv_intersections = format_intersections_csv(result)
            st.download_button(
                label="📥 Intersections CSV",
                data=csv_intersections,
                file_name=f"intersections_s{result.scale_s:.4f}_N{result.target_N}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # XYZ export
        with col_exp4:
            xyz_data = format_xyz(result, include_intersections=True)
            st.download_button(
                label="📥 XYZ File",
                data=xyz_data,
                file_name=f"structure_s{result.scale_s:.4f}_N{result.target_N}.xyz",
                mime="text/plain",
                use_container_width=True
            )
        
        st.divider()
        st.write("**Configuration Summary:**")
        
        config_summary = {
            "Lattice Type": st.session_state.lattice_params["bravais"],
            "a (Å)": st.session_state.lattice_params["a"],
            "b/a": st.session_state.lattice_params["b_ratio"],
            "c/a": st.session_state.lattice_params["c_ratio"],
            "α (°)": st.session_state.lattice_params["alpha"],
            "β (°)": st.session_state.lattice_params["beta"],
            "γ (°)": st.session_state.lattice_params["gamma"],
            "Scale Factor (s)": result.scale_s,
            "Target N": result.target_N,
            "Sublattices": [f"{s['name']}: {s['sphere_size']:.4f}×a" for s in st.session_state.sublattices]
        }
        
        st.json(config_summary)

st.divider()
st.markdown("*Built with ❤️ for crystallographic analysis*")
