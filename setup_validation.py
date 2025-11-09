#!/usr/bin/env python3
"""
Setup and validation script for integrated crystallography app
Verifies all dependencies and module imports
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Verify Python 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        print(f"   Current: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]}")

def check_package(package_name, import_name=None):
    """Check if package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - NOT INSTALLED")
        return False

def check_file(filepath, description):
    """Check if required file exists"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def main():
    print("\n" + "="*60)
    print("Crystallography Analysis Suite - Setup Validation")
    print("="*60 + "\n")
    
    # Check Python
    print("Python Environment:")
    check_python_version()
    
    # Check core dependencies
    print("\nCore Dependencies:")
    packages = [
        ("streamlit", "streamlit"),
        ("NumPy", "numpy"),
        ("Plotly", "plotly"),
        ("SciPy (optional but recommended)", "scipy"),
        ("Matplotlib (optional)", "matplotlib"),
    ]
    
    all_core_ok = True
    optional_missing = []
    
    for pkg_name, import_name in packages:
        ok = check_package(pkg_name, import_name)
        if not ok and "optional" in pkg_name.lower():
            optional_missing.append(pkg_name)
        elif not ok:
            all_core_ok = False
    
    # Check module files
    print("\nModule Files:")
    all_modules_ok = True
    modules = [
        ("interstitial_engine.py", "Coordination engine"),
        ("position_calculator.py", "Position calculator"),
        ("integrated_streamlit_app.py", "Main application"),
    ]
    
    for module_file, description in modules:
        ok = check_file(module_file, description)
        if not ok:
            all_modules_ok = False
    
    # Try importing modules
    print("\nModule Imports:")
    modules_import_ok = True
    
    try:
        from interstitial_engine import LatticeParams, Sublattice, max_multiplicity_for_scale
        print("✅ interstitial_engine imports")
    except Exception as e:
        print(f"❌ interstitial_engine import failed: {e}")
        modules_import_ok = False
    
    try:
        from position_calculator import generate_metal_positions, generate_intersection_positions
        print("✅ position_calculator imports")
    except Exception as e:
        print(f"❌ position_calculator import failed: {e}")
        modules_import_ok = False
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    if all_core_ok and all_modules_ok and modules_import_ok:
        print("✅ All checks passed! Ready to run:")
        print("   streamlit run integrated_streamlit_app.py")
        if optional_missing:
            print(f"\n⚠️  Optional packages not installed: {', '.join(optional_missing)}")
            print("   Performance may be reduced without SciPy")
    else:
        print("❌ Setup incomplete. Issues found:")
        if not all_core_ok:
            print("   - Core dependencies missing")
        if not all_modules_ok:
            print("   - Module files missing")
        if not modules_import_ok:
            print("   - Module imports failed")
        sys.exit(1)
    
    print()

if __name__ == "__main__":
    main()
