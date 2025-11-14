#!/usr/bin/env python3
"""
Extract reference data from C++ test files

This script parses C++ test files to extract coefficient arrays
and create reference data without needing to compile C++.
"""

import re
import json
from pathlib import Path


def extract_cpp_array(content: str, start_pattern: str) -> list:
    """
    Extract array values from C++ test code.

    Looks for patterns like:
        REQUIRE_THAT(c, Approx(T{val1, val2, val3}));
    """
    # Find the section with the pattern
    match = re.search(
        rf'{start_pattern}.*?Approx\(T\{{([^}}]+)\}}\)',
        content,
        re.DOTALL
    )

    if not match:
        return None

    # Extract the array values
    array_str = match.group(1)

    # Parse numbers (handle scientific notation, negatives, etc.)
    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', array_str)

    return [float(n) for n in numbers]


def extract_test_parameters(content: str, test_start: str) -> dict:
    """Extract test parameters like h, psi, etc."""
    # Find the test block
    match = re.search(
        rf'{test_start}.*?REQUIRE',
        content,
        re.DOTALL
    )

    if not match:
        return {}

    block = match.group(0)

    params = {}

    # Extract h
    h_match = re.search(r'st\.nbs\(([^,]+),', block)
    if h_match:
        h_str = h_match.group(1).strip()
        try:
            params['h'] = float(h_str)
        except ValueError:
            pass

    # Extract psi
    psi_match = re.search(r'real psi\s*=\s*([^;]+);', block)
    if psi_match:
        try:
            params['psi'] = float(psi_match.group(1).strip())
        except ValueError:
            pass

    # Extract ray_outside
    if 'true' in block:
        params['ray_outside'] = True
    elif 'false' in block:
        params['ray_outside'] = False

    # Extract r, t
    r_match = re.search(r'r\s*==\s*(\d+)', block)
    if r_match:
        params['r'] = int(r_match.group(1))

    t_match = re.search(r't\s*==\s*(\d+)', block)
    if t_match:
        params['t'] = int(t_match.group(1))

    return params


def main():
    project_root = Path(__file__).parent.parent
    stencils_dir = project_root / "src" / "stencils"

    print("Extracting reference data from C++ test files...")
    print("=" * 60)

    # E2_1 test file
    e2_1_test = stencils_dir / "E2_1.t.cpp"
    if e2_1_test.exists():
        with open(e2_1_test, 'r') as f:
            content = f.read()

        print("\nE2_1 Test Cases:")
        print("-" * 60)

        # Extract floating BC test
        print("\n1. Floating BC (h=2, psi=1.0, ray_outside=false):")
        floating_coeffs = extract_cpp_array(content, r'st\.nbs\(2, bcs::Floating')
        if floating_coeffs:
            print(f"   Found {len(floating_coeffs)} coefficients")
            print(f"   First few: {floating_coeffs[:5]}")
            print(f"   Last few: {floating_coeffs[-5:]}")
        else:
            print("   Not found")

        # Extract Dirichlet BC test
        print("\n2. Dirichlet BC (h=0.5, psi=0.0, ray_outside=true):")
        dirichlet_coeffs = extract_cpp_array(content, r'st\.nbs\(0\.5, bcs::Dirichlet')
        if dirichlet_coeffs:
            print(f"   Found {len(dirichlet_coeffs)} coefficients")
            print(f"   First few: {dirichlet_coeffs[:5]}")
            print(f"   Last few: {dirichlet_coeffs[-5:]}")
        else:
            print("   Not found")

    # polyE2_1 test file
    poly_test = stencils_dir / "polyE2_1.t.cpp"
    if poly_test.exists():
        with open(poly_test, 'r') as f:
            content = f.read()

        print("\n" + "=" * 60)
        print("\npolyE2_1 Test Cases:")
        print("-" * 60)

        # Extract dirichlet test
        print("\n1. Dirichlet BC (h=1.0, psi=0.001, ray_outside=false):")

        # Find exact values
        match = re.search(
            r'T exact\{([^}]+)\};',
            content
        )
        if match:
            array_str = match.group(1)
            numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', array_str)
            coeffs = [float(n) for n in numbers]
            print(f"   Found {len(coeffs)} coefficients")
            print(f"   Values: {coeffs}")
        else:
            print("   Not found")

    # E2_2 test file
    e2_2_test = stencils_dir / "E2_2.t.cpp"
    if e2_2_test.exists():
        with open(e2_2_test, 'r') as f:
            content = f.read()

        print("\n" + "=" * 60)
        print("\nE2_2 Test Cases:")
        print("-" * 60)
        print("(Contains multiple test cases - review manually for extraction)")

    print("\n" + "=" * 60)
    print("\nManual Review Recommended:")
    print("- Check test files for additional test cases")
    print("- Verify parameter values (h, psi, ray_outside)")
    print("- Note boundary condition types (Floating, Dirichlet)")
    print("- Verify r and t values for array reshaping")

    print("\n" + "=" * 60)
    print("\nNext Steps:")
    print("1. Review extracted values above")
    print("2. Add to stencil_reference_data_minimal.json")
    print("3. Or run ./build_and_generate.sh for complete dataset")
    print("=" * 60)


if __name__ == "__main__":
    main()
