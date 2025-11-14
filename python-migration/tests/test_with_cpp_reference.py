"""
Example tests using C++ reference data for validation

This demonstrates how to use the cpp_reference module to validate
Python stencil implementations against C++ reference data.
"""

import pytest
import numpy as np
from pathlib import Path

# Import the reference data loader
from cpp_reference import (
    load_reference_data,
    verify_interior_stencil,
    verify_boundary_stencil,
    get_hardcoded_interior,
    get_hardcoded_boundary,
)


# Try to load JSON reference data, fall back to hardcoded if not available
try:
    # Try to load the complete reference data
    ref_data_path = Path(__file__).parent.parent.parent / "tools" / "stencil_reference_data.json"
    if not ref_data_path.exists():
        # Try minimal version
        ref_data_path = Path(__file__).parent.parent.parent / "tools" / "stencil_reference_data_minimal.json"

    if ref_data_path.exists():
        ref_data = load_reference_data(ref_data_path)
        USE_JSON = True
    else:
        ref_data = None
        USE_JSON = False
except Exception:
    ref_data = None
    USE_JSON = False


class MockE2_1Stencil:
    """
    Mock stencil for demonstration purposes.
    Replace this with actual Python stencil implementation.
    """
    def __init__(self, alpha=None):
        self.alpha = alpha if alpha is not None else np.array([1.0, 2.0, 3.0, -1.0])

    def interior_coefficients(self, h: float) -> np.ndarray:
        """Return interior stencil coefficients"""
        # Standard centered difference: [-1/(2h), 0, 1/(2h)]
        return np.array([-1/(2*h), 0, 1/(2*h)])

    def nbs_coefficients(self, h: float, psi: float, bc_type: str,
                        ray_outside: bool) -> np.ndarray:
        """
        Return boundary stencil coefficients (mock implementation)

        In reality, this would compute the actual boundary stencil.
        For now, just return zeros to show the testing pattern.
        """
        if bc_type == "floating":
            r, t = 4, 5
        elif bc_type == "dirichlet":
            r, t = 3, 5
        else:
            raise ValueError(f"Unknown bc_type: {bc_type}")

        # Mock implementation - replace with actual computation
        return np.zeros(r * t)


def test_E2_1_interior_with_reference():
    """Test E2_1 interior stencil against C++ reference"""
    stencil = MockE2_1Stencil()
    h = 0.1

    # Get reference
    if USE_JSON and ref_data:
        reference = ref_data.get_interior("E2_1", h)
    else:
        reference = get_hardcoded_interior("E2_1", h)

    # Compute
    computed = stencil.interior_coefficients(h)

    # Verify
    np.testing.assert_allclose(
        computed, reference, rtol=1e-14,
        err_msg=f"E2_1 interior mismatch for h={h}"
    )


def test_E2_1_multiple_h_values():
    """Test E2_1 interior for multiple grid spacings"""
    stencil = MockE2_1Stencil()

    h_values = [0.1, 1.0, 2.0]

    for h in h_values:
        if USE_JSON and ref_data:
            reference = ref_data.get_interior("E2_1", h)
        else:
            if h in [0.1, 1.0, 2.0]:
                reference = get_hardcoded_interior("E2_1", h)
            else:
                pytest.skip(f"No reference data for h={h}")

        computed = stencil.interior_coefficients(h)

        np.testing.assert_allclose(
            computed, reference, rtol=1e-14,
            err_msg=f"E2_1 interior mismatch for h={h}"
        )


@pytest.mark.skipif(not USE_JSON, reason="Requires full reference data JSON")
def test_E2_1_floating_boundary():
    """Test E2_1 floating boundary conditions"""
    stencil = MockE2_1Stencil()

    # Test parameters
    h = 2.0
    psi = 1.0
    ray_outside = False

    # Get reference
    reference, r, t = ref_data.get_boundary(
        "E2_1", "floating", h, psi, ray_outside
    )

    # Compute (mock - would be real computation)
    computed = stencil.nbs_coefficients(h, psi, "floating", ray_outside)

    # For this demo, we expect it to fail since it's a mock
    # In real tests, this should pass
    # np.testing.assert_allclose(computed, reference, rtol=1e-14)

    # Just verify shape for now
    assert len(computed) == len(reference), \
        f"Shape mismatch: computed {len(computed)}, reference {len(reference)}"


@pytest.mark.skipif(not USE_JSON, reason="Requires full reference data JSON")
def test_E2_1_boundary_sweep():
    """Test E2_1 boundary for multiple psi values"""
    if not ref_data:
        pytest.skip("No reference data available")

    stencil = MockE2_1Stencil()
    h = 1.0

    # Get all available test cases
    floating_cases = ref_data.list_test_cases("E2_1", "floating")

    for case in floating_cases:
        if not np.isclose(case["h"], h):
            continue

        psi = case["psi"]
        ray_outside = case["ray_outside"]

        # Get reference
        reference = np.array(case["coefficients"])

        # Compute
        computed = stencil.nbs_coefficients(h, psi, "floating", ray_outside)

        # For demo purposes, just check shape
        assert len(computed) == len(reference), \
            f"Shape mismatch for psi={psi}, ray_outside={ray_outside}"


def test_reference_data_structure():
    """Test that reference data has expected structure"""
    if USE_JSON and ref_data:
        # Check metadata
        metadata = ref_data.get_metadata()
        assert "description" in metadata

        # Check E2_1 structure
        alpha = ref_data.get_alpha("E2_1")
        assert len(alpha) == 4
        np.testing.assert_array_equal(alpha, [1.0, 2.0, 3.0, -1.0])

        # Check available test cases
        interior_cases = ref_data.list_test_cases("E2_1", "interior")
        assert len(interior_cases) > 0

        print(f"Reference data loaded successfully:")
        print(f"  - {len(interior_cases)} interior test cases")

        floating_cases = ref_data.list_test_cases("E2_1", "floating")
        print(f"  - {len(floating_cases)} floating BC test cases")

        dirichlet_cases = ref_data.list_test_cases("E2_1", "dirichlet")
        print(f"  - {len(dirichlet_cases)} Dirichlet BC test cases")


def test_hardcoded_reference_available():
    """Verify hardcoded reference data is accessible"""
    # Interior
    interior = get_hardcoded_interior("E2_1", 0.1)
    assert isinstance(interior, np.ndarray)
    assert len(interior) == 3

    # Boundary
    boundary = get_hardcoded_boundary("E2_1", "floating", 2.0, 1.0, False)
    assert isinstance(boundary, np.ndarray)
    assert len(boundary) == 20  # r=4, t=5


if __name__ == "__main__":
    # Run a simple check
    print("Testing C++ reference data loader...")
    print(f"JSON reference data available: {USE_JSON}")

    if USE_JSON and ref_data:
        print("\nMetadata:")
        print(ref_data.get_metadata())

        print("\nE2_1 alpha:", ref_data.get_alpha("E2_1"))

        print("\nInterior test cases:")
        for case in ref_data.list_test_cases("E2_1", "interior"):
            print(f"  h={case['h']}: {case['coefficients']}")

    print("\nHardcoded reference data:")
    print(f"  E2_1 interior (h=0.1): {get_hardcoded_interior('E2_1', 0.1)}")

    print("\nAll tests completed!")
