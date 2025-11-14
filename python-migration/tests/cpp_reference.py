"""
C++ Reference Data Loader

Provides convenient access to reference data generated from the C++ SHOCCS implementation
for validating the Python translation.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class StencilReferenceData:
    """Container for stencil reference data from C++"""

    def __init__(self, json_path: Optional[Path] = None):
        """
        Load reference data from JSON file.

        Args:
            json_path: Path to JSON reference file. If None, uses default location.
        """
        if json_path is None:
            # Try complete data first, fall back to minimal
            tools_dir = Path(__file__).parent.parent.parent / "tools"
            json_path = tools_dir / "stencil_reference_data.json"

            if not json_path.exists():
                json_path = tools_dir / "stencil_reference_data_minimal.json"

            if not json_path.exists():
                raise FileNotFoundError(
                    f"No reference data found. Tried:\n"
                    f"  - {tools_dir / 'stencil_reference_data.json'}\n"
                    f"  - {tools_dir / 'stencil_reference_data_minimal.json'}\n"
                    f"Run build_and_generate.sh to create complete data."
                )

        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def get_interior(self, stencil_type: str, h: float) -> np.ndarray:
        """
        Get interior stencil coefficients for given h.

        Args:
            stencil_type: One of "E2_1", "E2_2", "polyE2_1"
            h: Grid spacing

        Returns:
            numpy array of coefficients
        """
        interior_cases = self.data[stencil_type]["interior"]
        for case in interior_cases:
            if np.isclose(case["h"], h):
                return np.array(case["coefficients"])
        raise ValueError(f"No interior reference data for {stencil_type} with h={h}")

    def get_boundary(self,
                     stencil_type: str,
                     bc_type: str,
                     h: float,
                     psi: float,
                     ray_outside: bool) -> Tuple[np.ndarray, int, int]:
        """
        Get boundary stencil coefficients.

        Args:
            stencil_type: One of "E2_1", "E2_2", "polyE2_1"
            bc_type: "floating" or "dirichlet"
            h: Grid spacing
            psi: Boundary position parameter (0 < psi <= 1)
            ray_outside: True if ray points outside domain

        Returns:
            Tuple of (coefficients array, r, t) where:
            - coefficients: flat array of shape (r*t,)
            - r: number of rows
            - t: number of columns (stencil width)
        """
        bc_cases = self.data[stencil_type][bc_type]

        for case in bc_cases:
            if (np.isclose(case["h"], h) and
                np.isclose(case["psi"], psi) and
                case["ray_outside"] == ray_outside):
                return (
                    np.array(case["coefficients"]),
                    case["r"],
                    case["t"]
                )

        raise ValueError(
            f"No {bc_type} reference data for {stencil_type} with "
            f"h={h}, psi={psi}, ray_outside={ray_outside}"
        )

    def get_boundary_matrix(self,
                           stencil_type: str,
                           bc_type: str,
                           h: float,
                           psi: float,
                           ray_outside: bool) -> np.ndarray:
        """
        Get boundary stencil coefficients as a 2D matrix.

        Same as get_boundary but returns coefficients reshaped to (r, t).
        """
        coeffs, r, t = self.get_boundary(stencil_type, bc_type, h, psi, ray_outside)
        return coeffs.reshape(r, t)

    def get_alpha(self, stencil_type: str, alpha_type: str = "alpha") -> np.ndarray:
        """
        Get alpha parameters for a stencil.

        Args:
            stencil_type: Stencil name
            alpha_type: "alpha", "floating_alpha", "dirichlet_alpha", "interpolant_alpha"

        Returns:
            numpy array of alpha values
        """
        return np.array(self.data[stencil_type].get(alpha_type, []))

    def list_test_cases(self, stencil_type: str, test_type: str) -> List[Dict]:
        """
        List all available test cases for inspection.

        Args:
            stencil_type: Stencil name
            test_type: "interior", "floating", "dirichlet"

        Returns:
            List of test case dictionaries
        """
        return self.data[stencil_type][test_type]

    def get_metadata(self) -> Dict:
        """Get metadata about the reference data file."""
        return self.data.get("metadata", {})


# Convenience functions for common operations

def load_reference_data(json_path: Optional[Path] = None) -> StencilReferenceData:
    """Load reference data (convenience function)."""
    return StencilReferenceData(json_path)


def verify_interior_stencil(stencil, h: float, ref_data: StencilReferenceData,
                            stencil_type: str = "E2_1", rtol: float = 1e-14):
    """
    Verify interior stencil against reference data.

    Args:
        stencil: Python stencil object with interior_coefficients method
        h: Grid spacing
        ref_data: Reference data object
        stencil_type: Type of stencil
        rtol: Relative tolerance for comparison

    Returns:
        True if match, raises AssertionError otherwise
    """
    computed = stencil.interior_coefficients(h)
    reference = ref_data.get_interior(stencil_type, h)

    np.testing.assert_allclose(
        computed, reference, rtol=rtol,
        err_msg=f"Interior coefficients mismatch for h={h}"
    )
    return True


def verify_boundary_stencil(stencil, h: float, psi: float, bc_type: str,
                            ray_outside: bool, ref_data: StencilReferenceData,
                            stencil_type: str = "E2_1", rtol: float = 1e-14):
    """
    Verify boundary stencil against reference data.

    Args:
        stencil: Python stencil object with nbs_coefficients method
        h: Grid spacing
        psi: Boundary position
        bc_type: "floating" or "dirichlet"
        ray_outside: Boundary direction
        ref_data: Reference data object
        stencil_type: Type of stencil
        rtol: Relative tolerance for comparison

    Returns:
        True if match, raises AssertionError otherwise
    """
    computed = stencil.nbs_coefficients(h, psi, bc_type, ray_outside)
    reference, r, t = ref_data.get_boundary(stencil_type, bc_type, h, psi, ray_outside)

    # Flatten computed if it's 2D
    if computed.ndim == 2:
        computed = computed.flatten()

    np.testing.assert_allclose(
        computed, reference, rtol=rtol,
        err_msg=f"Boundary coefficients mismatch for h={h}, psi={psi}, bc_type={bc_type}"
    )
    return True


# Hardcoded reference data from C++ test files for immediate use
# This can be used while the full JSON file is being generated

HARDCODED_REFERENCE = {
    "E2_1": {
        "alpha": [1.0, 2.0, 3.0, -1.0],
        "interior": {
            0.1: np.array([-5.0, 0.0, 5.0]),
            1.0: np.array([-0.5, 0.0, 0.5]),
            2.0: np.array([-0.25, 0.0, 0.25]),
        },
        "floating": {
            # From E2_1.t.cpp, line 47-68
            (2.0, 1.0, False): np.array([
                3, -5, 0.5, 1.5, 0,
                -0.5, 0, 1, -0.5, 0,
                0.02631578947368421, -0.32894736842105265, 0.07894736842105263,
                0.2236842105263158, 0,
                0, 0, -0.25, 0, 0.25
            ]),
        },
        "dirichlet": {
            # From E2_1.t.cpp, line 80-96
            (0.5, 0.0, True): np.array([
                -0.8947368421052632, -0.3157894736842105, 1.3157894736842106,
                -0.10526315789473684, 0., 2., -4., 0., 2., 0., -6., -2., 20., 0., -12.
            ]),
        }
    },
    "polyE2_1": {
        "dirichlet": {
            # From polyE2_1.t.cpp, line 59-66
            (1.0, 0.001, False): np.array([
                -0.4395604395604396, -0.4166369491239419, 0.7128343378083233, 0.14336305087605816,
                -0.4295704295704296, -0.435, 0.7295704295704296, 0.135
            ]),
        }
    }
}


def get_hardcoded_interior(stencil_type: str, h: float) -> np.ndarray:
    """Get interior coefficients from hardcoded test data."""
    return HARDCODED_REFERENCE[stencil_type]["interior"][h]


def get_hardcoded_boundary(stencil_type: str, bc_type: str,
                          h: float, psi: float, ray_outside: bool) -> np.ndarray:
    """Get boundary coefficients from hardcoded test data."""
    return HARDCODED_REFERENCE[stencil_type][bc_type][(h, psi, ray_outside)]
