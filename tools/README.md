# SHOCCS Reference Data Generator

This directory contains tools for generating reference data from the C++ SHOCCS implementation to validate the Python translation.

## Overview

The Python migration requires bit-exact validation against the C++ implementation. This is achieved by:

1. **C++ Reference Generator** (`generate_stencil_reference.cpp`): Extracts stencil coefficients from C++ and exports to JSON
2. **Python Loader** (`../python-migration/tests/cpp_reference.py`): Loads and provides convenient access to reference data
3. **Example Tests** (`../python-migration/tests/test_with_cpp_reference.py`): Demonstrates validation patterns

## Quick Start

### Option 1: Use Minimal Reference Data (Immediate)

A minimal reference data file with key test cases is already available:

```bash
# From python-migration/tests directory
python test_with_cpp_reference.py
```

This uses `stencil_reference_data_minimal.json` with reference values extracted from C++ test files.

### Option 2: Generate Complete Reference Data (Comprehensive)

Build and run the C++ generator to create a comprehensive dataset:

```bash
cd /home/user/shoccs/tools
./build_and_generate.sh
```

This will:
- Configure CMake
- Build the `generate_stencil_reference` executable
- Run it to generate `stencil_reference_data.json`
- Include 100+ test cases covering multiple stencils and parameters

**Requirements:**
- CMake 3.16+
- C++20 compiler
- SHOCCS dependencies (fmt, range-v3, sol2, etc.)

If dependencies are missing, you can:
1. Install them via package manager
2. Set `SHOCCS_TPL_DIR` environment variable
3. Or use the minimal reference data for initial development

## Reference Data Structure

### JSON Format

```json
{
  "metadata": {
    "description": "...",
    "generator": "generate_stencil_reference.cpp"
  },
  "E2_1": {
    "alpha": [1.0, 2.0, 3.0, -1.0],
    "interior": [
      {
        "name": "h=0.1",
        "h": 0.1,
        "coefficients": [-5.0, 0.0, 5.0]
      }
    ],
    "floating": [
      {
        "name": "h=1.0_psi=0.5_ray_outside",
        "h": 1.0,
        "psi": 0.5,
        "ray_outside": true,
        "bc_type": "floating",
        "r": 4,
        "t": 5,
        "coefficients": [...]
      }
    ],
    "dirichlet": [...]
  },
  "E2_2": {...},
  "polyE2_1": {...}
}
```

### Test Case Coverage

**E2_1** (alpha = [1, 2, 3, -1]):
- Interior: h ∈ {0.1, 0.5, 1.0, 2.0}
- Floating: h ∈ {0.5, 1.0, 2.0}, psi ∈ {0.1, 0.25, 0.5, 0.75, 0.9, 1.0}, both ray directions
- Dirichlet: h ∈ {0.5, 1.0, 2.0}, psi ∈ {0.0, 0.1, 0.5, 0.9}, both ray directions

**E2_2**:
- Interior: h ∈ {0.1, 0.5, 1.0, 2.0}
- Floating: Selected cases for h=1.0

**polyE2_1**:
- floating_alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
- dirichlet_alpha = [-0.1, -0.2, -0.3]
- interpolant_alpha = [0.15, 0.25, 0.35, 0.45]
- Interior and boundary test cases

## Using Reference Data in Python Tests

### Basic Pattern

```python
from cpp_reference import load_reference_data

# Load reference data
ref_data = load_reference_data()

# Test interior stencil
def test_interior():
    stencil = MyStencil()
    h = 0.1

    computed = stencil.interior_coefficients(h)
    reference = ref_data.get_interior("E2_1", h)

    np.testing.assert_allclose(computed, reference, rtol=1e-14)

# Test boundary stencil
def test_boundary():
    stencil = MyStencil()

    computed = stencil.nbs_coefficients(h=1.0, psi=0.5,
                                       bc_type="floating",
                                       ray_outside=True)
    reference, r, t = ref_data.get_boundary("E2_1", "floating",
                                            1.0, 0.5, True)

    np.testing.assert_allclose(computed, reference, rtol=1e-14)
```

### Parametric Testing

```python
import pytest

@pytest.mark.parametrize("h", [0.1, 0.5, 1.0, 2.0])
def test_interior_multiple_h(h, ref_data):
    stencil = MyStencil()
    computed = stencil.interior_coefficients(h)
    reference = ref_data.get_interior("E2_1", h)
    np.testing.assert_allclose(computed, reference, rtol=1e-14)

@pytest.mark.parametrize("psi", [0.1, 0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("ray_outside", [False, True])
def test_floating_boundary_sweep(psi, ray_outside, ref_data):
    stencil = MyStencil()
    h = 1.0
    computed = stencil.nbs_coefficients(h, psi, "floating", ray_outside)
    reference, r, t = ref_data.get_boundary("E2_1", "floating",
                                            h, psi, ray_outside)
    np.testing.assert_allclose(computed, reference, rtol=1e-14)
```

### Convenience Functions

```python
from cpp_reference import verify_interior_stencil, verify_boundary_stencil

# One-line verification
verify_interior_stencil(my_stencil, h=0.1, ref_data, "E2_1")
verify_boundary_stencil(my_stencil, h=1.0, psi=0.5, bc_type="floating",
                       ray_outside=True, ref_data, "E2_1")
```

## Files

- **generate_stencil_reference.cpp**: C++ program to generate complete reference data
- **CMakeLists.txt**: Build configuration for generator
- **build_and_generate.sh**: Automated build and run script
- **stencil_reference_data_minimal.json**: Minimal reference data (committed to repo)
- **stencil_reference_data.json**: Complete reference data (generated, not committed)
- **README.md**: This file

## Implementation Details

### C++ Generator

The generator:
1. Creates E2_1, E2_2, and polyE2_1 stencil objects
2. Calls `interior(h, c)` and `nbs(h, bc_type, psi, ray_outside, c, extra)` methods
3. Exports coefficients with full double precision (17 significant digits)
4. Organizes by stencil type, boundary condition, and parameters

### Python Loader

The `StencilReferenceData` class provides:
- `get_interior(stencil_type, h)`: Get interior coefficients
- `get_boundary(stencil_type, bc_type, h, psi, ray_outside)`: Get boundary coefficients
- `get_boundary_matrix(...)`: Get boundary coefficients as 2D array
- `get_alpha(stencil_type)`: Get alpha parameters
- `list_test_cases(stencil_type, test_type)`: Inspect available cases

### Hardcoded Fallback

For immediate testing without building C++, the loader includes hardcoded values from:
- `E2_1.t.cpp` (lines 47-96)
- `polyE2_1.t.cpp` (lines 59-66)

## Validation Strategy

1. **Unit Tests**: Verify individual stencil methods against reference data
2. **Parametric Sweep**: Test across full parameter range (h, psi, ray_outside)
3. **Edge Cases**: Focus on psi near 0 and 1, extreme h values
4. **Bit-Exact Comparison**: Use `rtol=1e-14` for double precision validation
5. **Regression Testing**: Continuous validation against reference as Python code evolves

## Tolerance Guidelines

- **Interior stencils**: `rtol=1e-14` (near machine precision for doubles)
- **Boundary stencils**: `rtol=1e-14` for simple cases, `rtol=1e-12` for complex polynomial cases
- **If using different algorithms**: Document expected tolerance based on algorithm differences

## Extending Reference Data

To add new test cases:

1. **Edit `generate_stencil_reference.cpp`:**
   - Add new parameter values to test vectors
   - Add new stencil types if needed

2. **Rebuild and regenerate:**
   ```bash
   ./build_and_generate.sh
   ```

3. **Update Python tests:**
   - Use new test cases in parametric tests
   - Add specific tests for critical edge cases

## Troubleshooting

### CMake Configuration Fails

```bash
# Check if dependencies are available
pkg-config --list-all | grep -E "fmt|range-v3|sol2"

# Set TPL directory if using custom installation
export SHOCCS_TPL_DIR=/path/to/tpl
./build_and_generate.sh
```

### Build Fails

- Ensure C++20 support: `g++ --version` (need 10+)
- Check missing libraries in CMake output
- Try building minimal set: `cmake -DBUILD_TESTING=OFF`

### Reference Data Mismatch

1. Verify Python implementation follows C++ algorithm
2. Check for floating-point ordering issues (associativity)
3. Inspect intermediate values if available
4. Use higher tolerance if algorithm intentionally differs

## Future Enhancements

- [ ] Add interpolant reference data
- [ ] Add E4_2 stencil coverage
- [ ] Include mesh-dependent test cases
- [ ] Add convergence study data
- [ ] Generate NPZ format for large datasets
- [ ] Automated validation reports
