# Stencils Module - Quick Start Guide

## Installation

The stencils module is part of the SHOCCS Python migration. No additional dependencies beyond NumPy and Numba (already required).

## Basic Usage

### 1. Simple Interior Stencils

```python
import numpy as np
from shoccs.stencils import centered_diff_1st_order2, apply_stencil_1d

# Create a grid
h = 0.1
x = np.arange(0, 1.0, h)

# Define a function: f(x) = x²
f = x**2

# Get the stencil for first derivative
stencil = centered_diff_1st_order2(h)

# Apply at an interior point (i=5)
df = apply_stencil_1d(f, stencil, 5)

# Compare with exact: f'(x) = 2x
df_exact = 2 * x[5]
print(f"Approximation: {df:.6f}")
print(f"Exact: {df_exact:.6f}")
print(f"Error: {abs(df - df_exact):.2e}")
```

### 2. All Available Interior Stencils

```python
from shoccs.stencils import (
    centered_diff_1st_order2,  # 3-point, 2nd order, 1st derivative
    centered_diff_2nd_order2,  # 3-point, 2nd order, 2nd derivative
    centered_diff_1st_order4,  # 5-point, 4th order, 1st derivative
    centered_diff_2nd_order4,  # 5-point, 4th order, 2nd derivative
)

h = 0.1

s1 = centered_diff_1st_order2(h)  # Returns: [-5.0, 0.0, 5.0]
s2 = centered_diff_2nd_order2(h)  # Returns: [100.0, -200.0, 100.0]
s3 = centered_diff_1st_order4(h)  # Returns: 5 coefficients
s4 = centered_diff_2nd_order4(h)  # Returns: 5 coefficients
```

### 3. E2-Poly Stencils

```python
import numpy as np
from shoccs.stencils import (
    make_e2_poly_stencil,
    e2_poly_interior,
    e2_poly_nbs_floating,
    e2_poly_nbs_dirichlet,
)

# Create configuration
fa = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # Floating BC params
da = np.array([0.5, 1.5, 2.5])                  # Dirichlet BC params
ia = np.array([0.1, 0.2, 0.3, 0.4])             # Interpolation params

config = make_e2_poly_stencil(fa, da, ia)

# Interior stencil (same as centered_diff_1st_order2)
h = 0.1
interior_stencil = e2_poly_interior(h)

# Near-boundary stencils
psi = 0.5  # Boundary stretching parameter

# Floating boundary (left side)
nbs_float = e2_poly_nbs_floating(h, psi, config.fa, right=False)
# Returns 12 coefficients (R * T = 3 * 4)

# Dirichlet boundary (right side)
nbs_dirich = e2_poly_nbs_dirichlet(h, psi, config.da, right=True)
# Returns 8 coefficients ((R-1) * T = 2 * 4)
```

## Complete Example: Computing a Derivative

```python
import numpy as np
from shoccs.stencils import centered_diff_1st_order2, apply_stencil_1d

def compute_derivative(x, f, h):
    """
    Compute df/dx using 2nd-order centered differences.

    Args:
        x: Grid points (1D array)
        f: Function values at grid points
        h: Grid spacing (uniform)

    Returns:
        df: Derivative at interior points
    """
    n = len(x)
    df = np.zeros(n)

    # Get stencil
    stencil = centered_diff_1st_order2(h)

    # Apply at interior points
    for i in range(1, n - 1):
        df[i] = apply_stencil_1d(f, stencil, i)

    # Boundary points would need special treatment
    df[0] = np.nan
    df[-1] = np.nan

    return df


# Example usage
h = 0.01
x = np.arange(0, 2*np.pi, h)
f = np.sin(x)

df_numerical = compute_derivative(x, f, h)
df_exact = np.cos(x)

# Check error at interior points
interior = slice(1, -1)
error = np.max(np.abs(df_numerical[interior] - df_exact[interior]))
print(f"Maximum error: {error:.2e}")
```

## Performance Notes

### Numba JIT Compilation

All stencil functions are JIT-compiled with Numba:

```python
# First call triggers compilation (~0.1-0.5s overhead)
stencil = centered_diff_1st_order2(0.1)  # Compiles + runs

# Subsequent calls are fast (< 1μs)
stencil = centered_diff_1st_order2(0.1)  # Just runs
stencil = centered_diff_1st_order2(0.2)  # Just runs (reuses compilation)
```

### Memory Efficiency

Stencils return small NumPy arrays:
- 2nd-order: 3 elements (24 bytes)
- 4th-order: 5 elements (40 bytes)
- E2-Poly NBS: 8-12 elements (64-96 bytes)

No dynamic allocations in hot loops.

## Polynomial Exactness

Stencils reproduce polynomials exactly (to machine precision):

```python
# 2nd-order stencils are exact for polynomials up to degree 2 (1st deriv) or 3 (2nd deriv)
# 4th-order stencils are exact for polynomials up to degree 4 (1st deriv) or 5 (2nd deriv)

h = 0.1
x = np.arange(0, 1, h)

# Test with f(x) = x³, f'(x) = 3x²
f = x**3
df_exact = 3 * x**2

stencil = centered_diff_1st_order2(h)
df = apply_stencil_1d(f, stencil, 5)

error = abs(df - df_exact[5])
print(f"Error: {error:.2e}")  # Should be < 1e-12
```

## Common Patterns

### Pattern 1: Comparing Stencil Orders

```python
# Compare 2nd-order vs 4th-order accuracy
h = 0.1
x = np.arange(0, 1, h)
f = np.sin(2 * np.pi * x)
df_exact = 2 * np.pi * np.cos(2 * np.pi * x)

# 2nd-order
s2 = centered_diff_1st_order2(h)
error_2 = max(abs(apply_stencil_1d(f, s2, i) - df_exact[i])
              for i in range(1, len(x)-1))

# 4th-order
s4 = centered_diff_1st_order4(h)
error_4 = max(abs(apply_stencil_1d(f, s4, i) - df_exact[i])
              for i in range(2, len(x)-2))

print(f"2nd-order error: {error_2:.2e}")
print(f"4th-order error: {error_4:.2e}")
print(f"Ratio: {error_2/error_4:.1f}x more accurate")
```

### Pattern 2: Grid Refinement Study

```python
def convergence_study(f, df, grid_sizes):
    """Study convergence with grid refinement."""
    errors = []

    for h in grid_sizes:
        x = np.arange(0, 1, h)
        y = f(x)
        dy_exact = df(x)

        stencil = centered_diff_1st_order2(h)

        max_error = 0
        for i in range(1, len(x)-1):
            dy = apply_stencil_1d(y, stencil, i)
            max_error = max(max_error, abs(dy - dy_exact[i]))

        errors.append(max_error)

    return errors

# Example
f = lambda x: np.sin(2*np.pi*x)
df = lambda x: 2*np.pi*np.cos(2*np.pi*x)

grid_sizes = [0.1, 0.05, 0.025, 0.0125]
errors = convergence_study(f, df, grid_sizes)

for h, err in zip(grid_sizes, errors):
    print(f"h = {h:.4f}, error = {err:.2e}")
```

## Testing

Run the comprehensive test suite:

```bash
cd /home/user/shoccs/python-migration
python -m pytest tests/test_stencils.py -v
```

**28 tests covering:**
- Stencil coefficient correctness
- Polynomial exactness
- Convergence rates
- Numba compilation
- E2-Poly functionality

## Examples

See complete working examples:

```bash
# Interactive demonstration
python examples/stencil_demo.py

# Shows:
# - Basic stencil usage
# - Convergence rates
# - Polynomial exactness
# - E2-Poly configuration
```

## API Reference

### Interior Stencils

```python
centered_diff_1st_order2(h: float) -> np.ndarray
    # 3-point centered difference for 1st derivative, O(h²)
    # Returns: [c_{-1}, c_0, c_1]

centered_diff_2nd_order2(h: float) -> np.ndarray
    # 3-point centered difference for 2nd derivative, O(h²)
    # Returns: [c_{-1}, c_0, c_1]

centered_diff_1st_order4(h: float) -> np.ndarray
    # 5-point centered difference for 1st derivative, O(h⁴)
    # Returns: [c_{-2}, c_{-1}, c_0, c_1, c_2]

centered_diff_2nd_order4(h: float) -> np.ndarray
    # 5-point centered difference for 2nd derivative, O(h⁴)
    # Returns: [c_{-2}, c_{-1}, c_0, c_1, c_2]

apply_stencil_1d(data: np.ndarray, stencil: np.ndarray, index: int) -> float
    # Apply stencil at given index
```

### E2-Poly Stencils

```python
class E2PolyStencil:
    # Configuration container
    fa: np.ndarray  # Floating BC params (6)
    da: np.ndarray  # Dirichlet BC params (3)
    ia: np.ndarray  # Interpolation params (4)

e2_poly_interior(h: float) -> np.ndarray
    # Interior stencil (3-point centered)

e2_poly_interp_interior(y: float) -> np.ndarray
    # Interior interpolation (linear)

e2_poly_interp_wall(i, y, psi, fa, ia, right) -> np.ndarray
    # Wall interpolation

e2_poly_nbs_floating(h, psi, fa, right) -> np.ndarray
    # Floating boundary stencil (12 coefficients)

e2_poly_nbs_dirichlet(h, psi, da, right) -> np.ndarray
    # Dirichlet boundary stencil (8 coefficients)

make_e2_poly_stencil(fa, da, ia) -> E2PolyStencil
    # Factory function
```

## Tips and Best Practices

1. **Choose appropriate stencil order:** 4th-order is more accurate but requires more points
2. **Verify polynomial exactness:** Test with polynomials before using on real data
3. **Watch boundary points:** Interior stencils don't work at boundaries
4. **Check grid spacing:** Smaller h = better accuracy (until roundoff dominates)
5. **Use Numba-compiled functions:** Already done for you, but avoid wrapping in non-JIT code

## Troubleshooting

**Q: Getting NaN at boundaries?**
A: Interior stencils can't be applied at boundaries. Use E2-Poly NBS or one-sided stencils.

**Q: Large errors with smooth function?**
A: Check grid spacing (h). May need refinement or higher-order stencil.

**Q: First call is slow?**
A: Normal! Numba compilation happens on first call (~0.1-0.5s). Subsequent calls are fast.

**Q: How to apply to 2D/3D data?**
A: Apply 1D stencils along each dimension separately. See `operators` module (Phase 3).

## Next Steps

- **Phase 3:** Differential operators (gradient, divergence, Laplacian)
- **Phase 4:** Time integration schemes
- **Phase 5:** Boundary condition handling

---

For more details, see:
- `PHASE2_STENCILS_SUMMARY.md` - Complete implementation summary
- `tests/test_stencils.py` - Comprehensive test suite
- `examples/stencil_demo.py` - Interactive demonstrations
