# Phase 3 Technical Notes - From Stencil Validation

**For:** Operator Construction Team
**From:** Numerical Methods Expert
**Date:** 2025-11-14

---

## Key Technical Insights for Operator Construction

### 1. Stencil Array Structure

The stencils return **flat arrays** with specific sizes:

```python
# Interior stencils (3-point centered difference)
interior = e2_poly_interior(h)
# Shape: (3,) - coefficients for [i-1, i, i+1]

# Dirichlet Near-Boundary Stencil (NBS)
dirichlet = e2_poly_nbs_dirichlet(h, psi, da, right=False)
# Shape: (8,) - flat array representing (R-1)×T = 2×4 matrix
# Reshape to (2, 4) for row-based application

# Floating Near-Boundary Stencil (NBS)
floating = e2_poly_nbs_floating(h, psi, fa, right=False)
# Shape: (12,) - flat array representing R×T = 3×4 matrix
# Reshape to (3, 4) for row-based application
```

**Important for operator assembly:** You'll need to reshape NBS arrays when constructing matrices.

### 2. Boundary Orientation Convention

The `right` parameter controls boundary orientation:

```python
# Left boundary (ray pointing right, into domain)
left_stencil = e2_poly_nbs_dirichlet(h, psi, da, right=False)

# Right boundary (ray pointing left, into domain)
right_stencil = e2_poly_nbs_dirichlet(h, psi, da, right=True)
```

**Relationship:** `right = -reverse(left)`
- This means right boundary coefficients are negated and reversed
- When assembling operators, account for this in your indexing

### 3. Parameter Arrays (Alpha Values)

E2-Poly stencils require parameter arrays initialized from reference data:

```python
from cpp_reference import StencilReferenceData

ref_data = StencilReferenceData()

# Get parameters for boundary conditions
fa = ref_data.get_alpha("polyE2_1", "floating_alpha")    # 6 elements
da = ref_data.get_alpha("polyE2_1", "dirichlet_alpha")   # 3 elements
ia = ref_data.get_alpha("polyE2_1", "interpolant_alpha") # 4 elements

# Create stencil configuration
params = make_e2_poly_stencil(fa, da, ia)
```

**For Phase 3:** You'll need to load these once during operator initialization.

### 4. Grid Spacing (h) Scaling

Stencils scale predictably with h:

- **1st derivatives:** Coefficients ∝ 1/h
- **2nd derivatives:** Coefficients ∝ 1/h²

**Implication:** When assembling operators on non-uniform grids or with adaptive refinement, ensure each stencil uses the local grid spacing.

### 5. Numerical Stability Guarantees

Validation shows excellent stability:

| Parameter | Range Tested | Status |
|-----------|-------------|---------|
| h | 1e-10 to 1e6 | ✅ Stable |
| ψ | 1e-10 to 1.0 | ✅ Stable |
| Condition number | < 6 for all cases | ✅ Well-conditioned |

**Implication:** No special numerical handling required in operator construction. Direct assembly is safe.

### 6. Stencil Application Pattern

For operator matrices, the stencil application follows this pattern:

```python
# Interior point i (using 3-point stencil)
row[i] = [... 0, c[0], c[1], c[2], 0, ...]
#              at i-1   i    i+1

# Near-boundary (using 2×4 Dirichlet stencil)
# Row 0:
row[0] = [c[0], c[1], c[2], c[3], 0, 0, ...]
# Row 1:
row[1] = [c[4], c[5], c[6], c[7], 0, 0, ...]
```

### 7. Symmetry Properties for Optimization

Interior stencils have exact symmetries you can exploit:

```python
# 1st derivative: anti-symmetric
assert c[0] == -c[2]  # Exact to machine precision
assert c[1] == 0.0    # Exact

# 2nd derivative: symmetric
assert c[0] == c[2]   # Exact to machine precision
assert c[1] == -2 * c[0]  # Exact relationship
```

**Optimization opportunity:** In operator construction, you could:
- Store only c[2] for 1st derivative (c[0] = -c[2], c[1] = 0)
- Store only c[0] for 2nd derivative (c[1] = -2*c[0], c[2] = c[0])

This would halve storage for interior stencils.

### 8. Performance Characteristics

All stencil functions are Numba JIT-compiled:

```python
from numba import njit

@njit
def e2_poly_interior(h: float) -> np.ndarray:
    # ... compiled to machine code on first call
```

**Implications:**
- First call has compilation overhead (~100ms)
- Subsequent calls are very fast (~microseconds)
- In operator construction, pre-compute all stencils once
- Store results if operator will be reused

**Recommended pattern:**
```python
# Pre-compute all stencils needed for operator
stencils = {
    'interior': e2_poly_interior(h),
    'boundary_left': e2_poly_nbs_dirichlet(h, psi, da, right=False),
    'boundary_right': e2_poly_nbs_dirichlet(h, psi, da, right=True),
}

# Then use stored stencils in matrix assembly loop
for i in range(n):
    if interior_point(i):
        matrix[i, i-1:i+2] = stencils['interior']
    elif left_boundary(i):
        # ... use stencils['boundary_left']
```

### 9. Zero Coefficients

Some stencil coefficients are exactly zero:

```python
# 1st derivative centered difference
stencil = centered_diff_1st_order2(h)
assert stencil[1] == 0.0  # Exact
```

**For sparse matrices:** These zeros can be omitted in CSR/CSC format.

### 10. Testing Recommendations for Operators

Based on stencil validation, your Phase 3 tests should verify:

1. **Polynomial reproduction:**
   ```python
   # Operator should reproduce stencil behavior
   u = polynomial_function(x)
   Du_stencil = apply_stencils(u)
   Du_operator = operator @ u
   assert np.allclose(Du_stencil, Du_operator, rtol=1e-12)
   ```

2. **Boundary consistency:**
   ```python
   # Left and right boundaries should match
   assert operator[0, :] = correct_left_boundary_stencil
   assert operator[-1, :] = correct_right_boundary_stencil
   ```

3. **Matrix properties:**
   ```python
   # For 1st derivative operator on periodic domain
   assert np.sum(operator, axis=1).max() < 1e-14  # Row sums ~0
   ```

### 11. Common Pitfalls to Avoid

1. **Forgetting h scaling:**
   ```python
   # WRONG: Using stencil directly
   stencil = e2_poly_interior(h=1.0)  # Computed at h=1
   operator[i, i-1:i+2] = stencil     # But actual h is 0.1!

   # RIGHT: Use actual grid spacing
   stencil = e2_poly_interior(h=actual_h)
   operator[i, i-1:i+2] = stencil
   ```

2. **Boundary orientation:**
   ```python
   # WRONG: Same stencil for both boundaries
   left = e2_poly_nbs_dirichlet(h, psi, da, right=False)
   right = left  # This is incorrect!

   # RIGHT: Specify boundary orientation
   left = e2_poly_nbs_dirichlet(h, psi, da, right=False)
   right = e2_poly_nbs_dirichlet(h, psi, da, right=True)
   ```

3. **Assuming floating point zeros:**
   ```python
   # WRONG: Exact comparison
   if stencil[i] == 0.0:  # May miss near-zeros

   # RIGHT: Use tolerance
   if abs(stencil[i]) < 1e-14:  # Safe for sparse matrix construction
   ```

### 12. Interpolation Stencils

E2-Poly also provides interpolation:

```python
# Interior interpolation (between grid points)
interp = e2_poly_interp_interior(y)  # y in [-1, 1]
# Returns 2 coefficients for linear interpolation

# Wall interpolation (near boundaries)
interp = e2_poly_interp_wall(i, y, psi, fa, ia, right)
# Returns 4 coefficients
```

**Use cases in Phase 3:**
- Evaluating solution at off-grid points
- Enforcing boundary conditions
- Cut-cell interpolation

### 13. Memory Layout for Efficiency

For best cache performance when building operators:

```python
# GOOD: Pre-allocate full matrix
operator = np.zeros((n, n))

# BETTER: Use sparse matrix from start
from scipy.sparse import lil_matrix
operator = lil_matrix((n, n))

# BEST: Build in COO format, convert to CSR
rows, cols, data = [], [], []
for i in range(n):
    stencil = get_stencil(i)
    indices = get_indices(i)
    rows.extend([i] * len(indices))
    cols.extend(indices)
    data.extend(stencil)

operator = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
operator = operator.tocsr()  # Convert for fast matrix-vector products
```

### 14. Parameter Validation

Stencils are validated for:
- h > 0 (no explicit check, but h=0 would cause division by zero)
- 0 < ψ ≤ 1 (validated to work correctly)

**For Phase 3:** Add parameter validation in operator construction:
```python
def build_operator(h, psi, ...):
    assert h > 0, "Grid spacing must be positive"
    assert 0 < psi <= 1, "Psi must be in (0, 1]"
    # ... proceed with operator assembly
```

---

## Quick Reference: Function Signatures

```python
# Interior stencils
def centered_diff_1st_order2(h: float) -> np.ndarray  # Returns (3,)
def centered_diff_2nd_order2(h: float) -> np.ndarray  # Returns (3,)
def centered_diff_1st_order4(h: float) -> np.ndarray  # Returns (5,)
def centered_diff_2nd_order4(h: float) -> np.ndarray  # Returns (5,)

# E2-Poly stencils
def e2_poly_interior(h: float) -> np.ndarray  # Returns (3,)

def e2_poly_nbs_dirichlet(
    h: float,
    psi: float,
    da: np.ndarray,  # (3,)
    right: bool
) -> np.ndarray  # Returns (8,) = (R-1)*T

def e2_poly_nbs_floating(
    h: float,
    psi: float,
    fa: np.ndarray,  # (6,)
    right: bool
) -> np.ndarray  # Returns (12,) = R*T

# Interpolation
def e2_poly_interp_interior(y: float) -> np.ndarray  # Returns (2,)

def e2_poly_interp_wall(
    i: int,          # Row index (0 or 1)
    y: float,        # Offset
    psi: float,
    fa: np.ndarray,  # (6,)
    ia: np.ndarray,  # (4,)
    right: bool
) -> np.ndarray  # Returns (4,)
```

---

## Questions to Consider for Phase 3

1. **Matrix Format:** Will you use dense or sparse matrices? (Sparse recommended for large grids)

2. **Boundary Conditions:** How will you handle multiple BC types on different boundaries?

3. **Grid Handling:** Will you support:
   - Uniform grids only?
   - Stretched grids (variable h)?
   - Multi-block grids?

4. **Performance:** Will you:
   - Pre-compute and cache stencils?
   - Build operators on-the-fly?
   - Support operator reuse?

5. **Testing:** Will you:
   - Test against analytical solutions?
   - Compare with C++ operator construction?
   - Validate conservation properties?

---

## Contact

If you encounter any numerical issues during Phase 3 implementation, re-run the validation suite:

```bash
cd /home/user/shoccs/python-migration
python3 -m pytest tests/test_stencils.py tests/test_cpp_reference.py tests/test_convergence.py -v
python3 tests/analyze_stencils.py
```

All tests should pass. Any failures indicate a regression that needs investigation.

---

**Good luck with Phase 3!** The stencils are solid - build with confidence.
