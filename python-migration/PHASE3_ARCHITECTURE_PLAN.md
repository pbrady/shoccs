# Phase 3 Architecture Plan: Differential Operators

**Author:** Software Architect
**Date:** 2025-11-14
**Status:** DRAFT - Ready for Implementation

---

## Executive Summary

Phase 3 builds discrete differential operators (derivative, gradient, Laplacian) from the validated stencils in Phase 2. The Python design simplifies the C++ template-heavy approach while maintaining mathematical correctness.

**Key Principle:** Use SciPy sparse matrices + simple dataclasses, avoiding the C++ builder pattern and template complexity.

---

## C++ Analysis Summary

### Current C++ Architecture

The C++ implementation uses:

1. **derivative class** (~515 lines)
   - Multiple sparse matrices: `O` (block), `B`, `N`, `Bfx/rx`, `Bfy/ry`, `Bfz/rz` (CSR)
   - Template-based operator application with `Op` parameter
   - Complex builder classes for matrix construction
   - Visitor pattern for matrix traversal

2. **gradient class** (~37 lines)
   - Contains 3 `derivative` objects
   - Returns `std::function` for lazy evaluation
   - Simple composition pattern

3. **laplacian class** (~55 lines)
   - Contains 3 `derivative` objects
   - Accumulates results using template `plus_eq` operator
   - Two overloads: with/without Neumann conditions

### Matrix Structure Analysis

**Block Matrix (`O`):**
```cpp
// Composed of inner_blocks along each line
inner_block:
  - left_boundary:  dense matrix (R×T)
  - interior:       circulant (repeating stencil)
  - right_boundary: dense matrix (R×T)
```

**CSR Matrices (`B`, `N`, `Bf*/Br*`):**
- Standard compressed sparse row format
- `B`: Couples boundary data to domain
- `N`: Neumann boundary conditions
- `Bf*/Br*`: Cut-cell boundary coupling

**Key Insight:** Python doesn't need this complexity. We can use SciPy's CSR directly and build matrices once.

---

## Python Design: Simplified Architecture

### Design Principles

✅ **DO:**
- Use `scipy.sparse.csr_matrix` for all matrices
- Simple dataclasses to hold operator data
- Factory functions for construction (not builders)
- Functional composition (gradient = 3 derivatives)
- Clear separation: construction vs application

❌ **DON'T:**
- NO builder pattern (use factory functions)
- NO abstract base classes (only 3 concrete operators)
- NO template metaprogramming (Python doesn't need it)
- NO custom matrix classes (SciPy is sufficient)
- NO visitor pattern (not needed in Python)

### Core Data Structures

#### 1. DerivativeOperator (Dataclass)

```python
from dataclasses import dataclass
from typing import Optional
import scipy.sparse as sp

@dataclass
class DerivativeOperator:
    """
    1D derivative operator along a coordinate direction.
    
    Represents d/dx, d/dy, or d/dz on a computational domain
    with cut-cell boundaries.
    
    Attributes:
        direction: 0=x, 1=y, 2=z
        O: Interior/domain operator (CSR matrix)
        B: Boundary coupling matrix (CSR matrix)
        N: Neumann boundary operator (CSR matrix, optional)
        Bfx, Brx, Bfy, Bry, Bfz, Brz: Cut-cell boundary operators
        interior_stencil: Interior stencil coefficients (for reference)
    """
    direction: int
    O: sp.csr_matrix
    B: sp.csr_matrix
    N: Optional[sp.csr_matrix] = None
    Bfx: Optional[sp.csr_matrix] = None
    Brx: Optional[sp.csr_matrix] = None
    Bfy: Optional[sp.csr_matrix] = None
    Bry: Optional[sp.csr_matrix] = None
    Bfz: Optional[sp.csr_matrix] = None
    Brz: Optional[sp.csr_matrix] = None
    interior_stencil: Optional[np.ndarray] = None
    
    def apply(self, u: ScalarField) -> ScalarField:
        """
        Apply operator to scalar field: du = D @ u
        
        Args:
            u: Input scalar field (with D, Rx, Ry, Rz components)
            
        Returns:
            Derivative scalar field
        """
        # Allocate output
        du = u.zeros_like()
        
        # Update cut-cell boundary regions
        if self.Bfx is not None and u.Rx is not None:
            du.Rx += self.Bfx @ u.D
        if self.Brx is not None and u.Rx is not None:
            du.Rx += self.Brx @ u.Rx
        # (similar for y, z)
        
        # Update domain interior
        du.D = self.O @ u.D
        
        # Add boundary coupling
        if self.direction == 0 and u.Rx is not None:
            du.D += self.B @ u.Rx
        # (similar for y, z)
        
        return du
    
    def apply_with_neumann(self, u: ScalarField, nu: ScalarField) -> ScalarField:
        """Apply with Neumann boundary data."""
        du = self.apply(u)
        if self.N is not None:
            du.D += self.N @ nu.D
        return du
```

**Why this design?**
- Simple: All matrices are CSR, easy to understand
- Flexible: Optional matrices for complex boundaries
- Testable: Each component can be verified independently
- Efficient: SciPy's CSR matvec is highly optimized

#### 2. No Gradient/Laplacian Classes (Just Functions!)

```python
def gradient(u: ScalarField, 
             dx: DerivativeOperator,
             dy: DerivativeOperator, 
             dz: DerivativeOperator) -> VectorField:
    """
    Compute gradient: ∇u = (∂u/∂x, ∂u/∂y, ∂u/∂z)
    
    Args:
        u: Scalar field
        dx, dy, dz: Derivative operators in each direction
        
    Returns:
        Vector field with gradient components
    """
    return VectorField(
        x=dx.apply(u),
        y=dy.apply(u),
        z=dz.apply(u)
    )

def laplacian(u: ScalarField,
              dx: DerivativeOperator,
              dy: DerivativeOperator,
              dz: DerivativeOperator) -> ScalarField:
    """
    Compute Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²
    
    Args:
        u: Scalar field
        dx, dy, dz: 2nd derivative operators in each direction
        
    Returns:
        Scalar field with Laplacian
    """
    result = u.zeros_like()
    result.D = dx.O @ u.D + dy.O @ u.D + dz.O @ u.D
    # Add boundary contributions...
    return result
```

**Why functions instead of classes?**
- Gradient/Laplacian are just compositions of derivatives
- No state to maintain
- Simpler API: `gradient(u, dx, dy, dz)` vs `grad = Gradient(...); grad(u)`
- Easier to test
- More Pythonic (like NumPy's functions)

---

## Implementation Plan

### Phase 3.1: Simple 1D Derivative (1-2 days)

**Goal:** Build derivative operator for 1D uniform grid, Dirichlet BCs only.

**Files to create:**
- `src/shoccs/operators/__init__.py`
- `src/shoccs/operators/derivative.py`
- `tests/test_derivative_1d.py`

**Steps:**
1. ✅ Create `DerivativeOperator` dataclass (simple version)
2. ✅ Implement `build_derivative_1d()` factory function:
   ```python
   def build_derivative_1d(
       n: int,           # Number of grid points
       h: float,         # Grid spacing
       stencil,          # From Phase 2
       bc_left='dirichlet',
       bc_right='dirichlet'
   ) -> DerivativeOperator:
       """Build 1D derivative with uniform grid and simple BCs."""
   ```
3. ✅ Test polynomial reproduction (constant → 0, linear → exact)
4. ✅ Test symmetry properties
5. ✅ Validate against hand-calculated examples

**Testing strategy:**
```python
def test_derivative_1d_linear_function():
    """Derivative of x should be 1 everywhere."""
    n = 10
    h = 0.1
    x = np.linspace(0, 1, n)
    u = ScalarField(D=x)  # u = x
    
    D = build_derivative_1d(n, h, e2_poly_interior(h))
    du = D.apply(u)
    
    expected = np.ones(n)
    assert np.allclose(du.D, expected, atol=1e-12)
```

**Success criteria:**
- ✅ Polynomial exactness to 1e-12
- ✅ Matrix is correct size
- ✅ Boundary rows use correct stencils
- ✅ Interior rows use interior stencil

### Phase 3.2: Boundary Conditions (1-2 days)

**Goal:** Add Neumann and Floating BCs.

**Enhancements:**
1. Add BC enum (like C++):
   ```python
   from enum import Enum
   
   class BCType(Enum):
       DIRICHLET = "dirichlet"
       NEUMANN = "neumann"
       FLOATING = "floating"
   ```
2. Update `build_derivative_1d()` to handle all BC types
3. Build `N` matrix for Neumann conditions
4. Test all BC combinations

**Testing:**
- Dirichlet-Dirichlet
- Neumann-Neumann
- Dirichlet-Neumann
- Floating boundaries

### Phase 3.3: Multi-Dimensional (2-3 days)

**Goal:** Extend to 2D/3D Cartesian grids.

**New function:**
```python
def build_derivative(
    mesh: CartesianMesh,
    direction: int,      # 0=x, 1=y, 2=z
    stencil,
    grid_bcs: dict,      # Grid boundary conditions
    object_bcs: dict,    # Object boundary conditions
) -> DerivativeOperator:
    """
    Build derivative operator for 3D Cartesian mesh.
    
    Handles:
    - Interior points (interior stencil)
    - Grid boundaries (left/right in each direction)
    - Cut-cell boundaries (object interfaces)
    """
```

**Complexity:**
- Need to build block-diagonal structure for 3D
- Handle strides for different directions
- Build cut-cell boundary matrices

**Testing:**
- 2D grid, all-Dirichlet BCs
- 3D grid, polynomial reproduction
- Compare with 1D results on 1D slices

### Phase 3.4: Cut-Cell Boundaries (2-3 days)

**Goal:** Add cut-cell boundary handling.

**Implementation:**
1. Build `B` matrix (couples boundary data to domain)
2. Build `Bf*` and `Br*` matrices for each direction
3. Handle interpolation at cut-cells
4. Use mesh's `BoundaryPoint` data

**Testing:**
- Simple cut-cell case (sphere, cylinder)
- Verify boundary coupling
- Check interpolation accuracy

### Phase 3.5: Gradient and Laplacian (1 day)

**Goal:** Compose higher-level operators.

**Files:**
- `src/shoccs/operators/composite.py`
- `tests/test_gradient.py`
- `tests/test_laplacian.py`

**Implementation:**
```python
# Simple functions!
def gradient(u, dx, dy, dz):
    return VectorField(x=dx.apply(u), y=dy.apply(u), z=dz.apply(u))

def laplacian(u, dxx, dyy, dzz):
    result = u.zeros_like()
    result.D = dxx.O @ u.D + dyy.O @ u.D + dzz.O @ u.D
    # Handle boundaries...
    return result
```

**Testing:**
- Gradient of polynomial: ∇(x²+y²+z²) = (2x, 2y, 2z)
- Laplacian of polynomial: ∇²(x²+y²+z²) = 6
- Verify vector calculus identities

### Phase 3.6: Validation and Documentation (1 day)

**Goal:** Comprehensive validation and docs.

**Validation:**
1. Compare with analytical solutions
2. Check convergence rates (should match stencil order)
3. Verify conservation properties
4. Benchmark against C++ (if time permits)

**Documentation:**
- API reference
- Usage examples
- Performance notes
- Migration guide (C++ → Python mapping)

---

## Matrix Construction Strategy

### Approach: Direct CSR Construction

**Use SciPy's COO → CSR pattern:**

```python
from scipy.sparse import coo_matrix

def build_derivative_1d(n, h, stencil, bc_left, bc_right):
    """Build derivative matrix using COO format, convert to CSR."""
    
    rows = []
    cols = []
    data = []
    
    # Handle left boundary
    stencil_left = get_boundary_stencil(bc_left, h, right=False)
    r, t = stencil_left.shape  # e.g., (2, 4) for E2-poly Dirichlet
    for i in range(r):
        for j in range(t):
            rows.append(i)
            cols.append(j)
            data.append(stencil_left[i, j])
    
    # Handle interior
    interior = stencil  # e.g., [-1/2h, 0, 1/2h]
    p = len(interior) // 2  # half-width
    for i in range(r, n - r):
        for offset, coeff in enumerate(interior, start=-p):
            rows.append(i)
            cols.append(i + offset)
            data.append(coeff)
    
    # Handle right boundary
    # ... similar to left
    
    # Build CSR matrix
    O = coo_matrix((data, (rows, cols)), shape=(n, n))
    O = O.tocsr()  # Convert to CSR for fast matvec
    
    return DerivativeOperator(direction=0, O=O, B=None)
```

**Why COO → CSR?**
- COO is easy to build (append triplets)
- CSR is fast for matvec (what we need)
- SciPy handles conversion efficiently
- No manual indexing into CSR arrays

### Handling Block Structure (3D)

For 3D, the operator is block-diagonal:

```
O = [ O_line1    0         0      ]
    [   0      O_line2     0      ]
    [   0        0      O_line3   ]
```

**Strategy:**
1. Iterate over lines in the specified direction
2. Build each line's operator (1D problem)
3. Place into global matrix at correct row/col offsets
4. Use COO format, convert to CSR at end

```python
def build_derivative_3d(mesh, direction):
    rows, cols, data = [], [], []
    
    for line in mesh.lines(direction):
        # Build 1D operator for this line
        line_rows, line_cols, line_data = build_line_operator(line, ...)
        
        # Offset into global matrix
        row_offset = line.start_index
        col_offset = line.start_index
        
        rows.extend([r + row_offset for r in line_rows])
        cols.extend([c + col_offset for c in line_cols])
        data.extend(line_data)
    
    O = coo_matrix((data, (rows, cols)), shape=(n_total, n_total))
    return O.tocsr()
```

---

## Testing Strategy

### Test Pyramid

**Level 1: Unit Tests (Fast, Many)**
- Stencil application (covered in Phase 2)
- Matrix construction
- BC handling
- Data structure creation

**Level 2: Integration Tests (Medium)**
- 1D derivative operators
- 2D/3D operators
- Gradient/Laplacian composition
- Cut-cell handling

**Level 3: Validation Tests (Slow, Few)**
- Polynomial exactness
- Convergence rates
- Analytical solutions
- Conservation properties

### Key Test Cases

#### 1. Polynomial Reproduction

```python
def test_derivative_polynomial_exactness():
    """
    1st derivative: d/dx(x^2) = 2x (exact to roundoff)
    2nd derivative: d^2/dx^2(x^3) = 6x (exact to roundoff)
    """
    n = 20
    h = 0.05
    x = np.linspace(0, 1, n)
    
    # 1st derivative
    u = ScalarField(D=x**2)
    D1 = build_derivative_1d(n, h, e2_poly_interior(h), order=1)
    du = D1.apply(u)
    assert np.allclose(du.D[1:-1], 2*x[1:-1], atol=1e-12)
    
    # 2nd derivative
    u = ScalarField(D=x**3)
    D2 = build_derivative_1d(n, h, e2_poly_interior_2nd(h), order=2)
    d2u = D2.apply(u)
    assert np.allclose(d2u.D[1:-1], 6*x[1:-1], atol=1e-12)
```

#### 2. Convergence Rates

```python
def test_derivative_convergence():
    """Verify 2nd-order convergence for O(h^2) stencils."""
    f = lambda x: np.sin(2*np.pi*x)
    df = lambda x: 2*np.pi*np.cos(2*np.pi*x)
    
    errors = []
    for n in [10, 20, 40, 80]:
        h = 1.0 / n
        x = np.linspace(0, 1, n)
        u = ScalarField(D=f(x))
        D = build_derivative_1d(n, h, e2_poly_interior(h))
        du = D.apply(u)
        error = np.linalg.norm(du.D - df(x))
        errors.append(error)
    
    # Check 2nd-order convergence: error ~ h^2
    rates = np.log(errors[:-1] / errors[1:]) / np.log(2)
    assert np.all(rates > 1.9), f"Expected ~2, got {rates}"
```

#### 3. Gradient Identity

```python
def test_gradient_polynomial():
    """∇(x^2 + y^2 + z^2) = (2x, 2y, 2z)"""
    mesh = CartesianMesh(nx=10, ny=10, nz=10, 
                        xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1)
    x, y, z = mesh.coordinates()
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    u = ScalarField(D=(X**2 + Y**2 + Z**2))
    dx = build_derivative(mesh, direction=0, ...)
    dy = build_derivative(mesh, direction=1, ...)
    dz = build_derivative(mesh, direction=2, ...)
    
    grad_u = gradient(u, dx, dy, dz)
    
    assert np.allclose(grad_u.x.D, 2*X, atol=1e-12)
    assert np.allclose(grad_u.y.D, 2*Y, atol=1e-12)
    assert np.allclose(grad_u.z.D, 2*Z, atol=1e-12)
```

#### 4. Laplacian Identity

```python
def test_laplacian_polynomial():
    """∇²(x^2 + y^2 + z^2) = 6"""
    # Similar to gradient test
    lap_u = laplacian(u, dxx, dyy, dzz)
    assert np.allclose(lap_u.D, 6.0, atol=1e-12)
```

### Validation Against C++

If time permits, create comparison tests:

```python
def test_compare_with_cpp():
    """Load C++ operator output and compare."""
    # Load C++ matrices from file
    cpp_matrix = load_cpp_matrix("derivative_x.mtx")
    
    # Build Python operator
    py_op = build_derivative(...)
    
    # Compare matrices
    assert np.allclose(cpp_matrix.toarray(), py_op.O.toarray(), atol=1e-14)
```

---

## API Design

### User-Facing API

**Simple, functional interface:**

```python
from shoccs.geometry import CartesianMesh
from shoccs.fields import ScalarField, VectorField
from shoccs.stencils import e2_poly_interior, make_e2_poly_stencil
from shoccs.operators import build_derivative, gradient, laplacian

# 1. Create mesh
mesh = CartesianMesh(nx=50, ny=50, nz=50, 
                     xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1)

# 2. Define boundary conditions
grid_bcs = {
    'x': {'left': 'dirichlet', 'right': 'dirichlet'},
    'y': {'left': 'dirichlet', 'right': 'dirichlet'},
    'z': {'left': 'dirichlet', 'right': 'dirichlet'},
}

# 3. Build operators
dx = build_derivative(mesh, direction=0, order=1, bc=grid_bcs)
dy = build_derivative(mesh, direction=1, order=1, bc=grid_bcs)
dz = build_derivative(mesh, direction=2, order=1, bc=grid_bcs)

dxx = build_derivative(mesh, direction=0, order=2, bc=grid_bcs)
dyy = build_derivative(mesh, direction=1, order=2, bc=grid_bcs)
dzz = build_derivative(mesh, direction=2, order=2, bc=grid_bcs)

# 4. Create field
x, y, z = mesh.coordinates()
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
u = ScalarField(D=np.sin(np.pi*X) * np.sin(np.pi*Y) * np.sin(np.pi*Z))

# 5. Apply operators
grad_u = gradient(u, dx, dy, dz)          # Returns VectorField
lap_u = laplacian(u, dxx, dyy, dzz)       # Returns ScalarField

# 6. Extract data
print(f"Max gradient magnitude: {np.max(grad_u.x.D**2 + grad_u.y.D**2 + grad_u.z.D**2)}")
print(f"Max Laplacian: {np.max(lap_u.D)}")
```

**Why this API?**
- Explicit: User sees all steps
- Flexible: Can customize BCs, stencils
- Composable: Operators are reusable
- Pythonic: Follows NumPy/SciPy conventions
- Simple: No hidden state or magic

### Alternative: Factory Function for Common Cases

For convenience, provide a helper:

```python
def build_operators(mesh, order=2, bc='dirichlet'):
    """
    Build standard set of operators for a mesh.
    
    Returns:
        dict with keys: 'dx', 'dy', 'dz', 'dxx', 'dyy', 'dzz'
    """
    grid_bcs = {
        'x': {'left': bc, 'right': bc},
        'y': {'left': bc, 'right': bc},
        'z': {'left': bc, 'right': bc},
    }
    
    return {
        'dx': build_derivative(mesh, 0, order=1, bc=grid_bcs),
        'dy': build_derivative(mesh, 1, order=1, bc=grid_bcs),
        'dz': build_derivative(mesh, 2, order=1, bc=grid_bcs),
        'dxx': build_derivative(mesh, 0, order=2, bc=grid_bcs),
        'dyy': build_derivative(mesh, 1, order=2, bc=grid_bcs),
        'dzz': build_derivative(mesh, 2, order=2, bc=grid_bcs),
    }
```

---

## Performance Considerations

### Memory Efficiency

**Sparse matrices:**
- 1D operator on n points: ~3n non-zeros (3-point stencil)
- 3D operator on n³ points: ~3n³ non-zeros
- CSR storage: ~24 bytes per non-zero (8 for value, 8 for col index, 8 for row pointer)

**Example:** 100³ grid
- Dense matrix: 8 * (10⁶)² = 8 TB (impossible!)
- Sparse CSR: 24 * 3*10⁶ = 72 MB (feasible)

**Takeaway:** Sparse matrices are essential for 3D.

### Computational Efficiency

**SciPy CSR matvec:**
- Highly optimized C implementation
- ~1-2 μs per row for sparse matrices
- For 100³ grid: ~1-2 ms per derivative

**Numba potential:**
- Could JIT-compile custom matvec if needed
- Not necessary for initial implementation
- Profile first, optimize later

### Construction Time

**Building operators:**
- COO construction: O(nnz) where nnz = number of non-zeros
- COO → CSR conversion: O(nnz log nnz) for sorting
- For 100³ grid: < 1 second (acceptable)

**Strategy:**
- Build operators once, reuse many times
- Cache operators if mesh doesn't change
- Lazy construction only if needed

---

## File Structure

```
python-migration/
├── src/shoccs/operators/
│   ├── __init__.py           # Public API
│   ├── derivative.py         # DerivativeOperator + build functions
│   ├── composite.py          # gradient, laplacian functions
│   └── boundary_conditions.py # BCType enum, BC handling
├── tests/
│   ├── test_derivative_1d.py      # 1D operator tests
│   ├── test_derivative_3d.py      # 3D operator tests
│   ├── test_gradient.py           # Gradient tests
│   ├── test_laplacian.py          # Laplacian tests
│   └── test_convergence.py        # Convergence validation
├── examples/
│   ├── operator_demo.py           # Basic usage
│   └── poisson_solver.py          # Example: solve ∇²u = f
└── docs/
    └── operators_guide.md         # User documentation
```

---

## Implementation Timeline

| Phase | Task | Duration | Priority |
|-------|------|----------|----------|
| 3.1 | 1D Derivative (Dirichlet) | 1-2 days | HIGH |
| 3.2 | Boundary Conditions | 1-2 days | HIGH |
| 3.3 | Multi-Dimensional | 2-3 days | HIGH |
| 3.4 | Cut-Cell Boundaries | 2-3 days | MEDIUM |
| 3.5 | Gradient/Laplacian | 1 day | HIGH |
| 3.6 | Validation/Docs | 1 day | HIGH |

**Total:** 8-12 days (1.5-2.5 weeks)

**Critical path:** 3.1 → 3.2 → 3.3 → 3.5 → 3.6

**Can parallelize:** 3.4 (cut-cells) can be done after 3.3, doesn't block 3.5

---

## Success Criteria

### Correctness
- ✅ Polynomial reproduction to machine precision (< 1e-12)
- ✅ Convergence rates match stencil order
- ✅ All BC types work correctly
- ✅ Gradient/Laplacian identities verified

### Code Quality
- ✅ Simple, readable implementation
- ✅ No unnecessary abstraction
- ✅ Comprehensive tests (>90% coverage)
- ✅ Clear documentation

### Performance
- ✅ Operator construction < 1s for 100³ grid
- ✅ Derivative application < 10ms for 100³ grid
- ✅ Memory usage reasonable (< 100MB for 100³)

### Integration
- ✅ Works with Phase 1 (Fields, Mesh)
- ✅ Uses Phase 2 (Stencils) correctly
- ✅ Ready for Phase 4 (Time integration)

---

## Risk Mitigation

### Risk 1: Matrix Construction Complexity

**Concern:** 3D block structure is complex, easy to get wrong.

**Mitigation:**
1. Start with 1D (simple)
2. Test thoroughly at each step
3. Visualize matrices (spy plots)
4. Compare with hand-calculated examples

### Risk 2: Cut-Cell Boundaries

**Concern:** C++ implementation is 250+ lines, lots of edge cases.

**Mitigation:**
1. Implement in Phase 3.4 (after core works)
2. Start with simple cut-cell cases (sphere)
3. Defer complex cases to later
4. Document assumptions clearly

### Risk 3: Performance

**Concern:** Python might be too slow.

**Mitigation:**
1. Use SciPy sparse matrices (already optimized)
2. Profile before optimizing
3. Numba JIT as needed
4. Most time in matvec (already fast)

---

## Code Examples

### Example 1: Building and Applying a 1D Derivative

```python
import numpy as np
from shoccs.geometry import CartesianMesh
from shoccs.fields import ScalarField
from shoccs.stencils import e2_poly_interior
from shoccs.operators import build_derivative_1d

# Setup
n = 100
h = 0.01
x = np.linspace(0, 1, n)

# Build operator
D = build_derivative_1d(
    n=n,
    h=h,
    stencil=e2_poly_interior(h),
    bc_left='dirichlet',
    bc_right='dirichlet'
)

# Create field: u(x) = sin(2πx)
u = ScalarField(D=np.sin(2*np.pi*x))

# Apply operator
du = D.apply(u)

# Analytical derivative: du/dx = 2π cos(2πx)
exact = 2*np.pi*np.cos(2*np.pi*x)

# Check error (should be small except near boundaries)
error = np.linalg.norm(du.D[2:-2] - exact[2:-2])
print(f"Error (interior): {error:.3e}")  # Should be ~1e-3 for O(h²)
```

### Example 2: Gradient on 3D Grid

```python
from shoccs.geometry import CartesianMesh
from shoccs.fields import ScalarField
from shoccs.operators import build_derivative, gradient

# Create mesh
mesh = CartesianMesh(nx=20, ny=20, nz=20,
                     xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1)

# Build operators
dx = build_derivative(mesh, direction=0, order=1)
dy = build_derivative(mesh, direction=1, order=1)
dz = build_derivative(mesh, direction=2, order=1)

# Create field: u = x² + y² + z²
x, y, z = mesh.coordinates()
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
u = ScalarField(D=X**2 + Y**2 + Z**2)

# Compute gradient
grad_u = gradient(u, dx, dy, dz)

# Check: ∇u should be (2x, 2y, 2z)
assert np.allclose(grad_u.x.D, 2*X, atol=1e-10)
assert np.allclose(grad_u.y.D, 2*Y, atol=1e-10)
assert np.allclose(grad_u.z.D, 2*Z, atol=1e-10)
print("✓ Gradient test passed!")
```

### Example 3: Laplacian for Poisson Equation

```python
from scipy.sparse.linalg import spsolve
from shoccs.operators import build_derivative, laplacian

# Build 2nd derivative operators
mesh = CartesianMesh(nx=50, ny=50, nz=50, ...)
dxx = build_derivative(mesh, direction=0, order=2)
dyy = build_derivative(mesh, direction=1, order=2)
dzz = build_derivative(mesh, direction=2, order=2)

# Laplacian matrix (for linear solve)
L = dxx.O + dyy.O + dzz.O  # Simple addition of sparse matrices!

# Solve Poisson equation: ∇²u = f
x, y, z = mesh.coordinates()
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
f = -3*np.pi**2 * np.sin(np.pi*X) * np.sin(np.pi*Y) * np.sin(np.pi*Z)

# Solve (with proper BC handling)
u_flat = spsolve(L, f.flatten())
u = ScalarField(D=u_flat.reshape(mesh.shape))

# Analytical solution
u_exact = np.sin(np.pi*X) * np.sin(np.pi*Y) * np.sin(np.pi*Z)
error = np.linalg.norm(u.D - u_exact)
print(f"Poisson solve error: {error:.3e}")
```

---

## Comparison: C++ vs Python

| Aspect | C++ | Python |
|--------|-----|--------|
| **Matrix Types** | Custom `block`, `csr`, `dense`, `circulant` | `scipy.sparse.csr_matrix` only |
| **Construction** | Builder pattern (`OB_builder`) | Factory functions |
| **Templates** | Heavy use (`template<typename Op>`) | None needed |
| **Operator Application** | Overloaded `operator()` with `Op` | Simple `apply()` method |
| **Gradient/Laplacian** | Classes with `std::function` | Pure functions |
| **Code Complexity** | ~600 lines (derivative.cpp) | ~200 lines (estimated) |
| **Abstraction Level** | Low (manual matrix assembly) | Medium (SciPy handles details) |

**Key Insight:** Python can be 3x simpler by using SciPy and avoiding template metaprogramming.

---

## Questions for Review

Before implementation, confirm:

1. ✅ **Dataclass vs function for operators?**
   - Proposed: DerivativeOperator dataclass, gradient/laplacian as functions
   - Alternative: All functions?

2. ✅ **How to handle cut-cell boundaries initially?**
   - Proposed: Defer to Phase 3.4, start with simple grids
   - Alternative: Implement from the start?

3. ✅ **Testing depth?**
   - Proposed: Polynomial exactness + convergence + identities
   - Alternative: Compare every matrix element with C++?

4. ✅ **Performance targets?**
   - Proposed: "Good enough" (< 1s build, < 10ms apply for 100³)
   - Alternative: Match C++ speed exactly?

---

## Conclusion

Phase 3 builds on the solid foundation of Phases 1-2:
- **Phase 1:** Fields and Mesh ✅
- **Phase 2:** Stencils (validated to 1e-14) ✅
- **Phase 3:** Operators (this plan)

**Design philosophy:**
- Simplicity over cleverness
- SciPy over custom implementations
- Functions over classes (where appropriate)
- Testing over documentation (but both are good!)

**Next steps:**
1. Review this plan
2. Start with Phase 3.1 (1D derivative)
3. Iterate based on learnings

**Estimated timeline:** 1.5-2.5 weeks for full implementation.

---

**Architect sign-off:** Ready for implementation. Start with Phase 3.1!
