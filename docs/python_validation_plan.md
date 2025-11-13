# SHOCCS Python Migration: Validation and Testing Plan

**Date:** 2025-11-13  
**Version:** 1.0  
**Status:** Draft for Review  
**Reference:** Based on Brady & Livescu (2021) JCP paper - TEMO approach

---

## Executive Summary

This document defines a rigorous validation and testing strategy for the Python migration of the SHOCCS cut-cell solver. The plan prioritizes **scientific correctness** through systematic verification of numerical properties, ensuring the Python implementation matches or exceeds the C++ reference implementation's accuracy and stability.

### Key Principles

1. **Scientific Correctness First**: All numerical properties must be verified before performance optimization
2. **Incremental Validation**: Build confidence through layered testing (unit → integration → convergence)
3. **Regression Prevention**: Continuous comparison against C++ reference implementation
4. **Reproducible Results**: Deterministic test cases with documented tolerances
5. **Method of Manufactured Solutions**: Primary verification tool for discretization accuracy

---

## 1. Test Hierarchy

### 1.1 Level 0: Component Unit Tests

**Purpose:** Verify individual mathematical operations and data structures in isolation.

#### 1.1.1 Data Structure Tests
- **Field Structure** (`Field` class with D, Rx, Ry, Rz components)
  - Correct memory layout and indexing
  - Proper boundary region extraction
  - Field arithmetic operations (addition, scaling, inner products)
  - **Pass Criteria:** Exact match with reference values (machine precision)
  
- **Mesh Geometry** (`Mesh` class)
  - Grid point coordinates (1D, 2D, 3D)
  - Cell spacing calculations
  - Domain boundary identification
  - **Pass Criteria:** Coordinate errors < 1e-14 (floating point rounding only)

- **Cut-Cell Geometry** 
  - Level set computation for spheres, cylinders
  - Psi (cut-cell fraction) calculation
  - Boundary point identification (Rx, Ry, Rz sets)
  - Ray-casting for boundary normals
  - **Pass Criteria:** 
    - Psi values: absolute error < 1e-12
    - Boundary point counts: exact match with C++ reference

#### 1.1.2 Stencil Coefficient Tests
- **Interior Stencils** (E2, E4)
  - 2nd-order E2: 3-point stencil [-1/h², 2/h², -1/h²]
  - 4th-order E4: 5-point stencil
  - **Pass Criteria:** Coefficient errors < 1e-15 (analytical values)

- **Boundary Stencils** (NBS - Numerical Boundary Scheme)
  - Dirichlet boundary treatments
  - Neumann boundary treatments  
  - Floating boundary treatments
  - **Pass Criteria:** Match C++ reference coefficients to machine precision
  - **Critical Test:** Verify 2509-line E2_1 stencil polynomial evaluation

- **Geometry-Dependent Stencils**
  - Coefficients as function of psi ∈ [0, 1]
  - Alpha parameter optimization results
  - **Pass Criteria:** 
    - Match C++ reference for discrete psi values: {0.1, 0.3, 0.5, 0.7, 0.9}
    - Smooth variation (no discontinuities)

#### 1.1.3 Matrix Assembly Tests
- **CSR Matrix Construction**
  - Correct row pointers, column indices, values
  - Sparsity pattern verification
  - Matrix-vector product accuracy
  - **Pass Criteria:** 
    - Sparsity pattern: exact match
    - Matrix-vector product: relative error < 1e-14

- **Block Matrix Structures**
  - Circulant interior blocks
  - Boundary block assembly
  - **Pass Criteria:** Element-wise comparison with C++ reference

### 1.2 Level 1: Operator Unit Tests

**Purpose:** Verify discrete differential operators reproduce exact solutions for polynomials within their design order.

#### 1.2.1 Derivative Operators (∂/∂x, ∂/∂y, ∂/∂z)

**Test Case 1: 2nd-Order E2 with Polynomial Functions**
```python
# For 2nd-order scheme, should be exact for quadratic polynomials
# Test function (from derivative.t.cpp):
f(x,y,z) = x*(y+z) + y*(x+z) + z*(x+y) + 3*x*y*z
∂f/∂x = 2*(y+z) + 3*y*z
∂²f/∂x² = 0  (should be exactly zero for E2)
```
- **Grid:** Uniform 10×13×17 points, domain [0.1, 1.0] × [0.2, 2.0] × [0.3, 2.2]
- **Boundary Conditions:** Test all combinations
  - FFFFFF (all periodic/free)
  - DDFFFD (Dirichlet on x-boundaries, z-max)
  - NNDDDF (Neumann on x, Dirichlet on y)
- **Pass Criteria:**
  - First derivative: L∞ error < 1e-12 (machine precision for 2nd-order polynomial)
  - Second derivative: L∞ error < 1e-13 (should be exactly zero)

**Test Case 2: Operators with Embedded Objects**
```python
# Sphere embedded in domain
mesh = {
    extents: [25, 26, 27],
    domain: [0.1, 1.0] × [0.2, 2.0] × [0.3, 2.2],
    sphere: center=(0.45, 1.011, 1.31), radius=0.25
}
```
- **Boundary Conditions:** 
  - Grid: NNDDFF (Neumann x, Dirichlet y, Free z)
  - Object: Dirichlet or Floating
- **Pass Criteria:**
  - Interior fluid points: same as no-object case
  - Boundary region (Rx, Ry, Rz): L∞ error < 1e-11
  - Match C++ reference implementation within 1e-12

#### 1.2.2 Laplacian Operator (∇²)

**Test Case 1: Domain Laplacian**
```python
# Test function (from laplacian.t.cpp):
f(x,y,z) = x²*(y+z) + y²*(x+z) + z²*(x+y) + 3*x*y*z + x + y + z
∇²f = 2*(y+z) + 2*(x+z) + 2*(x+y)
```
- **Grid:** 5×6×7 points, domain [0.1, 1.0] × [0.2, 2.0] × [0.3, 2.2]
- **Boundary Conditions:** DDFFFD, DDFFND (with Neumann)
- **Pass Criteria:** L∞ error < 1e-11

**Test Case 2: Cut-Cell Laplacian**
- Same test function with embedded sphere
- Both Dirichlet and Floating object BCs
- **Pass Criteria:** 
  - Fluid domain D: L∞ error < 1e-11
  - Boundary regions Rx, Ry, Rz: L∞ error < 5e-11 (slightly relaxed near cut-cells)

#### 1.2.3 Gradient and Divergence Operators

**Test Cases:**
- Vector field gradient: ∇(scalar field)
- Divergence: ∇·(vector field)
- **Pass Criteria:** Component-wise L∞ error < 1e-11

### 1.3 Level 2: Integration Tests

**Purpose:** Verify coupled operator-integrator systems for time-dependent PDEs.

#### 1.3.1 Heat Equation System Tests

**Test Case 1: MMS Verification**
```python
# Manufactured solution (from heat.t.cpp):
u(t,x,y,z) = sin(t) + x²*(y+z) + y²*(x+z) + z²*(x+y) + 3*x*y*z + x + y + z
∂u/∂t = cos(t)
∇²u = 2*(y+z) + 2*(x+z) + 2*(x+y)

# PDE: ∂u/∂t = κ*∇²u + source(t,x,y,z)
# where source = ∂u/∂t - κ*∇²u (computed from MMS)
```
- **Grid:** 21×22×23 with embedded sphere (radius 0.25)
- **Diffusivity:** κ = 0.1
- **Boundary Conditions:**
  - Grid: xmin=Dirichlet, y=Neumann, zmax=Dirichlet
  - Object: Dirichlet or Floating
- **Integrator:** RK4
- **Time:** Single step (verify RHS computation)
- **Pass Criteria:**
  - Initial condition error: L∞ < 1e-14
  - RHS at fluid points: L∞ error < 1e-10 (relative to cos(t))
  - RHS at boundaries: correctly enforced (zero for Dirichlet)

**Test Case 2: 2D Heat Equation**
- Reduced problem: 21×22 grid (2D)
- Floating sphere boundary
- Bilinear test function (lower order for exact treatment)
- **Pass Criteria:** Same as 3D case

#### 1.3.2 Scalar Wave Equation Tests

**Test Case:** Wave propagation with embedded objects
```python
# System: ∂²u/∂t² = c²*∇²u
# Convert to first-order system:
# ∂u/∂t = v
# ∂v/∂t = c²*∇²u
```
- **Grid:** 71×71 with two embedded spheres
- **Boundary Conditions:** Mixed Dirichlet/Floating
- **Integrator:** RK4
- **Pass Criteria:** 
  - Single timestep RHS evaluation: L∞ error < 1e-10
  - Energy conservation (for conservative BCs): drift < 1e-8 per step

### 1.4 Level 3: Convergence Studies

**Purpose:** Verify theoretical order of accuracy through systematic grid refinement.

#### 1.4.1 Spatial Convergence - Heat Equation

**Test Setup:**
```python
# MMS with smooth manufactured solution
u_exact(t,x,y,z) = exp(-t) * sin(π*x) * sin(π*y) * sin(π*z)
source = ∂u_exact/∂t - κ*∇²u_exact
```

**Grid Sequence:** 
- 2D: [16×16, 32×32, 64×64, 128×128]
- 3D: [8×8×8, 16×16×16, 32×32×32, 64×64×64]

**Configurations:**
1. **No embedded objects** (baseline)
2. **Single sphere** (center domain, radius 0.2*L)
3. **Multiple spheres** (complex geometry)

**Boundary Conditions:**
- Periodic (cleanest convergence)
- Mixed Dirichlet/Neumann
- With embedded objects (Dirichlet and Floating)

**Numerical Parameters:**
- Time step: Δt = 0.5 * CFL_parabolic * h² (parabolic CFL)
- Final time: T = 0.1 (short time to minimize temporal error)
- Use small temporal errors: RK4 with Δt = O(h³) or smaller

**Measured Quantities:**
1. **L² Error Norm:**
   ```
   ε₂ = √(∑ᵢ (uₙᵤₘ - u_exact)² * Δx * Δy * Δz)
   ```
2. **L∞ Error Norm:**
   ```
   ε∞ = max |uₙᵤₘ - u_exact|
   ```
3. **Convergence Rate:**
   ```
   p = log(ε_coarse / ε_fine) / log(h_coarse / h_fine)
   ```

**Pass Criteria:**
- **E2 Scheme (2nd order):**
  - Interior domain: p ≥ 1.95 (L² norm)
  - Near boundaries: p ≥ 1.85 (some degradation expected)
  - Overall: p ≥ 1.9 ± 0.1
  
- **E4 Scheme (4th order):**
  - Interior domain: p ≥ 3.90 (L² norm)
  - Near boundaries: p ≥ 3.70
  - Overall: p ≥ 3.8 ± 0.2

**Regression Check:**
- Python convergence rate must be within 0.05 of C++ reference

#### 1.4.2 Temporal Convergence

**Test Setup:**
```python
# Same MMS, fixed fine spatial grid (h = 1/128)
# Vary time step size
```

**Time Step Sequence:** Δt = [0.01, 0.005, 0.0025, 0.00125]

**Pass Criteria:**
- **RK4 Integrator:** p ≥ 3.95 (4th order)
- **Euler Integrator:** p ≥ 0.98 (1st order)

#### 1.4.3 Cut-Cell Geometry Convergence

**Purpose:** Verify stability and accuracy as cut-cell fraction (psi) approaches degenerate limits.

**Test Setup:**
```python
# Fix grid, vary sphere radius to create different psi values
# Measure error at boundary regions
sphere_radii = [0.15, 0.18, 0.21, 0.24, 0.27]  # Creates varying psi
```

**Pass Criteria:**
- No catastrophic error growth as psi → 0 or psi → 1
- Error remains bounded: ε < 10 * ε_nominal for all psi ∈ [0.1, 0.9]
- Verify TEMO optimization maintains stability

---

## 2. Numerical Accuracy Requirements

### 2.1 Tolerance Definitions

| Component | Absolute Tolerance | Relative Tolerance | Notes |
|-----------|-------------------|-------------------|-------|
| **Geometry** | | | |
| Grid coordinates | 1e-14 | - | Machine precision |
| Psi (cut fraction) | 1e-12 | - | Geometric calculation |
| Boundary normals | 1e-13 | - | Level set gradients |
| **Stencil Coefficients** | | | |
| Interior (analytical) | 1e-15 | - | Exact formulas |
| Boundary (E2_1 polynomial) | 1e-14 | - | Complex polynomial evaluation |
| **Operators** | | | |
| Polynomial reproduction | 1e-12 | - | Design-order polynomials |
| General smooth functions | 1e-10 | 1e-8 | Truncation error |
| Cut-cell boundaries | 5e-11 | 5e-9 | Near irregular boundaries |
| **Time Integration** | | | |
| RK4 single step | - | 1e-10 | Per-step accuracy |
| Long-time integration | - | √T * 1e-10 | Accumulated error |
| **Convergence Rates** | | | |
| 2nd-order scheme (E2) | p ∈ [1.85, 2.15] | - | ±0.15 tolerance |
| 4th-order scheme (E4) | p ∈ [3.70, 4.30] | - | ±0.30 tolerance |
| **Regression vs C++** | | | |
| Stencil coefficients | 1e-14 | - | Direct comparison |
| Operator application | 1e-12 | - | Full operator chain |
| Convergence rates | 0.05 | - | Rate difference |

### 2.2 Special Cases

#### 2.2.1 Degenerate Cut-Cells
For psi < 0.1 or psi > 0.9 (near-degenerate cells):
- Allow 10× relaxation of boundary tolerances
- Monitor condition numbers of stencil matrices
- Flag if condition number > 1e8

#### 2.2.2 Multi-Dimensional Corners
Grid corners (e.g., xmin ∩ ymin):
- May show reduced convergence rate (p ≈ 1.5 for 2nd order)
- Verify error contribution is localized (doesn't contaminate interior)

#### 2.2.3 Floating Boundary Conditions
- Verify boundary values evolve consistently with PDE
- Check mass conservation for conservative PDEs

---

## 3. Regression Testing Against C++ Reference

### 3.1 Exact Reproducibility Tests

**Purpose:** Ensure Python implementation matches C++ bit-for-bit where possible.

#### 3.1.1 Deterministic Test Cases

**Test Database:**
1. Extract outputs from C++ test suite (all .t.cpp files)
2. Store as JSON reference data:
   ```json
   {
     "test_name": "E2_derivative_FFFFFF",
     "grid": {"nx": 10, "ny": 13, "nz": 17},
     "domain": {"min": [0.1, 0.2, 0.3], "max": [1.0, 2.0, 2.2]},
     "inputs": {
       "field_values": [...],  # Input field
       "boundary_values": [...]
     },
     "outputs": {
       "result": [...],  # Expected output
       "l2_error": 1.234e-12,
       "linf_error": 5.678e-12
     }
   }
   ```

3. Python test harness:
   ```python
   def test_regression_E2_derivative_FFFFFF():
       ref_data = load_reference("E2_derivative_FFFFFF.json")
       result = python_implementation(ref_data["inputs"])
       assert np.allclose(result, ref_data["outputs"]["result"], 
                          rtol=1e-12, atol=1e-14)
   ```

**Test Coverage:**
- All operator tests (derivative, laplacian, gradient)
- All boundary condition combinations
- All stencil types (E2, E4, E2_1, polyE2_1)
- Representative system tests (heat, wave)

**Pass Criteria:**
- Stencil coefficients: Match C++ within 1e-14
- Operator outputs: Match C++ within 1e-12
- Convergence rates: Match C++ within 0.05

#### 3.1.2 Cross-Implementation Validation

**Workflow:**
1. **C++ Driver:** Create C++ executable that exports:
   - Mesh geometry
   - Stencil coefficients
   - Operator matrices (CSR format)
   - RHS evaluations for manufactured solutions

2. **Python Loader:** Read C++ outputs and compare:
   ```python
   def compare_operators(cpp_matrix, python_matrix):
       # Compare sparse matrix structure
       assert (cpp_matrix.indptr == python_matrix.indptr).all()
       assert (cpp_matrix.indices == python_matrix.indices).all()
       # Compare values
       assert np.allclose(cpp_matrix.data, python_matrix.data, 
                          rtol=1e-14, atol=1e-15)
   ```

3. **Automated Regression Suite:**
   - Run nightly on all test cases
   - Flag any deviation > tolerance
   - Track convergence rate drift over time

### 3.2 Statistical Validation

For non-deterministic aspects (e.g., iterative solvers with different stopping criteria):
- Compare distributions of results
- Verify same statistical properties (mean, variance)
- Use Kolmogorov-Smirnov test for distribution matching

---

## 4. Critical Numerical Properties

### 4.1 Stability Analysis

**Purpose:** Ensure schemes remain stable for target applications.

#### 4.1.1 Eigenvalue Analysis

**Test:** Compute spectrum of discretized operators
```python
# For Laplacian operator on uniform grid
L = build_laplacian_matrix(mesh, stencil, bcs)
eigenvalues = scipy.sparse.linalg.eigs(L, k=100, which='LM')
```

**Verification:**
1. **Parabolic Operators (Laplacian):**
   - All eigenvalues should have Re(λ) ≤ 0
   - Check maximum eigenvalue: λ_max ≈ -2π²/h² (for periodic BC)
   - **Pass Criteria:** No positive real eigenvalues

2. **Hyperbolic Operators (Wave):**
   - Eigenvalues should be purely imaginary (conservative)
   - Check maximum frequency: ω_max ≈ π/h
   - **Pass Criteria:** |Re(λ)| / |Im(λ)| < 1e-10

3. **Cut-Cell Stability:**
   - Verify eigenvalues remain bounded as psi → 0, 1
   - No eigenvalue magnitude growth > 10× compared to regular cells
   - **Pass Criteria:** Stable for all psi ∈ [0.1, 0.9]

#### 4.1.2 CFL Condition Verification

**Hyperbolic (Wave Equation):**
```python
# CFL_hyperbolic = c * dt / h < CFL_max
# Verify stability with MMS over 1000 timesteps
```
- **Pass Criteria:** 
  - Stable for CFL ≤ 0.8 (RK4)
  - Exponential growth for CFL > 1.0 (verify instability detection)

**Parabolic (Heat Equation):**
```python
# CFL_parabolic = κ * dt / h² < CFL_max
```
- **Pass Criteria:**
  - Stable for CFL ≤ 0.5 (explicit RK4)
  - Verify stability for implicit methods (if implemented)

### 4.2 Conservation Properties

**Purpose:** Verify discrete conservation for conservative PDEs.

#### 4.2.1 Mass Conservation

**Test Case:** Heat equation with Neumann boundaries (no flux)
```python
# Setup: ∂u/∂t = κ∇²u, all Neumann BCs (∂u/∂n = 0)
# Theoretical: Total mass M = ∫∫∫ u dV should be constant

def test_mass_conservation():
    u0 = initialize_field()
    M0 = integrate_field(u0, mesh)
    
    # Time-step for 100 steps
    for n in range(100):
        u = rk4_step(u, dt)
        M = integrate_field(u, mesh)
        assert abs(M - M0) < 1e-10 * M0  # Relative conservation
```

**Pass Criteria:**
- Mass drift < 1e-10 * M₀ per timestep (relative)
- No systematic growth over 1000 steps

#### 4.2.2 Energy Conservation

**Test Case:** Wave equation with periodic or free boundaries
```python
# Energy: E = ∫∫∫ (½v² + ½c²|∇u|²) dV
# Should be exactly conserved for conservative discretization
```

**Pass Criteria:**
- Energy drift < 1e-8 * E₀ per timestep (RK4 is not perfectly conservative)
- No exponential growth (indicates instability)

### 4.3 Symmetry Properties

**Purpose:** Verify operators preserve expected symmetries.

#### 4.3.1 Self-Adjointness

**Test:** Laplacian with Dirichlet BCs should be symmetric
```python
L = build_laplacian_matrix(mesh, stencil, dirichlet_bcs)
assert np.allclose(L.toarray(), L.T.toarray(), rtol=1e-14)
```

**Pass Criteria:** Symmetry error < 1e-13

#### 4.3.2 Rotational Invariance (Where Expected)

**Test:** Sphere in cube should give isotropic results
```python
# Laplacian at sphere surface should be independent of orientation
# Test by rotating coordinate system
```

**Pass Criteria:** Results vary < 1e-10 under rotation

---

## 5. Validation Workflow and Tools

### 5.1 Testing Framework

**Python Framework:** `pytest` with custom plugins

**Directory Structure:**
```
tests/
├── unit/
│   ├── test_fields.py           # Field data structures
│   ├── test_mesh.py              # Mesh and geometry
│   ├── test_stencils.py          # Stencil coefficients
│   ├── test_operators.py         # Discrete operators
│   └── test_integrators.py       # Time integrators
├── integration/
│   ├── test_heat_system.py       # Heat equation
│   ├── test_wave_system.py       # Wave equation
│   └── test_mms.py               # MMS verification
├── convergence/
│   ├── test_spatial_convergence.py
│   ├── test_temporal_convergence.py
│   └── test_cutcell_convergence.py
├── regression/
│   ├── test_vs_cpp_reference.py
│   └── reference_data/           # JSON files from C++
│       ├── stencils/
│       ├── operators/
│       └── systems/
└── conftest.py                   # Pytest configuration
```

### 5.2 Test Execution

#### 5.2.1 Continuous Integration (CI)

**Fast Suite** (run on every commit, ~5 min):
```bash
pytest tests/unit -v --tb=short
pytest tests/integration -k "not slow" -v
```
- All unit tests
- Quick integration tests (small grids)
- Regression tests vs C++ reference

**Nightly Suite** (~30 min):
```bash
pytest tests/ -v --tb=short --durations=20
```
- Full test suite
- Convergence studies (multiple grids)
- Performance benchmarks

**Weekly Suite** (~2 hours):
```bash
pytest tests/convergence -v --grid-sizes="[64,128,256]"
```
- High-resolution convergence studies
- Extensive parameter sweeps
- Stability boundary searches

#### 5.2.2 Test Markers

```python
@pytest.mark.unit              # Unit test (fast)
@pytest.mark.integration       # Integration test
@pytest.mark.convergence       # Convergence study (slow)
@pytest.mark.regression        # Regression vs C++
@pytest.mark.slow              # Exclude from fast suite
@pytest.mark.parametrize       # Parameter sweep
```

### 5.3 Validation Tools

#### 5.3.1 Error Analysis Utilities

```python
# tests/utils/error_analysis.py

def compute_convergence_rate(errors, grid_sizes):
    """Compute convergence rate via least-squares fit."""
    log_h = np.log(grid_sizes)
    log_err = np.log(errors)
    p, _ = np.polyfit(log_h, log_err, 1)
    return -p  # Negative slope = convergence rate

def verify_order(errors, grid_sizes, expected_order, tol=0.15):
    """Check if convergence rate matches expected order."""
    rate = compute_convergence_rate(errors, grid_sizes)
    assert abs(rate - expected_order) < tol, \
        f"Convergence rate {rate:.3f} not within {expected_order}±{tol}"
    return rate

def plot_convergence(errors, grid_sizes, expected_order, save_path):
    """Generate convergence plot with reference line."""
    import matplotlib.pyplot as plt
    plt.loglog(grid_sizes, errors, 'o-', label='Measured')
    plt.loglog(grid_sizes, errors[0] * (grid_sizes/grid_sizes[0])**expected_order,
               '--', label=f'Order {expected_order}')
    plt.xlabel('Grid size h')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(save_path)
```

#### 5.3.2 Reference Data Management

```python
# tests/utils/reference_data.py

class ReferenceData:
    """Manage C++ reference data for regression testing."""
    
    def __init__(self, data_dir="tests/regression/reference_data"):
        self.data_dir = Path(data_dir)
    
    def load(self, test_name):
        """Load reference data for a test case."""
        path = self.data_dir / f"{test_name}.json"
        with open(path) as f:
            return json.load(f)
    
    def compare(self, result, reference, rtol=1e-12, atol=1e-14):
        """Compare result against reference data."""
        if isinstance(reference, dict):
            for key in reference:
                self.compare(result[key], reference[key], rtol, atol)
        elif isinstance(reference, (list, np.ndarray)):
            np.testing.assert_allclose(result, reference, 
                                       rtol=rtol, atol=atol)
        else:
            assert abs(result - reference) < atol
```

#### 5.3.3 Visualization Tools

```python
# tests/utils/visualization.py

def visualize_field(field, mesh, title="Field", save_path=None):
    """Plot 2D slice or 3D isosurface of field."""
    # For 3D: plot mid-plane slice
    # For 2D: contour plot
    # Mark cut-cell boundaries
    pass

def visualize_error(computed, exact, mesh, save_path=None):
    """Plot error distribution."""
    error = np.abs(computed - exact)
    # Highlight high-error regions
    # Mark cut-cell locations
    pass

def visualize_convergence_study(results, save_dir):
    """Generate full convergence study report."""
    # Multiple plots: L2 error, Linf error, convergence rate
    # Tables with numerical values
    # Save as PDF report
    pass
```

### 5.4 Automated Reporting

**Test Report Generation:**
```bash
pytest tests/ --html=report.html --self-contained-html
```

**Convergence Study Report:**
```python
# Automatically generates LaTeX/PDF report with:
# - Convergence plots
# - Error tables
# - Comparison with C++ reference
# - Performance metrics
```

---

## 6. Performance Benchmarks

**Purpose:** Ensure Python implementation achieves acceptable performance for production use.

### 6.1 Baseline Performance Targets

**Goal:** Python should be within 2-5× of C++ performance after optimization.

| Operation | C++ Time (reference) | Python Target | Notes |
|-----------|---------------------|---------------|-------|
| Stencil coefficient computation | 1.0× | 1.0× | One-time setup, not critical |
| Sparse matrix assembly | 1.0× | 2.0× | Dominated by data structure overhead |
| Matrix-vector product (CSR) | 1.0× | 1.2× | Use SciPy/Numba optimized kernels |
| Operator application (full) | 1.0× | 2.5× | Includes boundary handling |
| RK4 timestep | 1.0× | 3.0× | Acceptable for iterative development |
| Full simulation (1000 steps) | 1.0× | 4.0× | Wall-clock time |

### 6.2 Performance Test Cases

#### 6.2.1 Microbenchmarks

```python
@pytest.mark.benchmark
def test_bench_stencil_E2_1(benchmark):
    """Benchmark E2_1 stencil coefficient computation."""
    psi = 0.5
    alpha = [0.1, 0.2, 0.3, 0.4]
    h = 0.01
    result = benchmark(compute_E2_1_stencil, psi, alpha, h)

@pytest.mark.benchmark
def test_bench_csr_matvec(benchmark):
    """Benchmark CSR matrix-vector product."""
    n = 100000
    matrix = create_laplacian_matrix_1d(n)
    x = np.random.rand(n)
    result = benchmark(matrix.dot, x)
```

**Tools:** `pytest-benchmark` plugin

#### 6.2.2 System-Level Benchmarks

**Test Case:** Heat equation on 64³ grid, 1000 timesteps
```python
@pytest.mark.slow
@pytest.mark.benchmark
def test_bench_heat_equation_3d():
    mesh = create_mesh([64, 64, 64])
    system = HeatSystem(mesh, diffusivity=0.1)
    u0 = initialize_field(mesh)
    
    start = time.time()
    for n in range(1000):
        u = system.step(u0, dt=0.001)
    wall_time = time.time() - start
    
    print(f"Wall time: {wall_time:.2f} s")
    print(f"Time per step: {wall_time/1000:.4f} s")
    
    # Record for comparison with C++
    record_benchmark("heat_3d_64cubed_1000steps", wall_time)
```

### 6.3 Performance Profiling

**Tools:**
- `cProfile` for Python-level profiling
- `line_profiler` for line-by-line analysis
- `py-spy` for sampling profiler (low overhead)

**Workflow:**
1. Profile representative test case
2. Identify hotspots (>5% of runtime)
3. Optimize with Numba JIT or Cython
4. Verify correctness after optimization
5. Re-measure and iterate

**Optimization Targets (in priority order):**
1. Matrix-vector products (use SciPy optimized)
2. Stencil coefficient computation (Numba JIT)
3. Boundary condition application (vectorize)
4. Field operations (NumPy broadcasting)

### 6.4 Scalability Tests

**Strong Scaling:** Fixed problem size, vary resources
- Not primary concern for serial Python code
- Future: test with multiprocessing/Dask

**Weak Scaling:** Problem size grows with grid refinement
```python
def test_weak_scaling():
    grid_sizes = [32, 64, 128]
    for n in grid_sizes:
        mesh = create_mesh([n, n, n])
        time_per_dof = benchmark_heat_equation(mesh, nsteps=100)
        # Should remain approximately constant
        print(f"N={n}: {time_per_dof:.6f} s/DOF")
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- ✅ Set up test framework structure
- ✅ Implement reference data extraction from C++
- ✅ Create basic field and mesh data structures
- 🔲 Unit tests for geometry (Level 0.1)

### Phase 2: Core Operators (Weeks 3-4)
- 🔲 Implement stencil coefficient computation
- 🔲 Unit tests for stencils (Level 0.2)
- 🔲 Implement derivative, laplacian, gradient operators
- 🔲 Operator unit tests (Level 1)
- 🔲 Regression tests vs C++ reference

### Phase 3: Time Integration (Weeks 5-6)
- 🔲 Implement RK4, Euler integrators
- 🔲 Implement Heat and Wave systems
- 🔲 Integration tests (Level 2)
- 🔲 MMS verification tests

### Phase 4: Convergence Validation (Weeks 7-8)
- 🔲 Spatial convergence studies (Level 3.1)
- 🔲 Temporal convergence studies (Level 3.2)
- 🔲 Cut-cell geometry convergence (Level 3.3)
- 🔲 Stability analysis (Section 4.1)
- 🔲 Conservation property tests (Section 4.2)

### Phase 5: Performance Optimization (Weeks 9-10)
- 🔲 Profile and optimize hotspots
- 🔲 Numba JIT for stencil computation
- 🔲 Benchmark suite (Section 6)
- 🔲 Performance comparison report

### Phase 6: Documentation (Week 11)
- 🔲 Validation report with all test results
- 🔲 Convergence study plots and tables
- 🔲 Performance comparison analysis
- 🔲 User guide for running validation suite

---

## 8. Success Criteria

### 8.1 Minimum Viable Product (MVP)

**Must Have:**
- ✅ All unit tests pass (Levels 0-1)
- ✅ All integration tests pass (Level 2)
- ✅ Spatial convergence achieves theoretical order ± 0.15
- ✅ Regression tests match C++ within specified tolerances
- ✅ No stability issues for standard test cases
- ✅ Performance within 5× of C++ for reference benchmarks

**Good to Have:**
- Temporal convergence verification
- Cut-cell geometry convergence study
- Conservation property verification
- Performance within 3× of C++

**Nice to Have:**
- Extensive parameter sweeps
- Parallel execution support
- Automated LaTeX report generation

### 8.2 Scientific Publication Readiness

For publishing results using the Python implementation:
1. **Full convergence studies** demonstrating theoretical order
2. **Regression tests** showing equivalence to published C++ results
3. **Stability analysis** for all problem configurations
4. **Conservation** verification for conservative PDEs
5. **Performance** sufficient for production simulations

### 8.3 Production Readiness

For using Python implementation in production research:
1. All MVP criteria met
2. Extensive test coverage (>90% for core components)
3. Performance benchmarks documented
4. User documentation complete
5. CI/CD pipeline operational
6. Version control and release process established

---

## 9. Risk Mitigation

### 9.1 Numerical Risks

| Risk | Mitigation |
|------|------------|
| Floating-point differences between Python and C++ | Use IEEE 754 compliant libraries; document acceptable deviations |
| Library version incompatibilities | Pin all dependencies; test with multiple NumPy/SciPy versions |
| Optimization breaks correctness | Require all tests pass after each optimization |
| Cut-cell instabilities | Extensive psi-sweep tests; condition number monitoring |

### 9.2 Development Risks

| Risk | Mitigation |
|------|------------|
| Complex C++ code translation errors | Incremental migration; validate each component |
| Test suite maintenance burden | Automate reference data extraction; use parametrized tests |
| Performance unacceptable | Profile early; identify optimization targets; use Numba/Cython |
| Scope creep | Focus on MVP first; defer nice-to-have features |

---

## 10. References

### 10.1 Scientific References
- Brady & Livescu (2021). "Foundations for high-order, conservative cut-cell methods." *Journal of Computational Physics* 426:109794.
- Relevant sections of C++ codebase test files (*.t.cpp)

### 10.2 Software Tools
- **Testing:** pytest, pytest-benchmark, pytest-html
- **Numerical:** NumPy, SciPy, Numba
- **Validation:** JSON (reference data), matplotlib (visualization)
- **Performance:** cProfile, line_profiler, py-spy

---

## Appendix A: Example Test Case

```python
# tests/unit/test_derivative_operator.py

import pytest
import numpy as np
from shoccs import Mesh, Field, DerivativeOperator, Stencil, BoundaryConditions

class TestDerivativeE2:
    """Test suite for 2nd-order derivative operator."""
    
    @pytest.fixture
    def mesh(self):
        """Standard test mesh from derivative.t.cpp."""
        return Mesh(
            extents=[10, 13, 17],
            domain_min=[0.1, 0.2, 0.3],
            domain_max=[1.0, 2.0, 2.2]
        )
    
    def test_E2_polynomial_reproduction(self, mesh):
        """Verify E2 scheme exactly reproduces quadratic polynomial."""
        # Test function: f(x,y,z) = x*(y+z) + y*(x+z) + z*(x+y) + 3*x*y*z
        def f(xyz):
            x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
            return x*(y+z) + y*(x+z) + z*(x+y) + 3*x*y*z
        
        def dfdx_exact(xyz):
            x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
            return 2*(y+z) + 3*y*z
        
        def d2fdx2_exact(xyz):
            return np.zeros_like(xyz[..., 0])  # Should be exactly zero
        
        # Initialize field
        u = Field.from_function(mesh, f)
        
        # Apply operator
        stencil = Stencil.E2(order=2)
        bcs = BoundaryConditions(grid="FFFFFF")  # All periodic/free
        op = DerivativeOperator(direction=0, mesh=mesh, 
                                stencil=stencil, bcs=bcs)
        
        # First derivative
        dudx = op.apply(u)
        exact_dudx = Field.from_function(mesh, dfdx_exact)
        
        error_1st = np.max(np.abs(dudx.data - exact_dudx.data))
        assert error_1st < 1e-12, \
            f"First derivative error {error_1st} exceeds tolerance"
        
        # Second derivative
        d2udx2 = op.apply(dudx)
        exact_d2udx2 = Field.from_function(mesh, d2fdx2_exact)
        
        error_2nd = np.max(np.abs(d2udx2.data - exact_d2udx2.data))
        assert error_2nd < 1e-13, \
            f"Second derivative error {error_2nd} exceeds tolerance (should be zero)"
    
    @pytest.mark.regression
    def test_vs_cpp_reference(self, mesh):
        """Compare against C++ reference data."""
        from shoccs.testing import ReferenceData
        
        ref = ReferenceData()
        ref_data = ref.load("derivative_E2_FFFFFF")
        
        # Apply operator with same inputs
        u = Field(mesh, data=ref_data["inputs"]["field_values"])
        op = DerivativeOperator.from_reference(ref_data["operator_config"])
        result = op.apply(u)
        
        # Compare
        ref.compare(result.data, ref_data["outputs"]["result"], 
                   rtol=1e-12, atol=1e-14)
```

---

## Appendix B: Convergence Study Example

```python
# tests/convergence/test_spatial_convergence.py

import pytest
import numpy as np
from shoccs import Mesh, Field, HeatSystem, MMS

@pytest.mark.convergence
@pytest.mark.slow
class TestSpatialConvergence:
    """Spatial convergence studies for heat equation."""
    
    def test_heat_2d_convergence_E2(self):
        """2D heat equation convergence study with MMS."""
        
        # Manufactured solution
        def u_exact(t, x, y):
            return np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y)
        
        def source(t, x, y):
            kappa = 0.1
            dudt = -np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y)
            laplacian = -2*np.pi**2 * np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y)
            return dudt - kappa * laplacian
        
        # Grid sequence
        grid_sizes = [16, 32, 64, 128]
        errors_l2 = []
        errors_linf = []
        
        for n in grid_sizes:
            # Setup
            mesh = Mesh.cartesian_2d(n, n, domain=[0, 1, 0, 1])
            system = HeatSystem(
                mesh, 
                diffusivity=0.1,
                source=lambda t, xyz: source(t, xyz[:,0], xyz[:,1])
            )
            
            # Initial condition
            u = Field.from_function(mesh, lambda xyz: u_exact(0, xyz[:,0], xyz[:,1]))
            
            # Time-step with small dt to minimize temporal error
            dt = 0.5 * (mesh.h**3)  # O(h^3) temporal error for RK4
            t_final = 0.1
            nsteps = int(np.ceil(t_final / dt))
            
            for _ in range(nsteps):
                u = system.rk4_step(u, dt)
            
            # Compute error
            u_ex = Field.from_function(mesh, 
                                       lambda xyz: u_exact(t_final, xyz[:,0], xyz[:,1]))
            error = u.data - u_ex.data
            
            # Norms (weighted by cell volume)
            vol = mesh.cell_volume
            l2_error = np.sqrt(np.sum(error**2 * vol))
            linf_error = np.max(np.abs(error))
            
            errors_l2.append(l2_error)
            errors_linf.append(linf_error)
        
        # Compute convergence rate
        from shoccs.testing import compute_convergence_rate, plot_convergence
        
        h_values = 1.0 / np.array(grid_sizes)
        rate_l2 = compute_convergence_rate(errors_l2, h_values)
        rate_linf = compute_convergence_rate(errors_linf, h_values)
        
        # Verify order
        expected_order = 2.0
        tolerance = 0.15
        
        assert abs(rate_l2 - expected_order) < tolerance, \
            f"L2 convergence rate {rate_l2:.3f} not within {expected_order}±{tolerance}"
        
        assert abs(rate_linf - expected_order) < tolerance, \
            f"Linf convergence rate {rate_linf:.3f} not within {expected_order}±{tolerance}"
        
        # Generate plots
        plot_convergence(errors_l2, h_values, expected_order, 
                        save_path="convergence_heat_2d_E2_l2.pdf")
        plot_convergence(errors_linf, h_values, expected_order,
                        save_path="convergence_heat_2d_E2_linf.pdf")
```

---

**Document Status:** Draft for Review  
**Last Updated:** 2025-11-13  
**Authors:** Generated for SHOCCS Python Migration Project  
**Next Review:** Upon completion of Phase 1 implementation
