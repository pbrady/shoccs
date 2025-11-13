# SHOCCS Python Migration: Framework Analysis and Recommendation

**Date:** 2025-11-13
**Prepared for:** SHOCCS Cut-Cell PDE Solver Python Migration
**Author:** Technical Analysis based on Codebase Review and Framework Research

---

## Executive Summary

**RECOMMENDATION: NumPy/SciPy + Numba with Optional JAX for Alpha Optimization**

After comprehensive analysis of the SHOCCS codebase and evaluation of available Python frameworks, **JAX is NOT recommended as the primary framework** for the core solver implementation. Instead, a **hybrid approach** using NumPy/SciPy with Numba JIT compilation is recommended, with JAX reserved only for potential alpha parameter optimization if that proves valuable.

### Key Decision Factors

1. **Sparse Matrix Operations:** SHOCCS relies heavily on mature CSR matrix operations. JAX's sparse support is experimental and has severe autodiff limitations.

2. **Static Geometry:** All stencil coefficients are computed once at initialization. There's minimal benefit from JAX's autodiff during time-stepping.

3. **Irregular Stencils:** The 2509-line E2_1 stencil with complex polynomial expressions would suffer from JAX's compilation overhead and static shape requirements.

4. **Mature Requirements:** This is production research code, not experimental ML code. SciPy's 18+ years of sparse matrix optimization outweigh JAX's experimental features.

---

## Detailed Framework Analysis

### 1. JAX - **NOT RECOMMENDED** for Core Solver

#### Strengths
- ✅ Excellent autodiff for optimization problems
- ✅ Good JIT compilation for regular, dense array operations
- ✅ Strong functional programming paradigm
- ✅ PyTree system handles nested data structures well

#### Critical Weaknesses for SHOCCS

**❌ Experimental Sparse Matrix Support**
- `jax.experimental.sparse` is explicitly marked as experimental
- CSR format has limited operations (no vmap support)
- Backward rules return **dense gradients** for sparse inputs
- Cannot compute derivatives with respect to CSR matrices directly
- Most sparse operations fall back to COO (slower than CSR)

**❌ Memory and Performance Issues**
Research findings (GitHub Discussion #10559):
> "JAX does not seem to be able to implement automatic differentiation of sparse matrices yet, so full matrices need to be used. The memory and computational overhead brought by this are very expensive."

**❌ Compilation Overhead for Complex Stencils**
- Compilation cost grows **quadratically** with number of operations
- E2_1 stencil (2509 lines, ~5000 FLOPs) would trigger massive recompilation
- Geometry-dependent coefficients (psi, alpha) create many static argument combinations
- First execution would have unacceptable latency

**❌ Limited Benefit from Autodiff**
- Stencil coefficients computed **once at initialization**, not in time-stepping loop
- RK4 integrator uses simple accumulation patterns (no need for autodiff)
- Alpha parameter optimization is a **one-time offline task**, not runtime requirement

#### Code Example - What JAX Would Look Like (NOT Recommended)

```python
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsp

# Problem 1: CSR matrix operations with autodiff don't work well
def apply_derivative_operator(field_values, O_matrix_csr):
    """This would be problematic in JAX"""
    # CSR operations don't support gradient computation
    # Would need to convert to dense or use BCOO format
    result = jsp.csr_matvec(O_matrix_csr, field_values)  # No autodiff through CSR
    return result

# Problem 2: Complex stencil with geometry-dependent coefficients
@jax.jit
def compute_E2_1_coefficients(psi, alpha, h):
    """2509 lines of polynomial expressions - massive compilation overhead"""
    # Every new (psi, alpha) combination triggers recompilation
    # Static shape requirements make this inflexible
    t3 = alpha[0]
    t5 = alpha[2]
    t17 = -1 + psi
    # ... 2500 more lines ...
    # Quadratic compilation time in number of operations
    return coefficients

# Problem 3: Custom PyTree for field structure
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Field:
    """Would need custom PyTree registration"""
    def __init__(self, D, Rx, Ry, Rz):
        self.D = D    # Domain interior
        self.Rx = Rx  # X boundary region
        self.Ry = Ry  # Y boundary region
        self.Rz = Rz  # Z boundary region

    def tree_flatten(self):
        children = (self.D, self.Rx, self.Ry, self.Rz)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
```

---

### 2. PyTorch - **NOT RECOMMENDED**

#### Strengths
- ✅ Mature ecosystem
- ✅ Good autodiff for optimization
- ✅ Strong GPU support

#### Critical Weaknesses

**❌ No Autodiff for CSR Matrices**
PyTorch documentation explicitly states:
> "This function doesn't support computing derivatives with respect to CSR matrices."

**❌ Slower for Scientific Computing**
Research findings indicate PyTorch is "much slower" than JAX for differential equations due to Python interpreter overhead.

**❌ Machine Learning Focus**
- API design prioritizes neural networks, not PDE solvers
- Less intuitive for scientific computing compared to NumPy semantics
- Heavier dependency (larger installation, more complexity)

---

### 3. **NumPy/SciPy + Numba - RECOMMENDED (Primary Framework)**

#### Why This is the Right Choice

**✅ Mature Sparse Matrix Ecosystem**
- `scipy.sparse.csr_matrix` has 18+ years of optimization
- CSR format is 2.88x faster than dense for sparse operations
- Excellent performance for matrix-vector products
- Proven in production scientific computing codes

**✅ Numba JIT Eliminates Performance Concerns**
- JIT compilation for performance-critical loops
- Works seamlessly with NumPy arrays
- Can parallelize with `@njit(parallel=True)`
- Zero Python overhead for compiled functions

**✅ Perfect Match for SHOCCS Architecture**
- Static geometry: compute stencil coefficients once, use many times
- Explicit time-stepping: simple accumulation patterns
- CSR matrices for boundary coupling: SciPy's strength
- Circulant matrices for interior: efficient with NumPy's FFT

**✅ Numba Sparse Matrix Support**
Pass CSR components directly to JIT-compiled functions:

```python
from numba import njit
import numpy as np
from scipy.sparse import csr_matrix

@njit
def csr_matvec_numba(data, indptr, indices, x, result):
    """Fast CSR matrix-vector product"""
    n_rows = len(indptr) - 1
    for i in range(n_rows):
        tmp = 0.0
        for j in range(indptr[i], indptr[i+1]):
            tmp += data[j] * x[indices[j]]
        result[i] = tmp

# Usage
A = csr_matrix(...)  # Boundary coupling matrix
result = np.zeros(A.shape[0])
csr_matvec_numba(A.data, A.indptr, A.indices, x, result)
```

**✅ No Compilation Overhead for Complex Stencils**
- Compile stencil coefficient generation once
- Store coefficients as NumPy arrays
- Reuse without recompilation regardless of geometry parameters

#### Recommended Data Structure

```python
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix

@dataclass
class Field:
    """Cut-cell field with domain and boundary regions"""
    D: np.ndarray   # Domain interior points
    Rx: np.ndarray  # X-direction boundary points
    Ry: np.ndarray  # Y-direction boundary points
    Rz: np.ndarray  # Z-direction boundary points

    def __add__(self, other):
        """Field arithmetic"""
        if isinstance(other, Field):
            return Field(
                self.D + other.D,
                self.Rx + other.Rx,
                self.Ry + other.Ry,
                self.Rz + other.Rz
            )
        else:  # scalar
            return Field(
                self.D + other,
                self.Rx + other,
                self.Ry + other,
                self.Rz + other
            )

    def __mul__(self, scalar):
        """Scalar multiplication"""
        return Field(
            self.D * scalar,
            self.Rx * scalar,
            self.Ry * scalar,
            self.Rz * scalar
        )

    __rmul__ = __mul__

@dataclass
class DerivativeOperator:
    """Discrete derivative operator for cut-cell mesh"""
    O_interior: csr_matrix     # Interior operator (could be circulant)
    B_boundary: csr_matrix     # Boundary coupling
    N_neumann: csr_matrix      # Neumann BC operator
    Bfx: csr_matrix           # Boundary operators for Rx/Ry/Rz
    Brx: csr_matrix
    Bfy: csr_matrix
    Bry: csr_matrix
    Bfz: csr_matrix
    Brz: csr_matrix
    interior_c: np.ndarray    # Interior stencil coefficients

    def apply(self, field: Field) -> Field:
        """Apply derivative operator"""
        result_D = self.O_interior @ field.D
        result_D += self.B_boundary @ np.concatenate([field.Rx, field.Ry, field.Rz])

        result_Rx = self.Bfx @ field.D + self.Brx @ field.Rx
        result_Ry = self.Bfy @ field.D + self.Bry @ field.Ry
        result_Rz = self.Bfz @ field.D + self.Brz @ field.Rz

        return Field(result_D, result_Rx, result_Ry, result_Rz)
```

#### Stencil Coefficient Generation

```python
from numba import njit
import numpy as np

class E2_1_Stencil:
    """Second-order accurate stencil with 4 alpha parameters"""

    def __init__(self, alpha: np.ndarray):
        """
        Args:
            alpha: Array of 4 tuning parameters for accuracy/stability
        """
        self.alpha = np.asarray(alpha, dtype=np.float64)
        if len(self.alpha) < 4:
            self.alpha = np.pad(self.alpha, (0, 4 - len(self.alpha)))

    def interior_coefficients(self, h: float) -> np.ndarray:
        """Interior stencil coefficients (uniform grid)"""
        return np.array([-1/(2*h), 0, 1/(2*h)])

    def nbs_floating_coefficients(self, h: float, psi: float) -> np.ndarray:
        """
        Near-boundary stencil for floating BC

        This is the 2509-line monster. Compile once, use many times.
        No recompilation needed for different psi/alpha values.
        """
        return _compute_E2_1_floating(h, psi, self.alpha)

@njit
def _compute_E2_1_floating(h: float, psi: float, alpha: np.ndarray) -> np.ndarray:
    """
    Compiled once, then reused. Fast function call overhead.
    Translates the 2509 lines of C++ polynomial expressions.
    """
    t3 = alpha[0]
    t5 = alpha[2]
    t17 = -1 + psi
    t11 = -psi
    t22 = alpha[1]
    t9 = 2 * t5
    t24 = alpha[3]
    # ... 2500 more lines of polynomial expressions ...
    # These are just arithmetic operations - Numba handles them efficiently
    # No compilation overhead per-call

    coeffs = np.zeros(20)  # R * T = 4 * 5
    coeffs[0] = # ... computed value
    # ... fill in all coefficients
    return coeffs
```

#### Time Integration

```python
class RK4Integrator:
    """Classic RK4 time integration"""

    rki = np.array([0.0, 0.5, 0.5, 1.0])
    rkf = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])

    def __init__(self, system_size: tuple):
        """Pre-allocate work arrays"""
        self.rk_rhs = Field.zeros(system_size)
        self.system_rhs = Field.zeros(system_size)

    def step(self, system, u0: Field, dt: float, time: float) -> Field:
        """Take one RK4 step"""
        # Simple accumulation - no need for autodiff
        self.rk_rhs = Field.zeros_like(u0)
        u = u0

        for i in range(4):
            if i > 0:
                u = u0 + dt * self.rki[i] * self.system_rhs
                system.update_boundary(u, time + dt * self.rki[i])

            self.system_rhs = system.rhs(u, time + dt * self.rki[i])
            self.rk_rhs = self.rk_rhs + dt * self.rkf[i] * self.system_rhs

        u = u0 + self.rk_rhs
        system.update_boundary(u, time + dt)
        return u
```

---

### 4. CasADi - **Specialized Use Case Only**

#### Strengths
- ✅ Excellent symbolic differentiation
- ✅ Sparse Jacobian/Hessian computation
- ✅ Interfaces with optimization solvers (IPOPT, etc.)

#### Limitations
- ❌ Not a general-purpose array library
- ❌ Requires symbolic model construction
- ❌ Overkill for explicit time-stepping

#### Recommended Use Case
**Only if optimizing alpha parameters becomes a priority:**

```python
import casadi as ca

def optimize_alpha_parameters(target_accuracy, stability_constraint):
    """
    Use CasADi only for alpha optimization - not for the solver itself
    """
    # Symbolic alpha parameters
    alpha = ca.MX.sym('alpha', 4)

    # Define accuracy metric as function of alpha
    # This would involve eigenvalue analysis of stencil operator
    accuracy = ca.Function('accuracy', [alpha], [compute_spectral_radius(alpha)])

    # Optimization problem
    nlp = {
        'x': alpha,
        'f': accuracy(alpha),  # Minimize error
        'g': stability_constraint(alpha)
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp)
    solution = solver(x0=[0.1, 0.1, 0.1, 0.1])

    return solution['x']
```

---

## SHOCCS-Specific Implementation Strategy

### Phase 1: Direct Translation (NumPy/SciPy + Numba)

**Timeline:** 2-3 months for core functionality

1. **Data Structures**
   - `Field` class with D/Rx/Ry/Rz layout
   - Keep CSR matrices for boundary operators
   - Use NumPy arrays for circulant interior operators

2. **Stencil Translation**
   - Translate E2_1.cpp polynomial expressions line-by-line
   - Use `@njit` decorator for coefficient computation
   - Store coefficients as NumPy arrays
   - No recompilation needed for different geometries

3. **Operator Assembly**
   - Use `scipy.sparse.csr_matrix` builder
   - Build operators at initialization (same as C++)
   - Store as CSR matrices for efficient matvec

4. **Time Integration**
   - Direct translation of RK4/Euler integrators
   - Simple field arithmetic (no autodiff needed)
   - Numba JIT for performance-critical loops if needed

### Phase 2: Validation and Optimization

**Timeline:** 1-2 months

1. **Numerical Verification**
   - Compare against C++ reference solutions
   - Method of Manufactured Solutions tests
   - Convergence studies

2. **Performance Tuning**
   - Profile with cProfile/line_profiler
   - Add Numba JIT to hot loops
   - Parallelize with `@njit(parallel=True)` if needed
   - Consider sparse matrix formats (CSR vs BSR vs block)

3. **Developer Experience**
   - Clean Python API
   - Interactive visualization (matplotlib)
   - Jupyter notebooks for exploration

### Phase 3: (Optional) Alpha Parameter Optimization

**Timeline:** 1-2 weeks if pursued

1. **Use JAX or CasADi**
   - Separate module for alpha optimization
   - Not integrated into main solver
   - Run offline to generate optimal alpha values

2. **Approach**
   - Option A: JAX for eigenvalue analysis of stencil operators
   - Option B: CasADi for symbolic optimization with constraints
   - Feed results back into main NumPy/SciPy solver

---

## Performance Comparison Estimate

Based on research and similar migration projects:

| Framework | Relative Performance | Maturity | Ease of Use |
|-----------|---------------------|----------|-------------|
| C++ (baseline) | 1.0x | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| NumPy/SciPy + Numba | 0.5x - 0.8x | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| JAX (sparse ops) | 0.3x - 0.6x* | ⭐⭐ | ⭐⭐⭐ |
| PyTorch | 0.4x - 0.7x | ⭐⭐⭐⭐ | ⭐⭐⭐ |

*JAX performance suffers from sparse matrix limitations and compilation overhead

**Key Insight:** The 20-50% performance loss with NumPy/SciPy + Numba is acceptable given:
- Dramatically faster development iteration
- Interactive exploration capabilities
- Easier debugging and visualization
- No loss in numerical accuracy

---

## Code Migration Example: Derivative Operator

### Current C++ (simplified)
```cpp
// src/operators/derivative.cpp
void derivative::operator()(scalar_view field, scalar_span result) const {
    // Interior operation (circulant or CSR)
    O(field.D(), result.D());

    // Boundary coupling
    B(concat(field.Rx(), field.Ry(), field.Rz()), result.D());

    // Boundary region updates
    Bfx(field.D(), result.Rx());
    Brx(field.Rx(), result.Rx());
    // ... similar for Ry, Rz
}
```

### Proposed Python (NumPy/SciPy + Numba)
```python
# src/operators/derivative.py
from dataclasses import dataclass
from scipy.sparse import csr_matrix
import numpy as np

@dataclass
class DerivativeOperator:
    """Discrete derivative operator"""
    O: csr_matrix
    B: csr_matrix
    Bfx: csr_matrix
    Brx: csr_matrix
    Bfy: csr_matrix
    Bry: csr_matrix
    Bfz: csr_matrix
    Brz: csr_matrix

    def __call__(self, field: Field) -> Field:
        """Apply derivative operator"""
        # Interior operation
        result_D = self.O @ field.D

        # Boundary coupling
        boundary_concat = np.concatenate([field.Rx, field.Ry, field.Rz])
        result_D += self.B @ boundary_concat

        # Boundary region updates
        result_Rx = self.Bfx @ field.D + self.Brx @ field.Rx
        result_Ry = self.Bfy @ field.D + self.Bry @ field.Ry
        result_Rz = self.Bfz @ field.D + self.Brz @ field.Rz

        return Field(result_D, result_Rx, result_Ry, result_Rz)

# If performance is critical, wrap the matvec operations:
from numba import njit

@njit
def apply_derivative_numba(
    O_data, O_indices, O_indptr,
    B_data, B_indices, B_indptr,
    field_D, boundary_concat, result_D
):
    """Numba-optimized derivative application"""
    # CSR matvec for O
    csr_matvec(O_data, O_indices, O_indptr, field_D, result_D)

    # CSR matvec for B (accumulate)
    csr_matvec_accumulate(B_data, B_indices, B_indptr,
                         boundary_concat, result_D)
```

---

## Addressing Common Concerns

### "Won't Python be too slow?"

**Answer:** Not with Numba and mature SciPy sparse matrices.

- SciPy CSR operations are highly optimized C code (18+ years of development)
- Numba JIT compiles hot loops to machine code (same speed as C++)
- Most time spent in sparse matrix operations (already C code)
- Field arithmetic overhead is minimal with Numba

### "What about autodiff for parameter optimization?"

**Answer:** Use targeted autodiff tools only where needed.

- Stencil coefficients computed once at initialization (no need for runtime autodiff)
- If alpha optimization is needed, use JAX or CasADi in separate offline module
- Don't pay autodiff complexity cost for simple explicit time-stepping

### "JAX is more modern and trendy"

**Answer:** Choose tools for the problem, not the hype.

- SHOCCS is production research code, not an ML experiment
- Sparse matrix operations are core to the algorithm
- JAX's experimental sparse support has known limitations
- SciPy's maturity and stability are more valuable here

### "Won't we lose GPU acceleration?"

**Answer:** GPU acceleration is premature optimization.

1. **Get Python version working first** with NumPy/SciPy + Numba (CPU)
2. **Profile and validate** correctness and reasonable performance
3. **If GPU is needed later**, options include:
   - CuPy (drop-in NumPy/SciPy replacement for GPU)
   - JAX for specific GPU kernels (not whole solver)
   - Keep CPU version as reference implementation

For explicit PDE solvers, CPU performance is often sufficient, and memory bandwidth is usually the bottleneck (not compute).

---

## Migration Risk Assessment

### NumPy/SciPy + Numba (RECOMMENDED)
- **Technical Risk:** 🟢 LOW - Mature ecosystem, well-documented
- **Performance Risk:** 🟡 MEDIUM - 20-50% slower than C++, but acceptable
- **Maintenance Risk:** 🟢 LOW - Standard scientific Python stack
- **Learning Curve:** 🟢 LOW - Familiar to Python developers

### JAX (NOT RECOMMENDED for core solver)
- **Technical Risk:** 🔴 HIGH - Experimental sparse support, autodiff limitations
- **Performance Risk:** 🔴 HIGH - Compilation overhead, dense gradient memory issues
- **Maintenance Risk:** 🟡 MEDIUM - API changes likely in experimental features
- **Learning Curve:** 🟡 MEDIUM - Functional paradigm, PyTrees, static shapes

### PyTorch
- **Technical Risk:** 🔴 HIGH - No CSR autodiff, ML-focused API
- **Performance Risk:** 🟡 MEDIUM - Slower than JAX for scientific computing
- **Maintenance Risk:** 🟢 LOW - Large community, stable API
- **Learning Curve:** 🟡 MEDIUM - Tensor paradigm differs from NumPy

---

## Recommended Action Plan

### Immediate Next Steps (Week 1-2)

1. **Set up Python environment**
   ```bash
   conda create -n shoccs-python python=3.11
   conda install numpy scipy numba matplotlib pytest
   ```

2. **Create core data structures**
   - `Field` class with D/Rx/Ry/Rz
   - Mesh class with geometry info
   - Test with simple examples

3. **Translate one simple stencil**
   - Start with interior stencil (3-point centered difference)
   - Verify coefficients match C++ version
   - Add Numba JIT and benchmark

### Short Term (Month 1-2)

4. **Translate E2_1 stencil**
   - Line-by-line translation of polynomial expressions
   - Careful validation against C++ reference
   - Store coefficients for reuse

5. **Build derivative operator**
   - CSR matrix assembly
   - Boundary coupling logic
   - Test on simple geometries

6. **Implement RK4 integrator**
   - Field arithmetic
   - System RHS evaluation
   - Time-stepping loop

### Medium Term (Month 3-4)

7. **Full system tests**
   - Method of Manufactured Solutions
   - Convergence studies
   - Comparison with C++ results

8. **Performance optimization**
   - Profile bottlenecks
   - Add Numba JIT where needed
   - Consider parallel execution

9. **Developer tools**
   - Visualization utilities
   - Jupyter notebooks
   - Documentation

### Optional (Month 5+)

10. **Alpha parameter optimization** (if valuable)
    - Separate module using JAX or CasADi
    - Eigenvalue analysis
    - Constraint optimization
    - Feed results to main solver

---

## Conclusion

For the SHOCCS cut-cell PDE solver migration, **NumPy/SciPy + Numba is the clear winner**. This approach:

✅ Leverages mature, production-tested sparse matrix operations
✅ Provides Numba JIT for performance-critical code
✅ Matches SHOCCS architecture (static geometry, explicit time-stepping)
✅ Offers easy debugging and visualization
✅ Minimizes technical risk
✅ Accelerates research iteration cycles

**JAX should be avoided for the core solver** due to experimental sparse support, autodiff limitations, and compilation overhead. It may have a narrow role in alpha parameter optimization if that becomes a priority, but even there, CasADi might be more appropriate.

The goal is **faster scientific iteration**, not adopting the latest ML framework. Choose tools that match the problem structure, and SHOCCS is fundamentally a sparse matrix problem with static geometry - SciPy's sweet spot.

---

## References

### Framework Documentation
- JAX sparse matrices: https://docs.jax.dev/en/latest/jax.experimental.sparse.html
- SciPy sparse: https://docs.scipy.org/doc/scipy/reference/sparse.html
- Numba: https://numba.pydata.org/
- CasADi: https://web.casadi.org/docs/

### Key Research Findings
- JAX sparse limitations: GitHub issues #13118, #10559
- PyTorch CSR autodiff: "doesn't support computing derivatives with respect to CSR matrices"
- JAX vs Julia/PyTorch for scientific computing: Patrick Kidger analysis
- Sparse matrix performance: SciPy lectures on CSR format

### SHOCCS Codebase
- E2_1 stencil: `/home/user/shoccs/src/stencils/E2_1.cpp` (2509 lines)
- CSR operations: `/home/user/shoccs/src/matrices/csr.hpp`
- Derivative operator: `/home/user/shoccs/src/operators/derivative.hpp`
- RK4 integrator: `/home/user/shoccs/src/temporal/rk4.cpp`
