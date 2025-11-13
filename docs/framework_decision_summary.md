# SHOCCS Python Migration: Quick Decision Reference

**Date:** 2025-11-13
**TL;DR:** Use **NumPy/SciPy + Numba**, not JAX

---

## The Verdict

| Framework | Recommendation | Reason |
|-----------|---------------|--------|
| **NumPy/SciPy + Numba** | ✅ **USE THIS** | Mature sparse matrices, JIT performance, low risk |
| **JAX** | ❌ **AVOID** | Experimental sparse support, autodiff limitations |
| **PyTorch** | ❌ **AVOID** | No CSR autodiff, ML-focused API |
| **CasADi** | 🟡 **MAYBE** | Only for alpha parameter optimization (optional) |

---

## Why Not JAX? (The Critical Issues)

### 1. Sparse Matrix Autodiff is Broken
```python
# From JAX documentation and GitHub issues:
# - "Backward rules return dense gradients for sparse inputs"
# - "Cannot compute derivatives with respect to CSR matrices"
# - "Memory overhead from dense gradients is very expensive"
```

**Impact on SHOCCS:** Your boundary coupling matrices (B, Bfx, etc.) are CSR. JAX can't differentiate through them properly.

### 2. Compilation Overhead Kills Performance
```python
# E2_1 stencil has 2509 lines of polynomial expressions
# JAX compilation cost grows quadratically with operations
# Every new (psi, alpha) combination triggers recompilation
```

**Impact on SHOCCS:** First stencil evaluation would have unacceptable latency. Your geometry-dependent coefficients create many static argument combinations.

### 3. You Don't Need Autodiff Anyway
```python
# Where autodiff might help:
# 1. Time-stepping loop? NO - RK4 is just accumulation
# 2. Stencil coefficients? NO - computed once at initialization
# 3. Alpha optimization? MAYBE - but that's offline, one-time

# Reality: 99% of your code doesn't benefit from autodiff
```

**Impact on SHOCCS:** You'd pay the complexity cost of autodiff for zero runtime benefit.

### 4. Experimental Status = Production Risk
```python
# From JAX docs:
# "jax.experimental.sparse is experimental and subject to change"
# "Reference implementations, not recommended for performance-critical applications"
```

**Impact on SHOCCS:** You need stable, production-ready code for research. Not experimental features.

---

## Why NumPy/SciPy + Numba? (The Right Choice)

### ✅ SciPy Sparse: 18+ Years of Production Use

```python
from scipy.sparse import csr_matrix

# Mature, optimized C code underneath
# Used by: scikit-learn, networkx, PyData ecosystem
# Performance: 2.88x faster than dense for sparse ops
A = csr_matrix(...)
y = A @ x  # Fast, reliable, battle-tested
```

### ✅ Numba: JIT Performance Without Pain

```python
from numba import njit

@njit
def stencil_coefficients(psi, alpha):
    # Your 2509 lines of E2_1.cpp go here
    # Compiled once on first call
    # Subsequent calls: near-C++ speed
    return coeffs

# No recompilation overhead
# No static shape requirements
# Just fast function calls
```

### ✅ Perfect Match for SHOCCS Architecture

| SHOCCS Feature | NumPy/SciPy Solution |
|----------------|---------------------|
| CSR boundary coupling | `scipy.sparse.csr_matrix` (perfect fit) |
| Static geometry | Compute once, store as arrays (no autodiff needed) |
| Explicit time-stepping | Simple field arithmetic (no autodiff needed) |
| Complex stencils | Numba JIT (compile once, reuse forever) |
| Field structure (D/Rx/Ry/Rz) | Python dataclass (clean, simple) |

### ✅ Low Risk, High Reward

- **Technical Risk:** LOW - using mature, stable tools
- **Performance Risk:** MEDIUM - expect 0.5-0.8x C++ speed (acceptable)
- **Maintenance Risk:** LOW - standard scientific Python
- **Learning Curve:** LOW - NumPy is lingua franca

---

## What About GPU Acceleration?

**Phase 1 (Now):** Get it working on CPU with NumPy/SciPy + Numba

**Phase 2 (Later, if needed):** Add GPU support
- Option A: CuPy (drop-in NumPy/SciPy replacement for GPU)
- Option B: Keep CPU version as reference, write targeted GPU kernels

**Reality Check:** For explicit PDE solvers, CPU is often sufficient. Memory bandwidth usually limits performance, not compute.

---

## Decision Matrix

| Criterion | NumPy/SciPy + Numba | JAX | PyTorch |
|-----------|---------------------|-----|---------|
| Sparse matrix support | ⭐⭐⭐⭐⭐ (mature) | ⭐ (experimental) | ⭐⭐ (no autodiff) |
| Autodiff for sparse ops | N/A (don't need it) | ⭐ (broken) | ⭐ (not for CSR) |
| JIT compilation | ⭐⭐⭐⭐ (Numba) | ⭐⭐⭐⭐⭐ (XLA) | ⭐⭐⭐ (TorchScript) |
| Ease of migration | ⭐⭐⭐⭐⭐ (straightforward) | ⭐⭐ (paradigm shift) | ⭐⭐⭐ (tensor API) |
| Scientific computing focus | ⭐⭐⭐⭐⭐ (designed for it) | ⭐⭐⭐ (ML focus) | ⭐⭐ (ML focus) |
| Production stability | ⭐⭐⭐⭐⭐ (18+ years) | ⭐⭐⭐ (changing APIs) | ⭐⭐⭐⭐ (stable) |
| Performance for SHOCCS | ⭐⭐⭐⭐ (very good) | ⭐⭐ (compilation overhead) | ⭐⭐⭐ (decent) |
| **TOTAL** | **31 / 35** | **16 / 35** | **18 / 35** |

---

## Common Objections Answered

### "But JAX is more modern!"

**Response:** Modern ≠ appropriate. JAX is modern for ML/autodiff. SHOCCS is a sparse matrix problem with static geometry. SciPy is modern *for that problem*.

### "But I want to use autodiff for research!"

**Response:**
1. What would you differentiate? Time-stepping is just accumulation. Stencil coefficients are computed once.
2. If you want to optimize alpha parameters (one-time, offline), use JAX or CasADi in a separate script.
3. Don't pay the complexity cost everywhere for a feature you'll use rarely.

### "But Python will be too slow!"

**Response:**
- SciPy sparse ops are C code (as fast as your current C++)
- Numba JIT compiles hot loops to machine code
- Most time is in sparse matrix operations (already optimized)
- Expect 0.5-0.8x C++ speed - acceptable for research iteration

### "But everyone is using JAX now!"

**Response:**
- "Everyone" = ML researchers doing autodiff-heavy workflows
- You = PDE solver with explicit time-stepping and sparse matrices
- Different problems need different tools

---

## The Migration Path

### Week 1-2: Foundation
```python
# Create core data structures
class Field:
    D, Rx, Ry, Rz  # Done in 50 lines

# Translate simple stencil
def interior_stencil(h):
    return np.array([-1/(2*h), 0, 1/(2*h)])  # Done
```

### Month 1-2: Core Solver
```python
# Translate E2_1 stencil (2509 lines)
@njit
def E2_1_coefficients(psi, alpha):
    # Line-by-line translation
    # Compile once, verify against C++

# Build derivative operator
D_x = DerivativeOperator(...)  # CSR matrices

# Implement RK4
u_new = rk4.step(system, u, dt, t)
```

### Month 3-4: Validation
```python
# Method of Manufactured Solutions
# Convergence studies
# Performance profiling and optimization
```

### Optional: Alpha Optimization
```python
# If you decide this is valuable:
import jax  # Only for this offline task

def optimize_alpha(target_accuracy):
    # Use JAX or CasADi here
    # Feed results to main NumPy solver
```

---

## Final Recommendation

**Start with NumPy/SciPy + Numba immediately.**

This is the path of:
- ✅ Least technical risk
- ✅ Fastest time to working code
- ✅ Best match for SHOCCS architecture
- ✅ Easiest debugging and visualization
- ✅ Most stable long-term maintenance

**Avoid JAX for the core solver** due to:
- ❌ Experimental sparse matrix support
- ❌ CSR autodiff limitations
- ❌ Compilation overhead for complex stencils
- ❌ No benefit from autodiff in your workflow

**Consider JAX only for** (optional, later):
- 🟡 Alpha parameter optimization (one-time, offline)
- 🟡 If you want to experiment with differentiable solvers (research)

---

## One-Line Summary

**Use the boring, mature tools (NumPy/SciPy + Numba) that perfectly match your problem, not the shiny, new tools (JAX) designed for a different problem (ML with autodiff).**

---

## References

- Full analysis: `/home/user/shoccs/docs/python_migration_framework_analysis.md`
- Code examples: `/home/user/shoccs/docs/python_migration_code_examples.md`
- JAX sparse issues: GitHub #13118, #10559
- JAX docs: "experimental and subject to change"
- SciPy sparse: 18+ years of production use
