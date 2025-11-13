# SHOCCS Python Migration: Orchestrated Implementation Plan

**Date:** 2025-11-13
**Status:** Ready to Begin Implementation

## Executive Summary

This document coordinates the orchestrated implementation of SHOCCS Python migration using multiple specialized agents working in parallel. The migration uses **NumPy/SciPy + Numba** (not JAX) based on comprehensive technical analysis.

## Team Structure

### Oversight & Review
- **Numerical Methods Expert**: Validates numerical correctness, convergence, and scientific accuracy
- **Software Architect**: Reviews design decisions, ensures simplicity and code reuse

### Development Team
- **Developer 1 (Fields)**: Core data structures (ScalarField, VectorField, Field arithmetic)
- **Developer 2 (Geometry)**: Mesh, ray tracing, cut-cell geometry
- **Developer 3 (Stencils)**: Finite difference stencil implementations
- **Developer 4 (Operators)**: Discrete operators (derivative, gradient, Laplacian)
- **Developer 5 (Systems)**: PDE systems and time integrators

### Review Team
- **Reviewer 1**: Code review for Developer 1 & 2
- **Reviewer 2**: Code review for Developer 3 & 4
- **Reviewer 3**: Code review for Developer 5 & integration testing

## Phase 1: Foundation (Weeks 1-2)

### Developer 1: Core Field Structures
**Tasks:**
- [ ] Implement `ScalarField` class (D, Rx, Ry, Rz structure)
- [ ] Implement `VectorField` class (x, y, z components)
- [ ] Field arithmetic operations (+, -, *, /, scalar mult)
- [ ] Unit tests for field operations

**Success Criteria:**
- All field arithmetic tests pass
- Zero-copy views work correctly
- Field norms computed correctly

**Review by:** Reviewer 1, Software Architect

### Developer 2: Mesh and Basic Geometry
**Tasks:**
- [ ] Implement `CartesianMesh` class
- [ ] Coordinate generation and indexing
- [ ] Grid spacing calculations
- [ ] Unit tests for mesh

**Success Criteria:**
- Mesh matches C++ CartesianMesh behavior
- Indexing scheme validated

**Review by:** Reviewer 1, Numerical Expert

## Phase 2: Stencils (Weeks 3-4)

### Developer 3: Stencil Implementations
**Tasks:**
- [ ] Interior stencil (3-point centered difference)
- [ ] E2_1 stencil translation (2509 lines from C++)
- [ ] Numba JIT compilation setup
- [ ] Reference data generation from C++
- [ ] Stencil coefficient tests vs C++ reference

**Success Criteria:**
- All stencil coefficients match C++ to 1e-14 tolerance
- Numba compilation works
- Performance within 2x of C++

**Review by:** Reviewer 2, Numerical Expert (critical validation!)

**Architect Notes:** Keep stencils simple functions, no complex class hierarchy

## Phase 3: Operators (Weeks 5-7)

### Developer 4: Discrete Operators
**Tasks:**
- [ ] CSR matrix construction utilities
- [ ] `DerivativeOperator` class
- [ ] Operator builder functions
- [ ] Apply operator to fields (CSR @ field)
- [ ] Tests: polynomials, convergence

**Success Criteria:**
- Derivative of x² = 2x (analytically exact for interior)
- Operator matrices match C++ structure
- Sparse matrix operations validated

**Review by:** Reviewer 2, Numerical Expert

**Architect Notes:** Composition over inheritance - build gradient from 3 derivatives

### Developer 2: Cut-Cell Geometry (Parallel)
**Tasks:**
- [ ] Shape interfaces (Sphere, Rectangle)
- [ ] Ray tracing implementation
- [ ] Boundary point computation (psi, normals)
- [ ] Solid point identification
- [ ] Geometry validation vs C++

**Success Criteria:**
- Sphere geometry matches C++ reference
- Ray intersections correct
- Psi values validated

**Review by:** Reviewer 1, Numerical Expert

## Phase 4: Integration (Weeks 8-10)

### Developer 5: Time Integration & Systems
**Tasks:**
- [ ] RK4 integrator implementation
- [ ] Euler integrator (for testing)
- [ ] `HeatEquation` system class
- [ ] System RHS evaluation
- [ ] Boundary condition updates

**Success Criteria:**
- RK4 matches reference implementation
- Heat equation converges at 2nd order
- Time-step stability correct

**Review by:** Reviewer 3, Numerical Expert

### Developer 1: I/O and Utilities (Parallel)
**Tasks:**
- [ ] Field I/O (save/load)
- [ ] Visualization helpers
- [ ] Configuration loading (YAML/dict)
- [ ] Example scripts

**Review by:** Reviewer 3, Software Architect

## Phase 5: Validation (Weeks 11-12)

### All Developers: Convergence Studies
**Tasks:**
- [ ] MMS test implementation
- [ ] Grid refinement studies
- [ ] Convergence rate analysis
- [ ] Performance profiling
- [ ] Optimization (if needed)

**Success Criteria:**
- All convergence tests pass
- Performance within acceptable range (0.5-0.8x C++)
- Full test suite passes

**Review by:** Numerical Expert (final validation), Software Architect

## Review Process

### Code Review Checklist

**For All Code:**
- [ ] Passes pytest tests
- [ ] Type hints on public APIs
- [ ] Docstrings on all functions/classes
- [ ] No premature optimization
- [ ] Simple, readable code

**For Numerical Code:**
- [ ] Validated against C++ reference
- [ ] Convergence tests pass
- [ ] Numerical tolerances documented
- [ ] Edge cases handled

**For Architecture:**
- [ ] Follows composition over inheritance
- [ ] No unnecessary abstractions
- [ ] Reuses existing components
- [ ] Clear, simple interfaces

### Review Flow

```
Developer writes code
    ↓
Self-test (unit tests)
    ↓
Submit for review
    ↓
Assigned Reviewer checks ← Software Architect spot-checks
    ↓
Numerical Expert validates (if numerical code)
    ↓
Revise based on feedback
    ↓
Merge when approved by all reviewers
```

## Communication Protocol

### Daily Standups (Async)
Each agent reports:
1. What was completed
2. What's in progress
3. Any blockers

### Code Reviews
- Reviewer provides specific, actionable feedback
- Focus on correctness, simplicity, testability
- No bikeshedding on style

### Architecture Decisions
- Software Architect has final say on design
- Numerical Expert has final say on correctness
- Default to simplicity

### Conflict Resolution
1. Try to find simple solution both parties accept
2. If deadlock, Architect decides design, Expert decides correctness
3. Escalate to orchestrator (me) only if necessary

## Success Metrics

### Phase Completion Criteria
Each phase must have:
- [ ] All assigned tasks completed
- [ ] All tests passing
- [ ] Code reviewed and approved
- [ ] Documentation updated
- [ ] Validated by appropriate experts

### Overall Success
- [ ] Heat equation matches C++ reference
- [ ] Second-order convergence demonstrated
- [ ] Performance within 2x of C++
- [ ] Clean, maintainable codebase
- [ ] Comprehensive test coverage

## Risk Mitigation

### If Stencil Translation is Difficult
- Use automated C++ parser to extract expressions
- Manual verification of subset
- Extensive reference testing

### If Performance is Poor
- Profile first (don't guess)
- Add Numba JIT selectively
- Accept 2x slowdown as success

### If Schedule Slips
- Cut scope: Start with 2D only
- Prioritize: Heat equation before wave equation
- Parallelize: Multiple agents on independent tasks

## Documentation Requirements

Each component must have:
- **API documentation**: Docstrings with examples
- **Test documentation**: What each test validates
- **Migration notes**: Differences from C++
- **Performance notes**: Any optimization done

## Next Steps

1. Set up Python repository structure
2. Generate C++ reference data
3. Assign agents to Phase 1 tasks
4. Begin parallel development
5. Daily progress tracking

---

## Key Design Decisions (Approved)

1. **Framework**: NumPy/SciPy + Numba (NOT JAX)
2. **Data structures**: Dataclasses with simple field layout
3. **Interfaces**: Protocols, not ABCs
4. **Matrices**: SciPy CSR for all sparse operations
5. **Testing**: Reference tests against C++ at every level
6. **Optimization**: Profile first, optimize bottlenecks only

---

## References

- **Architecture Plan**: `/home/user/shoccs/docs/python_migration_architecture.md`
- **Validation Plan**: `/home/user/shoccs/docs/python_validation_plan.md`
- **Framework Analysis**: `/home/user/shoccs/docs/python_migration_framework_analysis.md`
- **C++ Codebase**: `/home/user/shoccs/src/`

---

**Status**: Ready to deploy agents and begin implementation
**Estimated Completion**: 12-14 weeks
**Next Update**: After Phase 1 completion
