# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SHOCCS (Stable High-Order Cut-Cell Solver) is a C++ Cartesian cut-cell solver for time-dependent PDEs (heat equation, scalar wave, Euler equations). It uses high-order finite difference operators on structured grids with embedded boundaries. The codebase uses Kokkos for parallel execution.

## Build Commands

```bash
# Build (from repo root; build/ is pre-configured with Ninja)
cmake --build build

# Build a single target
cmake --build build --target t-heat

# Run all tests
ctest --test-dir build

# Run tests by label (fields, matrices, operators, stencils, mesh, systems, temporal, simulation)
ctest --test-dir build -L fields

# Run a single test by name
ctest --test-dir build -R t-dense

# Reconfigure from scratch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON -DBUILD_BENCHMARKS=ON
```

## Benchmarks

```bash
# Run all benchmarks and compare against stored baseline
./scripts/bench_compare.sh

# Save a new baseline
./scripts/bench_compare.sh --save
```

Benchmark executables are in `build/benchmarks/` (bench_stencil, bench_block, bench_derivative, bench_expr, bench_selection, bench_rhs).

## Code Conventions

- **C++20** with concepts, `std::ranges`, and `std::span`. No C++23 features; missing C++23 utilities (stride, zip_transform, cartesian_product, bind_back) have project-local implementations in `src/fields/lazy_views.hpp`.
- **Namespace:** Everything lives under `ccs`. Core type aliases in `src/shoccs_config.hpp`: `real = double`, `integer = long`, `int3 = std::array<int,3>`, `real3 = std::array<real,3>`.
- **Kokkos types** in `src/kokkos_types.hpp`: `execution_space`, `memory_space`, `device_view<T>`. Currently host-only (`DefaultHostExecutionSpace`).
- **Test files:** Named `*.t.cpp`, use Catch2 v3. Tests needing Kokkos provide a custom `main()` with `Kokkos::ScopeGuard` and link `Catch2::Catch2` (not `WithMain`). Simple tests use the `add_unit_test()` CMake helper which links `Catch2::Catch2WithMain`.
- **CMake targets** follow the pattern `shoccs-<subsystem>` (e.g., `shoccs-matrices`, `shoccs-operators`). Test executables are `t-<name>`.

## Architecture

### Data Flow

Lua config → `simulation::builder` → mesh + operators + system + integrator → `simulation_cycle::run()` (time-stepping loop)

### Key Subsystems (dependency order)

1. **Indexing** (`src/indexing.hpp`, `index_extents.hpp`, `index_view.hpp`) — Multi-dimensional index mapping for structured grids. All subsystems depend on this.

2. **Fields** (`src/fields/`) — Field storage and algebra. `field_registry` owns all buffers as `Kokkos::View<real*>`. `field_ref` is a lightweight handle (slot index). Expression templates (`expr.hpp`) enable `dst = a + alpha * b` syntax that dispatches to `Kokkos::parallel_for`. Selection descriptors (`selection_desc.hpp`) describe contiguous/strided/gather subsets of a field for BC application.

3. **Mesh** (`src/mesh/`) — Cartesian grid with cut-cell geometry. `cartesian` holds 1D coordinate arrays and grid spacings. `object_geometry` performs ray-casting intersection with embedded shapes (spheres, rectangles) to compute `psi` parameters for cut-cell stencils.

4. **Matrices** (`src/matrices/`) — Small per-line operators, not global sparse systems. Composite structure: `inner_block = [dense_left | circulant_interior | dense_right]`, wrapped in `block` for multi-line application. CSR for sparse boundary coupling. Matrix-vector products use explicit loops (no KokkosKernels).

5. **Stencils** (`src/stencils/`) — Finite difference coefficients. Named by scheme (E2, E4, E6, E8) and order. Each provides interior circulant + boundary dense + optional CSR cut-cell coefficients.

6. **Operators** (`src/operators/`) — Discrete differential operators (derivative, gradient, laplacian) built from stencils + matrices. Applied per-direction using `operator_visitor` pattern across `block`/`inner_block`.

7. **Systems** (`src/systems/`) — PDE implementations. Each system provides `rhs()` (spatial discretization), `update_boundary()`, `timestep_size()`, and initialization. Concrete: `heat`, `scalar_wave`, `hyperbolic_eigenvalues`. Uses `system` variant wrapper for type erasure.

8. **Temporal** (`src/temporal/`) — Time integrators (`euler`, `rk4`) operating on `field_ref` slots in the registry. `step_controller` manages adaptive time stepping. Slot arithmetic helpers in `slot_ops.hpp`.

9. **Simulation** (`src/simulation/`) — Builder pattern assembles the full simulation from Lua config. `simulation_cycle` owns the time-stepping loop.

10. **I/O** (`src/io/`) — XDMF/binary field output, spdlog-based logging.

### Historical Context

The codebase was migrated from range-v3 to Kokkos. The `plans/` directory contains the migration plans and architectural decisions from that effort. Key decisions are documented in `plans/meta.md`.
