# Phase 16: Parallelize Remaining Serial Hot-Path Loops

**Goal:** Replace serial `for` loops on field-sized data in system methods with `Kokkos::parallel_for` and `Kokkos::parallel_reduce`. These are the easy cases — element-wise maps and reductions with no nesting concerns. The nested circulant-inside-block case (Kokkos TeamPolicy with team/thread/vector levels) is deferred to Phase 17.

**Depends on:** Phase 15 (fences and safety assertions in place)

**Read first:**
- `src/systems/heat.cpp` (eval_at_locations, rhs, stats, write, initialize)
- `src/systems/scalar_wave.cpp` (eval_at_locations, rhs, stats, write, initialize)
- `src/fields/expr.hpp` (assign, parallel_for pattern)
- `src/fields/selection_desc.hpp` (gather_selection, contiguous_selection)
- `src/kokkos_types.hpp` (execution_space, memory_space)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -R "t-heat|t-scalar_wave"
ctest --test-dir build -R t-simulation_cycle
ctest --test-dir build
```

---

## Strategy

All targets follow the same pattern: extract raw `real*` pointers from spans before the lambda, launch `Kokkos::parallel_for` or `Kokkos::parallel_reduce`, fence after.

For `eval_at_locations`, the callable `func` captures mesh coordinate data. Since MMS functions are small stateless lambdas, they are trivially copyable and safe for `KOKKOS_LAMBDA` on `DefaultHostExecutionSpace`. The `cartesian_product_view` iteration is replaced with flat-index decomposition (`i/(ny*nz)`, `(i/nz)%ny`, `i%nz`).

For `stats()` reductions, `Kokkos::parallel_reduce` with custom reducers (`Max`, `Min`, or a combined struct) replaces the serial scan.

---

## Items

### 16.1 — Parallelize `eval_at_locations` (TDD)

This function is called in both `heat.cpp` and `scalar_wave.cpp` from `rhs()`, `update_boundary()`, `stats()`, `initialize()`, and `write()`. Parallelizing it once benefits all call sites.

- [x] **16.1a** Write a test for parallel `eval_at_locations`:
  - Created 8×9×10 mesh with `f(x,y,z) = x+y+z`, floating sphere BC.
  - Verifies: stats error=0 at t=0, D-buffer corner values match x+y+z, Rx/Ry/Rz values finite and in-range.
  - File: `src/systems/heat.t.cpp` (TEST_CASE "heat - eval_at_locations correctness")
  - Test: `ctest --test-dir build -R t-heat` — PASS

- [x] **16.1b** Parallelize the D-buffer loop in `eval_at_locations` (heat.cpp):
  - Replaced `cartesian_product` range-for with flat `Kokkos::parallel_for` using index decomposition.
  - Parallelized Rx/Ry/Rz loops using `parallel_for` over `mesh_object_info` positions.
  - Added `Kokkos::fence()` after all four `parallel_for` calls.
  - **Thread-safety:** Discovered Lua MMS is not thread-safe (Lua state is single-threaded).
    Added `manufactured_solution::is_thread_safe()` virtual method (defaults to `true`, `lua_mms`
    overrides via `static constexpr bool thread_safe = false`). `eval_at_locations` takes a
    `bool parallel` parameter; callers pass `m_sol.is_thread_safe()`. Gauss MMS runs parallel,
    Lua MMS falls back to serial loops.
  - Files: `src/systems/heat.cpp`, `src/mms/manufactured_solutions.hpp`, `src/mms/lua_mms.hpp`
  - Test: `ctest --test-dir build -R t-heat` — PASS
  - Test: `ctest --test-dir build -R t-simulation_cycle` — PASS

- [x] **16.1b-fix** Add a test that exercises the parallel `eval_at_locations` path:
  - Added TEST_CASE "heat - eval_at_locations parallel path (gaussian MMS)" using
    `type = "gaussian"` MMS so `is_thread_safe()` returns `true` and the
    `Kokkos::parallel_for` branch is exercised.
  - Verifies: stats error=0 at t=0, D-buffer corner values match Gaussian formula
    `exp(-0.5*(x^2+y^2+z^2))`, R-buffer values finite and in (0, 1].
  - File: `src/systems/heat.t.cpp`
  - Test: `ctest --test-dir build -R t-heat` — PASS
  - Test: `ctest --test-dir build -R t-simulation_cycle` — PASS

- [x] **16.1c** Apply the same parallelization to `eval_at_locations` in `scalar_wave.cpp`:
  - Replaced serial `cartesian_product` range-for and R-buffer loops with
    `Kokkos::parallel_for` using flat index decomposition, same as heat.cpp.
  - No `bool parallel` parameter needed — scalar_wave only uses simple constexpr
    math lambdas (`neg_G_at`, `solution_at`) which are always thread-safe.
  - Added `Kokkos::fence()` after all four `parallel_for` calls.
  - File: `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R t-scalar_wave` — PASS
  - Test: `ctest --test-dir build -R t-simulation_cycle` — PASS

### 16.2 — Parallelize `dot_spans` in `scalar_wave::rhs()`

- [x] **16.2a** Parallelize the `dot_spans` lambda:
  - Replaced serial `for` loop with `Kokkos::parallel_for` using `RangePolicy<execution_space>`.
  - Extracted raw pointers from spans before the lambda for safe capture.
  - Added `Kokkos::fence()` after each `parallel_for` call.
  - The lambda is called 4 times (D, Rx, Ry, Rz) — each call is independently parallel.
  - File: `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R t-scalar_wave` — PASS
  - Test: `ctest --test-dir build -R t-simulation_cycle` — PASS

### 16.3 — Parallelize `stats()` reductions (TDD)

- [x] **16.3a** Write a test for parallel stats reduction:
  - Added TEST_CASE "heat - stats reduction correctness" using Gaussian MMS
    (8×9×10 mesh, floating sphere BC).
  - After initialize (u = sol, error = 0), perturbs D[0] by +0.42 and
    D[719] by −0.35 at known fluid corners, and Rx[0] by +0.13.
  - Verifies all 11 stats fields: overall Linf = 0.42, u_min = exp(−1.5) − 0.35,
    u_max = 1.42, err_d = 0.42 at idx 0, err_rx = 0.13 at idx 0, err_ry = err_rz = 0.
  - File: `src/systems/heat.t.cpp`
  - Test: `ctest --test-dir build -R t-heat` — PASS

- [x] **16.3b** Parallelize the fluid-D reduction in `heat::stats()`:
  - Replaced serial `for (int k = 0; k < fd.count(); ++k)` min/max/error loop with two
    `Kokkos::parallel_reduce` calls:
    1. `Kokkos::MinMax<real>` with `MinMaxScalar<real>` for u_min/u_max
    2. `Kokkos::MaxLoc<real, int>` with `ValLocScalar<real, int>` for err_d/err_d_idx
  - Extracted raw `real*` pointers from spans; copied `gather_selection fd` by value for lambda capture.
  - Added guard for empty fluid case (`fd.count() > 0`) on MaxLoc result.
  - Added `Kokkos::fence()` after both reductions.
  - File: `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R t-heat` — PASS
  - Test: `ctest --test-dir build -R t-simulation_cycle` — PASS

- [x] **16.3c** Same for `scalar_wave::stats()`:
  - Replaced serial `for (int k = 0; k < fd.count(); ++k)` min/max/error loop with two
    `Kokkos::parallel_reduce` calls:
    1. `Kokkos::MinMax<real>` with `MinMaxScalar<real>` for u_min/u_max
    2. `Kokkos::MaxLoc<real, int>` with `ValLocScalar<real, int>` for err_d/err_d_idx
  - Extracted raw `real*` pointers from spans; copied `gather_selection fd` by value for lambda capture.
  - Added guard for empty fluid case (`fd.count() > 0`) on MaxLoc result.
  - Added `Kokkos::fence()` after both reductions.
  - File: `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R t-scalar_wave` — PASS
  - Test: `ctest --test-dir build -R t-simulation_cycle` — PASS

- [x] **16.3d** Parallelize the R-component reductions in `stats()`:
  - Replaced serial `component_stats` lambda with a `for (int dir = 0; dir < 3; ++dir)` loop
    using two `Kokkos::parallel_reduce` calls per direction:
    1. `Kokkos::MinMax<real>` with `MinMaxScalar<real>` for per-direction u_min/u_max
    2. `Kokkos::MaxLoc<real, int>` with `ValLocScalar<real, int>` for per-direction comp_err/comp_idx
  - Added guard for empty non-dirichlet case (`nd.count() == 0`) via `continue`.
  - Added `Kokkos::fence()` after all R-component reductions.
  - Files: `src/systems/heat.cpp`, `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R "t-heat|t-scalar_wave"` — PASS
  - Test: `ctest --test-dir build -R t-simulation_cycle` — PASS

### 16.4 — Parallelize `write()` error computation

- [x] **16.4a** Parallelize the error-fill loops in `heat::write()`:
  - Replaced serial D-buffer `for (int k = 0; k < fd.count(); ++k)` error loop with
    `Kokkos::parallel_for` using `RangePolicy<execution_space>` and `Kokkos::abs`.
  - Replaced serial R-direction loops with `Kokkos::parallel_for` per direction,
    with `nd.count() == 0` guard via `continue`.
  - Extracted raw `real*` pointers from spans/vectors; copied `gather_selection fd` by value
    for lambda capture.
  - Added `Kokkos::fence()` after all parallel_for calls.
  - File: `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R t-heat` — PASS
  - Test: `ctest --test-dir build -R t-simulation_cycle` — PASS

- [x] **16.4b** Same for `scalar_wave::write()`:
  - Same pattern as 16.4a: `Kokkos::parallel_for` for D-buffer and R-direction error
    computation, with raw pointer extraction, empty-guard, and `Kokkos::fence()`.
  - File: `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R t-scalar_wave` — PASS
  - Test: `ctest --test-dir build -R t-simulation_cycle` — PASS

### 16.5 — Parallelize `initialize()` fill and copy

- [x] **16.5a** Replace `std::ranges::fill` and `std::ranges::copy` with Kokkos equivalents in `heat::initialize()`:
  - Replaced `std::ranges::fill(u.D, 0.0)` with `Kokkos::parallel_for` zeroing D buffer.
  - Replaced serial gather-scatter loop with `assign_selected` from `selection_desc.hpp`.
  - Replaced `std::ranges::copy` for Rx/Ry/Rz with `Kokkos::parallel_for` per component.
  - Added `Kokkos::fence()` after all parallel dispatches.
  - File: `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R t-heat` — PASS
  - Test: `ctest --test-dir build -R t-simulation_cycle` — PASS

- [x] **16.5b** Same for `scalar_wave::initialize()`:
  - Same pattern as 16.5a: `Kokkos::parallel_for` for D fill and R copies,
    `assign_selected` for fluid gather-scatter, `Kokkos::fence()` at end.
  - File: `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R t-scalar_wave` — PASS
  - Test: `ctest --test-dir build -R t-simulation_cycle` — PASS

### 16.6 — Parallelize `scalar_span::operator=` fill

- [x] **16.6a** Replace `std::ranges::fill` in `scalar_span::operator=(real)` with `Kokkos::parallel_for`:
  - Hot-path caller: `du = 0` in `laplacian.cpp` (called from `rhs()` every timestep).
  - Replaced 4 `std::ranges::fill` calls with `Kokkos::parallel_for` using
    `RangePolicy<execution_space>` and raw pointer extraction per component (D, Rx, Ry, Rz).
  - Added `Kokkos::fence()` after all four dispatches.
  - Replaced `#include <algorithm>` with `#include "kokkos_types.hpp"` (fields library
    already links `Kokkos::kokkos`).
  - File: `src/fields/scalar.hpp`
  - Test: `ctest --test-dir build -L fields` — PASS (4 tests)
  - Test: `ctest --test-dir build -R "t-heat|t-scalar_wave"` — PASS
  - Test: `ctest --test-dir build -R t-simulation_cycle` — PASS
  - Note: t-laplacian "E2 with Floating Objects" failure is pre-existing (verified).

### 16.7 — Full regression

- [x] **16.7a** Run full test suite, verify no regressions:
  - `cmake --build build && ctest --test-dir build`
  - 41/44 tests pass; 3 failures all pre-existing:
    - t-object_geometry (mesh), t-E2_1 (stencils), t-laplacian (operators)
  - No new regressions introduced by Phase 16.

---

## Ordering Constraints

```
16.1a (test) → 16.1b (heat eval_at_locations) → 16.1b-fix (parallel path test) → 16.1c (scalar_wave eval_at_locations)
16.2a (dot_spans) — independent
16.3a (test) → 16.3b (heat stats) → 16.3c (scalar_wave stats) → 16.3d (R-component stats)
16.4a (heat write) — depends on 16.1b (eval_at_locations must be parallel first, since write calls it)
16.4b (scalar_wave write) — depends on 16.1c
16.5a-b (initialize) — independent (cold path, low priority)
16.6a (scalar_span fill) — independent
16.7a (regression) — last
```

---

## Non-Goals

- **Nested parallelism for circulant-inside-block** — deferred to Phase 17 (Kokkos TeamPolicy with team/thread/vector levels).
- **Dense matrix parallelization** — rows are 2-6 elements, not worth a kernel launch. Stays serial in nested context.
- **GPU device migration** — Phase 14 (unchanged).

---

## Completion Criteria

- `eval_at_locations` in both systems uses `parallel_for` for D and R buffers.
- `dot_spans` in `scalar_wave::rhs()` uses `parallel_for`.
- `stats()` reductions in both systems use `parallel_reduce`.
- `write()` error computation in both systems uses `parallel_for`.
- `initialize()` fill/copy uses `Kokkos::deep_copy` or `parallel_for`.
- All `Kokkos::fence()` calls present after parallel dispatches.
- All tests pass.
