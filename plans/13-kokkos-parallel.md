# Phase 13: Kokkos Parallel Execution for Operators

**Goal:** Replace the serial loops in the matrix-vector product stack (`block`, `circulant`) with `Kokkos::parallel_for` on `DefaultHostExecutionSpace`, establishing the parallel execution framework for future GPU migration.

**Depends on:** Phase 8 (registry), Phase 10 (expression templates with `parallel_for`)

**Read first:**
- `src/matrices/block.hpp` (serial `for` over independent lines — the outer parallelism target)
- `src/matrices/inner_block.hpp` + `inner_block.cpp` (per-line: left dense + circulant + right dense)
- `src/matrices/dense.hpp` + `dense.cpp` (`std::inner_product`, strided access)
- `src/matrices/circulant.cpp` (convolution stencil, dominant per-line cost)
- `src/matrices/csr.hpp` + `csr.cpp` (sparse boundary terms)
- `src/matrices/common.hpp` (`matrix_base` with offset/stride)
- `src/operators/derivative.hpp` + `derivative.cpp` (orchestrator calling all matrix types)
- `src/kokkos_types.hpp` (execution/memory space aliases)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -L matrices
ctest --test-dir build -L operators
ctest --test-dir build
```

---

## Strategy

The matrix-vector product stack is the **dominant computational cost** per time step. The parallelization proceeds outside-in:

1. **`block::operator()`**: The serial `for (auto&& block : blocks)` loop over Ny×Nz independent lines becomes `Kokkos::parallel_for`. Each line writes to a disjoint output region — embarrassingly parallel.

2. **`circulant::operator()`**: The per-line interior stencil is a 1D convolution. Each output row is independent. Use a flat `Kokkos::parallel_for` over rows within each `operator()` call.

3. **`dense::operator()`**: The NBS boundary stencils are tiny (2–5 rows). These are best left serial within each line's thread, or batched across lines. **Not parallelized in Phase 13.**

4. **`csr::operator()`**: Sparse boundary terms, small per-line. Uses `+=` accumulation. **Not parallelized in Phase 13.**

5. **Matrix coefficient storage**: `dense::v`, `circulant::v`, `csr::w/v/u` remain `std::vector` / `std::span` for now (host-only). Migration to `Kokkos::View` is deferred to Phase 14.

6. **Kokkos linkage**: `shoccs-matrices` already receives `Kokkos::kokkos` transitively through `fields` (INTERFACE library). No explicit CMake link changes are needed for library targets. Only `#include "kokkos_types.hpp"` must be added to source files that call `Kokkos::parallel_for`.

7. **Lambda capture**: Use `[=]` (not `KOKKOS_LAMBDA`) because the parallelized code calls host-only functions (`inner_block::operator()`, `std::inner_product`). On `DefaultHostExecutionSpace`, `KOKKOS_LAMBDA` expands to `[=]` anyway, but using `[=]` explicitly signals host-only intent and prevents compilation errors if someone later attempts a GPU backend change. Phase 14 must address device compatibility separately. **Note:** `[=]` makes captured copies const, so any callable captured (like `eq_t`/`plus_eq_t`) must have const-qualified `operator()`. This was fixed in 13.3a.

---

## Items

### 13.1 — Kokkos Test Initialization for Matrices/Operators

After 13.2–13.3 add `Kokkos::parallel_for` to matrix operators, every test that
calls `block::operator()`, `circulant::operator()`, or `inner_block::operator()`
(which chains to circulant) will require `Kokkos::initialize()` to have been called.
This step converts the affected tests from `add_unit_test` (using
`Catch2::Catch2WithMain`) to manual test executables with custom `main()` using
`Kokkos::ScopeGuard` (per D-R9).

**Affected tests** (call parallelized operator() paths or construct objects that create Kokkos Views):
- Matrix: `t-block`, `t-circulant`, `t-inner_block`
- Operator: `t-derivative`, `t-gradient`, `t-laplacian`, `t-eigenvalue_visitor`

**NOT affected** (don't call parallelized operators or construct Kokkos-dependent objects):
- `t-dense`, `t-csr`, `t-unit_stride_visitor`, `t-coefficient_visitor` (matrix internals only)

> **Note:** `t-eigenvalue_visitor` calls `visit()` not `operator()`, but it constructs `derivative` objects whose constructor creates Kokkos Views, requiring Kokkos initialization.

- [x] **13.1a** Add Kokkos custom `main()` to matrix test files and update CMakeLists.txt:
  - Files: `src/matrices/block.t.cpp`, `src/matrices/circulant.t.cpp`, `src/matrices/inner_block.t.cpp`, `src/matrices/CMakeLists.txt`
  - In each test `.cpp` file, add after existing includes:
    ```cpp
    #include <Kokkos_Core.hpp>
    #include <catch2/catch_session.hpp>

    // Custom main: Kokkos must be initialized before parallel_for calls.
    int main(int argc, char* argv[])
    {
        Kokkos::ScopeGuard kokkos(argc, argv);
        return Catch::Session().run(argc, argv);
    }
    ```
  - In `src/matrices/CMakeLists.txt`, replace the 3 `add_unit_test(...)` calls for block, circulant, inner_block with explicit test definitions:
    ```cmake
    if (BUILD_TESTING)
      add_executable(t-block block.t.cpp)
      target_link_libraries(t-block Catch2::Catch2 shoccs-matrices shoccs-random Kokkos::kokkos)
      add_test(NAME t-block COMMAND t-block)
      set_tests_properties(t-block PROPERTIES LABELS "matrices")

      add_executable(t-circulant circulant.t.cpp)
      target_link_libraries(t-circulant Catch2::Catch2 shoccs-matrices shoccs-random Kokkos::kokkos)
      add_test(NAME t-circulant COMMAND t-circulant)
      set_tests_properties(t-circulant PROPERTIES LABELS "matrices")

      add_executable(t-inner_block inner_block.t.cpp)
      target_link_libraries(t-inner_block Catch2::Catch2 shoccs-matrices shoccs-random Kokkos::kokkos)
      add_test(NAME t-inner_block COMMAND t-inner_block)
      set_tests_properties(t-inner_block PROPERTIES LABELS "matrices")
    endif()
    ```
  - Keep `add_unit_test(dense ...)`, `add_unit_test(csr ...)`, `add_unit_test(unit_stride_visitor ...)`, `add_unit_test(coefficient_visitor ...)` unchanged.
  - Test: `cmake --build build && ctest --test-dir build -L matrices` — all matrix tests must still pass (no functional changes yet).
  - ~40 lines of diff across 4 files.

- [x] **13.1b** Add Kokkos custom `main()` to operator test files and update CMakeLists.txt:
  - Files: `src/operators/derivative.t.cpp`, `src/operators/gradient.t.cpp`, `src/operators/laplacian.t.cpp`, `src/operators/eigenvalue_visitor.t.cpp`, `src/operators/CMakeLists.txt`
  - Same pattern as 13.1a: add `Kokkos::ScopeGuard` + `Catch::Session` main to each test file.
  - In `src/operators/CMakeLists.txt`, replaced the 4 `add_unit_test(...)` calls with explicit test definitions.
  - Test results: `t-derivative`, `t-gradient`, `t-eigenvalue_visitor` all pass. `t-laplacian` has a **pre-existing** numerical tolerance failure in "E2 with Floating Objects" (`rx_vec` approx mismatch) — previously this was masked by a SIGABRT crash (no Kokkos init). All matrix tests still pass.
  - Must come after: 13.1a (to verify pattern works)

### 13.2 — Parallelize block::operator()

- [x] **13.2a** Add Kokkos include to `block.hpp` and replace the serial loop with `Kokkos::parallel_for`:
  - File: `src/matrices/block.hpp`
  - Add `#include "kokkos_types.hpp"` after `#include "inner_block.hpp"`.
  - Replace the body of `operator()`:
    ```cpp
    // BEFORE:
    for (auto&& block : blocks) { block(x, b, op); }

    // AFTER:
    const auto n = static_cast<int>(blocks.size());
    const auto* bp = blocks.data();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, n),
        [=](int i) { bp[i](x, b, op); });
    ```
  - Capture rationale:
    - `bp` (`const inner_block*`): pointer to vector data, captured by value.
    - `x` (`std::span<const real>`): pointer+size, captured by value (read-only shared input).
    - `b` (`std::span<real>`): pointer+size, captured by value (each block writes to disjoint region via `row_offset`/`stride`).
    - `op` (`eq_t` or `plus_eq_t`): stateless trivially-copyable callable, captured by value.
  - Correctness: Each `inner_block[i]` accesses a disjoint region of `b` determined by its `row_offset` and `stride`. For a 3D mesh with stride=3 (3 Y-lines), block 0 writes {0,3,6,...}, block 1 writes {1,4,7,...}, block 2 writes {2,5,8,...}. No data races.
  - Edge case: `blocks.size() == 0` → `RangePolicy(0, 0)` is a no-op.
  - ~10 lines of diff in 1 file.
  - Test: `ctest --test-dir build -R t-block && ctest --test-dir build -R t-inner_block`
  - Must come after: 13.1a

### 13.3 — Parallelize circulant::operator()

- [x] **13.3a** Add Kokkos include to `circulant.cpp` and replace the serial row loops with `Kokkos::parallel_for`:
  - Files: `src/matrices/circulant.cpp`, `src/types.hpp`
  - Added `#include "kokkos_types.hpp"` and `#include <omp.h>` (guarded by `KOKKOS_ENABLE_OPENMP`).
  - Extracted all member accesses (`rows()`, `v.data()`, `v.size()`, `x.data()`, `b.data()`) into local variables before lambdas for `[=]` capture.
  - Both `st == 1` and `st != 1` cases use `Kokkos::parallel_for` with `RangePolicy<execution_space>`.
  - **Nesting guard:** Kokkos forbids nested `parallel_for` (deadlocks on OpenMP even with 1 thread). Added `omp_in_parallel()` check — when called from within `block::operator()`'s `parallel_for`, falls back to serial loops. Standalone calls use `parallel_for`.
  - **const-qualified `eq_t`/`plus_eq_t`:** Added `const` to `operator()` on both types in `src/types.hpp`. Required because `[=]` capture makes the copy const inside lambdas, and Kokkos functors must have const-callable `operator()`. These are stateless types so const is correct.
  - Test: `t-circulant`, `t-block`, `t-inner_block` all pass.
  - Must come after: 13.1a

### 13.4 — Full Regression Verification

- [x] **13.4a** Run the full matrix and operator test suites and verify no regressions:
  - `ctest --test-dir build -L matrices` — 7/7 pass
  - `ctest --test-dir build -L operators` — 3/4 pass (1 expected failure)
  - `ctest --test-dir build` (full test suite) — 41/44 pass (3 pre-existing failures)
  - **Expected failure:** `t-laplacian` "E2 with Floating Objects" has a pre-existing numerical tolerance failure (`rx_vec` approx mismatch) that predates Phase 13 — see 13.1b notes. This is NOT a regression from parallelization.
  - **Other pre-existing failures** (unrelated to Phase 13, not modified by any Phase 13 commit):
    - `t-object_geometry` (mesh label): floating-point tolerance in sphere intersection tests — last modified in commit 6190a0c (Phase 3 range-v3 migration).
    - `t-E2_1` (stencils label): coefficient tolerance mismatch — last modified in commit 6190a0c (Phase 3 range-v3 migration).
  - No file changes — verification only.
  - Must come after: 13.2a, 13.3a, 13.1b

---

## Completion Criteria

- `block::operator()` dispatches via `Kokkos::parallel_for` over lines.
- `circulant::operator()` dispatches via `Kokkos::parallel_for` over rows.
- All matrix and operator tests pass with identical numerical results (except `t-laplacian` "E2 with Floating Objects" — pre-existing tolerance failure unrelated to Phase 13).
- Kokkos is initialized in all test executables that invoke parallelized operators.

## Notes

### Why no explicit Kokkos CMake linking for library targets
`shoccs-matrices` links `PUBLIC fields`, and `fields` links `INTERFACE Kokkos::kokkos` (see `src/fields/CMakeLists.txt:4`). This propagates Kokkos include paths and link libraries transitively to `shoccs-matrices` and all its downstream consumers (`shoccs-operators`, etc.). Only test executables need explicit `Kokkos::kokkos` in their `target_link_libraries` — and this is only for clarity/consistency with the established pattern (D-R9), since the transitive dependency already provides it.

### Nested parallel_for: block → inner_block → circulant
After 13.2a and 13.3a, calling `block::operator()` would create nested `Kokkos::parallel_for`:
```
block::operator()  →  parallel_for over blocks (13.2a)
  → inner_block::operator()
    → circulant::operator()  →  parallel_for over rows (13.3a)  ← NESTED
```
**Kokkos forbids nested `parallel_for`** — it deadlocks on OpenMP even with 1 thread, because the inner fence cannot complete while the outer parallel region is active. This is documented as undefined behavior in the Kokkos specification regardless of execution space.

**Resolution (13.3a):** `circulant::operator()` checks `omp_in_parallel()` at runtime. When nested (called from within `block::operator()`'s dispatch), it falls back to serial loops. When called standalone (e.g., `t-circulant` tests), it uses `Kokkos::parallel_for`. The effective parallelism is therefore at the block level (outer loop). Hierarchical parallelism (e.g., `TeamPolicy` for block × row) is deferred to Phase 14.

### What is NOT parallelized (and why)
- **`dense::operator()`**: NBS boundary stencils are 2–5 rows. Parallelization overhead dominates. Already runs in parallel indirectly (each line's dense operations execute on the thread handling that block).
- **`csr::operator()`**: Sparse boundary coupling terms. Small, uses `+=` accumulation. Not on the critical path.
- **`inner_block::operator()`**: Not parallelized internally. Delegates to `left_boundary(x,b,op); interior(x,b,op); right_boundary(x,b,op)` serially — these write to disjoint subregions. Parallelism comes from the block level (13.2a). Note: `interior()` is a `circulant` whose `operator()` gains its own `parallel_for` in 13.3a, creating the nested dispatch described above.
