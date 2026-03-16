# Phase 13: Kokkos Parallel Execution for Operators

**Goal:** Replace the serial loops in the matrix-vector product stack (`block`, `inner_block`, `dense`, `circulant`, `csr`) with `Kokkos::parallel_for` and `Kokkos::TeamPolicy` on `DefaultHostExecutionSpace`, establishing the parallel execution framework for future GPU migration.

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

2. **`circulant::operator()`**: The per-line interior stencil is a 1D convolution. Each output row is independent. This is the inner parallelism target, either via `TeamPolicy` (one team per line, threads parallelize rows) or via a flat `parallel_for` over all output rows across all lines.

3. **`dense::operator()`**: The NBS boundary stencils are tiny (3×4 rows). These are best left serial within each line's thread, or batched across lines.

4. **Matrix coefficient storage**: `dense::v`, `circulant::v`, `csr::w/v/u` remain `std::vector` for now (host-only). Migration to `Kokkos::View` is deferred to Phase 14.

---

## Items

### 13.1 — Parallelize block::operator() (TDD)

- [ ] **13.1a** Write a benchmark/test for `block::operator()` parallel dispatch:
  - Create a `block` with N inner_blocks (e.g., N=64 for a 8×8 mesh).
  - Time the serial version vs `Kokkos::parallel_for(N, ...)` version.
  - Verify identical results.
  - File: `src/matrices/block.t.cpp` (extend) or new benchmark file
  - Test: `ctest --test-dir build -R t-block`

- [ ] **13.1b** Replace the serial loop in `block::operator()` with `Kokkos::parallel_for`:
  - `Kokkos::parallel_for(blocks.size(), [&](int i) { blocks[i](x, b, op); })`.
  - The `std::span` parameters `x` and `b` are shared across all iterations; each `inner_block` accesses a disjoint region via its `row_offset`/`col_offset`/`stride`.
  - File: `src/matrices/block.hpp`
  - Test: `ctest --test-dir build -L matrices`

### 13.2 — Parallelize circulant interior (TDD)

- [ ] **13.2a** Write tests verifying parallel circulant produces identical results:
  - File: `src/matrices/circulant.t.cpp` (extend)
  - Test: `ctest --test-dir build -R t-circulant`

- [ ] **13.2b** Replace the serial row loop in `circulant::operator()` with `Kokkos::parallel_for`:
  - For `st == 1`: `Kokkos::parallel_for(rows(), [=](int i) { auto dot = std::inner_product(...); op(b[i], dot); })`.
  - For `st != 1`: same but with strided indexing.
  - File: `src/matrices/circulant.cpp`
  - Test: `ctest --test-dir build -R t-circulant`

### 13.3 — Link Kokkos to matrix/operator targets

- [ ] **13.3a** Add `Kokkos::kokkos` to `target_link_libraries` for `shoccs-matrices` and `shoccs-operators` in their respective CMakeLists.txt files.
  - Files: `src/matrices/CMakeLists.txt`, `src/operators/CMakeLists.txt`
  - Test: `cmake --build build`

### 13.4 — Verify operator correctness

- [ ] **13.4a** Run the full operator test suite and verify no regressions:
  - `ctest --test-dir build -L operators`
  - Compare results with pre-parallelization baseline for `t-derivative`, `t-gradient`, `t-laplacian`.
  - No file changes — verification only.

---

## Completion Criteria

- `block::operator()` dispatches via `Kokkos::parallel_for` over lines.
- `circulant::operator()` dispatches via `Kokkos::parallel_for` over rows.
- All matrix and operator tests pass with identical numerical results.
- Kokkos is linked to the matrices and operators CMake targets.
