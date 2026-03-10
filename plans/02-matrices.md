# Phase 2: Matrices Subsystem

**Goal:** Migrate matrix types and their matrix-vector product implementations from range-v3 view pipelines to Kokkos kernels or std::ranges equivalents.

**Depends on:** Phase 0, Phase 1 (for `fields/tuple_fwd.hpp` concepts used by visitors)

**Read first:**
- `src/matrices/dense.hpp` + `dense.cpp` (heavy: zip_with, inner_product, chunk, repeat_n, stride, zip)
- `src/matrices/circulant.hpp` + `circulant.cpp` (heavy: sliding, zip_with, inner_product, repeat_n, stride, zip)
- `src/matrices/csr.hpp` + `csr.cpp` (moderate: sort, sliding, enumerate, transform)
- `src/matrices/coefficient_visitor.hpp` + `coefficient_visitor.cpp` (moderate: chunk, drop, for_each, zip)
- `src/matrices/unit_stride_visitor.hpp` + `unit_stride_visitor.cpp` (light: rs::size)
- `src/matrices/inner_block.hpp` + `inner_block.cpp` (no range-v3)
- `src/matrices/block.hpp` (no range-v3)
- `src/matrices/common.hpp` (no range-v3)
- `src/matrices/matrix_visitor.hpp` (no range-v3)
- `src/matrices/CMakeLists.txt`
- `plans/meta.md` (decision D6)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -L matrices
```

---

## Items

### Resolve Decisions

- [ ] **2.1** Resolve Decision D6 (Matrix-vector product migration). Update `plans/meta.md`.

### Core Matrix Types

- [ ] **2.2** Migrate `dense.cpp`: Replace the `vs::zip_with(inner_product, vs::chunk, vs::repeat_n)` MatVec pattern with explicit loops or a Kokkos kernel. Replace `vs::stride`/`vs::zip` accumulation loop. Remove range-v3 includes.
  - Files: `src/matrices/dense.hpp`, `src/matrices/dense.cpp`
  - Test: `ctest --test-dir build -R t-dense`

- [ ] **2.3** Migrate `dense.hpp`: Replace `rs::input_range` concept constraint on constructor with `std::ranges::input_range`. Replace `rs::copy`/`vs::take`.
  - Files: `src/matrices/dense.hpp`
  - Test: `ctest --test-dir build -R t-dense`

- [ ] **2.4** Migrate `circulant.cpp`: Replace `vs::sliding` + `vs::zip_with(inner_product, vs::repeat_n)` convolution pattern. Replace `vs::stride`/`vs::zip` for strided application. This is the core stencil-application kernel.
  - Files: `src/matrices/circulant.hpp`, `src/matrices/circulant.cpp`
  - Test: `ctest --test-dir build -R t-circulant`

- [ ] **2.5** Migrate `csr.cpp`: Replace `rs::sort` with `std::ranges::sort` or `std::sort`. Replace `vs::sliding(2) | vs::enumerate` in builder's `to_csr()`. Replace `vs::transform` for field extraction.
  - Files: `src/matrices/csr.hpp`, `src/matrices/csr.cpp`
  - Test: `ctest --test-dir build -R t-csr`

- [ ] **2.6** Migrate `csr.hpp`: Replace `ranges::input_range` concept on constructor template. Replace `rs::begin`/`rs::end`.
  - Files: `src/matrices/csr.hpp`
  - Test: `ctest --test-dir build -R t-csr`

### Visitors

- [ ] **2.7** Migrate `coefficient_visitor.cpp`: Replace `vs::chunk | vs::for_each(vs::drop)` Dirichlet column-skipping pattern. Replace `vs::zip` in all three `visit()` overloads.
  - Files: `src/matrices/coefficient_visitor.hpp`, `src/matrices/coefficient_visitor.cpp`
  - Test: `ctest --test-dir build -R t-coefficient_visitor`

- [ ] **2.8** Migrate `unit_stride_visitor.hpp/cpp`: Replace `rs::size`, `rs::begin`, `rs::end` with std equivalents.
  - Files: `src/matrices/unit_stride_visitor.hpp`, `src/matrices/unit_stride_visitor.cpp`
  - Test: `ctest --test-dir build -R t-unit_stride_visitor`

### Test Migration

- [ ] **2.9** Migrate all 7 test files in `src/matrices/` to remove range-v3 includes.
  - Test: `ctest --test-dir build -L matrices` — all pass.

---

## Completion Criteria

- All 7 matrix test files pass.
- No `#include <range/v3/...>` remains in `src/matrices/`.
- Decision D6 is recorded in `meta.md`.
- `shoccs-matrices` library no longer links range-v3 directly (may still get it transitively until fields is done).
