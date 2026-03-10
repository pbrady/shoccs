# Phase 3: Stencils Subsystem

**Goal:** Migrate stencil coefficient construction from range-v3 view pipelines to standard C++ or Kokkos equivalents.

**Depends on:** Phase 0

**Read first:**
- `src/stencils/stencil.hpp` (no range-v3)
- `src/stencils/stencil.cpp` (no range-v3)
- `src/stencils/E2_1.cpp` (2509 lines — heaviest; uses copy, fill, reverse, concat, repeat, take, transform)
- `src/stencils/E2_2.cpp` (light: fill, reverse)
- `src/stencils/E4_2.cpp` (light: fill, reverse)
- `src/stencils/E4u_1.cpp` (moderate: copy, fill, reverse, concat, repeat, take, transform)
- `src/stencils/E6u_1.cpp` (moderate: same pattern)
- `src/stencils/E8u_1.cpp` (moderate: same pattern)
- `src/stencils/polyE2_1.cpp` (moderate: 3 concat+repeat+take pipelines, reverse)
- `src/stencils/CMakeLists.txt`
- `plans/meta.md`

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -L stencils
```

---

## Items

### Common Pattern: `rs::copy(vs::concat(a, vs::repeat(0.0)) | vs::take(n), rs::begin(out))`

This pattern zero-pads a coefficient array `a` to length `n`. It appears in E2_1, E4u_1, E6u_1, E8u_1, and polyE2_1. Replace with a simple helper function:
```cpp
void copy_zero_padded(span<const real> src, span<real> dst) {
    auto n = std::min(src.size(), dst.size());
    std::copy_n(src.begin(), n, dst.begin());
    std::fill(dst.begin() + n, dst.end(), 0.0);
}
```

- [ ] **3.1** Create a shared `copy_zero_padded` utility (either in `stencil.hpp` or a small stencils utility header). This replaces the `concat | repeat | take` pipeline used across all stencil files.
  - Test: build succeeds.

- [ ] **3.2** Migrate `E2_2.cpp`: Replace `rs::fill` with `std::fill` or `std::ranges::fill`. Replace `ranges::reverse` with `std::ranges::reverse`.
  - Test: `ctest --test-dir build -R t-E2_2`

- [ ] **3.3** Migrate `E4_2.cpp`: Same pattern as E2_2.
  - Test: `ctest --test-dir build -R t-E4_2`

- [ ] **3.4** Migrate `E4u_1.cpp`: Replace `copy(concat(a, repeat(0.0)) | take(n), begin(out))` with `copy_zero_padded`. Replace `ranges::reverse`.
  - Test: `ctest --test-dir build -R t-E4u_1`

- [ ] **3.5** Migrate `E6u_1.cpp`: Same pattern as E4u_1.
  - Test: `ctest --test-dir build -R t-E6u_1`

- [ ] **3.6** Migrate `E8u_1.cpp`: Same pattern as E4u_1.
  - Test: `ctest --test-dir build -R t-E8u_1`

- [ ] **3.7** Migrate `polyE2_1.cpp`: Three `copy_zero_padded` replacements plus `ranges::reverse`.
  - Test: `ctest --test-dir build -R t-polyE2_1`

- [ ] **3.8** Migrate `E2_1.cpp` (2509 lines — largest): Has the most complex coefficient tables. Same `concat|repeat|take` pattern but many instances. Also uses `vs::transform` for coefficient generation.
  - Test: `ctest --test-dir build -R t-E2_1`

- [ ] **3.9** Remove all range-v3 includes from stencil source files. Verify no transitive range-v3 dependency remains.
  - Test: `ctest --test-dir build -L stencils` — all pass.

---

## Completion Criteria

- All 7 stencil test files pass.
- No `#include <range/v3/...>` remains in `src/stencils/`.
- The `shoccs-stencils` library no longer depends on range-v3.
