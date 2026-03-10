# Phase 1: Fields Subsystem

**Goal:** Migrate the fields subsystem from range-v3 view composition to Kokkos-compatible patterns. This is the highest-complexity phase due to 4 custom view adaptors and deep range-v3 integration.

**Depends on:** Phase 0

**Read first:**
- `src/fields/tuple_fwd.hpp` (concept hub — 15 rs/vs uses)
- `src/fields/selector.hpp` (71 rs/vs uses — highest in codebase)
- `src/fields/tuple_utils.hpp` (16 rs/vs uses — algorithmic workhorse)
- `src/fields/view_tuple.hpp` (12 rs/vs uses)
- `src/fields/tuple_math.hpp` (6 rs/vs uses)
- `src/fields/field.hpp` (14 rs/vs uses)
- `src/fields/field_utils.hpp` (4 rs/vs uses)
- `src/fields/algorithms.hpp` (9 rs/vs uses)
- `src/fields/container_tuple.hpp` (1 rs/vs use)
- `src/fields/tuple_pipe.hpp` (3 rs/vs uses)
- `src/fields/scalar.hpp`
- `src/fields/vector.hpp`
- `src/fields/tuple.hpp`
- `src/fields/field_math.hpp`
- `src/fields/selector_fwd.hpp`
- `src/fields/field_fwd.hpp`
- `src/fields/CMakeLists.txt`
- `plans/meta.md` (decisions D2, D4, D5)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -L fields
```

---

## Items

### Resolve Decisions

- [ ] **1.1** Resolve Decision D4 (Field storage migration). Read the field data structure hierarchy and decide. Update `plans/meta.md`.

- [ ] **1.2** Resolve Decision D5 (Selector/view adaptor replacement). Read `selector.hpp` thoroughly — it has `plane_view<0,1,2>`, `multi_slice_view`, `optional_view`, `predicate_view`. Update `plans/meta.md`.

### Foundation Types (tuple_fwd, concepts)

- [ ] **1.3** Migrate `src/fields/tuple_fwd.hpp`: Replace range-v3 concept usage (`rs::range`, `rs::viewable_range`, `rs::input_range`, etc.) with C++20 `std::ranges` equivalents. Replace `rs::ref_view` with `std::ranges::ref_view`. Replace `vs::view_closure` with a project-local equivalent or `std::ranges` equivalent. Remove `ranges::enable_view` specializations (use `std::ranges::enable_view` or `std::ranges::view_interface`).
  - Test: `ctest --test-dir build -L fields` — expect many failures initially; this is the keystone header.

### Utilities and Algorithms

- [ ] **1.4** Migrate `src/fields/tuple_utils.hpp`: Replace `rs::copy`, `rs::copy_n`, `rs::fill` with `std::ranges` equivalents. Replace `vs::all`, `vs::common`, `vs::zip_with` with std or manual equivalents.
  - Test: `ctest --test-dir build -R t-tuple_utils`

- [ ] **1.5** Migrate `src/fields/algorithms.hpp`: Replace `rs::minmax`, `rs::min`, `rs::max` with `std::ranges` equivalents.
  - Test: `ctest --test-dir build -R t-algorithms`

### View/Container Tuple

- [ ] **1.6** Migrate `src/fields/container_tuple.hpp`: Replace `rs::begin`/`rs::end` with `std::ranges::begin`/`end`.
  - Test: `ctest --test-dir build -R t-container_tuple`

- [ ] **1.7** Migrate `src/fields/view_tuple.hpp`: Replace `vs::all`/`vs::all_t` with `std::views::all`/`std::ranges::views::all_t`. Replace `rs::equal` with `std::ranges::equal`. The `single_view<A>` class inherits from `vs::all_t<A>` — this inheritance needs redesign.
  - Test: `ctest --test-dir build -R t-view_tuple`

### Math Operations

- [ ] **1.8** Migrate `src/fields/tuple_math.hpp`: Replace `vs::zip`, `vs::zip_with`, `vs::repeat_n` with std equivalents or manual loops. `vs::zip_with` has no direct C++23 equivalent — use `std::views::zip` + `std::views::transform` or manual iteration.
  - Test: `ctest --test-dir build -R t-tuple_math`

- [ ] **1.9** Migrate `src/fields/tuple_pipe.hpp`: Replace `vs::view_closure` concept checks.
  - Test: `ctest --test-dir build -R t-tuple_pipe`

### Field Type

- [ ] **1.10** Migrate `src/fields/field.hpp`: Replace `rs::swap_ranges` with `std::ranges::swap_ranges`. Replace `rs::size`, `rs::begin`, `rs::end` with std equivalents.
  - Test: `ctest --test-dir build -R t-field`

- [ ] **1.11** Migrate `src/fields/field_utils.hpp`: Replace `vs::zip`/`vs::zip_with` with std equivalents.
  - Test: `ctest --test-dir build -R t-field_utils`

- [ ] **1.12** Migrate `src/fields/field_math.hpp`: Depends on field_utils migration.
  - Test: `ctest --test-dir build -R t-field_math`

### Selectors (Highest Risk)

- [ ] **1.13** Replace `plane_view<0>` (X-plane): Currently uses `vs::drop_exactly | vs::take_exactly`. Replace with index-range or `std::views::drop`/`std::views::take` or Kokkos subview.
  - Test: `ctest --test-dir build -R t-selector`

- [ ] **1.14** Replace `plane_view<1>` (Y-plane): **Most complex custom view_adaptor.** Has hand-written iterator with strided access. Replace with pre-computed index array or custom iterator without range-v3 base classes.
  - Test: `ctest --test-dir build -R t-selector`

- [ ] **1.15** Replace `plane_view<2>` (Z-plane): Uses `vs::drop_exactly | vs::stride`. Replace with index arithmetic or `std::views` equivalents.
  - Test: `ctest --test-dir build -R t-selector`

- [ ] **1.16** Replace `multi_slice_view`: Custom `rs::view_adaptor` for discontiguous fluid regions. Replace with pre-computed index array of fluid cell indices.
  - Test: `ctest --test-dir build -R t-selector`

- [ ] **1.17** Replace `optional_view`: Custom `rs::view_adaptor` that conditionally empties a range. Replace with simple conditional or wrapper.
  - Test: `ctest --test-dir build -R t-selector`

- [ ] **1.18** Replace `predicate_view`: Custom `rs::view_adaptor` for filtered ranges. Replace with `std::views::filter` or index array.
  - Test: `ctest --test-dir build -R t-selector`

- [ ] **1.19** Replace `selection<L,R,Fn>` and `rs::semiregular_box_t`, `rs::make_view_closure`, `rs::bind_back`, `rs::compose` usage in selectors. These are range-v3 internal utilities.
  - Test: `ctest --test-dir build -R t-selector`

### Test Migration

- [ ] **1.20** Migrate all 15 test files in `src/fields/` to remove `#include <range/v3/all.hpp>` and use std equivalents or direct Kokkos.
  - Test: `ctest --test-dir build -L fields` — all pass.

---

## Completion Criteria

- All 15 field test files pass.
- No `#include <range/v3/...>` remains in `src/fields/`.
- Decisions D4 and D5 are recorded in `meta.md`.
- The `fields` INTERFACE library no longer links `range-v3::range-v3`.
