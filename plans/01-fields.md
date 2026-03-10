# Phase 1: Fields Subsystem

**Goal:** Migrate the fields subsystem from range-v3 view composition to C++20 `std::ranges` and project-local utilities. This is the highest-complexity phase due to 4 custom `view_adaptor` classes and deep range-v3 integration.

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
- `plans/meta.md` (decisions D2, D4, D5, D8)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -L fields
```

---

## Items

### Resolve Decisions

- [x] **1.1** Resolve Decision D4 (Field storage migration). Keep `std::vector` for Phase 1; defer `Kokkos::View` to GPU phase. Updated `plans/meta.md`.

- [x] **1.2** Resolve Decision D5 (Selector/view adaptor replacement). Replace `rs::view_adaptor` with custom `std::ranges::view_interface` classes; create project-local range utilities. Added D8 for C++20 utility strategy. Updated `plans/meta.md`.

### Utility Foundation

These project-local utilities replace range-v3 internal APIs that have no C++20 `std::ranges` equivalent. They must be created before migrating any production headers. See decision D8.

- [x] **1.2a** Create `src/fields/ccs_range_utils.hpp` with core range-v3 replacements:
  - `ccs::view_closure<Fn>`: Wraps a callable `Fn` and provides `template<typename Rng> friend auto operator|(Rng&& rng, view_closure fn)` for pipe syntax. Must be default-constructible if `Fn` is.
  - `ccs::make_view_closure(fn)`: Factory returning `view_closure{fn}`.
  - `ccs::bind_back(fn, args...)`: Returns a lambda that prepends forwarded arguments before the bound trailing arguments. Used in selector function composition.
  - `ccs::compose(f, g)`: Returns a lambda `[f,g](args...) { return f(g(args...)); }`. Used in selector `apply` chains.
  - `ccs::semiregular_box<Fn>`: Wrapper using `std::optional<Fn>` to make any callable default-constructible. Provides `operator()` forwarding.
  - Files: Create `src/fields/ccs_range_utils.hpp`.
  - Test: `cmake --build build` (header-only, compilation test).
  - Must come before: 1.3, 1.9, 1.13–1.19.

- [x] **1.2b** Create `src/fields/lazy_views.hpp` with lazy view replacements for range-v3 views with no C++20 equivalent:
  - `ccs::zip_transform_view<F, Rngs...>`: Lazy view that yields `f(*it1, *it2, ...)`. Must model `std::ranges::view_interface`. Iterators must support `==`, `++`, `*`. Needed by `tuple_math.hpp` (binary operators), `tuple_utils.hpp` (`lift`), and `field_utils.hpp` (`transform_scalar`/`transform_vector`). Support at least the binary (2-range) case; variadic is needed for `field_utils.hpp`.
  - `ccs::repeat_n_view<T>`: Lazy view of `n` copies of value `v`. Models `std::ranges::random_access_range` and `std::ranges::sized_range`. Used in `tuple_math.hpp` binary scalar operators (`vs::repeat_n(v, sz)`).
  - `ccs::stride_view<Rng>`: Lazy view that yields every `n`-th element of `Rng`. Models `std::ranges::view_interface`. Needed by `plane_view<2>` (z-plane selector) and test code. Must support bidirectional or random-access iteration if the base range does.
  - Helper factory functions: `ccs::zip_transform(f, rngs...)`, `ccs::repeat_n(v, n)`, `ccs::stride(rng, n)`.
  - Files: Create `src/fields/lazy_views.hpp`.
  - Test: `cmake --build build` (header-only, compilation test).
  - Must come before: 1.2c, 1.4, 1.8, 1.13, 1.15.

- [x] **1.2c** Fix `zip_transform_view` to propagate iterator category and support `sized_range` in `src/fields/lazy_views.hpp`:
  - **Problem:** The current `zip_transform_iterator` hardcodes `iterator_category = std::input_iterator_tag` and lacks `operator--`, `operator+=`/`-=`/`[]`/`+`/`-`/`<=>`. Downstream production code depends on `sized_range` and `random_access_range` when base ranges support them: `field.hpp:90,92` (`nscalars()`/`nvectors()` require `sized_range`), `field.hpp:103,120` (`scalars(i)` requires `random_access_range`), and `tuple_math.t.cpp:219` (`.size()` on view math results). The range-v3 `zip_with` and C++23 `zip_transform_view` both propagate the weakest category from base ranges.
  - Compute `iterator_concept` as the weakest iterator concept among all base range iterators (input/forward/bidirectional/random-access).
  - Add conditional `operator--` (when all bases are bidirectional), `operator+=`/`-=`/`[]`/`+`/`-`/`<=>` (when all bases are random-access) to `zip_transform_iterator`, guarded by `requires` clauses.
  - Add `size()` to `zip_transform_view` when all `Rngs` model `std::ranges::sized_range`, returning the minimum `std::ranges::size(rng)` across base ranges.
  - Files: `src/fields/lazy_views.hpp`
  - Test: `cmake --build build` (header-only, compilation test).
  - Must come before: 1.4d, 1.8b, 1.8c, 1.11.

### Foundation Types (tuple_fwd, concepts)

- [x] **1.3** Migrate `src/fields/tuple_fwd.hpp`. This is the keystone header — changes here affect all downstream files. Apply the following substitutions:
  - [x] **1.3a** Replace range-v3 concept usage with C++20 equivalents. **Note:** The `All` concept uses `(std::is_lvalue_reference_v<T> || std::ranges::view<std::remove_cvref_t<T>>)` instead of `std::ranges::viewable_range<T>` to preserve range-v3's narrower semantics — C++20's `viewable_range` additionally allows rvalue movable non-view types, which would break `viewable_range_by_value` CTAD deduction guides (would strip references from containers, causing copies instead of references).
    - Files: `src/fields/tuple_fwd.hpp`
  - [x] **1.3b** Replace `rs::ref_view<Rng>` with `std::ranges::ref_view<Rng>` in `is_ref_view_impl`.
    - Files: `src/fields/tuple_fwd.hpp`
  - [x] **1.3c** Replace `vs::view_closure<Fn>` with `ccs::view_closure<Fn>` in `is_view_closure_impl`. `ViewClosure` now checks for `ccs::view_closure` — range-v3 `vs::view_closure` types no longer satisfy it. Test `range_concepts.t.cpp` line 135 will fail until migrated in 1.20a.
    - Files: `src/fields/tuple_fwd.hpp`
  - [x] **1.3d** Replace `vs::common(...)` with `std::views::common(...)` in `constructible_from_range_impl`.
    - Files: `src/fields/tuple_fwd.hpp`
  - [x] **1.3e** Removed `rs::common_tuple<Args...>` specialization from `is_tuple_like_impl`. Test `range_concepts.t.cpp` line 173 (`NumericTuple` with `rs::common_tuple`) will fail until migrated in 1.20a.
    - Files: `src/fields/tuple_fwd.hpp`
  - [x] **1.3f** Replaced `namespace ranges { enable_view }` with `namespace std::ranges { enable_view }`. Range-v3's `ranges::enable_view` is no longer specialized (not in scope without range-v3 include). Downstream files that still mix range-v3 views with ccs::tuple may see compilation changes; these resolve when those files are migrated.
    - Files: `src/fields/tuple_fwd.hpp`
  - [x] **1.3g** Removed all 3 range-v3 includes. Added `#include <ranges>` and `#include "ccs_range_utils.hpp"`.
    - Files: `src/fields/tuple_fwd.hpp`
  - Test results: 8 fields targets compile (t-range_concepts, t-tuple_utils, t-container_tuple, t-single_view, t-algorithms, t-field, t-field_utils, t-field_math). 6 targets with pre-existing compilation failures have slightly increased error counts due to C++20 concepts evaluating differently for range-v3 types; these will resolve when those headers/tests are migrated. All runtime test failures are pre-existing.

### Utilities and Algorithms

- [x] **1.4** Migrate `src/fields/tuple_utils.hpp`:
  - [x] **1.4a** Replace algorithms: `rs::copy` → `std::ranges::copy`, `rs::copy_n` → `std::ranges::copy_n`, `rs::fill` → `std::ranges::fill`. Replace `rs::begin`/`rs::end`/`rs::size` → `std::ranges::begin`/`end`/`size`.
    - Files: `src/fields/tuple_utils.hpp` (`resize_and_copy`, `to`, `ssize` functions)
  - [x] **1.4b** ~~Replace `vs::all`/`vs::all_t` in `to()` function~~ — REMOVED: `vs::all` and `vs::all_t` are not used in `tuple_utils.hpp`. The `<range/v3/view/all.hpp>` include is unused (no direct `vs::all` usage in the file body) and its removal is already covered by 1.4f.
  - [x] **1.4c** Replace `vs::common(...)` → `std::views::common(...)` in `to()` function (line 342). Also replaced `rs::common_range` → `std::ranges::common_range` and `rs::begin`/`rs::end` → `std::ranges::begin`/`end` in the same function.
    - Files: `src/fields/tuple_utils.hpp`
  - [x] **1.4d** Replace `vs::zip_with(fn, rngs...)` with `ccs::zip_transform(fn, rngs...)` in `lift()` function (lines 381, 385).
    - Depends on: 1.2b
    - Files: `src/fields/tuple_utils.hpp`
  - [x] **1.4e** Replace `rs::range` and `rs::sized_range` concept usage in template constraints with `std::ranges` equivalents. Changed `rs::range... Args` → `std::ranges::range... Args` in `lift()` and `rs::sized_range X` → `std::ranges::sized_range X` in `ssize()`.
    - Files: `src/fields/tuple_utils.hpp`
  - [x] **1.4f** Remove all 7 range-v3 includes. Add `#include <algorithm>`, `#include <ranges>`, and `#include "lazy_views.hpp"`.
    - Files: `src/fields/tuple_utils.hpp`
  - Test: `ctest --test-dir build -R t-tuple_utils` — 15 passed, 2 failed (pre-existing: `resize_and_copy tuples to tuples`). Downstream targets (t-container_tuple, t-field, t-field_utils, t-field_math) now fail because they relied on transitive range-v3 includes through tuple_utils.hpp; will be fixed by items 1.5–1.9.

- [x] **1.5** Migrate `src/fields/algorithms.hpp`:
  - Replaced `rs::minmax` → `std::ranges::minmax`, `rs::min` → `std::ranges::min`, `rs::max` → `std::ranges::max`, `rs::minmax_result` → `std::ranges::minmax_result`.
  - Replaced `rs::begin`/`rs::end` → `std::ranges::begin`/`end`.
  - Replaced `rs::range_value_t` → `std::ranges::range_value_t`.
  - Removed 3 range-v3 includes. Added `#include <algorithm>`, `#include <ranges>`.
  - Files: `src/fields/algorithms.hpp`
  - Test: `ctest --test-dir build -R t-algorithms` — 1 passed.

### View/Container Tuple

- [x] **1.6** Migrate `src/fields/container_tuple.hpp`:
  - Replaced `rs::begin(r)`/`rs::end(r)` → `std::ranges::begin(r)`/`std::ranges::end(r)` in the range constructor (line 27).
  - No direct range-v3 includes to remove (container_tuple.hpp gets range-v3 transitively through `tuple_utils.hpp`, which is already migrated in 1.4).
  - Files: `src/fields/container_tuple.hpp`
  - Test: `ctest --test-dir build -R t-container_tuple` — 1 passed.

- [x] **1.7** Migrate `src/fields/view_tuple.hpp`:
  - [x] **1.7a** Replace `vs::all`/`vs::all_t` with `std::views::all`/`std::views::all_t` throughout (lines 34, 44, 48, 53, 75, etc.). Used in `view_tuple_base` member `std::tuple<vs::all_t<Args>...> v` and constructors/assignment.
    - Files: `src/fields/view_tuple.hpp`
  - [x] **1.7b** Replace `rs::equal` with `std::ranges::equal` in `operator==` (lines 115, 125).
    - Files: `src/fields/view_tuple.hpp`
  - [x] **1.7c** Redesign `single_view<A>` (lines 186–235). Currently inherits from `vs::all_t<A>` (a range-v3 view type). Replace with inheritance from `std::views::all_t<A>` (a `std::ranges::ref_view<A>` or `std::ranges::owning_view<A>`). The key behaviors to preserve:
    - `single_view<A>` makes a 1-element `view_tuple` directly iterable as a range.
    - Assignment uses placement-new destroy-and-reconstruct semantics.
    - The `using view = vs::all_t<A>` alias switches to `std::views::all_t<A>`.
    - Files: `src/fields/view_tuple.hpp`
  - [x] **1.7d** Remove `#include <range/v3/algorithm/equal.hpp>`, `#include <range/v3/view/all.hpp>`. Add `#include <algorithm>`, `#include <ranges>`.
    - Files: `src/fields/view_tuple.hpp`
  - Test: `t-view_tuple` target fails to compile because `view_tuple.t.cpp` still uses range-v3 types (will be fixed in 1.20d). Downstream targets (t-single_view, t-container_tuple, t-field, t-field_utils, t-field_math) all compile and pass.

### Math Operations

- [x] **1.8** Migrate `src/fields/tuple_math.hpp`:
  - [x] **1.8a** Replaced `vs::zip(out, in)` in compound-assignment operators with index-based iteration using `std::ranges::begin`/`end`.
    - Files: `src/fields/tuple_math.hpp`
  - [x] **1.8b** Replaced `vs::zip_with(f, rng, vs::repeat_n(v, sz))` in binary scalar operators with `ccs::zip_transform(f, FWD(rng), ccs::repeat_n(v, sz))`.
    - Files: `src/fields/tuple_math.hpp`
  - [x] **1.8c** Replaced `vs::zip_with(f, a, b)` in binary tuple-tuple operators with `ccs::zip_transform(f, FWD(a), FWD(b))`.
    - Files: `src/fields/tuple_math.hpp`
  - [x] **1.8d** Replaced `rs::size(rng)` → `std::ranges::size(rng)`.
    - Files: `src/fields/tuple_math.hpp`
  - [x] **1.8e** Removed 4 range-v3 includes. Added `#include <ranges>` and `#include "lazy_views.hpp"`.
    - Files: `src/fields/tuple_math.hpp`
  - **Cascading fixes required:** Removing range-v3 includes from `tuple_math.hpp` broke transitive include chains for downstream files. Fixed `selector_fwd.hpp` (1.12b), `field_utils.hpp` (1.11), added `<range/v3/view/view.hpp>` to `tuple_pipe.hpp`, added `<functional>` to `tuple.hpp`, and added `<range/v3/view/repeat_n.hpp>` to `field.t.cpp`.
  - **Infrastructure fix:** Wrapped `F` in `ccs::semiregular_box<F>` inside `zip_transform_view` (`lazy_views.hpp`). Lambdas (non-assignable) passed as `F` prevented `zip_transform_view` from satisfying `std::ranges::view` (which requires `movable`). Range-v3's `zip_with` handled this automatically via `semiregular_box_t<F>`.
  - Test: `t-tuple_math` fails to compile because test file still uses range-v3 types (will be fixed in 1.20c). All previously-passing downstream targets still pass (t-field, t-field_math, t-field_utils, t-single_view, t-container_tuple, t-algorithms).

- [x] **1.9** Migrate `src/fields/tuple_pipe.hpp`:
  - Replaced `vs::view_closure<ViewFn>` with `ccs::view_closure<ViewFn>` in the two `operator|` overloads (lines 54, 63–64). Replaced `#include <range/v3/view/view.hpp>` with `#include "ccs_range_utils.hpp"`.
  - Done concurrently with 1.19 (selector utility fn migration).
  - Files: `src/fields/tuple_pipe.hpp`
  - Test: `t-tuple_pipe` test file still uses range-v3 types (will compile after 1.20h2). All downstream targets (t-field, t-field_utils, t-field_math, t-single_view, t-container_tuple, t-algorithms) build and pass.

### Tuple Type

- [x] **1.9a** Migrate `src/fields/tuple.hpp`:
  - Replaced `vs::view_closure<ViewFn>` → `ccs::view_closure<ViewFn>` in deduction guides (lines 153–156).
  - Done concurrently with 1.19 (selector utility fn migration).
  - Files: `src/fields/tuple.hpp`
  - Test: `t-tuple` test file still uses range-v3 types (will compile after 1.20f). All downstream targets build and pass.

### Field Type

- [x] **1.10** Migrate `src/fields/field.hpp`:
  - Replaced `rs::swap_ranges` → `std::ranges::swap_ranges` (4 occurrences in swap method).
  - Replaced `rs::size` → `std::ranges::size`, `rs::begin`/`rs::end` → `std::ranges::begin`/`end`.
  - Replaced `rs::sized_range` → `std::ranges::sized_range`, `rs::random_access_range` → `std::ranges::random_access_range`.
  - Replaced `rs::range_reference_t` → `std::ranges::range_reference_t` (4 occurrences in assignment operators).
  - Removed `#include <range/v3/algorithm/swap_ranges.hpp>`. Added `#include <algorithm>`.
  - Files: `src/fields/field.hpp`
  - Test: `ctest --test-dir build -R t-field` — passed. All downstream targets (t-field_utils, t-field_math, t-single_view, t-container_tuple) also pass.

- [x] **1.11** Migrate `src/fields/field_utils.hpp`:
  - Replaced `vs::zip(FWD(t).scalars()...)` in `for_each_scalar`/`for_each_vector` with index-based iteration: `for (int i = 0; i < n; ++i) f(t.scalars()[i]...);` using the first argument's `nscalars()`/`nvectors()`.
  - Replaced `vs::zip_with(f, FWD(t).scalars()...)` in `transform_scalar`/`transform_vector` with `ccs::zip_transform(f, FWD(t).scalars()...)`.
  - Added `#include "lazy_views.hpp"`.
  - Files: `src/fields/field_utils.hpp`
  - Test: `ctest --test-dir build -R t-field_utils` — passed.

- [x] **1.12** Migrate `src/fields/field_math.hpp`: Verified — no `rs::`, `vs::`, or `range/v3` includes present. File uses only C++20 concepts and project-local utilities transitively.
  - Files: `src/fields/field_math.hpp`
  - Test: `ctest --test-dir build -R t-field_math` — passed.

- [x] **1.12a** Migrate `src/fields/field_fwd.hpp`:
  - Replaced `rs::range_value_t` → `std::ranges::range_value_t` (2 occurrences), `rs::range_reference_t` → `std::ranges::range_reference_t` (2 occurrences).
  - Files: `src/fields/field_fwd.hpp`
  - Test: `cmake --build build` — passed. All downstream field tests pass.

- [x] **1.12b** Migrate `src/fields/selector_fwd.hpp`:
  - Replaced `namespace ranges { ... enable_view ... }` with `namespace std::ranges { ... enable_view ... }`.
  - Files: `src/fields/selector_fwd.hpp`
  - Test: builds successfully as part of downstream targets.

### Selectors (Highest Risk)

All selector items depend on: 1.2a (ccs_range_utils.hpp) and 1.3 (tuple_fwd.hpp migration).

- [x] **1.13** Replace `plane_view<0>` (X-plane) in `src/fields/selector.hpp`:
  - Replaced `x_plane_t<Rng>` type alias: `vs::drop_exactly(int{}) | vs::take_exactly(integer{})` → `std::views::drop(int{}) | std::views::take(integer{})`.
  - Replaced `apply_` method: `vs::drop_exactly(i * n) | vs::take_exactly(n)` → `std::views::drop(i * n) | std::views::take(n)`.
  - Replaced `rs::semiregular_box_t<Fn>` → `ccs::semiregular_box<Fn>`.
  - Files: `src/fields/selector.hpp` (lines 173–202)
  - Test: `t-selector` test file still uses range-v3 (will compile after 1.20e). All downstream targets (t-field, t-field_utils, t-field_math, t-single_view, t-container_tuple, t-algorithms) build and pass.

- [x] **1.14** Replace `plane_view<1>` (Y-plane) in `src/fields/selector.hpp`:
  - Rewrote as a class inheriting from `std::ranges::view_interface<plane_view<1, Rng, Fn>>` with a custom `iterator` class implementing the non-contiguous y-plane stride pattern.
  - Iterator stores base iterator + grid dimensions (nx, ny, nz) + logical position (i, k). Implements all random-access operations: `++`, `--`, `+=`, `-=`, `[]`, `+`, `-`, `==`, `<=>`.
  - `begin()` positions at `j * nz`, `end()` positions at `(nx-1)*ny*nz + j*nz + nz`.
  - Added explicit `size() const` returning `nx * nz`.
  - Replaced `rs::view_adaptor` → `std::ranges::view_interface`, `rs::range_difference_t` → `std::ranges::range_difference_t`, `rs::semiregular_box_t` → `ccs::semiregular_box`, `rs::begin`/`rs::advance` → `std::ranges::begin`/`advance`.
  - Files: `src/fields/selector.hpp` (lines 208–393)
  - Test: `t-selector` test file still uses range-v3 (will compile after 1.20e). All downstream targets (t-field, t-field_utils, t-field_math, t-single_view, t-container_tuple, t-algorithms) build and pass.

- [x] **1.15** Replace `plane_view<2>` (Z-plane) in `src/fields/selector.hpp`:
  - Changed `z_plane_t<Rng>` type alias: `vs::drop_exactly(int{}) | vs::stride(integer{})` → `ccs::stride(rng | std::views::drop(int{}), integer{})`.
  - Changed constructor: `FWD(rng) | vs::drop_exactly(k) | vs::stride(extents[2])` → `ccs::stride(FWD(rng) | std::views::drop(k), extents[2])`.
  - Replaced `rs::semiregular_box_t<Fn>` → `ccs::semiregular_box<Fn>`.
  - Files: `src/fields/selector.hpp` (lines 401–425)
  - Test: `t-selector` test file still uses range-v3 (will compile after 1.20e). All downstream targets (t-field, t-field_utils, t-field_math, t-single_view, t-container_tuple, t-algorithms) build and pass.

- [x] **1.16** Replace `multi_slice_view` in `src/fields/selector.hpp`:
  - Rewrote as a class inheriting from `std::ranges::view_interface<multi_slice_view<Rng, Fn>>` with a custom `iterator` class implementing the slice-navigation logic.
  - Iterator stores: `base_iter`, `slice_it` (current slice), `last_slice_`, `integer i_` (position in base range), `integer multi_i_` (logical position). Models `std::random_access_iterator`.
  - Implements all random-access operations: `++`, `--`, `+=`, `-=`, `[]`, `+`, `-`, `==`, `<=>`.
  - Fixed potential UB in `operator--` and `operator+=` (negative) when iterator is at end position (slice iterator past end of span) — now properly decrements slice iterator before dereferencing.
  - `begin()` positions at first slice's first element; `end()` positions past last slice with `multi_i` = total element count.
  - Replaced `rs::view_adaptor` → `std::ranges::view_interface`, `rs::range_difference_t` → `std::ranges::range_difference_t`, `rs::semiregular_box_t` → `ccs::semiregular_box`, `rs::begin`/`rs::end`/`rs::advance` → `std::ranges::begin`/`end`/`advance`.
  - Files: `src/fields/selector.hpp`
  - Test: `t-selector` test file still uses range-v3 (will compile after 1.20e). All downstream targets (t-field, t-field_utils, t-field_math, t-single_view, t-container_tuple, t-algorithms) build and pass.

- [x] **1.17** Replace `optional_view` in `src/fields/selector.hpp`:
  - Rewrote as a class inheriting from `std::ranges::view_interface<optional_view<Rng, Fn>>` with explicit `begin()`, `end()`, and `base()` methods.
  - `begin()` returns `keep_bounds ? std::ranges::begin(base_) : std::ranges::end(base_)` (empty range when `keep_bounds` is false).
  - `end()` returns `std::ranges::end(base_)`.
  - Added `base()` accessor (needed by `apply()` for nested view composition).
  - Replaced `rs::view_adaptor` → `std::ranges::view_interface`, `rs::semiregular_box_t` → `ccs::semiregular_box`.
  - Removed `adaptor` inner class, `friend rs::range_access`, `begin_adaptor`/`end_adaptor` methods.
  - Files: `src/fields/selector.hpp`
  - Test: `t-selector` test file still uses range-v3 (will compile after 1.20e). All downstream targets (t-field, t-field_utils, t-field_math, t-single_view, t-container_tuple, t-algorithms) build and pass.

- [x] **1.18** Replace `predicate_view` in `src/fields/selector.hpp`:
  - Rewrote as a class inheriting from `std::ranges::view_interface<predicate_view<Rng, Pred, Fn>>` with a custom `iterator` class implementing filter-style bidirectional iteration.
  - Iterator stores: `base_it_`, `base_end_`, `pred_it_`, `pred_end_`. Advances both base and predicate iterators in lockstep, skipping elements where predicate is false (`satisfy_forward`). Supports `operator--` via `satisfy_reverse` logic (decrement until predicate is true).
  - Models `std::bidirectional_iterator` (no random access, matching original).
  - `begin()` caches the initial satisfied position (both base and predicate iterators) via `cached_begin_` and `cached_pred_begin_` members.
  - `end()` returns iterator at `std::ranges::end(base_)` / `std::ranges::end(pred_)`.
  - Added `base()` accessor (needed by `apply()` for nested view composition).
  - Replaced `rs::view_adaptor` → `std::ranges::view_interface`, `rs::semiregular_box_t` → `ccs::semiregular_box`, `rs::begin`/`rs::end`/`rs::size`/`rs::iterator_t` → `std::ranges` equivalents. Removed `adaptor` inner class, `friend rs::range_access`, `begin_adaptor`/`end_adaptor` methods.
  - Files: `src/fields/selector.hpp`
  - Test: `t-selector` test file still uses range-v3 (will compile after 1.20e). All downstream targets (t-field, t-field_utils, t-field_math, t-single_view, t-container_tuple, t-algorithms) build and pass.

- [x] **1.18a** Restore dropped size assertion in `predicate_view` constructor (`src/fields/selector.hpp`):
  - Added `assert(std::ranges::size(base_) == std::ranges::size(pred_));` to the constructor body.
  - Added `#include <cassert>` to `selector.hpp`.
  - Files: `src/fields/selector.hpp`
  - Test: `cmake --build build` — all 6 downstream field targets compile and pass.

- [x] **1.19** Replace range-v3 utility usage in selector function objects (`selection`, `plane_selection_fn`, `multi_slice_fn`, `optional_view_fn`, `predicate_view_fn`):
  - [x] **1.19a** Replaced `rs::semiregular_box_t<Fn>` → `ccs::semiregular_box<Fn>` in `selection` struct.
  - [x] **1.19b** Replaced `rs::make_view_closure(...)` → `ccs::make_view_closure(...)` in `selection_view`.
  - [x] **1.19c** Replaced `rs::bind_back(...)` → `ccs::bind_back(...)` and `rs::compose(...)` → `ccs::compose(...)` in `plane_selection_base_fn`.
  - [x] **1.19d** Replaced `rs::make_view_closure(rs::bind_back(...))` → `ccs::make_view_closure(ccs::bind_back(...))` and `rs::bind_back(*this, plane_coord)` → `ccs::bind_back(*this, plane_coord)` in `plane_selection_fn::operator()`.
  - [x] **1.19e** Replaced `rs::make_view_closure`, `rs::bind_back`, `rs::compose` → `ccs::` equivalents in `multi_slice_fn` and `multi_slice_base_fn`.
  - [x] **1.19f** Replaced `rs::bind_back` → `ccs::bind_back`, `rs::make_view_closure` → `ccs::make_view_closure` in `optional_view_fn`.
  - [x] **1.19g** Replaced `rs::bind_back`, `rs::compose`, `rs::make_view_closure` → `ccs::` equivalents in `predicate_view_fn` and `predicate_view_base_fn`.
  - [x] **1.19h** Removed 3 range-v3 includes. Added `#include <ranges>`, `#include "ccs_range_utils.hpp"`, `#include "lazy_views.hpp"`.
  - **Infrastructure fix:** Added `std::invocable<Fn const&, Rng>` / `std::invocable<Fn, Rng>` constraints to `ccs::view_closure::operator|` in `ccs_range_utils.hpp`. Without this, the unconstrained `operator|` made `is_pipeable` return true for leaf ranges (e.g., `std::vector<double>`) that can't actually be called by the closure's inner function, causing the wrong `tuple_pipe` overload to be selected.
  - Files: `src/fields/selector.hpp`, `src/fields/ccs_range_utils.hpp`
  - Test: `t-selector` test file still uses range-v3 (will compile after 1.20e). All 6 downstream field test targets build and pass.

### Test Migration

Migrate test files to remove `#include <range/v3/all.hpp>` and all `rs::`/`vs::` usage. Replace with `std::ranges`/`std::views` (C++20) and project-local utilities. Common replacements across all test files:
- `vs::iota(a, b)` → `std::views::iota(a, b)` (C++20)
- `vs::transform(f)` → `std::views::transform(f)` (C++20)
- `rs::equal(a, b)` → `std::ranges::equal(a, b)` (C++20)
- `rs::size(r)` → `std::ranges::size(r)` (C++20)
- `rs::begin(r)`/`rs::end(r)` → `std::ranges::begin(r)`/`std::ranges::end(r)` (C++20)
- `vs::all(x)` → `std::views::all(x)` (C++20)
- `vs::take(n)` / `vs::take_exactly(n)` → `std::views::take(n)` (C++20)
- `vs::drop_exactly(n)` → `std::views::drop(n)` (C++20)
- `vs::repeat_n(v, n)` → `std::vector<T>(n, v)` (eager, test-only) or `ccs::repeat_n(v, n)` (lazy)
- `vs::concat(a, b, ...)` → `std::vector{...}` with values listed, or helper that copies multiple ranges into a vector
- `vs::zip(a, b)` → index-based iteration
- `vs::zip_with(f, a, b)` → `ccs::zip_transform(f, a, b)` or construct expected vector manually
- `vs::stride(n)` → `ccs::stride(rng, n)` or manual iteration
- `rs::to<T>()` → `T(std::ranges::begin(r), std::ranges::end(r))`
- `vs::generate_n(f, n)` → manual loop building a vector
- `vs::join` → `std::views::join` (C++20)
- `rs::random_access_range<T>` → `std::ranges::random_access_range<T>` (C++20)
- `rs::output_range<T, V>` → `std::ranges::output_range<T, V>` (C++20)
- `rs::common_tuple<...>` → remove (no longer needed)
- `vs::cartesian_product(a, b, ...)` → nested for loops (test-only; C++23 `std::views::cartesian_product` not available in C++20)
- `rs::accumulate(rng, init, op)` → `std::accumulate(std::ranges::begin(r), std::ranges::end(r), init, op)` (from `<numeric>`)
- `rs::empty_view<T>{}` → `std::ranges::empty_view<T>{}` (C++20)
- `rs::minmax(rng)` → `std::ranges::minmax(rng)`, `rs::minmax_result<V>` → `std::ranges::minmax_result<V>` (C++20)
- `rs::min(a, b)` → `std::ranges::min(a, b)`, `rs::max(a, b)` → `std::ranges::max(a, b)` (C++20)
- `vs::repeat(v)` → `ccs::repeat_n(v, n)` where count is known, or manual pattern
- `rs::make_view_closure(fn)` → `ccs::make_view_closure(fn)` (where used in test code)

- [x] **1.20a** Migrate `src/fields/range_concepts.t.cpp`: Removed `#include <range/v3/all.hpp>` and `<iostream>`. Added `#include <algorithm>`, `#include <ranges>`, `#include "ccs_range_utils.hpp"`, `#include "lazy_views.hpp"`. Replaced all `rs::`/`vs::` usage with `std::ranges`/`std::views`/`ccs::` equivalents. Rewrote ViewClosure test with `ccs::make_view_closure`. Removed `rs::common_tuple` test lines. Rewrote `generate_n` test with manual loop. Rewrote x_plane_view/z_plane_view classes and tests to use `std::views::drop`/`take`/`ccs::stride`. Removed `vs::zip` expansion test entirely. Rewrote z_plane_view test to use vector instead of range-v3 streaming.
  - Files: `src/fields/range_concepts.t.cpp`
  - Test: `ctest --test-dir build -R t-range_concepts` — 1 passed. All downstream field targets unaffected.

- [x] **1.20a1** Add value verification to `z_plane_view` test in `src/fields/range_concepts.t.cpp` (line 278–280). Replaced `REQUIRE(std::ranges::size(zpv) > 0)` with `REQUIRE(std::ranges::equal(zpv, std::vector{0, 4, 8, 12, 16, 20}))` (z-plane 0 with extents {2,3,4}: drop(0), stride(4) over [0..23] yields every 4th element).
  - Files: `src/fields/range_concepts.t.cpp`
  - Test: `ctest --test-dir build -R t-range_concepts` — 1 passed.

- [x] **1.20b** Migrate `src/fields/tuple_utils.t.cpp` (611 lines, heaviest `rs::` usage): Removed `#include <range/v3/all.hpp>` and `#include <iostream>`. Added `#include <algorithm>`, `#include <numeric>`, `#include <ranges>`, `#include "lazy_views.hpp"`. Replaced:
  - `vs::zip` in for loops → index-based iteration (3 occurrences).
  - `vs::zip_with(..., vs::repeat(1))` → `ccs::zip_transform(..., ccs::repeat_n(1, size))`.
  - `vs::zip_with(std::plus{}, ...)` → `ccs::zip_transform(std::plus{}, ...)`.
  - `vs::repeat_n(-1, n)` → `std::vector<int>(n, -1)` (5 occurrences).
  - `rs::accumulate(rng, init)` → `std::accumulate(std::ranges::begin(r), std::ranges::end(r), init)` (10 occurrences).
  - `rs::equal` → `std::ranges::equal`, `rs::size` → `std::ranges::size`, `vs::iota` → `std::views::iota`, `vs::all` → `std::views::all`, `vs::transform` → `std::views::transform`.
  - `rs::max`/`rs::min`/`rs::minmax`/`rs::minmax_result` → `std::ranges::` equivalents.
  - Files: `src/fields/tuple_utils.t.cpp`
  - Test: `ctest --test-dir build -R t-tuple_utils` — 1 passed (all 17 subcases). All 8 previously-passing downstream targets still pass.

- [x] **1.20c** Migrate `src/fields/tuple_math.t.cpp`: Removed `#include <range/v3/...>` (6 headers). Added `#include <algorithm>`, `#include <ranges>`, `#include "lazy_views.hpp"`. Replaced `vs::zip_with(std::plus{}, a, b)` → `ccs::zip_transform(std::plus{}, a, b)` (4 occurrences). Replaced `vs::iota` → `std::views::iota`, `rs::size` → `std::ranges::size`, `vs::repeat(6)` → `ccs::repeat_n(6, 2 * i + 1)`. Added extra parentheses around all `REQUIRE((x == y))` comparisons involving `ccs::tuple` types to prevent Catch2 `ExprLhs` decomposition, which triggers C++20 ambiguous reversed operator== errors.
  - Files: `src/fields/tuple_math.t.cpp`
  - Test: `ctest --test-dir build -R t-tuple_math` — 1 passed (all 5 test cases). All 9 previously-passing downstream field targets still pass.

- [ ] **1.20d** Migrate `src/fields/view_tuple.t.cpp` (456 lines): Remove range-v3 includes. Replace:
  - `vs::repeat_n` (line 133) and `vs::zip_with` (lines 352, 366–367) → project-local or manual equivalents.
  - `rs::empty_view<T>{}` → `std::ranges::empty_view<T>{}`.
  - `rs::make_view_closure(fn)` → `ccs::make_view_closure(fn)`.
  - `rs::equal` → `std::ranges::equal`, `rs::size` → `std::ranges::size`.
  - Files: `src/fields/view_tuple.t.cpp`
  - Test: `ctest --test-dir build -R t-view_tuple`

- [ ] **1.20e** Migrate `src/fields/selector.t.cpp`: This is the largest test file (723 lines, 266 `rs::`/`vs::` occurrences). Split into 3 sub-items due to diff size. Remove `#include <range/v3/all.hpp>`. Add `#include <algorithm>`, `#include <ranges>`, `#include "lazy_views.hpp"`.

  Common patterns across all sub-items:
  - `vs::repeat_n(v, n)` → `std::vector<int>(n, v)` (~66 occurrences total)
  - `vs::concat(a, b, ...)` → `std::vector<int>{...}` with values listed out, or a test-local `concat_to_vec` helper (~24 occurrences total)
  - `vs::iota(a, b)` → `std::views::iota(a, b)` (many occurrences)
  - `rs::equal(a, b)` → `std::ranges::equal(a, b)`, `rs::size(r)` → `std::ranges::size(r)`
  - `ViewClosure<...>` static_asserts (lines 26–31, 188) remain valid since selector closures use `ccs::view_closure` after migration

  - [ ] **1.20e1** Migrate plane selector tests (lines 1–181, ~61 `rs::`/`vs::` occurrences): Tests for `planes construction`, `planes extraction`, `planes assignment`, `planes scalar extraction/assignment`, `planes vector extraction/assignment`.
    - Replace `vs::stride(n) | vs::take_exactly(n)` → `ccs::stride(rng, n)` piped to `std::views::take(n)` (lines 70, 115).
    - Replace `vs::repeat_n` in REQUIRE comparisons → `std::vector<int>(n, v)`.
    - Replace `vs::iota` → `std::views::iota`.
    - Replace `rs::equal` → `std::ranges::equal` (~20 occurrences in this section).
    - Files: `src/fields/selector.t.cpp` (lines 1–181)
    - Test: `ctest --test-dir build -R t-selector`

  - [ ] **1.20e2** Migrate multi_slice tests (lines 183–461, ~127 `rs::`/`vs::` occurrences): Tests for `multi_slice construction`, `multi_slice extraction`, `multi_slice assignment`, `multi_slice scalar extraction/assignment`, `multi_slice vector extraction/assignment`, `default operators`.
    - This section has the heaviest `vs::concat(vs::repeat_n(...), vs::iota(...), ...)` nesting. Each `vs::concat(...)` should be replaced with the computed `std::vector<int>{...}` literal. For example: `vs::concat(vs::repeat_n(-1, 1), vs::repeat_n(-2, 7), vs::repeat_n(-1, 14), vs::repeat_n(-2, 2))` → `std::vector<int>{-1, -2,-2,-2,-2,-2,-2,-2, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -2,-2}` (or use a helper).
    - Consider adding a test-local helper: `template<typename... Rngs> auto concat_vec(Rngs&&... rngs)` that copies ranges into a single `std::vector`.
    - Replace `rs::equal`, `rs::size` → `std::ranges` equivalents (~30 occurrences).
    - Files: `src/fields/selector.t.cpp` (lines 183–461)
    - Test: `ctest --test-dir build -R t-selector`

  - [ ] **1.20e3** Migrate optional/predicate tests (lines 462–723, ~78 `rs::`/`vs::` occurrences): Tests for `optional tuple`, `optional scalar`, `optional vector`, `multi_slice math`, `predicate extraction`, `predicate assignment`, `predicate scalar extraction/assignment`.
    - Replace `vs::repeat_n`, `vs::iota`, `vs::transform`, `vs::stride` patterns.
    - The `dble` lambda at line 14 uses `lift(std::plus{})` which calls `vs::zip_with` internally — after migration this uses `ccs::zip_transform`. No test code change needed, but verify the `dble(vs::iota(...))` calls work with `std::views::iota`.
    - `vs::stride(2)` at line 687 → `ccs::stride(rng, 2)` (note: used as `vs::iota(0, 12) | vs::stride(2)`, replace with manual vector or pipe through `ccs::stride`).
    - Replace `rs::equal`, `rs::size` → `std::ranges` equivalents.
    - Remove the `#include <range/v3/all.hpp>` and finalize includes.
    - Files: `src/fields/selector.t.cpp` (lines 462–723)
    - Test: `ctest --test-dir build -R t-selector`

- [ ] **1.20f** Migrate `src/fields/tuple.t.cpp`: Remove range-v3 includes. Replace `vs::take_exactly` (line 98), `vs::concat` (line 191), `vs::generate_n` + `rs::to` (line 305), `vs::repeat_n` (lines 471, 485), `rs::equal`, `rs::size`.
  - Files: `src/fields/tuple.t.cpp`
  - Test: `ctest --test-dir build -R t-tuple`

- [ ] **1.20g** Migrate `src/fields/scalar.t.cpp` (266 lines) and `src/fields/vector.t.cpp` (354 lines): Remove range-v3 includes. Replace:
  - `vs::repeat_n` (~20 occurrences across both files) → `std::vector<T>(n, v)` or `ccs::repeat_n`.
  - `vs::cartesian_product(a, b, ...)` → nested for loops (used in both files for constructing test grids; no C++20 equivalent).
  - `vs::generate_n(f, n)` → manual loop building a vector (in `scalar.t.cpp`).
  - `rs::equal` → `std::ranges::equal` (heavily used in both files, ~118 occurrences combined).
  - Files: `src/fields/scalar.t.cpp`, `src/fields/vector.t.cpp`
  - Test: `ctest --test-dir build -R t-scalar && ctest --test-dir build -R t-vector`

- [ ] **1.20h1** Migrate `src/fields/container_tuple.t.cpp` (327 lines, heaviest of remaining tests): Remove range-v3 includes. Replace `rs::equal` → `std::ranges::equal`, `rs::size` → `std::ranges::size`, `rs::begin`/`rs::end` → `std::ranges::begin`/`end` (~49 `rs::` occurrences). Replace `vs::iota` → `std::views::iota`, `vs::transform` → `std::views::transform`.
  - Files: `src/fields/container_tuple.t.cpp`
  - Test: `ctest --test-dir build -R t-container_tuple`

- [ ] **1.20h2** Migrate `src/fields/tuple_pipe.t.cpp` (160 lines), `src/fields/single_view.t.cpp` (27 lines), and `src/fields/algorithms.t.cpp` (65 lines): Remove range-v3 includes. Replace `rs::equal` → `std::ranges::equal`, `rs::size` → `std::ranges::size`, `vs::iota` → `std::views::iota`, `vs::transform` → `std::views::transform`.
  - Files: `src/fields/tuple_pipe.t.cpp`, `src/fields/single_view.t.cpp`, `src/fields/algorithms.t.cpp`
  - Test: `ctest --test-dir build -R "t-tuple_pipe|t-single_view|t-algorithms"`

- [ ] **1.20h3** Migrate `src/fields/field.t.cpp` (169 lines), `src/fields/field_utils.t.cpp` (60 lines), and `src/fields/field_math.t.cpp` (106 lines): Remove range-v3 includes where present. Replace `rs::begin`/`rs::end` → `std::ranges::begin`/`end`, `rs::equal` → `std::ranges::equal`, `vs::repeat_n` → `std::vector<T>(n, v)` or `ccs::repeat_n`, `vs::iota` → `std::views::iota`. Note: `field.t.cpp` has no direct `#include <range/v3/...>` (gets range-v3 transitively) but may need `#include <algorithm>` and `#include <ranges>` after transitive includes are removed.
  - Files: `src/fields/field.t.cpp`, `src/fields/field_utils.t.cpp`, `src/fields/field_math.t.cpp`
  - Test: `ctest --test-dir build -R "t-field$|t-field_utils|t-field_math"`

- [ ] **1.20i** Migrate or remove `src/fields/view_tuple_seg.cpp`. This is a standalone scratch/debug executable (`add_executable(seg ...)` in CMakeLists.txt) with 7 range-v3 includes (`equal`, `all`, `concat`, `iota`, `repeat_n`, `take`, `zip_with`). It is **not** a unit test. Options:
  - (a) Delete the file and remove the `seg` target from CMakeLists.txt (preferred — it appears to be unused scratch code with commented-out lines).
  - (b) Migrate its range-v3 usage to `std::ranges`/`std::views` and project-local equivalents.
  - Files: `src/fields/view_tuple_seg.cpp`, `src/fields/CMakeLists.txt`
  - Test: `cmake --build build`

### CMake Cleanup

- [ ] **1.21** Remove `range-v3::range-v3` from the `fields` INTERFACE library link and clean up the `seg` target:
  - In `src/fields/CMakeLists.txt` line 4: change `target_link_libraries(fields INTERFACE range-v3::range-v3 Boost::boost)` to `target_link_libraries(fields INTERFACE Boost::boost)`.
  - If `view_tuple_seg.cpp` was not deleted in 1.20i, ensure the `seg` target (line 22–23) no longer depends on range-v3. If deleted, remove the `add_executable(seg ...)` and `target_link_libraries(seg ...)` lines.
  - Verify no `#include <range/v3/...>` remains in any `src/fields/` file: `grep -rn 'range/v3' src/fields/` should return nothing.
  - Files: `src/fields/CMakeLists.txt`
  - Test: `cmake --build build && ctest --test-dir build -L fields` — all pass, no range-v3 headers.

---

## Ordering Constraints

```
1.2a (ccs_range_utils.hpp) ──┬── 1.3c → 1.3 (tuple_fwd.hpp)
                              ├── 1.19 (selector utility fns)
                              └── 1.9, 1.9a (MUST come after 1.19)

1.19 (selector utility fns) ──── 1.9, 1.9a (tuple_pipe.hpp, tuple.hpp view_closure migration)

1.2b (lazy_views.hpp) ──── 1.2c (zip_transform fix) ──┬── 1.4d (tuple_utils.hpp lift)
                                                        ├── 1.8b, 1.8c (tuple_math.hpp)
                                                        └── 1.11 (field_utils.hpp) [DONE]
1.2b (lazy_views.hpp) ───────┬── 1.15 (z-plane stride_view)
                              └── 1.19h (selector includes)

1.3 (tuple_fwd.hpp) ─────────┬── 1.4–1.12b (all downstream headers)
                              └── 1.13–1.19 (selectors)

1.4 (tuple_utils.hpp) ───────── 1.6, 1.7, 1.8, 1.9 (all include tuple_utils)

1.13–1.19 (selectors) ───────── 1.20e1, 1.20e2, 1.20e3 (selector tests)

1.2a, 1.2b ──────────────────── 1.20a (range_concepts.t.cpp — needs ccs_range_utils + lazy_views)

1.4–1.19 (all headers) ──────── 1.20a–1.20h3 (test migration)

1.20a–1.20h3, 1.20e1–1.20e3, 1.20i (tests + seg.cpp) ── 1.21 (CMake cleanup)
```

---

## Completion Criteria

- All 15 field test files pass.
- No `#include <range/v3/...>` remains in `src/fields/`.
- Decisions D4 and D5 are recorded in `meta.md`.
- The `fields` INTERFACE library no longer links `range-v3::range-v3`.
