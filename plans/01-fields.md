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

- [ ] **1.3** Migrate `src/fields/tuple_fwd.hpp`. This is the keystone header — changes here affect all downstream files. Apply the following substitutions:
  - [ ] **1.3a** Replace range-v3 concept usage with C++20 equivalents:
    - `rs::range<T>` → `std::ranges::range<T>`
    - `rs::viewable_range<T>` → `std::ranges::viewable_range<T>`
    - `rs::input_range<T>` → `std::ranges::input_range<T>`
    - `rs::output_range<T, V>` → `std::ranges::output_range<T, V>`
    - `rs::common_range<T>` → `std::ranges::common_range<T>`
    - `rs::sized_range<T>` → `std::ranges::sized_range<T>`
    - `rs::range_value_t<T>` → `std::ranges::range_value_t<T>`
    - `rs::range_reference_t<T>` → `std::ranges::range_reference_t<T>`
    - `rs::range_difference_t<T>` → `std::ranges::range_difference_t<T>`
    - `rs::iterator_t<T>` → `std::ranges::iterator_t<T>`
    - `rs::sentinel_t<T>` → `std::ranges::sentinel_t<T>`
    - `rs::difference_type_t<I>` → `std::iter_difference_t<I>`
    - Files: `src/fields/tuple_fwd.hpp`
    - Test: `cmake --build build` — expect many failures; this is foundational.
  - [ ] **1.3b** Replace `rs::ref_view<Rng>` with `std::ranges::ref_view<Rng>` in the `is_ref_view_impl` specialization (line 586).
    - Files: `src/fields/tuple_fwd.hpp`
  - [ ] **1.3c** Replace `vs::view_closure<Fn>` with `ccs::view_closure<Fn>` in `is_view_closure_impl` (line 539). Update the `ViewClosure` and `ViewClosures` concepts accordingly.
    - Depends on: 1.2a
    - Files: `src/fields/tuple_fwd.hpp`
  - [ ] **1.3d** Replace `vs::common(...)` with `std::views::common(...)` in `constructible_from_range_impl` (line 333).
    - Files: `src/fields/tuple_fwd.hpp`
  - [ ] **1.3e** Remove `rs::common_tuple<Args...>` specialization from `is_tuple_like_impl` (line 129). This type is only produced by range-v3's `vs::zip`; after migration, no code generates `rs::common_tuple` values. The `NumericTuple` test using `rs::common_tuple` (in `range_concepts.t.cpp`) will be updated in item 1.20.
    - Files: `src/fields/tuple_fwd.hpp`
  - [ ] **1.3f** Replace `ranges::enable_view` specializations (lines 711–721) with `std::ranges::enable_view` in namespace `std::ranges`. Note: the specialization for `ccs::tuple<Args...>` sets `enable_view = false` (multi-element tuples are not views); the single-`All` specialization sets it to `true`.
    - Files: `src/fields/tuple_fwd.hpp`
  - [ ] **1.3g** Remove `#include <range/v3/range/concepts.hpp>`, `#include <range/v3/view/common.hpp>`, `#include <range/v3/view/view.hpp>`. Add `#include <ranges>` and `#include "ccs_range_utils.hpp"`.
    - Files: `src/fields/tuple_fwd.hpp`
  - Test (all of 1.3): `cmake --build build` — expect many compilation errors in downstream files until they are also migrated. Use iterative approach: fix tuple_fwd.hpp, then fix callers.

### Utilities and Algorithms

- [ ] **1.4** Migrate `src/fields/tuple_utils.hpp`:
  - [ ] **1.4a** Replace algorithms: `rs::copy` → `std::ranges::copy`, `rs::copy_n` → `std::ranges::copy_n`, `rs::fill` → `std::ranges::fill`. Replace `rs::begin`/`rs::end`/`rs::size` → `std::ranges::begin`/`end`/`size`.
    - Files: `src/fields/tuple_utils.hpp` (`resize_and_copy`, `to`, `ssize` functions)
  - [x] **1.4b** ~~Replace `vs::all`/`vs::all_t` in `to()` function~~ — REMOVED: `vs::all` and `vs::all_t` are not used in `tuple_utils.hpp`. The `<range/v3/view/all.hpp>` include is unused (no direct `vs::all` usage in the file body) and its removal is already covered by 1.4f.
  - [ ] **1.4c** Replace `vs::common(...)` → `std::views::common(...)` in `to()` function (line 342).
    - Files: `src/fields/tuple_utils.hpp`
  - [ ] **1.4d** Replace `vs::zip_with(fn, rngs...)` with `ccs::zip_transform(fn, rngs...)` in `lift()` function (lines 381, 385).
    - Depends on: 1.2b
    - Files: `src/fields/tuple_utils.hpp`
  - [ ] **1.4e** Replace `rs::range` and `rs::sized_range` concept usage in template constraints with `std::ranges` equivalents.
    - Files: `src/fields/tuple_utils.hpp`
  - [ ] **1.4f** Remove `#include <range/v3/algorithm/copy.hpp>`, `#include <range/v3/algorithm/copy_n.hpp>`, `#include <range/v3/algorithm/fill.hpp>`, `#include <range/v3/view/all.hpp>`, `#include <range/v3/view/common.hpp>`, `#include <range/v3/view/view.hpp>`, `#include <range/v3/view/zip_with.hpp>`. Add `#include <algorithm>`, `#include <ranges>`, and `#include "lazy_views.hpp"`.
    - Files: `src/fields/tuple_utils.hpp`
  - Test: `ctest --test-dir build -R t-tuple_utils`

- [ ] **1.5** Migrate `src/fields/algorithms.hpp`:
  - Replace `rs::minmax` → `std::ranges::minmax`, `rs::min` → `std::ranges::min`, `rs::max` → `std::ranges::max`, `rs::minmax_result` → `std::ranges::minmax_result`.
  - Replace `rs::begin`/`rs::end` → `std::ranges::begin`/`end`.
  - Replace `rs::range_value_t` → `std::ranges::range_value_t`.
  - Remove `#include <range/v3/algorithm/max.hpp>`, etc. Add `#include <algorithm>`, `#include <ranges>`.
  - Files: `src/fields/algorithms.hpp`
  - Test: `ctest --test-dir build -R t-algorithms`

### View/Container Tuple

- [ ] **1.6** Migrate `src/fields/container_tuple.hpp`:
  - Replace `rs::begin(r)`/`rs::end(r)` → `std::ranges::begin(r)`/`std::ranges::end(r)` in the range constructor (line 27).
  - No direct range-v3 includes to remove (container_tuple.hpp gets range-v3 transitively through `tuple_utils.hpp`, which is already migrated in 1.4).
  - Files: `src/fields/container_tuple.hpp`
  - Test: `ctest --test-dir build -R t-container_tuple`

- [ ] **1.7** Migrate `src/fields/view_tuple.hpp`:
  - [ ] **1.7a** Replace `vs::all`/`vs::all_t` with `std::views::all`/`std::views::all_t` throughout (lines 34, 44, 48, 53, 75, etc.). Used in `view_tuple_base` member `std::tuple<vs::all_t<Args>...> v` and constructors/assignment.
    - Files: `src/fields/view_tuple.hpp`
  - [ ] **1.7b** Replace `rs::equal` with `std::ranges::equal` in `operator==` (lines 115, 125).
    - Files: `src/fields/view_tuple.hpp`
  - [ ] **1.7c** Redesign `single_view<A>` (lines 186–235). Currently inherits from `vs::all_t<A>` (a range-v3 view type). Replace with inheritance from `std::views::all_t<A>` (a `std::ranges::ref_view<A>` or `std::ranges::owning_view<A>`). The key behaviors to preserve:
    - `single_view<A>` makes a 1-element `view_tuple` directly iterable as a range.
    - Assignment uses placement-new destroy-and-reconstruct semantics.
    - The `using view = vs::all_t<A>` alias switches to `std::views::all_t<A>`.
    - Files: `src/fields/view_tuple.hpp`
  - [ ] **1.7d** Remove `#include <range/v3/algorithm/equal.hpp>`, `#include <range/v3/view/all.hpp>`. Add `#include <algorithm>`, `#include <ranges>`.
    - Files: `src/fields/view_tuple.hpp`
  - Test: `ctest --test-dir build -R t-view_tuple`

### Math Operations

- [ ] **1.8** Migrate `src/fields/tuple_math.hpp`:
  - [ ] **1.8a** Replace `vs::zip(out, in)` in compound-assignment operators (line 47) with index-based iteration: `auto it_o = std::ranges::begin(out); auto it_i = std::ranges::begin(in); for (; it_o != std::ranges::end(out); ++it_o, ++it_i) *it_o OP *it_i;`. This avoids needing a lazy zip view for in-place mutation.
    - Files: `src/fields/tuple_math.hpp`
  - [ ] **1.8b** Replace `vs::zip_with(f, rng, vs::repeat_n(v, sz))` in binary scalar operators (lines 84, 96) with `ccs::zip_transform(f, FWD(rng), ccs::repeat_n(v, sz))` or equivalently `rng | std::views::transform([f, v](auto&& x) { return f(FWD(x), v); })`. The `transform` approach is simpler and avoids needing `repeat_n_view`.
    - Depends on: 1.2b (only if using `ccs::zip_transform`; `std::views::transform` approach needs no dependency)
    - Files: `src/fields/tuple_math.hpp`
  - [ ] **1.8c** Replace `vs::zip_with(f, a, b)` in binary tuple-tuple operators (line 108) with `ccs::zip_transform(f, FWD(a), FWD(b))`.
    - Depends on: 1.2b
    - Files: `src/fields/tuple_math.hpp`
  - [ ] **1.8d** Replace `rs::size(rng)` → `std::ranges::size(rng)` (line 83).
    - Files: `src/fields/tuple_math.hpp`
  - [ ] **1.8e** Remove `#include <range/v3/view/repeat.hpp>`, `#include <range/v3/view/repeat_n.hpp>`, `#include <range/v3/view/zip.hpp>`, `#include <range/v3/view/zip_with.hpp>`. Add `#include <ranges>` and `#include "lazy_views.hpp"`.
    - Files: `src/fields/tuple_math.hpp`
  - Test: `ctest --test-dir build -R t-tuple_math`

- [ ] **1.9** Migrate `src/fields/tuple_pipe.hpp`:
  - Replace `vs::view_closure<ViewFn>` with `ccs::view_closure<ViewFn>` in the two `operator|` overloads that accept/match view closures (lines 52, 62).
  - Depends on: 1.2a
  - Files: `src/fields/tuple_pipe.hpp`
  - Test: `ctest --test-dir build -R t-tuple_pipe`

### Tuple Type

- [ ] **1.9a** Migrate `src/fields/tuple.hpp`:
  - Replace `vs::view_closure<ViewFn>` → `ccs::view_closure<ViewFn>` in deduction guides (lines 152–154).
  - Depends on: 1.2a
  - Files: `src/fields/tuple.hpp`
  - Test: `ctest --test-dir build -R t-tuple`

### Field Type

- [ ] **1.10** Migrate `src/fields/field.hpp`:
  - Replace `rs::swap_ranges` → `std::ranges::swap_ranges` (lines 138, 142–143).
  - Replace `rs::size` → `std::ranges::size`, `rs::begin`/`rs::end` → `std::ranges::begin`/`end` (lines 51–52, 90, 92, 103).
  - Replace `rs::sized_range` → `std::ranges::sized_range`, `rs::random_access_range` → `std::ranges::random_access_range` (lines 90, 103, 120).
  - Replace `rs::range_reference_t` → `std::ranges::range_reference_t` (lines 72, 81).
  - Replace `rs::output_range` → `std::ranges::output_range` (concept usage removed if already handled by tuple_fwd).
  - Remove `#include <range/v3/algorithm/swap_ranges.hpp>`. Add `#include <algorithm>`.
  - Files: `src/fields/field.hpp`
  - Test: `ctest --test-dir build -R t-field`

- [ ] **1.11** Migrate `src/fields/field_utils.hpp`:
  - Replace `vs::zip(t.scalars()...)` with index-based iteration in `for_each_scalar` and `for_each_vector` (lines 11, 17): iterate `for (int i = 0; i < t.nscalars(); ++i) f(t.scalars(i)...);` (or equivalent using `std::ranges::size`). Note: `t` is a parameter pack `T&&... t`; use the first argument's size.
  - Replace `vs::zip_with(f, t.scalars()...)` with `ccs::zip_transform(f, t.scalars()...)` in `transform_scalar` and `transform_vector` (lines 30, 36).
  - Add `#include "lazy_views.hpp"` for `ccs::zip_transform`. No direct range-v3 includes to remove (range-v3 is accessed transitively through `field_fwd.hpp` → `selector.hpp`).
  - Depends on: 1.2b
  - Files: `src/fields/field_utils.hpp`
  - Test: `ctest --test-dir build -R t-field_utils`

- [ ] **1.12** Migrate `src/fields/field_math.hpp`: No range-v3 includes — only depends on `field_utils.hpp` which provides range concepts transitively. Verify that after 1.11, this file compiles with no range-v3 usage.
  - Files: `src/fields/field_math.hpp`
  - Test: `ctest --test-dir build -R t-field_math`

- [ ] **1.12a** Migrate `src/fields/field_fwd.hpp`:
  - Replace `rs::range_value_t` → `std::ranges::range_value_t`, `rs::range_reference_t` → `std::ranges::range_reference_t` (lines 76–85).
  - Files: `src/fields/field_fwd.hpp`
  - Test: `cmake --build build`

- [ ] **1.12b** Migrate `src/fields/selector_fwd.hpp`:
  - Replace `ranges::enable_view` specialization (line 46) with `std::ranges::enable_view` in namespace `std::ranges`.
  - Files: `src/fields/selector_fwd.hpp`
  - Test: `cmake --build build`

### Selectors (Highest Risk)

All selector items depend on: 1.2a (ccs_range_utils.hpp) and 1.3 (tuple_fwd.hpp migration).

- [ ] **1.13** Replace `plane_view<0>` (X-plane) in `src/fields/selector.hpp`:
  - Currently: `x_plane_t<Rng>` = `decltype(rng | vs::drop_exactly(n) | vs::take_exactly(m))`. The class inherits from this composed type.
  - Replace: Change `x_plane_t<Rng>` to use `std::views::drop(n) | std::views::take(m)` (C++20). Update the type alias and the `apply_` static method (line 186).
  - Replace `rs::semiregular_box_t<Fn>` → `ccs::semiregular_box<Fn>` (line 181).
  - Files: `src/fields/selector.hpp` (lines 173–202)
  - Test: `ctest --test-dir build -R t-selector`

- [ ] **1.14** Replace `plane_view<1>` (Y-plane) in `src/fields/selector.hpp`:
  - Currently: Inherits `rs::view_adaptor<plane_view<1, Rng, Fn>, Rng>` with a custom `adaptor` class using `rs::adaptor_base`, `rs::range_access`, `rs::begin`, `rs::advance`, `rs::difference_type_t`.
  - Replace: Rewrite as a class inheriting from `std::ranges::view_interface<plane_view<1, Rng, Fn>>`. Implement:
    - Store the base range (by value), `index_extents n`, `diff_t j`, and `ccs::semiregular_box<Fn> f`.
    - Define a custom `iterator` class with: `operator*` (dereference base iterator), `operator++` (next with stride logic from current adaptor lines 247–255), `operator--` (prev with reverse stride, lines 258–267), `operator+=` (advance with division-based skip, lines 270–303), `operator==`, `operator-` (distance_to, line 309).
    - `begin()` returns iterator at position `j * nz`, `end()` returns iterator at position `(nx-1) * ny * nz + j * nz + nz`.
    - The iterator must model `std::random_access_iterator` since the current adaptor provides random-access operations.
  - Replace `rs::semiregular_box_t<Fn>` → `ccs::semiregular_box<Fn>`.
  - Replace `rs::range_difference_t<Rng>` → `std::ranges::range_difference_t<Rng>`.
  - Replace `rs::begin`, `rs::advance` → `std::ranges::begin`, `std::ranges::advance`.
  - Files: `src/fields/selector.hpp` (lines 208–330)
  - Test: `ctest --test-dir build -R t-selector` — verify x/y/z plane extraction, assignment, and `apply` on scalar/vector types.

- [ ] **1.15** Replace `plane_view<2>` (Z-plane) in `src/fields/selector.hpp`:
  - Currently: `z_plane_t<Rng>` = `decltype(rng | vs::drop_exactly(k) | vs::stride(n))`. Inherits from this type.
  - Replace: Change `z_plane_t<Rng>` to use `std::views::drop(k)` piped into `ccs::stride(rng, n)`. Or define `z_plane_t<Rng>` using the project-local `ccs::stride_view`.
  - Replace `rs::semiregular_box_t<Fn>` → `ccs::semiregular_box<Fn>`.
  - Depends on: 1.2b (stride_view)
  - Files: `src/fields/selector.hpp` (lines 332–356)
  - Test: `ctest --test-dir build -R t-selector`

- [ ] **1.16** Replace `multi_slice_view` in `src/fields/selector.hpp`:
  - Currently: Inherits `rs::view_adaptor<multi_slice_view<Rng, Fn>, Rng>` with a custom `adaptor` class (~135 lines, lines 459–614).
  - Replace: Rewrite as a class inheriting from `std::ranges::view_interface<multi_slice_view<Rng, Fn>>`. Implement:
    - Store: base range (by value), `std::span<const index_slice> slices`, `ccs::semiregular_box<Fn> f`.
    - Define a custom `iterator` with the same slice-navigation logic (current adaptor lines 470–595). The iterator stores: `slice_it` (current slice), `last_slice`, `integer i` (position in base range), `integer multi_i` (logical position), and a base iterator.
    - Must model at least `std::bidirectional_iterator` (current view supports `next`, `prev`, `advance`, `distance_to`).
  - Replace all `rs::` calls with `std::ranges::` equivalents.
  - Replace `rs::semiregular_box_t` → `ccs::semiregular_box`.
  - Files: `src/fields/selector.hpp` (lines 455–667)
  - Test: `ctest --test-dir build -R t-selector` — verify multi_slice extraction, assignment, and `apply`.

- [ ] **1.17** Replace `optional_view` in `src/fields/selector.hpp`:
  - Currently: Inherits `rs::view_adaptor<optional_view<Rng, Fn>, Rng>` with a simple adaptor that returns `begin` or `end` based on a bool.
  - Replace: Rewrite as a class inheriting from `std::ranges::view_interface<optional_view<Rng, Fn>>`. Implement:
    - `begin()`: returns `keep_bounds ? std::ranges::begin(base) : std::ranges::end(base)`.
    - `end()`: returns `std::ranges::end(base)`.
    - This is a very simple view — ~30 lines of replacement code.
  - Replace `rs::semiregular_box_t` → `ccs::semiregular_box`.
  - Files: `src/fields/selector.hpp` (lines 676–764)
  - Test: `ctest --test-dir build -R t-selector`

- [ ] **1.18** Replace `predicate_view` in `src/fields/selector.hpp`:
  - Currently: Inherits `rs::view_adaptor<predicate_view<Rng, Pred, Fn>, Rng>` with filter-style iteration (lines 780–877).
  - Replace: Rewrite as a class inheriting from `std::ranges::view_interface<predicate_view<Rng, Pred, Fn>>`. Implement:
    - Store: base range, predicate range `Pred`, `ccs::semiregular_box<Fn> f`, cached begin.
    - Define a custom `iterator` that skips elements where the predicate is false (same `satisfy_forward`/`satisfy_reverse` logic from lines 821–839).
    - Must model `std::bidirectional_iterator` (current view supports `next` and `prev` but not `advance`/`distance_to`).
    - Note: The predicate is a *range* (not a callable) — iteration advances both the base and predicate iterators in lockstep.
  - Replace `rs::semiregular_box_t` → `ccs::semiregular_box`.
  - Replace `rs::begin`/`rs::end`/`rs::size`/`rs::iterator_t` → `std::ranges` equivalents.
  - Files: `src/fields/selector.hpp` (lines 776–940)
  - Test: `ctest --test-dir build -R t-selector`

- [ ] **1.19** Replace range-v3 utility usage in selector function objects (`selection`, `plane_selection_fn`, `multi_slice_fn`, `optional_view_fn`, `predicate_view_fn`):
  - [ ] **1.19a** In `selection<L,R,Fn>` struct (line 48): Replace `rs::semiregular_box_t<Fn>` → `ccs::semiregular_box<Fn>`.
    - Files: `src/fields/selector.hpp`
  - [ ] **1.19b** In `selection_view` (line 123): Replace `rs::make_view_closure(...)` → `ccs::make_view_closure(...)`.
    - Files: `src/fields/selector.hpp`
  - [ ] **1.19c** In `plane_selection_base_fn` (lines 370–391): Replace `rs::bind_back(...)` → `ccs::bind_back(...)` and `rs::compose(...)` → `ccs::compose(...)`.
    - Files: `src/fields/selector.hpp`
  - [ ] **1.19d** In `plane_selection_fn::operator()` (lines 415–416, 420): Replace `rs::make_view_closure(rs::bind_back(...))` → `ccs::make_view_closure(ccs::bind_back(...))` on lines 415–416. Also replace `rs::bind_back(*this, plane_coord)` → `ccs::bind_back(*this, plane_coord)` on line 420.
    - Files: `src/fields/selector.hpp`
  - [ ] **1.19e** In `multi_slice_fn::operator()` (line 647) and `multi_slice_base_fn` (lines 652–665): Replace `rs::make_view_closure`, `rs::bind_back`, `rs::compose` → `ccs::` equivalents.
    - Files: `src/fields/selector.hpp`
  - [ ] **1.19f** In `optional_view_fn` (lines 738–764): Replace `rs::bind_back` → `ccs::bind_back`, `rs::make_view_closure` → `ccs::make_view_closure`.
    - Files: `src/fields/selector.hpp`
  - [ ] **1.19g** In `predicate_view_fn` and `predicate_view_base_fn` (lines 882–938): Replace `rs::bind_back`, `rs::compose`, `rs::make_view_closure` → `ccs::` equivalents.
    - Files: `src/fields/selector.hpp`
  - [ ] **1.19h** Remove `#include <range/v3/view/drop_exactly.hpp>`, `#include <range/v3/view/stride.hpp>`, `#include <range/v3/view/take_exactly.hpp>` from `selector.hpp`. Add `#include <ranges>`, `#include "ccs_range_utils.hpp"`, `#include "lazy_views.hpp"`.
    - Files: `src/fields/selector.hpp`
  - Test: `ctest --test-dir build -R t-selector`

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

- [ ] **1.20a** Migrate `src/fields/range_concepts.t.cpp` (313 lines, 36 `rs::`/`vs::` occurrences): This is the concept test file. Remove `#include <range/v3/all.hpp>`. Add `#include <algorithm>`, `#include <ranges>`, `#include "ccs_range_utils.hpp"`, `#include "lazy_views.hpp"`.
  - Replace `rs::output_range` → `std::ranges::output_range`, `rs::range_value_t` → `std::ranges::range_value_t` in concept checks (lines 18–28).
  - Replace `vs::all(x)` → `std::views::all(x)` (line 55), `vs::iota(a, b)` → `std::views::iota(a, b)` (lines 94, 253, 273), `vs::transform(f)` → `std::views::transform(f)` (lines 65–66, 188).
  - Replace `rs::equal` → `std::ranges::equal` (lines 61, 190, 256), `rs::begin`/`rs::end` → `std::ranges::begin`/`end` (line 254), `rs::sized_range` → `std::ranges::sized_range` (lines 259, 280), `rs::random_access_range` → `std::ranges::random_access_range` (lines 260, 281).
  - **ViewClosure test (lines 132–138)**: After 1.3c, `ViewClosure` checks for `ccs::view_closure<Fn>`, not range-v3's `vs::view_closure`. `std::views::transform(...)` returns a standard library closure type which is NOT a `ccs::view_closure`. Rewrite: `using I = decltype(ccs::make_view_closure([](auto&& rng) { return rng; }));` and test `ViewClosure<I>`, `ViewClosures<std::tuple<I, I>>`.
  - **NumericTuple/common_tuple test (lines 172–174)**: Remove the 3 lines using `rs::common_tuple` (type no longer exists after 1.3e).
  - **From test (lines 91–107)**: Replace `vs::zip_with(std::plus{}, vs::iota(0, 10), vs::iota(1, 11))` (line 95) with `ccs::zip_transform(std::plus{}, std::views::iota(0, 10), std::views::iota(1, 11))`.
  - **TupleLike test (line 79)**: Replace `vs::take_exactly(5)` → `std::views::take(5)`. Test intent preserved (verifying non-TupleLike range types).
  - **generate_n test (lines 195–202)**: Rewrite. Replace `vs::generate_n(f, n) | vs::join | rs::to<T>()` with manual loop: `std::vector<int> v; for (int i = 0; i < 3; i++) { v.push_back(0); v.push_back(1); }`.
  - **x_plane_view/z_plane_view test classes and test cases (lines 204–282)**: Rewrite type aliases: `det::X<Rng>` uses `std::views::drop | std::views::take` instead of `vs::drop_exactly | vs::take_exactly`. `det::Z<Rng>` uses `std::views::drop` piped into `ccs::stride` instead of `vs::drop_exactly | vs::stride`. Update constructors accordingly. Also rewrite the local `U` type alias at line 264 (mirrors `det::Z` pattern with `vs::drop_exactly | vs::stride`). The z_plane_view test case (lines 269–282) uses `vs::iota`, `rs::sized_range`, `rs::random_access_range` — replace per general rules above.
  - **vs::zip "expansion" test (lines 284–301)**: Remove entirely (including the `wrapper` struct at lines 284–288 which is only used by this test). This only tested range-v3's `vs::zip` with parameter pack expansion, which is no longer needed.
  - Depends on: 1.2a (ccs_range_utils.hpp for `ccs::make_view_closure`), 1.2b (lazy_views.hpp for `ccs::zip_transform`, `ccs::stride`)
  - Files: `src/fields/range_concepts.t.cpp`
  - Test: `ctest --test-dir build -R t-range_concepts`

- [ ] **1.20b** Migrate `src/fields/tuple_utils.t.cpp` (611 lines, heaviest `rs::` usage): Remove `#include <range/v3/all.hpp>`. Add `#include <algorithm>`, `#include <numeric>`, `#include <ranges>`. Replace:
  - `vs::zip` in for loops → index-based iteration (lines 117, 123, 161).
  - `vs::zip_with`/`vs::repeat`/`vs::repeat_n` in lift tests → `ccs::zip_transform` or manual equivalents (lines 271, 282, 401, 424, 442, 460).
  - `rs::accumulate(rng, init, op)` → `std::accumulate(std::ranges::begin(r), std::ranges::end(r), init, op)` (~5 occurrences).
  - `rs::equal` → `std::ranges::equal`, `rs::size` → `std::ranges::size` (~100 occurrences combined).
  - `rs::minmax`/`rs::min`/`rs::max`/`rs::minmax_result` → `std::ranges::minmax`/`min`/`max`/`minmax_result`.
  - Files: `src/fields/tuple_utils.t.cpp`
  - Test: `ctest --test-dir build -R t-tuple_utils`

- [ ] **1.20c** Migrate `src/fields/tuple_math.t.cpp`: Remove range-v3 includes. Replace `vs::zip_with(std::plus{}, a, b)` test constructions (lines 42, 85, 158, 214) with `ccs::zip_transform` or manual expected values.
  - Files: `src/fields/tuple_math.t.cpp`
  - Test: `ctest --test-dir build -R t-tuple_math`

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
                              ├── 1.9  (tuple_pipe.hpp)
                              ├── 1.9a (tuple.hpp)
                              └── 1.19 (selector utility fns)

1.2b (lazy_views.hpp) ──── 1.2c (zip_transform fix) ──┬── 1.4d (tuple_utils.hpp lift)
                                                        ├── 1.8b, 1.8c (tuple_math.hpp)
                                                        └── 1.11 (field_utils.hpp)
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
