# Phase 8: Field Registry and Handle Types

**Goal:** Introduce the centralized `field_registry` and trivially-copyable handle types alongside the existing tuple infrastructure. Prove the design via TDD and a span-bridge that lets existing operators work unchanged with registry-backed storage.

**Depends on:** Phases 0–7 (range-v3 removal complete)

**Read first:**
- `src/fields/handle.hpp` (prototype handle types)
- `src/fields/field.hpp` (`field`/`field_span`/`field_view` types, constructors, assignment)
- `src/fields/field_fwd.hpp` (`system_size`, `Field` concept)
- `src/fields/scalar.hpp` (`scalar_real`/`scalar_span`/`scalar_view`)
- `src/fields/vector.hpp` (`vector_real`/`vector_span`/`vector_view`)
- `src/fields/container_tuple.hpp` (owning storage)
- `src/fields/view_tuple.hpp` (`view_tuple_base`, `single_view`, re-anchoring)
- `src/fields/tuple.hpp` (dual-inheritance `tuple`)
- `src/fields/selector_fwd.hpp` (`si::D`, `vi::Dx`, etc.)
- `src/kokkos_types.hpp` (Kokkos aliases)
- `plans/meta.md` (decisions)
- `plans/dsl-restructuring-proposal.md` (design doc, especially §13.9)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -R t-handle
ctest --test-dir build -R t-field_registry
ctest --test-dir build
```

---

## Items

### 8.0 — Commit design artifacts

- [x] **8.0a** Commit the design artifacts from the design conversation:
  - `plans/dsl-restructuring-proposal.md`
  - `plans/kokkos-view-migration-impact.md`
  - `src/fields/handle.hpp`
  - `plans/08-registry-and-handles.md` (this file)
  - Update `plans/meta.md` with new decisions D-R1 through D-R6 and the phase table.
  - **Done:** commit 6bfd230.

### 8.1 — Handle type tests (TDD)

- [x] **8.1a** Create `src/fields/handle.t.cpp` with tests for `handle.hpp`:
  - `field_layout` arithmetic: `total_buffers`, `vector_base` for `<1,0>`, `<2,1>`, `<4,4>`.
  - `scalar_handle` accessors: `D()`, `Rx()`, `Ry()`, `Rz()`, `all()`, `R()` return correct `buf_handle.id` values.
  - `vector_handle` accessors: `x()`, `y()`, `z()`, `Dx()`, `xRy()`, etc.
  - `make_scalar_handle` / `make_vector_handle` consteval validation: verify that valid-index
    calls produce handles with expected `.base` values. Use `constexpr auto` variables (the
    factories are `consteval`, so any call that compiles is proven correct). Example:
    ```cpp
    constexpr auto layout = field_layout<2, 1>{.n_scalars = 2, .n_vectors = 1};
    constexpr auto s0 = make_scalar_handle(layout, 0);
    STATIC_REQUIRE(s0.base == 0);
    ```
    Invalid-index rejection (e.g., `make_scalar_handle(layout, 5)`) cannot be tested at runtime —
    `consteval` failures are compile errors, not exceptions. The existing `static_assert`s in
    `handle.hpp::detail` already cover this; the Catch2 test only validates well-formed calls.
  - `handle_sel::D`, `handle_sel::R`, `handle_sel::Rx` dispatch for both scalar and vector handles.
    Scalar dispatch returns a single `buf_handle`; vector dispatch returns `std::array<buf_handle, N>`.
    Verify specific `.id` values (e.g., `handle_sel::D(s0).id == 0`,
    `handle_sel::D(v0)[0].id == 8` for layout `<2,1>`).
  - `handle_for_each` visits all 4 scalar buffers / all 12 vector buffers in correct order.
    Collect visited `buf_handle.id` values into a `std::vector<int>` and compare to expected
    sequence (e.g., scalar `{0, 1, 2, 3}`, vector `{8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}`
    for layout `<2,1>` handle at index 0).
  - Paired `handle_for_each` visits corresponding pairs: collect `(a.id, b.id)` pairs and verify
    they match expected pairings between two scalar handles or two vector handles.
  - Trivial copyability: `static_assert` on all handle types (already in `handle.hpp`; replicate
    in test file for documentation).
  - Note: `handle.hpp` is pure compile-time arithmetic — no Kokkos dependency, no custom main.
    Many checks are already `static_assert`s in `handle.hpp` `detail` namespace; the Catch2 tests
    complement these with runtime `REQUIRE`/`CHECK` and `SECTION`-based organization.
  - File: `src/fields/handle.t.cpp` (new)
  - CMake: `add_unit_test(handle "fields" fields)` in `src/fields/CMakeLists.txt`
  - Test: `ctest --test-dir build -R t-handle`

### 8.2 — Field registry (TDD)

- [x] **8.2a** Create `src/fields/field_registry.t.cpp` with tests.

  **Kokkos initialization:** This test allocates `Kokkos::View`, which requires an initialized
  Kokkos runtime. Provide a custom `main()` using Catch2 v3's `catch2/catch_session.hpp`:
  ```cpp
  #include <Kokkos_Core.hpp>
  #include <catch2/catch_session.hpp>
  int main(int argc, char* argv[]) {
      Kokkos::ScopeGuard kokkos(argc, argv);
      return Catch::Session().run(argc, argv);
  }
  ```

  **CMake:** Cannot use `add_unit_test()` (it links `Catch2::Catch2WithMain` which conflicts
  with the custom main). Instead, manually add in `src/fields/CMakeLists.txt`:
  ```cmake
  if (BUILD_TESTING)
    add_executable(t-field_registry field_registry.t.cpp)
    target_link_libraries(t-field_registry Catch2::Catch2 fields Kokkos::kokkos)
    add_test(NAME t-field_registry COMMAND t-field_registry)
    set_tests_properties(t-field_registry PROPERTIES LABELS "fields")
  endif()
  ```
  Note: `Kokkos::kokkos` is linked directly so the test compiles before 8.2b adds
  Kokkos to the `fields` INTERFACE target. After 8.2b the explicit link is harmlessly
  redundant.

  **Test cases:**
  - `field_ref` trivial copyability: `static_assert(std::is_trivially_copyable_v<field_ref>)`.
  - `field_ref` size: `static_assert(sizeof(field_ref) == 12)` (3 × `int`).
  - Construction: `field_registry<4, 2, 1>` compiles; verify `buffers_per_slot == 20`
    (= `field_layout<2,1>::total_buffers` = 2×4 + 1×12).
  - `allocate_scalar(int slot, int scalar_index, int d_sz, int rx_sz, int ry_sz, int rz_sz)`:
    After calling with `(0, 0, 100, 5, 3, 2)`, verify:
    - `reg.size(ref, sh.D()) == 100`, `reg.size(ref, sh.Rx()) == 5`, etc.
    - `reg.data(ref, sh.D())` is non-null.
    - `ref` is `field_ref{.slot=0, .n_scalars=1, .n_vectors=0}`.
  - `allocate_vector(int slot, int vector_index, int d_sz, int rx_sz, int ry_sz, int rz_sz)`:
    After calling with `(0, 0, 100, 5, 3, 2)`, verify 12 Views (3 components × 4 buffers)
    with domain size 100 and boundary sizes 5, 3, 2 respectively.
  - `view(field_ref, buf_handle)` returns `Kokkos::View<real*>&`.
  - `data(field_ref, buf_handle)` returns valid `real*`; writing through it is visible via `view()`.
  - `size(field_ref, buf_handle)` returns `view().extent(0)`.
  - Unallocated slots: `size() == 0`, `data() == nullptr`.
  - Sequential scalar allocation: allocate scalar 0 then scalar 1 in the same slot
    (using `field_registry<4, 2, 1>`). Verify returned `field_ref` has `n_scalars=1` after
    the first call and `n_scalars=2` after the second. Verify both scalars' D buffers are
    accessible and point to different memory.
  - Mixed scalar+vector allocation: allocate scalar 0 AND vector 0 in the same slot.
    Verify `field_ref` has `n_scalars=1, n_vectors=1`. Verify scalar D buffer
    (at index `0*4+0 = 0`) and vector x.D buffer (at index `2*4+0*12+0 = 8`)
    are both accessible and point to different memory. This confirms the flat layout
    doesn't cause interference between scalar and vector regions.
  - `deep_copy_slot(int dst, int src)`: allocate scalar 0 in both slot 0 and slot 1
    with matching sizes `(100, 5, 3, 2)`. Fill source slot's D buffer with `42.0`,
    deep-copy, verify destination's D buffer contains `42.0`.
  - `swap_slots(int a, int b)`: allocate scalar 0 in slot 0 with size 100 and in slot 1
    with size 200; after swap, slot 0's D buffer has extent 200 and slot 1's has extent 100.
    Also fill slot 0's D buffer with `1.0` and slot 1's with `2.0` before swap;
    verify slot 0's D buffer contains `2.0` and slot 1's contains `1.0` after swap
    (confirms data, not just extent, was swapped).
  - File: `src/fields/field_registry.t.cpp` (new)
  - Test: `ctest --test-dir build -R t-field_registry`

- [x] **8.2a-fix** Strengthen `deep_copy_slot` and `swap_slots` tests to verify beyond buffer index 0.

  **Problem:** Both tests only check the D buffer of scalar 0 (buffer index 0 out of 20).
  An implementation that copies/swaps only the first buffer would pass all current tests.

  **`deep_copy_slot` fix:** After filling D with `42.0`, also fill the Rx buffer (size 5) of
  scalar 0 with `7.0` in the source slot. After `deep_copy_slot`, verify the destination's
  Rx buffer contains `7.0` (in addition to the existing D check).

  **`swap_slots` fix:** Add an `SECTION("boundary buffer extents are swapped")` that checks
  `reg.size(swapped_ref0, sh.Rx()) == 10` and `reg.size(swapped_ref1, sh.Rx()) == 5`
  (the two slots were allocated with Rx sizes 5 and 10 respectively).

  - File: `src/fields/field_registry.t.cpp` (edit)
  - Test: `ctest --test-dir build -R t-field_registry`

- [x] **8.2b** Create `src/fields/field_registry.hpp` to pass the tests.

  **`field_ref` struct** (defined in `field_registry.hpp`):
  ```cpp
  struct field_ref {
      int slot       = -1;
      int n_scalars  = 0;
      int n_vectors  = 0;
      constexpr bool operator==(const field_ref&) const = default;
  };
  static_assert(std::is_trivially_copyable_v<field_ref>);
  ```

  **`field_registry` class template:**
  ```cpp
  template <int MaxSlots, int MaxS, int MaxV>
  class field_registry {
  public:
      using layout_type = field_layout<MaxS, MaxV>;
      static constexpr int buffers_per_slot = layout_type::total_buffers;
  private:
      static constexpr int total_views_ = MaxSlots * buffers_per_slot;
      std::array<Kokkos::View<real*>, total_views_> buffers_{};
      std::array<field_ref, MaxSlots> metadata_{};  // per-slot allocation tracking
  };
  ```

  **Per-slot metadata:** The `metadata_` array tracks how many scalars and vectors have been
  allocated in each slot. On construction, each entry is default-initialized to
  `field_ref{.slot=-1, .n_scalars=0, .n_vectors=0}`. On first allocation in a slot,
  `.slot` is set to the slot index. The `allocate_scalar`/`allocate_vector` methods
  enforce sequential allocation: `scalar_index` must equal `metadata_[slot].n_scalars`
  (and similarly for vectors). This prevents gaps in the buffer layout.

  **Index formula:** `slot * buffers_per_slot + buf_handle.id`.

  **Methods:**
  - `allocate_scalar(int slot, int scalar_index, int d_sz, int rx_sz, int ry_sz, int rz_sz)
    -> field_ref`:
    **Preconditions (asserted):**
    - `slot >= 0 && slot < MaxSlots`
    - `scalar_index >= 0 && scalar_index < MaxS`
    - `scalar_index == metadata_[slot].n_scalars` (sequential allocation)

    Constructs a `scalar_handle` at runtime via direct arithmetic:
    `scalar_handle{scalar_index * layout_type::scalar_stride}`. **Note:**
    `make_scalar_handle_unchecked` is `consteval` and cannot be called with
    the runtime `scalar_index` parameter.
    Allocates 4 `Kokkos::View<real*>` with labels (e.g., `"s0_D"`, `"s0_Rx"`, ...) and the
    corresponding sizes. Stores them at the computed index offsets. Use
    `Kokkos::View<real*>(label, extent)` constructor for allocation.
    Updates `metadata_[slot]`: sets `.slot = slot`, increments `.n_scalars`.
    Returns a copy of the updated `metadata_[slot]`.
  - `allocate_vector(int slot, int vector_index, int d_sz, int rx_sz, int ry_sz, int rz_sz)
    -> field_ref`:
    **Preconditions (asserted):**
    - `slot >= 0 && slot < MaxSlots`
    - `vector_index >= 0 && vector_index < MaxV`
    - `vector_index == metadata_[slot].n_vectors` (sequential allocation)

    Constructs a `vector_handle` at runtime via direct arithmetic:
    `vector_handle{layout_type::vector_base + vector_index * layout_type::vector_stride}`.
    Allocates 12 Views (3 components × 4 buffers each), all components share the same
    4 sizes (matching current `system_size` which uses one `scalar<integer>` for all components).
    Updates `metadata_[slot]`: sets `.slot = slot`, increments `.n_vectors`.
    Returns a copy of the updated `metadata_[slot]`.
  - `view(field_ref ref, buf_handle h) -> Kokkos::View<real*>&`:
    Returns `buffers_[ref.slot * buffers_per_slot + h.id]`.
  - `view(field_ref ref, buf_handle h) const -> const Kokkos::View<real*>&`.
  - `data(field_ref ref, buf_handle h) -> real*`:
    Returns `view(ref, h).data()`.
  - `data(field_ref ref, buf_handle h) const -> const real*`:
    Const overload, returns `view(ref, h).data()`. Needed for `extract_scalar_view` (8.3b).
  - `size(field_ref ref, buf_handle h) const -> int`:
    Returns `static_cast<int>(view(ref, h).extent(0))`.
  - `deep_copy_slot(int dst, int src)`:
    For each of the `buffers_per_slot` Views: if source has non-zero extent,
    call `Kokkos::deep_copy(buffers_[dst_idx], buffers_[src_idx])`.
    Destination must already be allocated with matching extents (or caller re-allocates first).
  - `swap_slots(int a, int b)`:
    **Precondition (asserted):** both slots must have matching allocation structures
    (`metadata_[a].n_scalars == metadata_[b].n_scalars && metadata_[a].n_vectors == metadata_[b].n_vectors`).
    This is always true for the time-integration use case (both slots hold the same field layout).
    For each of the `buffers_per_slot` Views: `std::swap(buffers_[a_idx], buffers_[b_idx])`.
    Also swaps `metadata_[a]` and `metadata_[b]` (and fixes `.slot` fields to match their new indices).
    This swaps `Kokkos::View` objects (lightweight reference-counted handles, ~24 bytes each),
    not the underlying data.

  **Includes:** `#include "handle.hpp"`, `#include "kokkos_types.hpp"`, `#include "shoccs_config.hpp"`.

  **CMake change in `src/fields/CMakeLists.txt`:**
  Add `Kokkos::kokkos` to the `fields` INTERFACE link libraries:
  ```cmake
  target_link_libraries(fields INTERFACE Boost::boost Kokkos::kokkos)
  ```
  This propagates Kokkos include paths and link flags to all targets that depend on `fields`.
  Existing tests (which don't include any Kokkos headers) are unaffected — they just gain
  Kokkos on the link line with no behavioral change.

  - Files: `src/fields/field_registry.hpp` (new), `src/fields/CMakeLists.txt` (edit)
  - Test: `ctest --test-dir build -R t-field_registry`
  - Ordering: must come after 8.2a (TDD: tests first, then implementation)

### 8.2c — Add missing slot bounds assertions in bulk operations (review follow-up)

- [x] **8.2c** Add `assert(slot >= 0 && slot < MaxSlots)` guards to `deep_copy_slot` and `swap_slots`.

  **Problem:** `allocate_scalar`/`allocate_vector` consistently assert
  `slot >= 0 && slot < MaxSlots`, but `deep_copy_slot` and `swap_slots` do not.
  Calling either with an out-of-range slot index silently causes out-of-bounds
  access on `buffers_[]` (UB). In `swap_slots`, the existing metadata-matching
  assertion (`metadata_[a].n_scalars == metadata_[b].n_scalars`) is itself UB
  when `a` or `b` is out of range, since it reads `metadata_[]` before any
  bounds check.

  **Fix in `field_registry.hpp`:**
  - `deep_copy_slot(int dst, int src)`: add at the top:
    ```cpp
    assert(dst >= 0 && dst < MaxSlots);
    assert(src >= 0 && src < MaxSlots);
    ```
  - `swap_slots(int a, int b)`: add at the top (before the metadata assertion):
    ```cpp
    assert(a >= 0 && a < MaxSlots);
    assert(b >= 0 && b < MaxSlots);
    ```

  - File: `src/fields/field_registry.hpp` (edit)
  - Test: `ctest --test-dir build -R t-field_registry` (existing tests still pass)

### 8.3 — Span bridge

**Scope:** Phase 8 implements only `extract_scalar_span`/`extract_scalar_view`. Vector span
extraction (`extract_vector_span`/`extract_vector_view`) follows the same pattern but is deferred
to Phase 9 when the system/integrator migration needs it. The scalar-only bridge is sufficient
to prove the design in the integration test (8.4a).

- [x] **8.3a** Add tests to `field_registry.t.cpp` for span extraction:
  - `extract_scalar_span(registry, field_ref, scalar_handle)` → `scalar_span` with correct D/Rx/Ry/Rz spans.
    Verify each span's `.data()` matches the corresponding `reg.data(ref, bh)` pointer and
    `.size()` matches `reg.size(ref, bh)`.
  - `extract_scalar_view(registry, field_ref, scalar_handle)` → `scalar_view` (const).
    Pass `const field_registry&` to enforce const-correctness.
  - Writing through the `scalar_span`'s D buffer modifies the registry's underlying View data.
    Write `42.0` through `get<si::D>(span)[0]`, verify `reg.view(ref, sh.D())(0) == 42.0`.
  - `scalar_view` elements match the registry data but type is `std::span<const real>`.
    Verify `std::is_same_v<decltype(get<si::D>(view)), std::span<const real>>`.
  - Extract from two different slots: verify spans point to different underlying data.
    Compare `.data()` pointers from slot 0 and slot 1.
  - Test: `ctest --test-dir build -R t-field_registry`
  - Ordering: must come after 8.2b (registry implementation must exist)

- [x] **8.3b** Implement span extraction in `field_registry.hpp`.

  **Type recap:** `scalar_span = scalar<std::span<real>> = tuple< tuple<std::span<real>>, tuple<std::span<real>, std::span<real>, std::span<real>> >`.
  `scalar_view = scalar<std::span<const real>>` — same structure with `const real`.

  **Construction pattern:**
  ```cpp
  template <int MaxSlots, int MaxS, int MaxV>
  scalar_span extract_scalar_span(field_registry<MaxSlots, MaxS, MaxV>& reg,
                                  field_ref ref, scalar_handle h)
  {
      auto sp = [&](buf_handle bh) -> std::span<real> {
          return {reg.data(ref, bh),
                  static_cast<std::size_t>(reg.size(ref, bh))};
      };
      return scalar_span{tuple{sp(h.D())},
                          tuple{sp(h.Rx()), sp(h.Ry()), sp(h.Rz())}};
  }
  ```
  For `extract_scalar_view`, same pattern but:
  - Takes `const field_registry&` (const-qualified registry).
  - Returns `scalar_view` (uses `std::span<const real>`).
  - Needs a `data(field_ref, buf_handle) const -> const real*` overload on `field_registry`.

  **Include dependency:** `field_registry.hpp` must include `scalar.hpp` (for `scalar_span`/`scalar_view` types)
  and `tuple.hpp` (for `tuple` constructor). This creates no circular dependency since
  `scalar.hpp` → `tuple.hpp` → no field_registry.

  - File: `src/fields/field_registry.hpp`
  - Test: `ctest --test-dir build -R t-field_registry`
  - Ordering: must come after 8.3a (TDD)

### 8.4 — Integration test with existing operators

- [x] **8.4a** Write an integration test that:
  1. Creates a `field_registry<2, 1, 0>` with 2 slots (input + output), 1 scalar, 0 vectors.
  2. Allocates scalar in both slots with mesh-like sizes: D=512 (8×8×8), Rx=5, Ry=3, Rz=2.
  3. Fills input slot's D buffer with known values (e.g., `i * 0.5` via Kokkos host accessor).
  4. Extracts `scalar_view` from input slot and `scalar_span` from output slot via span bridge.
  5. Performs element-wise operation through the extracted spans using `tuple_math`'s `+=` operator:
     ```cpp
     auto input = extract_scalar_view(reg, in_ref, sh);
     auto output = extract_scalar_span(reg, out_ref, sh);
     output += input;  // uses tuple_math::operator+= on the nested tuple spans
     ```
     If `tuple_math` `+=` between `scalar_span` and `scalar_view` doesn't compile (type mismatch
     in the `OutputTuple` concept), fall back to a manual element-wise loop over `get<si::D>()`:
     ```cpp
     auto d_in  = get<si::D>(input);   // std::span<const real>
     auto d_out = get<si::D>(output);  // std::span<real>
     for (int i = 0; i < (int)d_in.size(); ++i) d_out[i] += d_in[i];
     ```
  6. Verifies the result is written into the registry's output slot storage by reading
     the View data directly: `reg.view(out_ref, sh.D())(i)` equals the expected value.
  - This proves the registry can back the existing operator stack via the span bridge.
  - File: `src/fields/field_registry.t.cpp` (extend)
  - Test: `ctest --test-dir build -R t-field_registry`
  - Ordering: must come after 8.3b
  - **Done:** Two integration tests added: (1) manual element-wise loop through extracted spans,
    (2) `tuple_math` `operator+=` on full `scalar_span += scalar_view`. Both compile and pass,
    proving the registry can back the existing operator stack via the span bridge.

### 8.5 — field_ref SBO fitness test

- [x] **8.5a** Write a test demonstrating `field_ref` fits in `std::function` SBO:

  **Approach:** Use `sizeof` and `static_assert` to verify that `field_ref` is small enough
  for SBO. Most `std::function` implementations use a 16–32 byte SBO buffer.
  A lambda capturing `field_ref` (12 bytes) by value fits within SBO. A lambda capturing
  `field_span` (contains two `std::vector`s, ≥48 bytes on most platforms) does not.

  **Concrete tests:**
  - `static_assert(sizeof(field_ref) == 12)` — 3 ints.
  - `static_assert(std::is_trivially_copyable_v<field_ref>)`.
  - `REQUIRE(sizeof(field_ref) <= 24)` — well within any SBO threshold.
  - `REQUIRE(sizeof(field_span) > 48)` — too large for SBO (contains 2 `std::vector`s of
    scalar/vector spans, which themselves contain `std::vector`s).
  - Optionally: construct `std::function<void(int)>` capturing `field_ref` by value,
    invoke it, and verify it works. The SBO guarantee is platform-specific, so the test
    focuses on size comparison rather than allocation detection.

  - File: `src/fields/field_registry.t.cpp` (extend)
  - Test: `ctest --test-dir build -R t-field_registry`
  - Ordering: must come after 8.2b
  - **Done:** Three test sections added: (1) `field_ref` is 12 bytes and trivially copyable (well
    within SBO), (2) `field_span` is >= 48 bytes (too large for typical 16–32 byte SBO), (3)
    `std::function` capturing `field_ref` by value works correctly.

---

## Completion Criteria

- `t-handle` passes with full coverage of handle arithmetic and selectors.
- `t-field_registry` passes: allocation, access, copy, swap, span bridge.
- Span bridge extracts working `scalar_span`/`scalar_view` from registry storage.
- `field_ref` is trivially copyable and fits in SBO.
- All existing tests still pass (no regressions).
- No existing code is modified (pure additions, except adding `Kokkos::kokkos` to `fields` link libraries).
