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

- [ ] **8.0a** Commit the design artifacts from the design conversation:
  - `plans/dsl-restructuring-proposal.md`
  - `plans/kokkos-view-migration-impact.md`
  - `src/fields/handle.hpp`
  - `plans/08-registry-and-handles.md` (this file)
  - Update `plans/meta.md` with new decisions D-R1 through D-R6 and the phase table.

### 8.1 — Handle type tests (TDD)

- [ ] **8.1a** Create `src/fields/handle.t.cpp` with tests for `handle.hpp`:
  - `field_layout` arithmetic: `total_buffers`, `vector_base` for `<1,0>`, `<2,1>`, `<4,4>`.
  - `scalar_handle` accessors: `D()`, `Rx()`, `Ry()`, `Rz()`, `all()`, `R()` return correct `buf_handle.id` values.
  - `vector_handle` accessors: `x()`, `y()`, `z()`, `Dx()`, `xRy()`, etc.
  - `make_scalar_handle` / `make_vector_handle` consteval validation (compile-time bounds checking).
  - `handle_sel::D`, `handle_sel::R`, `handle_sel::Rx` dispatch for both scalar and vector handles.
  - `handle_for_each` visits all 4 scalar buffers / all 12 vector buffers in correct order.
  - Paired `handle_for_each` visits corresponding pairs.
  - Trivial copyability: `static_assert` on all handle types.
  - File: `src/fields/handle.t.cpp` (new)
  - CMake: `add_unit_test(handle "fields" fields)` in `src/fields/CMakeLists.txt`
  - Test: `ctest --test-dir build -R t-handle`

### 8.2 — Field registry (TDD)

- [ ] **8.2a** Create `src/fields/field_registry.t.cpp` with tests:
  - Construction: `field_registry<MaxSlots, MaxS, MaxV>` with correct slot/buffer counts.
  - `allocate_scalar(slot, scalar_index, sizes)`: 4 Views with correct extents (D, Rx, Ry, Rz).
  - `allocate_vector(slot, vector_index, sizes)`: 12 Views with correct extents.
  - `view(field_ref, buf_handle)` returns the correct View.
  - `data(field_ref, buf_handle)` returns valid `real*` after allocation.
  - `size(field_ref, buf_handle)` returns correct extent.
  - Unallocated slots: `size() == 0`, `data() == nullptr`.
  - `deep_copy_slot(dst_slot, src_slot)`: data copied between scalar slots.
  - `swap_slots(a, b)`: slot indices swap (Views themselves don't move).
  - File: `src/fields/field_registry.t.cpp` (new)
  - CMake: `add_unit_test(field_registry "fields" fields Kokkos::kokkos)`
  - Test: `ctest --test-dir build -R t-field_registry`

- [ ] **8.2b** Create `src/fields/field_registry.hpp` to pass the tests:
  - `template <int MaxSlots, int MaxScalars, int MaxVectors> class field_registry`
  - Storage: `std::array<Kokkos::View<real*>, MaxSlots * Layout::total_buffers> buffers_`
  - `field_ref` struct: `{ int slot; int n_scalars; int n_vectors; }` — trivially copyable.
  - Methods: `allocate_scalar`, `allocate_vector`, `view`, `data`, `size`, `deep_copy_slot`, `swap_slots`.
  - Slot offset: `slot * Layout::total_buffers + buf_handle.id`.
  - Link Kokkos to the fields target if needed.
  - File: `src/fields/field_registry.hpp` (new)
  - Test: `ctest --test-dir build -R t-field_registry`

### 8.3 — Span bridge

- [ ] **8.3a** Add tests to `field_registry.t.cpp` for span extraction:
  - `extract_scalar_span(registry, field_ref, scalar_handle)` → `scalar_span` with correct D/Rx/Ry/Rz spans.
  - `extract_scalar_view(registry, field_ref, scalar_handle)` → `scalar_view` (const).
  - Writing through the `scalar_span` modifies registry data.
  - `scalar_view` is read-only.
  - Test: `ctest --test-dir build -R t-field_registry`

- [ ] **8.3b** Implement span extraction in `field_registry.hpp`:
  - `scalar_span extract_scalar_span(field_registry&, field_ref, scalar_handle)`
  - `scalar_view extract_scalar_view(const field_registry&, field_ref, scalar_handle)`
  - Constructs `scalar<std::span<real>>` from the registry's View `data()`/`extent(0)`.
  - File: `src/fields/field_registry.hpp`
  - Test: `ctest --test-dir build -R t-field_registry`

### 8.4 — Integration test with existing operators

- [ ] **8.4a** Write an integration test that:
  1. Creates a `field_registry` with 2 slots (input + output).
  2. Allocates a scalar with mesh-like sizes (e.g., 8×8×8 domain, small Rx/Ry/Rz).
  3. Extracts `scalar_view` and `scalar_span` via the span bridge.
  4. Passes them to an existing operator (e.g., matrix product or a simulated derivative).
  5. Verifies the result is written into the registry's storage.
  - This proves the registry can back the existing operator stack.
  - File: `src/fields/field_registry.t.cpp` (extend)
  - Test: `ctest --test-dir build -R t-field_registry`

### 8.5 — field_ref SBO fitness test

- [ ] **8.5a** Write a test demonstrating `field_ref` fits in `std::function` SBO:
  - Capture `field_ref` (12 bytes) in a `std::function<void(int)>` — verify no heap allocation.
  - Compare: capture `field_span` in a `std::function<void(int)>` — it heap-allocates.
  - Use Kokkos or a custom allocator to detect heap allocations, or simply check `sizeof`.
  - File: `src/fields/field_registry.t.cpp` (extend)
  - Test: `ctest --test-dir build -R t-field_registry`

---

## Completion Criteria

- `t-handle` passes with full coverage of handle arithmetic and selectors.
- `t-field_registry` passes: allocation, access, copy, swap, span bridge.
- Span bridge extracts working `scalar_span`/`scalar_view` from registry storage.
- `field_ref` is trivially copyable and fits in SBO.
- All existing tests still pass (no regressions).
- No existing code is modified (pure additions).
