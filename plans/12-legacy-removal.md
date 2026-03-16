# Phase 12: Remove Legacy Tuple Infrastructure

**Goal:** Delete the old `container_tuple`/`view_tuple` dual-inheritance hierarchy, the `mp_list`-based selector dispatch, the `zip_transform_view` lazy evaluation layer, and all supporting concepts. Rewrite the ~130 affected test cases to use the new handle/registry/expression system.

**Depends on:** Phases 8–11 (all callers migrated to registry + handles)

**Read first:**
- All files in `src/fields/` (to understand what remains)
- `plans/dsl-restructuring-proposal.md` §8 (file impact table)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build
```

---

## Strategy

This phase is primarily deletion. By this point, no production code should reference the old tuple types. The work is:
1. Verify no production code uses the old types.
2. Rewrite test files to use the new system.
3. Delete old infrastructure files.
4. Clean up CMakeLists.txt.

---

## Items

### 12.1 — Verify no production code references old types

- [ ] **12.1a** Grep for `container_tuple`, `view_tuple`, `single_view`, `tuple_pipe`, `OwningTuple`, `NestedTuple`, `list_index`, `selection_view_fn`, `zip_transform_view` in all non-test `.hpp`/`.cpp` files under `src/`. Verify zero matches (excluding the files about to be deleted).
  - If any remain, create follow-up items to migrate them.
  - No file changes.
  - Test: grep only.

### 12.2 — Rewrite field test files

Each old test file needs a new counterpart using the handle/registry system. The old tests document expected behavior; the new tests verify the same behavior through the new API.

- [ ] **12.2a** Rewrite `scalar.t.cpp` (9 TEST_CASEs, 290 lines):
  - Test `scalar_handle` construction, `handle_sel::D/Rx/Ry/Rz` dispatch.
  - Test scalar field arithmetic via expression templates.
  - File: `src/fields/scalar.t.cpp` (rewrite)
  - Test: `ctest --test-dir build -R t-scalar`

- [ ] **12.2b** Rewrite `vector.t.cpp` (8 TEST_CASEs, 373 lines):
  - Test `vector_handle` construction, component access, `handle_sel` dispatch.
  - File: `src/fields/vector.t.cpp` (rewrite)
  - Test: `ctest --test-dir build -R t-vector`

- [ ] **12.2c** Rewrite `selector.t.cpp` (23 TEST_CASEs, 734 lines — largest test file):
  - Test selection descriptors (`contiguous_selection`, `strided_selection`, `gather_selection`).
  - Test `assign_selected`, `fill_selected` with each descriptor type.
  - Test the mesh's selection descriptor builders.
  - File: `src/fields/selector.t.cpp` (rewrite)
  - Test: `ctest --test-dir build -R t-selector`

- [ ] **12.2d** Rewrite `tuple.t.cpp`, `container_tuple.t.cpp`, `view_tuple.t.cpp`, `single_view.t.cpp`, `tuple_pipe.t.cpp`:
  - These test infrastructure that is being deleted. Replace with tests for `field_registry`, `field_ref`, `handle_for_each`.
  - Some may be deleted entirely if their functionality is covered by `handle.t.cpp` and `field_registry.t.cpp`.
  - Files: 5 test files (rewrite or delete)
  - Test: `ctest --test-dir build -L fields`

- [ ] **12.2e** Rewrite `tuple_utils.t.cpp` (17 TEST_CASEs, 613 lines):
  - Replace tests for `for_each`/`transform`/`resize_and_copy` with tests for `handle_for_each` and expression template evaluation.
  - File: `src/fields/tuple_utils.t.cpp` (rewrite or delete)
  - Test: `ctest --test-dir build -L fields`

- [ ] **12.2f** Rewrite `tuple_math.t.cpp` and `field_math.t.cpp`:
  - Replace `zip_transform_view`-based arithmetic tests with expression template tests.
  - Files: 2 test files (rewrite)
  - Test: `ctest --test-dir build -L fields`

- [ ] **12.2g** Rewrite `range_concepts.t.cpp`:
  - Replace `zip_transform_view`/`stride_view` range-concept tests with expression/selection descriptor tests.
  - File: `src/fields/range_concepts.t.cpp` (rewrite or delete)
  - Test: `ctest --test-dir build -L fields`

- [ ] **12.2h** Rewrite `algorithms.t.cpp`:
  - Test `reduce_max`, `reduce_min`, `reduce_sum` and the `dot` implementation using expression templates.
  - File: `src/fields/algorithms.t.cpp` (rewrite)
  - Test: `ctest --test-dir build -L fields`

- [ ] **12.2i** Rewrite `field.t.cpp` and `field_utils.t.cpp`:
  - Test `field_registry` construction, slot allocation, and `field_ref` access patterns.
  - Files: 2 test files (rewrite)
  - Test: `ctest --test-dir build -L fields`

### 12.3 — Delete old infrastructure files

- [ ] **12.3a** Delete the following files:
  - `src/fields/container_tuple.hpp` (69 lines)
  - `src/fields/view_tuple.hpp` (341 lines)
  - `src/fields/tuple_pipe.hpp` (111 lines)
  - `src/fields/ccs_range_utils.hpp` (221 lines — `semiregular_box`, `view_closure`, `bind_back`, `compose`)
  - `src/fields/tuple_fwd.hpp` (716 lines — old concepts; new concepts live in handle/registry headers)
  - `src/fields/tuple.hpp` (243 lines)
  - `src/fields/tuple_utils.hpp` (393 lines)
  - `src/fields/tuple_math.hpp` (122 lines)
  - Files deleted: 8, ~2216 lines removed.

- [ ] **12.3b** Delete or gut the following files:
  - `src/fields/selector.hpp` — remove the old `plane_view`, `multi_slice_view`, `predicate_view`, `optional_view`, `selection_view_fn`. Keep any still-needed utilities.
  - `src/fields/selector_fwd.hpp` — remove `list_index` types; `si::D` etc. are now in `handle.hpp` as `handle_sel::D`.
  - `src/fields/lazy_views.hpp` — remove `zip_transform_view` (~400 lines). Keep `stride_view`, `repeat_n_view`, `cartesian_product_view`, `linear_distribute` if still used by mesh/stencil code.

- [ ] **12.3c** Simplify `src/fields/scalar.hpp` and `src/fields/vector.hpp`:
  - If `scalar<T>` and `vector<T>` are still used as types for the span bridge, keep them as simple structs (not inheriting from `ccs::tuple`). Otherwise, they may become type aliases for handle bundles.
  - `scalar_real`/`scalar_span`/`scalar_view` may remain if the operator layer still consumes them.
  - Files: `src/fields/scalar.hpp`, `src/fields/vector.hpp`

- [ ] **12.3d** Simplify `src/fields/field.hpp`:
  - Remove the `detail::field<S,V>` template if no longer needed.
  - The `field`/`field_span`/`field_view` aliases may point to `field_ref` or remain as compatibility shims.
  - Depends on what the operator layer needs.
  - File: `src/fields/field.hpp`

### 12.4 — Clean up CMakeLists.txt and matchers

- [ ] **12.4a** Update `src/fields/CMakeLists.txt`: remove deleted source files from the `fields` library, update test entries.
- [ ] **12.4b** Update `src/fields/matchers.hpp` if it references old tuple types.
- [ ] **12.4c** Remove `src/mesh/selections.hpp` if its `YPlaneView`/`FView` are no longer used (they were already dead code — not included by any TU).

### 12.5 — Final verification

- [ ] **12.5a** Full build and test: `cmake --build build && ctest --test-dir build`.
  - All tests pass.
  - `grep -r 'container_tuple\|view_tuple\|single_view\|tuple_pipe\|OwningTuple\|NestedTuple' src/` returns only comments/documentation.

---

## Completion Criteria

- Old tuple infrastructure is deleted (~2200+ lines removed).
- All 141 field test cases are rewritten or deleted, replaced by new tests.
- No production code references `container_tuple`, `view_tuple`, `single_view`, `tuple_pipe`, `zip_transform_view`, `semiregular_box`, or `list_index`.
- Full build succeeds. All tests pass.
- The `src/fields/` directory contains: `handle.hpp`, `field_registry.hpp`, `expr.hpp`, `selection_desc.hpp`, `scalar.hpp` (simplified), `vector.hpp` (simplified), `field.hpp` (simplified or deleted), `lazy_views.hpp` (trimmed), `algorithms.hpp` (using reductions), `matchers.hpp`, and their test files.
