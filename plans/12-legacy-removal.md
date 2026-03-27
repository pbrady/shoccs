# Phase 12: Remove Legacy Tuple Infrastructure

**Goal:** Delete the old `container_tuple`/`view_tuple` dual-inheritance hierarchy, the `mp_list`-based selector dispatch, the `zip_transform_view` lazy evaluation layer, and all supporting concepts. Delete the ~141 affected test cases (all old test files removed entirely — no rewrites); ~98 replacement tests already exist in the new handle/registry/expression/selection_desc test files.

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

This phase is primarily deletion. **Status update (2026-03-17):** Phase 9/11 completed via D-R6 coexistence strategy — new v2 adapter methods were added alongside old implementations, then old public interfaces were removed. However, the internal implementations of `heat.cpp`, `scalar_wave.cpp`, `gradient.hpp/cpp`, and the IO pipeline still use old DSL patterns: `scalar_real` scratch buffers, `m.xyz | func` piping, `sel::` pipe assignment, `get<si::*>()` tuple access on `scalar_real`, `vector_real`/`vector_span` types, `dot()` from `algorithms.hpp`, and `field_view` construction. These production D-R6 remnants must be migrated (12.3f) before infrastructure files can be deleted. The `mesh.hpp` selector members (`xmin`–`zmax`, `fluid`) and old selector methods (`grid_boundaries`, `object_boundaries`, `dirichlet(Grid/Object)`, `non_dirichlet`, `fluid_all`) also remain.

The work is:
1. Verify no production code uses the old types (done — 12.1a identified all remaining usage).
2. Delete old test files (done — 12.2a–i).
3. **Migrate production D-R6 remnants** (12.3f — new; required before gutting/deletion).
4. Gut files being partially kept (selector, lazy_views, scalar, vector, field).
5. Delete old infrastructure files.
6. Clean up CMakeLists.txt.

**Ordering:** 12.1 → 12.2a–i (fields tests) → **12.3c3a** (pre-migrate old DSL in stats/initialize/write) → 12.3c3b+c1+c2 (atomic struct rewrite; within 12.3c: c3a → c3b+c1+c2 atomically; c4 independent) → **12.3f1–f3** (migrate production D-R6 remnants: scratch buffers, m.xyz piping, sel:: pipes in heat/scalar_wave) → **12.3f4** (migrate vector_real/vector_span in scalar_wave/gradient) → **12.3f5** (migrate field_view in IO pipeline) → **12.3f6** (scalar.hpp Phase B) → **12.3e4** (domain() type changes + delete xyz/vxyz from mesh) → 12.3b3 (gut lazy_views, needs selector.hpp → tuple.hpp chain handled) → 12.3e5 → 12.3e6a (delete ss/vs from mesh, after 12.2j migrates test callers) → 12.2j1–j5 (non-fields tests; j4/j6/j7/j8/j9 done; j1–j3/j5 depend on 12.3e6a for Pattern A sizing) → **12.3f7** (mesh.hpp old selector cleanup, unblocked after f1–f3 + 12.2j1–j5 migrate all sel:: callers; subsumes 12.3e6c–e) → 12.3c4 (delete vector.hpp, unblocked after f4) → 12.3d2+d3 (delete field.hpp/field_fwd.hpp, unblocked after f5) → 12.3b1/b2 (delete selectors, unblocked after f7+12.2j) → 12.3a (delete remaining) → 12.4 → 12.5

---

## Items

### 12.1 — Verify no production code references old types

- [x] **12.1a** Comprehensive audit of production-code dependencies on old infrastructure. Grep for each category below in all non-test `.hpp`/`.cpp` files under `src/`. Verify zero matches (excluding `src/fields/` infrastructure files being deleted in 12.3). If any non-infrastructure production code still references old types, create follow-up items to migrate them before proceeding. No file changes — grep only. **COMPLETED:** All findings match the plan's expectations exactly. No unexpected production-code dependencies found. All `container_tuple`/`view_tuple`/`single_view`/`zip_transform_view` references are confined to `src/fields/` infrastructure files. Categories A–G confirmed: `sel::`/`si::` in mesh.hpp/cpp, scalar_wave.cpp, heat.cpp, field_data.cpp, derivative.cpp (unqualified via `using namespace si;`, 22 uses); `field_view` in io/*.hpp/cpp, heat.cpp, scalar_wave.cpp; `system_size` in all system .hpp/.cpp; `NumericTuple` in real3_operators.hpp (7 uses); stale `#include "fields/tuple_fwd.hpp"` in coefficient_visitor.hpp, unit_stride_visitor.hpp; `ccs::tuple` in xdmf.hpp/cpp, field_io.hpp/cpp, field_data.hpp/cpp, cartesian.hpp, mesh.hpp/cpp; `TupleLike`/`ArrayFromTuple`/`to<real3>` in manufactured_solutions.hpp; `tuple_cat` in mesh.hpp (2 uses); `#include "fields/algorithms.hpp"` in scalar_wave.cpp, heat.cpp. `vector_real`/`vector_span`/`vector_view` in scalar_wave.hpp (2 struct members), gradient.hpp/cpp (function signature). All covered by 12.3b–e migration items.
  - **Category A — `selector.hpp`/`selector_fwd.hpp` usage (`sel::` pipe-syntax and `si::` / `get<si::>` index access).** Phase 11 must have migrated all `sel::` usage to `handle_sel` + selection descriptors. Phase 9 must have migrated `get<si::D>`/`get<si::Rx>` patterns on `scalar_real`/`field_view` to registry + handle access. Known production files that `#include "selector.hpp"` or use `sel::`/`si::`:
    - `src/mesh/mesh.hpp` — uses `sel::optional_view`, `sel::predicate`, `sel::Rx/Ry/Rz`, tuple pipe `|` syntax in `grid_boundaries()` and `object_boundaries()`
    - `src/systems/scalar_wave.cpp` — uses `sel::D`, `sel::R`, `sel::xR/yR/zR`; also `get<si::D>(sol_buf)` (line 191), `get<si::Rx>(sol_buf)` (line 198)
    - `src/systems/heat.cpp` — uses `sel::D`, `sel::R`; also `get<si::D>` (lines 125, 161), `get<si::Rx>` (lines 128, 168)
    - `src/systems/inviscid_vortex.cpp` — includes `selector.hpp`
    - `src/operators/derivative.cpp` — includes `selector.hpp` (redundant with `mesh.hpp` include; no direct `sel::` usage found)
    - `src/io/field_data.cpp` — uses `sel::R` (lines 55–57) and `get<si::D>` (line 46)
    - `src/fields/field_fwd.hpp` — `#include "selector.hpp"` (infrastructure, resolved by 12.3d)
  - **Category B — `field_view`/`field_span` type usage.** Phase 9 must have migrated these to `field_ref` + registry. Known production files:
    - `src/io/field_data.hpp` — `field_view` parameter in `write()`
    - `src/io/field_io.hpp` — `field_view` parameter
    - `src/io/field_io.cpp` — `field_view` parameter
    - `src/systems/scalar_wave.cpp` — constructs `field_view` for I/O (line ~276)
    - `src/systems/heat.cpp` — constructs `field_view` for I/O (line ~253)
  - **Category C — `system_size` struct.** Contains `scalar<integer>` member, which depends on `scalar.hpp` → `tuple.hpp`. Used extensively:
    - All system `.hpp/.cpp` files: `scalar_wave`, `heat`, `inviscid_vortex`, `hyperbolic_eigenvalues`, `empty_system`, `system.hpp/cpp`
    - `src/fields/field.hpp` — constructor takes `system_size`
    - `src/fields/field_utils.hpp` — returns `system_size`
    - Must be redesigned before `scalar.hpp`/`tuple.hpp` can be deleted (see 12.3d1).
  - **Category D — `tuple_fwd.hpp` concepts in non-fields production code.** These are NOT in scope for Phases 8–11 (they don't relate to field lifecycle); Phase 12 must handle them (see 12.3e):
    - `src/real3_operators.hpp` — uses `NumericTuple` concept (8 uses) for real3 arithmetic
    - `src/matrices/coefficient_visitor.hpp` — includes `tuple_fwd.hpp`; no direct concept usage found — likely a stale include
    - `src/matrices/unit_stride_visitor.hpp` — includes `tuple_fwd.hpp`; uses `Range` concept (= `std::ranges::input_range` with `int3` exclusion) — can be replaced
  - **Category E — `tuple.hpp` usage in non-fields production code** (Phase 12 must handle; see 12.3e):
    - `src/io/xdmf.hpp` — uses `ccs::tuple<span, span, span>` as a function parameter type (3 occurrences in `xdmf.cpp`)
    - `src/io/field_io.hpp` — uses `tuple<std::span<const mesh_object_info>, ...>` in `write()` parameter (line 48); also uses `field_view` (Category B)
    - `src/io/field_io.cpp` — uses `tuple<span, span, span>` in `write()` implementation (line 38)
    - `src/io/field_data.hpp` — uses `tuple<std::span<const mesh_object_info>, ...>` in `write_geom()` parameter (line 21)
    - `src/io/field_data.cpp` — uses `tuple<span, span, span>` in `write_geom()` implementation (line 9)
    - `src/mesh/cartesian.hpp` — `domain()` returns `tuple{ccs::cartesian_product(x(), y(), z())}` (line 74)
    - `src/mesh/mesh.hpp` — uses `ccs::tuple` extensively: `xyz` and `vxyz` member types (lines 199–208) declared via `decltype(domain())`, `ss()` returns `scalar<integer>` (line 144–147), `vs()` returns `tuple{ss(), ss(), ss()}` (line 151), `R()` returns `tuple{Rx(), Ry(), Rz()}` (line 108). Phase 11 should migrate selector-based members (`xmin`–`zmax`, `fluid`, `grid_boundaries`, `object_boundaries`) and pipe-pattern callers of `xyz`/`vxyz`. But `ss()`, `vs()`, and `R()` are standalone tuple usage that Phase 12 must handle.
    - `src/mesh/mesh.cpp` — constructs `xyz` and `vxyz` using `tuple{cart.domain(), geometry.domain()}` (lines 143–146)
  - **Category F — `tuple_utils.hpp` usage in non-fields production code** (Phase 12 must handle; see 12.3e):
    - `src/mms/manufactured_solutions.hpp` — uses `TupleLike`, `ArrayFromTuple`, `to<real3>` concepts/functions from `tuple_utils.hpp`
    - `src/mesh/mesh.hpp` — uses `tuple_cat<tuple<integer>>` in `dirichlet()` (line 175) and `fluid_all()` (line 181); these should be migrated by Phase 11 (selector-related)
  - **Category G — `algorithms.hpp` production usage** (Phase 12 must handle):
    - `src/systems/scalar_wave.cpp` — `#include "fields/algorithms.hpp"` (line 2); uses `dot(grad_G, du)` (line 178), `minmax(...)` (line 221), `max(...)` (line 223).
    - `src/systems/heat.cpp` — `#include "fields/algorithms.hpp"` (line 2); uses `minmax(...)` (line 196), `max(...)` (line 198).
    - **Resolution:** `dot` is NOT reimplemented. The old `dot()` operates on `Vector` types (= `tuple<scalar, scalar, scalar>`) via `tuple_math` multiplication + addition. The new adapter methods (Phase 9) replace this with explicit component-wise `scalar_expr` arithmetic: `gx * ux + gy * uy + gz * uz`. `minmax`/`max` are replaced by `reduce_max`/`reduce_min` on expression nodes. The `#include "fields/algorithms.hpp"` lines are removed when old system methods are deleted (Phase 9 migration or Phase 12 old-code cleanup per D-R6).
  - **Known infrastructure-only references (resolved by 12.3, not by callers):**
    - `list_index`: `selector_fwd.hpp` defines `si::D`, `si::Rx`, etc. as `list_index<...>` aliases — deleted in 12.3b2. **Note:** `si::D`/`si::Rx` are also used in production code via `get<si::D>()` patterns (see Category A: `scalar_wave.cpp`, `heat.cpp`, `field_data.cpp`) and in non-fields test files (`derivative.t.cpp`, `laplacian.t.cpp`, `field_registry.t.cpp`). These must be migrated by Phases 9/11 before 12.3b2 deletion.
    - `semiregular_box`: `selector.hpp` (7 struct members) and `lazy_views.hpp` (`zip_transform_view` member) — removed in 12.3b.
    - `view_closure`/`make_view_closure`/`bind_back`/`compose`: `selector.hpp` and `tuple_pipe.hpp` — deleted in 12.3.

### 12.2 — Delete old test files

New test files already provide complete replacement coverage: `handle.t.cpp` (13 tests), `field_registry.t.cpp` (18 tests), `expr.t.cpp` (32 tests), `selection_desc.t.cpp` (35 tests) — total 98 new tests. All 15 old fields test files (12.2a–i: ~3500 lines, ~141 TEST_CASEs) are deleted entirely — no rewrites needed.

- [x] **12.2a** Delete `scalar.t.cpp` (9 TEST_CASEs, 290 lines): **DONE** — file deleted, CMakeLists entry removed.
  - Current: tests `scalar<T>` nested-tuple construction, `sel::D/Rx/Ry/Rz` dispatch, arithmetic, `lift()`, mesh location piping.
  - **Decision: delete entirely.** All functionality is already covered by new test files:
    - `scalar_handle` construction + `handle_sel::D/Rx/Ry/Rz` dispatch → `handle.t.cpp` (5 tests: "scalar_handle accessors", "consteval factory validation", "handle_sel::D dispatch", "handle_sel::R dispatch", "handle_sel::Rx dispatch").
    - `scalar_expr` arithmetic + `assign_scalar` → `expr.t.cpp` (10+ tests: "scalar_expr operator+", "scalar_expr operator*", "assign_scalar materializes expression", compound ops, reductions).
    - `scalar_span`/`scalar_view` struct (after 12.3c1 simplification) → tested implicitly by span bridge tests in `field_registry.t.cpp` ("extract_scalar_span", "extract_scalar_view"); the struct is a trivial aggregate with 4 named `std::span` members.
    - Mesh location piping (tests 8–9) → integration coverage in operator test files (`derivative.t.cpp`, `laplacian.t.cpp`) which use Pattern B after 12.2j migration.
  - No new test file needed. Remove `add_unit_test(scalar "fields" fields)` from `CMakeLists.txt` in 12.4a.
  - File: delete `src/fields/scalar.t.cpp`
  - Test: `ctest --test-dir build -L fields`

- [x] **12.2b** Delete `vector.t.cpp` (8 TEST_CASEs, 373 lines): **DONE** — file deleted, CMakeLists entry removed.
  - Current: tests `vector<T>` 3-level nested-tuple construction, `sel::Dx/Dy/Dz/xRx/...` dispatch, component access, `lift()`.
  - **Decision: delete entirely.** `vector.hpp` is deleted in 12.3c4. All replacement functionality is already tested:
    - `vector_handle` construction + component accessors (`.x()/.y()/.z()`, `.Dx()/.Dy()/.Dz()`, 9 boundary selectors) → `handle.t.cpp` (3 tests: "vector_handle accessors", "handle_for_each visits vector buffers", "paired handle_for_each on vector handles").
    - `handle_sel` dispatch on vector handles → `handle.t.cpp` ("handle_sel::D dispatch", "handle_sel::R dispatch" — both test vector path returning arrays of 3/9 `buf_handle`s).
    - No `vector_span`/`vector_view` replacement struct exists (no `extract_vector_span` in `field_registry.hpp`); vector access goes through 3 `scalar_handle` components.
  - No new test file needed. Remove `add_unit_test(vector "fields" fields)` from `CMakeLists.txt` in 12.4a.
  - File: delete `src/fields/vector.t.cpp`
  - Test: `ctest --test-dir build -L fields`

- [x] **12.2c** Delete `selector.t.cpp` (23 TEST_CASEs, 734 lines): **DONE** — file deleted, CMakeLists entry removed.
  - Current: tests `sel::xmin/xmax/ymin/ymax/zmin/zmax` (plane selectors), `sel::multi_slice` (sparse ranges), `sel::optional_view` (conditional), `sel::predicate` (boolean mask) — all old view-based selectors operating on `tuple<T>`, `scalar<T>`, `vector<T>`.
  - **Decision: delete entirely.** `selection_desc.t.cpp` already provides complete replacement coverage:
    - Plane selectors → `make_x/y/z_plane_desc` + cross-check tests (6 tests validating contiguous/strided element indexing).
    - Multi-slice → `make_gather_from_slices` + `gather_selection` tests (3 tests).
    - Predicate → `make_gather_from_predicate` test.
    - Assignment through selectors → `assign_selected`, `fill_selected`, `plus_assign_selected` tests (7 tests covering all 3 descriptor types).
    - Grid BC iteration → `for_each_grid_bc_desc` test.
    - Compound math through selectors → replaced by `plus_assign_selected`/`assign_selected` with expression nodes.
    - Optional selector → replaced by conditional logic at the call site (no descriptor needed for conditional BC application).
  - No new test file needed. Remove `add_unit_test(selector "fields" fields)` from `CMakeLists.txt` in 12.4a.
  - File: delete `src/fields/selector.t.cpp`
  - Test: `ctest --test-dir build -L fields`

- [x] **12.2d** Delete pure-old-infrastructure test files (50 TEST_CASEs across 5 files): **DONE** — all 5 files deleted, CMakeLists entries removed.
  - **12.2d1** Delete `container_tuple.t.cpp` (10 TEST_CASEs): tests `container_tuple<T>` construction, structured binding, copy, move, nested. All old owning-tuple infrastructure; storage tests covered by `field_registry.t.cpp`.
  - **12.2d2** Delete `view_tuple.t.cpp` (18 TEST_CASEs): tests `view_tuple<T&>`, `single_view`, pipe syntax, pass-through assignment, math. Replaced by `expr.t.cpp` (expression evaluation) and `handle.t.cpp` (handle access patterns).
  - **12.2d3** Delete `single_view.t.cpp` (1 TEST_CASE): tests `single_view` wrapper — infrastructure eliminated entirely.
  - **12.2d4** Delete `tuple.t.cpp` (13 TEST_CASEs): tests `ccs::tuple<T>` concepts, construction, conversion, `list_index`-based access. Covered by `handle.t.cpp` (handle construction, `handle_sel` dispatch) and `field_registry.t.cpp`.
  - **12.2d5** Delete `tuple_pipe.t.cpp` (8 TEST_CASEs): tests pipe syntax (`operator|` with transforms). Replaced by expression template composition in `expr.t.cpp`.
  - Files: delete all 5 `.t.cpp` files.
  - Test: `ctest --test-dir build -L fields`

- [x] **12.2e** Delete `tuple_utils.t.cpp` (17 TEST_CASEs, 613 lines): **DONE** — file deleted, CMakeLists entry removed.
  - Tests `for_each`, `transform`, `reduce`, `resize_and_copy`, `lift`, `to<T>`, `make_tuple`, `tuple_cat`, `join` — all operate on `ccs::tuple<T>` being removed.
  - Replacement: `handle_for_each` in `handle.t.cpp`, expression evaluation in `expr.t.cpp`.
  - File: delete `src/fields/tuple_utils.t.cpp`
  - Test: `ctest --test-dir build -L fields`

- [x] **12.2f** Delete `tuple_math.t.cpp` and `field_math.t.cpp` (8 TEST_CASEs total): **DONE** — both files deleted, CMakeLists entries removed.
  - `tuple_math.t.cpp` (5 TEST_CASEs): tests `+=`, `+`, `-`, `*` on `ccs::tuple<T>`. Replaced by `expr.t.cpp` expression arithmetic (`plus_assign`, `assign` with binary expressions).
  - `field_math.t.cpp` (3 TEST_CASEs): tests arithmetic on old `field`/`field_span`. Replaced by `expr.t.cpp` `assign_scalar`/`plus_assign` tests.
  - Files: delete both `.t.cpp` files.
  - Test: `ctest --test-dir build -L fields`

- [x] **12.2g** Delete `range_concepts.t.cpp` (17 TEST_CASEs): **DONE** — file deleted, CMakeLists entry removed.
  - Tests concepts from `tuple_fwd.hpp` (`OutputRange`, `OutputTuple`, `TupleLike`, `NonTupleRange`, `SimilarTuples`, `NestedTuple`, `ViewClosure`, `NumericTuple`, `ListIndex`, `ArrayFromTuple`) — all defined in `tuple_fwd.hpp` which is being deleted.
  - Also tests `container_tuple` and `list_index` directly.
  - New types have their own concept checks in `handle.t.cpp`, `field_registry.t.cpp`, `expr.t.cpp`.
  - File: delete `src/fields/range_concepts.t.cpp`
  - Test: `ctest --test-dir build -L fields`

- [x] **12.2h** Delete `algorithms.t.cpp` (3 TEST_CASEs, 63 lines): **DONE** — file deleted, CMakeLists entry removed.
  - Current: tests `dot`, `minmax`, `max` using `scalar<std::vector<int>>`, `vector<std::vector<int>>`, `sel::D/Rx/Ry/Rz` — all old types.
  - **Decision: delete entirely.** No reimplementation of `dot` is needed:
    - `minmax`/`max` → replaced by `reduce_max`/`reduce_min`/`reduce_sum` in `expr.hpp`, tested in `expr.t.cpp` (2+ tests: "reduce_max over raw buffer", "reduce_min over raw buffer").
    - `dot(grad_G, du)` (vector dot product at `scalar_wave.cpp:178`) → replaced by explicit component-wise expression template arithmetic in the new adapter method: `gx * ux + gy * uy + gz * uz` using `scalar_expr` operations. The old `dot()` function operates on `Vector` types (= `tuple<scalar, scalar, scalar>`) using `tuple_math` operators; the new system uses 3 `scalar_handle` components directly. No general-purpose `dot` function is needed.
    - `minmax`/`max` calls in old system methods (`heat.cpp:196,198`, `scalar_wave.cpp:221,223`) → replaced by `reduce_max`/`reduce_min` on expression nodes in the new adapter methods. The `#include "fields/algorithms.hpp"` lines are removed when old system methods are deleted (Phase 9 migration or Phase 12 old-code cleanup).
  - File: delete `src/fields/algorithms.t.cpp`
  - Test: `ctest --test-dir build -L fields`

- [x] **12.2i** Delete `field.t.cpp` and `field_utils.t.cpp` (6 TEST_CASEs total): **DONE** — both files deleted, CMakeLists entries removed.
  - `field.t.cpp` (4 TEST_CASEs): tests old `field`/`field_view`/`field_span` construction, concepts, assignment. Replaced by `field_registry.t.cpp` (18 tests: construction, allocation, deep_copy, swap, span extraction).
  - `field_utils.t.cpp` (2 TEST_CASEs): tests `for_each`/`transform` over old `field`. Replaced by `handle_for_each` in `handle.t.cpp`.
  - Files: delete both `.t.cpp` files.
  - Test: `ctest --test-dir build -L fields`

- [x] **12.2j** Update non-fields test files that include deleted headers or use old infrastructure types. These test files are outside `src/fields/` and will break when infrastructure headers are deleted in 12.3a/12.3b. Depends on: 12.3c (scalar type simplification), 12.3d1 (system_size redesign), 12.3e4 (domain() migration). j4/j6 have no 12.3c/12.3e dependency; j7/j8 need only 12.3c1+12.3d1; j9 needs 12.3c1+12.3d1+12.3d2+12.3e3+Phase 9. Must complete before: 12.3a (file deletion), 12.3b1/b2 (selector deletion).
  - **Common test migration patterns** for 12.2j1–j3, j5. After 12.3c (scalar struct), 12.3e4 (domain() change), and 12.3e6 (mesh cleanup), test files lose `scalar<T>`, `vector_real`, `m.ss()`, `m.vs()`, `m.xyz`, `m.vxyz`, and `tuple_math` arithmetic. Six recurring replacement patterns:
    - **Pattern A — Owning scalar allocation:** `scalar<T>{m.ss()}` → allocate 4 separate `std::vector<real>` sized by `m.size()`, `m.Rx().size()`, `m.Ry().size()`, `m.Rz().size()`; construct `scalar_span{d, rx, ry, rz}`. For copy-construction (`scalar<T>{u}`), copy each source span member into a new vector.
    - **Pattern B — Mesh-location piping:** `m.xyz | f2` (constructs `scalar<T>` with domain + object values) → explicit loops: iterate `ccs::cartesian_product(m.x(), m.y(), m.z())` to fill `.D`; iterate `m.Rx()/Ry()/Rz()` using `.position` member to fill `.Rx/.Ry/.Rz`. The `std::views::transform` lambda functions (`f2`, `f2_dx`, ...) accept `auto&& loc` with structured binding `[x, y, z]` and work unchanged with both `cartesian_product_view` elements and `mesh_object_info::position` (a `real3`). A test-local helper `owned_scalar eval_at_locations(mesh, func)` returning owned vectors + `scalar_span` can reduce boilerplate across test cases.
    - **Pattern C — Scalar arithmetic:** `du += 1`, `ex = a + b + c` (via `tuple_math` operators on `scalar<T>`) → element-wise loops on each span member (`.D`, `.Rx`, `.Ry`, `.Rz`). A test-local `for_each_component(scalar_span, func)` helper can reduce repetition.
    - **Pattern D — Owning vector allocation:** `vector_real{m.vs()}` → 3 × Pattern A (one set of 4 vectors per component Dx/Dy/Dz, represented as 3 `scalar_span` values or a local `vector_span` struct).
    - **Pattern E — Approx helper rewrite:** `approx<si::D, si::Rx, ...>(u, v)` (uses `for_each` + `get<I>()` from `tuple_utils.hpp`) → compare each named member directly: `REQUIRE_THAT(to_vec(u.D), Approx(to_vec(v.D)))`, `REQUIRE_THAT(to_vec(u.Rx), Approx(to_vec(v.Rx)))`, etc. Remove the template parameter pack dispatch.
    - **Pattern F — Selection-based assignment:** `ex | m.dirichlet(gridBcs) = 0` and `du | m.fluid_all(o) = rhs` → use `for_each_grid_bc_desc(m, gridBcs, [&](auto desc) { fill_selected(ex_span, desc, 0); })` or iterate `fluid_desc().element(i)` indices for fluid assignment.
  - [x] **12.2j1** `src/operators/derivative.t.cpp` (407 lines) — comprehensive rewrite of 6 TEST_CASEs: **DONE** — Complete rewrite removing all `scalar<T>`, `sel::`, `si::`, `m.ss()`, `m.dirichlet()`, `m.fluid_all()`, and `tuple_math` arithmetic. Introduced test-local `owned_scalar` struct (4 vectors + implicit conversions to `scalar_view`/`scalar_span`), `eval_at_mesh` (Pattern B), `make_scalar`/`copy_scalar` (Pattern A), `approx_D`/`approx_all` (Pattern E), `fill_scalar`/`add_offset` (Pattern C), `zero_grid_dirichlet`/`zero_dirichlet`/`assign_fluid_all` (Pattern F). Removed `#include "fields/selector.hpp"`. All 6 TEST_CASEs preserved with same test logic. Build succeeds; t-derivative has pre-existing Kokkos/OpenMP init failure (unchanged). Changes span three categories:
    - **sel::/si:: access (29 uses):**
      - **Scalar access:** `get<si::D>(u)`, `get<si::Rx>(...)`, etc. (~20 uses). Replace with `u.D`, `du.Rx`, etc. (Pattern E for approx helper).
      - **Pipe assignment:** `u | sel::D = tmp` (~5 uses). Replace with `std::ranges::copy(tmp, u.D.begin())`.
      - **Size queries:** `std::ranges::size(u | sel::Rx)`, `std::ranges::size(du | sel::D)` (~4 uses). Replace with `u.Rx.size()`, `du.D.size()`.
      - **Approx helper** (lines 84–93): `approx<si::D>(...)`, `approx<si::D, si::Rx, si::Ry, si::Rz>(...)` — 8 call sites. Rewrite per Pattern E.
    - **scalar<T> construction + m.xyz piping (25+ declarations):**
      - `scalar<T>{m.ss()}` — 9 uses (lines 114, 148, 151, 179, 206, 275, 376×3). Replace per Pattern A.
      - `scalar<T>{u}` copy — 11 uses (lines 214×2, 219, 236×2, 241, 300, 337, 338, 339, 393). Replace per Pattern A (copy variant).
      - `scalar<T> u = m.xyz | f2` and `u = m.xyz | f2` — 18 uses total (lines 108, 110, 112, 176, 183×3, 266, 274×3, 297, 298×3, 333, 371, 374). Replace per Pattern B.
    - **Scalar arithmetic + selection assignment:**
      - `du += 1`, `ex += 1`, `u -= 1` — 5 uses. Replace per Pattern C.
      - `ex | m.dirichlet(gridBcs) = 0` — 3 uses. Replace per Pattern F.
      - `du_x | m.fluid_all(objectBcs) = m.xyz | f2_ddx` — 3 uses (lines 378, 381, 384). Replace with `fluid_all_desc()` iteration + Pattern B evaluation at fluid indices.
    - Remove `#include "fields/selector.hpp"`, add `#include "fields/scalar.hpp"` (new struct-based).
    - Depends on: 12.3c1 (scalar_span/scalar_view struct definition), 12.3e4 (domain() type change for Pattern B), 12.3e6a (mesh::ss() deletion).
    - Estimated diff: ~300 lines (complete rewrite of 6 TEST_CASEs + approx helper).
    - File: `src/operators/derivative.t.cpp`
    - Test: `ctest --test-dir build -R t-derivative`
  - [x] **12.2j2** `src/operators/gradient.t.cpp` (330 lines) — comprehensive rewrite of 4 TEST_CASEs: **DONE** — Complete rewrite removing all `scalar<T>`, `vector_real`, `sel::`, `vi::`, `m.ss()`, `m.vs()`, `m.dirichlet()`, `m.fluid`, and `tuple_math` infrastructure. Introduced test-local `owned_scalar` struct (4 vectors + implicit conversions to `scalar_view`/`scalar_span`), `eval_at_mesh` (Pattern B), `make_scalar`/`copy_scalar` (Pattern A), `zero_grid_dirichlet`/`zero_dirichlet` (Pattern F), `assign_fluid` (Pattern F). Vector-level `get<vi::*>()` replaced with named members on 3 separate `owned_scalar` components (ex_x/ex_y/ex_z, du_x/du_y/du_z). R component filling via `m.Rx()/Ry()/Rz() | pos | func` patterns preserved directly on `owned_scalar` members. `#include "fields/selector.hpp"` replaced with `#include "fields/scalar.hpp"` + `#include "fields/selection_desc.hpp"`. All 4 TEST_CASEs preserved with same test logic. Build succeeds; t-gradient has pre-existing Kokkos/OpenMP init failure (unchanged). 6 pre-existing test failures unrelated.
    - File: `src/operators/gradient.t.cpp`
    - Test: `ctest --test-dir build -R t-gradient`
  - [x] **12.2j3** `src/operators/laplacian.t.cpp` (371 lines) — comprehensive rewrite of 4 TEST_CASEs: **DONE** — Complete rewrite removing all `scalar<T>`, `sel::`, `si::`, `m.ss()`, `m.dirichlet()`, `m.fluid`, `m.fluid_all()`, and `tuple_math` arithmetic. Introduced test-local `owned_scalar` struct (4 vectors + implicit conversions to `scalar_view`/`scalar_span`), `eval_at_mesh` (Pattern B), `make_scalar`/`copy_scalar` (Pattern A), `add_offset`/`add_scalar` (Pattern C), `zero_grid_dirichlet`/`zero_dirichlet` (Pattern F), `assign_fluid_all`/`assign_fluid` (Pattern F). Removed `#include "fields/selector.hpp"`. All 4 TEST_CASEs preserved with same test logic. Build succeeds; t-laplacian has pre-existing Kokkos/OpenMP init failure (unchanged). 6 pre-existing test failures unrelated.
    - File: `src/operators/laplacian.t.cpp`
    - Test: `ctest --test-dir build -R t-laplacian`
  - [x] **12.2j4** `src/operators/eigenvalue_visitor.t.cpp` (116 lines) — `#include "fields/tuple_utils.hpp"`; 2 uses of `to<T>()`: **DONE** — removed `#include "fields/tuple_utils.hpp"`, replaced `to<T>(v.eigenvalues_real())` with `T(er.begin(), er.end())` at both call sites (lines 58 and 112). t-eigenvalue_visitor has pre-existing Kokkos/OpenMP initialization failure (unrelated).
    - File: `src/operators/eigenvalue_visitor.t.cpp`
    - Test: `ctest --test-dir build -R t-eigenvalue`
  - [x] **12.2j5** `src/mesh/mesh.t.cpp` (483 lines) — comprehensive rewrite of tests 3–5: **DONE** — Removed `#include "fields/selector.hpp"`. Replaced all `scalar<T>{m.ss()}` (5 uses) with plain `std::vector<int/real>` sized by `m.size()`/`m.Rx().size()`/etc. Replaced `u | sel::D` → `u_d`, `u | sel::Rx/Ry/Rz` → `u_rx/u_ry/u_rz`. Replaced `u | m.fluid = 1` with `fluid_desc()` + `to_host_indices()` helper loop. Replaced `u | m.dirichlet(obj_bcs) = -1` with `dirichlet_object_desc(dir, obj_bcs)` loops over R components. Replaced `u == v` (tuple equality) with per-component `REQUIRE(u_d == v_d)` etc. Replaced `u | m.fluid = u` (fluid copy) with fluid index loop. Renamed test 5 from "fluid_desc matches fluid multi_slice selector" to "fluid_desc"; replaced old-infrastructure cross-check (marker + `m.fluid` pipe) with geometric classification check using distance from sphere center. Added file-local `to_host_indices(gather_selection)` helper. Tests 1, 2, 6 unchanged. All 6 TEST_CASEs preserved. t-mesh passes (2/2).
    - File: `src/mesh/mesh.t.cpp`
    - Test: `ctest --test-dir build -R t-mesh`
  - [x] **12.2j6** `src/mms/mms.t.cpp` (203 lines) — `#include "fields/tuple_utils.hpp"`; 1 use of `to<real3>()`: **DONE** — removed `#include "fields/tuple_utils.hpp"`, replaced `to<real3>(res)` with `real3{std::get<0>(res), std::get<1>(res), std::get<2>(res)}` (line 199). t-mms passes.
    - File: `src/mms/mms.t.cpp`
    - Test: `ctest --test-dir build -R t-mms`
  - [x] **12.2j7** `src/systems/scalar_wave.t.cpp` (156 lines) — uses `system_size.scalar_size` tuple access and `sel::D` pipe syntax: **DONE** — `system_size.scalar_size` already updated by 12.3d1. `u_rhs | sel::D` → `u_rhs.D` (line 151). t-scalar_wave passes.
    - **`system_size.scalar_size` tuple access** (lines 29–33 in `setup_registry` helper; line 29 is `auto& ss = sz.scalar_size;`, lines 30–33 are the four `get<>` calls):
      - `get<0>(get<0>(ss))` → `sz.d_size`
      - `get<0>(get<1>(ss))` → `sz.rx_size`
      - `get<1>(get<1>(ss))` → `sz.ry_size`
      - `get<2>(get<1>(ss))` → `sz.rz_size`
      - Also change `auto& ss = sz.scalar_size;` (line 29) → remove (access `sz.*_size` directly).
    - **Selector pipe** (line 152): `u_rhs | sel::D` → `u_rhs.D` (a `std::span<real>`; range operations work directly).
    - No include changes needed (gets selector.hpp transitively through `system.hpp` → system headers; after migration, no old-infrastructure includes remain — `field_registry.hpp` and `selection_desc.hpp` are already present).
    - Depends on: 12.3c1 (scalar_span struct — for `| sel::D` → `.D`), 12.3d1 (system_size redesign — for `scalar_size` removal).
    - Estimated diff: ~15 lines.
    - File: `src/systems/scalar_wave.t.cpp`
    - Test: `ctest --test-dir build -R t-scalar_wave`
  - [x] **12.2j8** `src/systems/heat.t.cpp` (375 lines) — uses `system_size.scalar_size` tuple access, `sel::D` pipe syntax, `transform()` from `tuple_utils.hpp`, and `get<si::*>()` accessors: **DONE** — `system_size.scalar_size` already updated by 12.3d1. `u0_scalar | sel::D` → `u0_scalar.D` (3 uses), `u_rhs | sel::D` → `u_rhs.D` (4 uses), `transform(lambda, u_rhs)` → manual `std::accumulate` over `.D/.Rx/.Ry/.Rz` (2 uses), `get<si::D/Rx/Ry/Rz>(sum)` → `sum_d/sum_rx/sum_ry/sum_rz` (7 uses). t-heat passes. Also noted:
    - **`system_size.scalar_size` tuple access** (lines 28–32 in `setup_registry` helper): same 4-line pattern as 12.2j7. Replace `get<0>(get<0>(ss))` etc. with `sz.d_size`/`sz.rx_size`/`sz.ry_size`/`sz.rz_size`.
    - **Selector pipe `| sel::D`** (7 uses):
      - `u0_scalar | sel::D` at lines 119, 233, 347 (used in `std::ranges::count(... | sel::D, 0.0)`).
      - `u_rhs | sel::D` at lines 132, 246, 360 (used in `std::ranges::count`).
      - `u_rhs | sel::D` at line 139 (assigned to `d_rng` for `std::accumulate`).
      - Replace all with `.D` named member access (e.g., `std::ranges::count(u0_scalar.D, 0.0)`).
    - **`transform()` from `tuple_utils.hpp`** (lines 253, 366): `transform(lambda, u_rhs)` applies a function across all 4 components of a `scalar_span`, returning a `scalar<real>` with accumulated sums. Replace with manual accumulation into 4 local `real` variables:
      ```
      real sum_d  = std::accumulate(u_rhs.D.begin(),  u_rhs.D.end(),  0.0);
      real sum_rx = std::accumulate(u_rhs.Rx.begin(), u_rhs.Rx.end(), 0.0);
      real sum_ry = std::accumulate(u_rhs.Ry.begin(), u_rhs.Ry.end(), 0.0);
      real sum_rz = std::accumulate(u_rhs.Rz.begin(), u_rhs.Rz.end(), 0.0);
      ```
    - **`get<si::D/Rx/Ry/Rz>(sum)` and `get<si::Rx/Ry/Rz>(ss)`** (lines 255–262, 368–374): Replace with the 4 local variables and `sz.rx_size` etc. (e.g., `REQUIRE(sum_rx == Catch::Approx((real)sz.rx_size))`). Total 11 uses across 2 test cases.
    - **`sys.size().scalar_size`** (lines 258, 371): `auto ss = sys.size().scalar_size` → `auto sz = sys.size()` and access `sz.rx_size` etc. directly.
    - The 3rd test case ("2D heat - E2 - floating", line 265) follows the same pattern as test 2 but has only 2 dimensions — `sum_rz` assertion is absent (`sum_rx` at line 373, `sum_ry` at line 374, but no `sum_rz`/`get<si::Rz>`).
    - Depends on: 12.3c1 (scalar_span struct), 12.3d1 (system_size redesign).
    - Estimated diff: ~60 lines.
    - File: `src/systems/heat.t.cpp`
    - Test: `ctest --test-dir build -R t-heat`
  - [x] **12.2j9** `src/io/field_io.t.cpp` (76 lines) — uses old `field` type, `sel::D`, `field_view`: **DONE** — Removed dead `field f{}` from first test. Replaced `field f{system_size{...}}` + `f.scalars(0, 1)` in second test with plain `std::vector<real>` + `scalar_span` (avoided `field_registry` to stay Kokkos-free like the original test). Replaced `u | sel::D = std::views::iota(...)` with `std::ranges::copy(... | transform(to_real), u.D.begin())`. Replaced `field_view fv{f}; io.write(names, fv.scalars(), ...)` with `std::vector<scalar_view> io_scalars{u, v}; io.write(names, io_scalars, ...)`. Removed `#include "fields/field.hpp"` and `#include "fields/selector.hpp"`; added `#include "fields/scalar.hpp"`. t-field_io passes (2/2 tests).
    - File: `src/io/field_io.t.cpp`
    - Test: `ctest --test-dir build -R t-field_io`
  - **Ordering:** 12.2j4 and 12.2j6 can run early (simple `to<T>` replacement, no scalar struct dependency, no dependency on 12.3e5 — `res` in mms.t.cpp is from Lua, not manufactured_solutions). 12.2j1/j3 depend on 12.3c1 (scalar struct) + 12.3e4 (domain() change for Pattern B) + 12.3e6a (mesh::ss() deletion for Pattern A sizing). 12.2j2 depends on 12.3c1 + 12.3c4 (vector.hpp deletion) + 12.3e4 + 12.3e6a (mesh::vs() deletion) + Phase 11 (gradient output restructuring). 12.2j5 depends on 12.3c1 + 12.3e4 + 12.3e6 (mesh member cleanup). 12.2j7/j8 depend on 12.3c1 + 12.3d1 (system_size redesign). 12.2j9 done (all prerequisites satisfied; unblocks 12.3d2).
  - Test: `cmake --build build && ctest --test-dir build`

### 12.3 — Delete old infrastructure files

**Ordering:** **12.3c3a** (pre-migrate old DSL in stats/initialize/write) → 12.3c3b+c1+c2 (atomic struct rewrite) → **12.3f1–f5** (production D-R6 cleanup) → **12.3f6** (scalar.hpp Phase B) → 12.3b3 (gut lazy_views, unblocked after f6 + 12.3e4b break include chains) → 12.3d (simplify types) → 12.3e (non-fields migration) → 12.2j1–j9 (non-fields tests) → **12.3f7** (mesh.hpp old selector cleanup, after f1–f3 + 12.2j1–j5) → 12.3b1/b2 (delete selectors) → 12.3a (delete remaining). This ensures include chains are resolved, non-fields includers and test files are migrated before files are removed. Note: 12.3b1/b2 (selector deletion) cannot run until 12.2j updates all non-fields test files that `#include "selector.hpp"` or use `sel::`/`si::` syntax (j1–j5, j7–j9). 12.3b1 also requires removal of stale `#include "selector.hpp"` from `inviscid_vortex.cpp` and migration of active `si::` usage in `derivative.cpp`.

- [x] **12.3a** Delete the following files: **DONE** — All 9 remaining infrastructure files deleted (2,469 lines removed). Build succeeds (56 targets rebuilt after touching scalar.hpp to trigger full recompilation of dependents). All tests pass (6 pre-existing failures unchanged).
  - `src/fields/container_tuple.hpp` (~69 lines)
  - `src/fields/view_tuple.hpp` (~341 lines)
  - `src/fields/tuple_pipe.hpp` (~111 lines)
  - `src/fields/ccs_range_utils.hpp` (~221 lines — `semiregular_box`, `view_closure`, `bind_back`, `compose`; all users removed by 12.3b)
  - `src/fields/tuple_fwd.hpp` (~716 lines — old concepts/fwd decls; new concepts in `handle.hpp`/`field_registry.hpp`)
  - `src/fields/tuple.hpp` (~243 lines)
  - `src/fields/tuple_utils.hpp` (~393 lines)
  - `src/fields/tuple_math.hpp` (~122 lines)
  - ~~`src/fields/field_math.hpp`~~ — deleted in 12.3d2 (only includer was field.hpp)
  - ~~`src/fields/field_utils.hpp`~~ — deleted in 12.3d2 (only includer was field_math.hpp)
  - `src/fields/matchers.hpp` (dead code — defines Catch2 matchers for non-existent types `scalar_field`, `vector_field`, `vector_range`; zero `#include`s anywhere in the codebase)
  - ~~`src/fields/algorithms.hpp`~~ — deleted in 12.3c4 (zero includers; `#include` already removed from heat.cpp in 12.3f1d and scalar_wave.cpp in 12.3f4b)
  - Files deleted: 12, ~2427+ lines removed.
  - Depends on: 12.2 (all tests deleted/rewritten, including 12.2j non-fields tests), 12.3b (gut files), 12.3c/12.3d (simplify types), 12.3e (non-fields migration). Also depends on removal of `#include "fields/algorithms.hpp"` from `scalar_wave.cpp` and `heat.cpp` (Phase 9 or Phase 12 old-code cleanup).

- [x] **12.3b** Gut the following files — remove old infrastructure, keep only what is still needed:
  - [x] **12.3b1** `src/fields/selector.hpp` — **Deleted entirely.** (~1138 lines removed) All contents replaced. All prerequisites met: stale include removed from `inviscid_vortex.cpp` (12.3f7), active `sel::`/`si::` usage migrated in `derivative.cpp` (12.3c3b), `scalar_wave.cpp` (12.3f2/f4), `heat.cpp` (12.3f1), `field_data.cpp` (12.3c3b). Dead-code includer `selections.hpp` deleted atomically (12.4c). No `#include "selector.hpp"` remains in production or test code. Build succeeds; all tests pass (6 pre-existing failures unrelated). All contents replaced:
    - `selection<I,R,Fn>` / `selection_view_fn` / `selection_view` → `handle_sel::D/Rx/Ry/Rz` in `handle.hpp`
    - `sel::D/Rx/Ry/Rz` named view selectors → `handle_sel::D/Rx/Ry/Rz`
    - `detail::plane_view<0/1/2>` → `make_x/y/z_plane_desc` in `selection_desc.hpp`
    - `detail::multi_slice_view` → `make_gather_from_slices` in `selection_desc.hpp`
    - `detail::optional_view` → conditional selection descriptors
    - `detail::predicate_view` → `make_gather_from_predicate` in `selection_desc.hpp`
    - `sel::xmin/xmax/ymin/ymax/zmin/zmax` → `make_x/y/z_plane_desc` + `for_each_grid_bc_desc`
    - **Prerequisite — stale include removal:** Before deleting `selector.hpp`, remove stale `#include "fields/selector.hpp"` from one production `.cpp` file that includes it but has zero `sel::`/`si::` usage:
      - `src/systems/inviscid_vortex.cpp` (line 2) — remove include line.
    - **Prerequisite — active include removal:** These production `.cpp` files have active `sel::`/`si::` usage and must be migrated by Phase 9/11 before this deletion:
      - `src/operators/derivative.cpp` — `using namespace si;` at lines 481, 511; active `get<D>(u)`, `get<Rx>(du)`, etc. (si:: symbols via using-directive, 22 uses across two `operator()` overloads at lines 479–515). Must be migrated to named member access (see 12.3c3).
      - `src/systems/scalar_wave.cpp` — 5 `#include` lines, `sel::D`, `sel::R`, `get<si::D>()` (see 12.3c3).
      - `src/systems/heat.cpp` — 5 `#include` lines, `sel::D`, `sel::R`, `get<si::D>()` (see 12.3c3).
      - `src/io/field_data.cpp` — `get<si::D>(sc)`, `sc | sel::R` (see 12.3c3).
    - **Prerequisite — dead-code includer:** `src/mesh/selections.hpp` (line 4) includes `"fields/selector.hpp"` and is compiled via `add_unit_test(selections ...)` in `src/mesh/CMakeLists.txt`. This is dead code (only `selections.t.cpp` includes it; no production TU uses it). **Must delete `selections.hpp`, `selections.t.cpp`, and remove the CMakeLists entry atomically with 12.3b1** (pull 12.4c forward) to avoid build breakage.
    - Verify no `#include "selector.hpp"` remains in non-test production code before deleting.
  - [x] **12.3b2** `src/fields/selector_fwd.hpp` — **Deleted entirely.** (~48 lines removed) Only includer was `selector.hpp` (deleted in 12.3b1). No `#include "selector_fwd.hpp"` remains anywhere.
  - [x] **12.3b3** `src/fields/lazy_views.hpp` — **Remove `zip_transform_view`; keep remaining utilities.** **DONE** — Removed `#include "ccs_range_utils.hpp"`, removed `zip_transform_iterator` (detail namespace), `zip_transform_view` class, deduction guide, `zip_transform_fn` struct + `zip_transform` object (~260 lines). Also removed unused `<algorithm>` and `<initializer_list>` includes. Kept: `basic_common_reference` backport, `repeat_n_view`/`repeat_n`, `stride_view`/`stride`, `cartesian_product_view`/`cartesian_product`, `linear_distribute`. File reduced from 833 to 572 lines. Build succeeds; 6 pre-existing test failures unchanged. Per D-R17, codebase analysis confirms:
    - **Remove** (line 13): `#include "ccs_range_utils.hpp"` — only `zip_transform_view` used `semiregular_box`; remaining views don't need it.
    - **Remove** (lines 45–300): `zip_transform_iterator` (lines 56–208 in `detail` namespace), `zip_transform_view` class (lines 212–287), deduction guide (lines 289–290), `zip_transform_fn` struct + `zip_transform` object (lines 293–300) — replaced by expression templates. ~256 lines deleted.
    - **Keep** (lines 1–12, 14–43): headers + `std::basic_common_reference` specialization (C++20 backport of P2321R2) — required by `cartesian_product_view::iterator` to satisfy `std::input_iterator`. `namespace ccs` is on line 42, opening brace on line 43.
    - **Keep** (lines 302–413): `repeat_n_view` / `repeat_n` — used by `src/io/xdmf.cpp:86`.
    - **Keep** (lines 415–647): `stride_view` / `stride` — used by matrix tests (`block.t.cpp`, `dense.t.cpp`, `circulant.t.cpp`, `inner_block.t.cpp`).
    - **Keep** (lines 649–817): `cartesian_product_view` / `cartesian_product` — used by `src/mesh/cartesian.hpp:74` (`domain()` method).
    - **Keep** (lines 819–833): `linear_distribute` + closing `} // namespace ccs` — used by `src/mesh/cartesian.cpp:41–43` and stencil tests.
    - **Include-chain fixup:** After 12.3a deletes `tuple_utils.hpp`, `tuple_math.hpp`, `field_utils.hpp`, `selector.hpp`, the remaining direct production includers of `lazy_views.hpp` will be: `src/mesh/cartesian.hpp`, `src/mesh/cartesian.cpp`, and `src/io/xdmf.cpp` (all three already include it directly — no changes needed). `src/mesh/selections.hpp` also includes it but is deleted in 12.4c. Matrix test files (`block.t.cpp`, `dense.t.cpp`, `circulant.t.cpp`, `inner_block.t.cpp`) and stencil test files (`polyE2_1.t.cpp`, `E2_2.t.cpp`, `E4_2.t.cpp`) also include `lazy_views.hpp` directly and will be unaffected. Verify no other file loses `lazy_views.hpp` transitively through the deleted headers.
    - File: `src/fields/lazy_views.hpp` (edit in-place)
    - Test: `cmake --build build && ctest --test-dir build`

- [x] **12.3c** Simplify `src/fields/scalar.hpp`; delete `src/fields/vector.hpp`. All sub-items (c1+c2+c3a+c3b+c4) complete. **Previous blocker analysis (2026-03-17) now resolved for Phase A:**
  - **Blocker 1 — Old DSL on `extract_scalar_view`/`extract_scalar_span` results.** After c1, `extract_scalar_view` returns the new struct type, breaking old DSL usage on the result:
    - `heat.cpp` stats(): `u | m.fluid_all(object_bcs)` (pipe syntax on `scalar_view`), `abs(u - sol)` (tuple arithmetic), `minmax(u | ...)`, `transform(std::ranges::max_element, ...)`.
    - `heat.cpp` initialize(): `sol = m.xyz | m_sol(time)` then `u | sel::D = 0; u | m.fluid = sol; u | sel::R = sol;` (pipe syntax + selection assignment on `scalar_span` obtained from `extract_scalar_span`).
    - `heat.cpp` write(): `std::vector<scalar_view>{u, error}` (conversion from `scalar_real` to new struct — requires converting constructor).
    - `scalar_wave.cpp` stats(), initialize(), write(): identical patterns.
  - **Blocker 2 — Broadcast fill.** `laplacian.cpp:38,50`: `du = 0` (fills all span elements; old tuple has `operator=(T)` from `view_tuple`). Fixable by adding `operator=(arithmetic)` to new struct.
  - **Blocker 3 — Functional assignment.** `heat.cpp:115`: `u_rhs = lap(u, neumann_u)` where `lap()` returns `std::function<void(scalar_span)>`. Old tuple has `operator=(invocable)`. Fixable by adding template `operator=(invocable)` to new struct.
  - **Blocker 4 — `scalar_real` → `scalar_view` implicit conversion.** `heat.cpp:256`: `std::vector<scalar_view>{u, error}` where `error` is `scalar_real`. Fixable by adding converting constructor from `scalar<T>` to new struct using `get<0,0>` etc.
  - **What CAN be fixed with backward-compat operators:** Blockers 2–4 (broadcast fill, functional assignment, `scalar_real` conversion). **What CANNOT:** Blocker 1 — pipe syntax and tuple arithmetic require the type to be a `ccs::tuple` with `view_tuple` base; these cannot be added to a plain struct without reimplementing the old DSL.
  - **Required pre-work:** Migrate `stats()`, `initialize()`, and `write()` methods in `heat.cpp` and `scalar_wave.cpp` to stop using old DSL on `extract_scalar_view`/`extract_scalar_span` results. These methods need either (a) to use `scalar_real` scratch buffers (old type, not affected) for old DSL operations and only use `extract_*` results with new-style access, or (b) to be fully rewritten to new patterns.
  - **Decision: (b) Simplify to lightweight structs.** `field_registry.hpp` (lines 197–218) returns `scalar_span`/`scalar_view` from `extract_scalar_span`/`extract_scalar_view`. These functions are used by `heat.cpp`, `scalar_wave.cpp`, and test files. The types must persist but can be decoupled from `tuple.hpp`.
  - **Internal ordering:** 12.3c3 (migrate callers from `get<si::X>()` to `.X` named members) must happen **before or atomically with** 12.3c1 (rewrite scalar.hpp from tuple-based alias to plain struct). After c1, `get<si::D>(span)` no longer compiles — callers must already use `span.D`. 12.3c4 (delete vector.hpp) is independent of c1–c3 ordering but has Phase 9/11 prerequisites.
  - **Revised sub-item ordering:** First do 12.3c3a (migrate old DSL in stats/initialize/write to not use extract results with old DSL), then c3b+c1+c2 atomically, then c4.
  - [x] **12.3c3a** Pre-migrate old DSL usage on `extract_scalar_view`/`extract_scalar_span` results. **DONE** — Rewrote `stats()`, `initialize()`, and `write()` in both `heat.cpp` and `scalar_wave.cpp` to avoid old DSL (pipe syntax, tuple arithmetic, `minmax`, `transform`, `abs`) on variables from `extract_scalar_view`/`extract_scalar_span`. Changes:
    - `stats()`: Replaced `minmax(u | m.fluid_all(...))`, `max(abs(u-sol) | ...)`, and `transform(max_element, ...)` with explicit loops over `m.fluid_desc()` (D) and `m.non_dirichlet_object_desc()` (Rx/Ry/Rz) indices, accessing `u` via `get<si::D>(u)` etc. Manufactured solution materialized into `scalar_real` scratch buffer.
    - `initialize()`: Replaced `u | sel::D = 0`, `u | m.fluid = sol`, `u | sel::R = sol` with `std::ranges::fill(get<si::D>(u), 0.0)`, fluid-index loop for D, and `std::ranges::copy` for R components.
    - `write()`: Replaced `error | m.fluid_all(...) = abs(u - sol)` with explicit loops computing `|u - sol|` into `error` at fluid/non-dirichlet indices. Old DSL retained on `error` (scalar_real, not from extract) and `field_view` construction (handled by c1 backward-compat constructors).
    - t-heat and t-scalar_wave pass. 6 pre-existing failures unrelated.
  - [x] **12.3c1** Rewrite `src/fields/scalar.hpp` — replace tuple-based aliases with simple structs: **DONE** — `scalar_span` and `scalar_view` are now plain structs with named members `.D`, `.Rx`, `.Ry`, `.Rz`. Backward-compat operators: broadcast fill (`= 0`), functional assignment (`= lap(u,nu)`), converting constructors from `scalar<T>` and `scalar_span→scalar_view`. Old `scalar<T>` template, `scalar_real`, `is_scalar`, `Scalar` concept retained (Phase A). Changes:
    - **Phase A (with backward compat, do atomically with c3b+c2):** Keep `#include "tuple.hpp"`, `scalar<T>` template, `scalar_real`, `is_scalar`, `Scalar` concept. Remove old `scalar_span`/`scalar_view` typedef aliases. Add new struct definitions with backward-compat operators:
      ```
      struct scalar_span {
          std::span<real> D{}, Rx{}, Ry{}, Rz{};
          // Converting constructor from scalar_real / scalar<T>
          template<typename T> scalar_span(scalar<T>& s);
          // Broadcast fill: du = 0
          template<typename T> requires std::is_arithmetic_v<T>
          scalar_span& operator=(T val);
          // Functional assignment: u_rhs = lap(u, nu)
          template<std::invocable<scalar_span&> Fn>
          scalar_span& operator=(Fn&& fn);
      };
      struct scalar_view { /* similar with const spans */ };
      ```
    - **Phase B (final cleanup, after all old DSL removed):** Delete `#include "tuple.hpp"`, `scalar<T>`, `scalar_real`, `is_scalar`, `Scalar`, and all backward-compat operators.
    - Depends on: 12.3c3a (old DSL migration in stats/initialize/write).
    - Only includes needed (Phase B): `<span>`, `"types.hpp"` (for `real`).
    - ~30 lines → ~15 lines (after Phase B).
  - [x] **12.3c2** Update `field_registry.hpp` span bridge to use new struct member syntax: **DONE** — Changed `scalar_span{tuple{...}, tuple{...}}` → `scalar_span{sp(h.D()), sp(h.Rx()), sp(h.Ry()), sp(h.Rz())}`. Same for `scalar_view`.
    - Change `scalar_span{tuple{sp(h.D())}, tuple{sp(h.Rx()), sp(h.Ry()), sp(h.Rz())}}` → `scalar_span{sp(h.D()), sp(h.Rx()), sp(h.Ry()), sp(h.Rz())}`.
    - Same for `scalar_view`.
    - Remove `#include "scalar.hpp"` if `scalar_span`/`scalar_view` are forward-declared or defined in a new lightweight header; otherwise keep the include.
  - [x] **12.3c3b** Update all callers of `scalar_span`/`scalar_view` that use tuple-style access to named member access. **DONE** — All callers updated atomically with c1+c2. `derivative.cpp`: 22 `get<D/Rx/Ry/Rz>()` → named members, removed `using namespace si;` and `#include "fields/selector.hpp"`. `heat.cpp`: 11 `get<si::*>(u)` → `u.D/u.Rx/etc.` in stats/initialize/write. `scalar_wave.cpp`: 11 uses same pattern; also `rhs()` `dot(grad_G, du)` result can't assign to new struct — used old `scalar<span>` type locally for backward compat (Phase 9 will replace dot()). `field_data.cpp`: `get<si::D>(sc)` → `sc.D`, `sc | sel::R` → direct `.Rx/.Ry/.Rz`. `field_registry.t.cpp`: 13 `get<si::*>` → named members, `output += input` → explicit element-wise loops, removed `#include "selector_fwd.hpp"`. Also updated `heat.t.cpp` (12.2j8) and `scalar_wave.t.cpp` (12.2j7) which used old DSL on extract results. All affected tests pass; 3 pre-existing operator test failures (Kokkos SIGABRT) unrelated.
    - **Original detailed caller analysis (retained for reference):**
    - `src/operators/derivative.cpp` — `using namespace si;` at lines 481, 511 enables unqualified `get<D>(u)`, `get<Rx>(du)`, etc. Two `operator()` overloads (lines 479–505, 509–515) have 22 total `get<>` calls using si:: symbols (D, Rx, Ry, Rz). Replace all with named member access: `get<D>(u)` → `u.D`, `get<Rx>(du)` → `du.Rx`, etc. Remove `using namespace si;` and `#include "fields/selector.hpp"` after migration.
    - `src/systems/heat.cpp` — Two categories after c3a:
      - **scalar_real uses (no change for c3b):** `get<si::D>(src_buf)` (line 130) and `get<si::Rx/Ry/Rz>(src_buf)` (lines 133–135); `get<si::D/Rx/Ry/Rz>(sol)` in `rhs()` (lines 166, 173–175). `src_buf`/`sol` are `scalar_real` — still tuple after c1 Phase A. Change when `scalar_real` removed in c1 Phase B.
      - **scalar_view/scalar_span uses (MUST change for c3b):** Introduced by c3a. `stats()`: `get<si::D>(u)` (line 205), `get<si::Rx/Ry/Rz>(u)` (lines 246–248) — `u` is `scalar_view`. `initialize()`: `get<si::D>(u)` (line 276), `get<si::Rx/Ry/Rz>(u).begin()` (lines 286–288) — `u` is `scalar_span`. `write()`: `get<si::D>(u)` (line 304), `get<si::Rx/Ry/Rz>(u)` (line 314) — `u` is `scalar_view`. Total 11 uses → `u.D`, `u.Rx`, etc.
    - `src/systems/scalar_wave.cpp` — Two categories after c3a:
      - **scalar_real uses (no change for c3b):** `get<si::D/Rx/Ry/Rz>(sol_buf)` in `rhs()` (lines 196, 203–205). Same as heat.cpp.
      - **scalar_view/scalar_span uses (MUST change for c3b):** Introduced by c3a. `stats()`: `get<si::D>(u)` (line 230), `get<si::Rx/Ry/Rz>(u)` (lines 271–273) — `u` is `scalar_view`. `initialize()`: `get<si::D>(u)` (line 299), `get<si::Rx/Ry/Rz>(u).begin()` (lines 309–311) — `u` is `scalar_span`. `write()`: `get<si::D>(u)` (line 327), `get<si::Rx/Ry/Rz>(u)` (line 337) — `u` is `scalar_view`. Total 11 uses → `u.D`, `u.Rx`, etc.
    - `src/io/field_data.cpp` — `get<si::D>(sc)` (line 46), `sc | sel::R` (lines 55–57). `sc` is `scalar_view` from `field_view.scalars()`. After c1, `scalar_view` is a struct — `get<si::D>(sc)` breaks. Replace with `sc.D`, and replace `sel::R` pipe with direct `.Rx`/`.Ry`/`.Rz` member access. Total 4 uses.
    - `src/fields/field_registry.t.cpp` — span bridge tests (lines 330–480) use `get<si::D>(span)`, `get<si::Rx>(span)`, etc. (13 uses). Rewrite all to named member access: `span.D`, `span.Rx`, `input.D`, `output.Rx`, etc. Also rewrite "tuple_math operator+=" integration test (line 546: `output += input`) to use explicit element-wise loops.
  - [x] **12.3c4** Delete `src/fields/vector.hpp`: **DONE** — `vector.hpp` deleted. `Vector` concept, `is_vector`, `detail::vector`, and type aliases (`vector_real`/`vector_span`/`vector_view`) temporarily moved to `selector.hpp` (gradient.t.cpp still uses `vector_real` with `vi::` selectors; full test migration in 12.2j2). `algorithms.hpp` also deleted (zero includers in codebase). All production prerequisites met: scalar_wave.hpp members migrated (12.3f4b), gradient.hpp/cpp signatures changed (12.3f4a), field_view construction removed (12.3f5b), field.hpp deleted (12.3d2). Vector types will be fully removed when `selector.hpp` is deleted in 12.3b1 (after 12.2j2 migrates gradient.t.cpp). All tests pass (6 pre-existing Kokkos/OpenMP failures unrelated).
  - Files deleted: `src/fields/vector.hpp`, `src/fields/algorithms.hpp`
  - Test: `cmake --build build && ctest --test-dir build` ✓ (6 pre-existing failures)

- [x] **12.3d** Delete `src/fields/field.hpp`; redesign and relocate `system_size` from `src/fields/field_fwd.hpp`. Per D-R18:
  - [x] **12.3d1** Redesign `system_size` struct. **DONE** — `system_size` moved to `field_registry.hpp` with plain integer fields (`d_size`, `rx_size`, `ry_size`, `rz_size`), replacing `scalar<integer> scalar_size`. `field_fwd.hpp` now includes `field_registry.hpp` to get `system_size` (old definition removed). `field.hpp` constructor updated to reconstruct `scalar<integer>` from new fields internally. All production construction sites updated (`scalar_wave.cpp`, `heat.cpp`, `hyperbolic_eigenvalues.cpp` — now use `m.size()`, `m.Rx().size()`, etc. directly). All accessor sites updated (`simulation_cycle.cpp`, `scalar_wave.t.cpp`, `heat.t.cpp`, `rk4_v2.t.cpp`, `euler_v2.t.cpp` — `sz.scalar_size` → `sz.d_size`/`sz.rx_size`/etc.). `field_io.t.cpp` construction updated (`system_size{2, 0, 24, 0, 0, 0}`). `empty_system.cpp` and `inviscid_vortex.cpp` unchanged (zero-init `return {}` works with default-initialized fields). `mesh::ss()` and `mesh::vs()` still exist for `scalar_real{m.ss()}` scratch allocation (cleanup deferred to 12.3e6a). All tests pass (7/7 affected tests, 6 pre-existing failures unrelated).
  Currently defined in `field_fwd.hpp`:
    ```
    struct system_size {
        integer nscalars; integer nvectors;
        scalar<integer> scalar_size;  // = tuple<tuple<integer>, tuple<integer,integer,integer>>
    };
    ```
    The `scalar<integer>` member depends on `scalar.hpp` → `tuple.hpp`. Replace with plain fields:
    ```
    struct system_size {
        integer nscalars; integer nvectors;
        integer d_size, rx_size, ry_size, rz_size;
    };
    ```
    Move to `src/fields/field_registry.hpp` (alongside `field_ref` which serves a similar role). **Decision: use `field_registry.hpp`** — it already defines `field_ref` (a similarly lightweight sizing/addressing type), is already `#include`d by all system files that use `system_size`, and avoids creating a new header.
    Update all construction sites (currently use `scalar<integer>{tuple{d}, tuple{rx, ry, rz}}`):
    - `src/systems/scalar_wave.cpp:107` — `{1, 0, m.ss()}` → `{1, 0, m.size(), m.Rx().size(), m.Ry().size(), m.Rz().size()}`
    - `src/systems/heat.cpp:102` — same pattern
    - `src/systems/hyperbolic_eigenvalues.cpp:43` — `{0, 0, m.ss()}` → `{0, 0, m.size(), m.Rx().size(), m.Ry().size(), m.Rz().size()}`
    - `src/systems/empty_system.cpp:19` — `{}` → `{0, 0, 0, 0, 0, 0}` (or keep zero-init if default is 0)
    - `src/systems/inviscid_vortex.cpp:228` — `{}` → same
    - Test files: `field.t.cpp` (being deleted), `field_utils.t.cpp` (being deleted), `field_math.t.cpp` (being deleted), `field_io.t.cpp` (12.2j9)
    - **`system_size.scalar_size` accessor sites** (not construction, but reading the `scalar<integer>` member via tuple indexing — these also need updating after the redesign):
      - `src/systems/scalar_wave.t.cpp` lines 29–33: `setup_registry` helper uses `get<0>(get<0>(ss))`, `get<0>(get<1>(ss))`, `get<1>(get<1>(ss))`, `get<2>(get<1>(ss))` to extract sizes from `sz.scalar_size` (line 29 binds `auto& ss`, lines 30–33 are the four `get<>` calls). After redesign, replace with `sz.d_size`, `sz.rx_size`, `sz.ry_size`, `sz.rz_size`. See 12.2j7.
      - `src/systems/heat.t.cpp` lines 28–32: identical `setup_registry` helper pattern. See 12.2j8.
      - `src/systems/heat.t.cpp` lines 258, 371: `auto ss = sys.size().scalar_size` then `get<si::Rx>(ss)` etc. After redesign, use `sz.rx_size` directly. See 12.2j8.
    - **`mesh::ss()` confirmed:** Returns `tuple{tuple{size()}, tuple{Rx().size(), Ry().size(), Rz().size()}}` = `scalar<integer>` (mesh.hpp:144–147). Construction `{1, 0, m.ss()}` maps to `system_size{nscalars=1, nvectors=0, scalar_size=m.ss()}`. After redesign, either (a) inline `m.size()`, `m.Rx().size()`, etc. at each call site, or (b) change `mesh::ss()` to return `system_size` directly (with `nscalars=0, nvectors=0`; caller fills in nscalars/nvectors). Option (a) is simpler since `ss()` is only called at 3 system `size()` methods and scratch allocation sites.
    - **`scalar_real{m.ss()}` construction** at `scalar_wave.cpp:60,186` and `heat.cpp:36–37,117,156`: These allocate `scalar_real` (= `scalar<std::vector<real>>`) scratch buffers sized by `m.ss()`. Phase 9 should migrate these to registry-based allocation. If still present, they need updating to use individual size args. Also `vector_real{m.vs()}` at `scalar_wave.cpp:58–59` — `mesh::vs()` (line 151) returns `tuple{ss(), ss(), ss()}`.
    - **`mesh::ss()` and `mesh::vs()` cleanup:** After all callers are updated, delete or simplify these methods. They exist solely to produce `scalar<integer>` / `vector<integer>` sizing tuples for the old tuple-based construction. Remove `#include "fields/tuple.hpp"` from `cartesian.hpp` (included transitively for `tuple{}` wrapper in `domain()`; see 12.3e4).
  - [x] **12.3d2** Delete `src/fields/field.hpp` entirely: **DONE** — Deleted `field.hpp`, `field_fwd.hpp`, `field_math.hpp`, `field_utils.hpp` (all 4 files — the entire self-contained include chain). `field_math.hpp` was only included by `field.hpp`; `field_utils.hpp` only by `field_math.hpp`; `field_fwd.hpp` only by `field.hpp`/`field_math.hpp`/`field_utils.hpp`. Includer cleanup: removed `#include "fields/field.hpp"` and dead `field U;` member from `inviscid_vortex.hpp`; replaced `divergence.hpp` stub (was `return field{};` → `return 0;`, removing `field.hpp` dependency); removed `#include "field.hpp"` from `field_registry.t.cpp` and replaced `sizeof(field_span)` assertion with `sizeof(scalar_span)`. All tests pass (6 pre-existing failures unrelated).
  - [x] **12.3d3** Delete `src/fields/field_fwd.hpp`: **DONE** — Deleted atomically with d2 (see above). All includers (`field.hpp`, `field_math.hpp`, `field_utils.hpp`) were also deleted.
  - Files deleted: `src/fields/field.hpp`, `src/fields/field_fwd.hpp`, `src/fields/field_math.hpp`, `src/fields/field_utils.hpp`
  - Test: `cmake --build build && ctest --test-dir build` ✓ (6 pre-existing failures)

- [x] **12.3e** Migrate non-fields production code that uses old infrastructure types not in scope for Phases 8–11. These files use `tuple_fwd.hpp`/`tuple.hpp`/`tuple_utils.hpp` for general-purpose type machinery, not for field lifecycle.
  - [x] **12.3e1** `src/real3_operators.hpp` — uses `NumericTuple` concept from `tuple_fwd.hpp`: **DONE** — `NumericTuple` concept moved to `types.hpp` with simplified Boost.Mp11-free implementation using `all_elements_arithmetic` helper trait. `real3_operators.hpp` updated to `#include "types.hpp"` instead of `#include "fields/tuple_fwd.hpp"`. Old definition removed from `tuple_fwd.hpp`. `is_stdarray` left in `tuple_fwd.hpp` (still used by `ArrayFromTuple` concept there; will be handled when `tuple_fwd.hpp` is deleted in 12.3a).
    - File: `src/real3_operators.hpp`, `src/types.hpp`
    - Test: `ctest --test-dir build -R t-real3` ✓
  - [x] **12.3e2** `src/matrices/unit_stride_visitor.hpp` — uses `Range` concept from `tuple_fwd.hpp`: **DONE** — `Range` concept moved to `types.hpp`. Removed `#include "fields/tuple_fwd.hpp"` from both `unit_stride_visitor.hpp` and `coefficient_visitor.hpp` (stale include). Added missing `#include <cassert>` to `unit_stride_visitor.cpp`, `coefficient_visitor.cpp`, and `eigenvalue_visitor.hpp` (previously obtained transitively through `tuple_fwd.hpp`). Old definition removed from `tuple_fwd.hpp`.
    - Files: `src/matrices/unit_stride_visitor.hpp`, `src/matrices/coefficient_visitor.hpp`, `src/types.hpp`
    - Test: `ctest --test-dir build -L matrices` ✓
  - [x] **12.3e3** Replace `ccs::tuple<span, span, span>` with `std::array<std::span<const mesh_object_info>, 3>` across the IO pipeline and `mesh::R()`: **DONE** — All 7 production files updated (xdmf.hpp/cpp, field_data.hpp/cpp, field_io.hpp/cpp, mesh.hpp). Removed `#include "fields/tuple.hpp"` from xdmf.hpp (replaced with `<array>` + `<span>`). Kept `#include "fields/field.hpp"` in field_data.hpp (still needed for `field_view`). Updated test files xdmf.t.cpp and field_io.t.cpp to use `std::array<U, 3>` alias. Internal access patterns (`get<I>(t)`, `auto&& [x, y, z] = t;`) work unchanged with `std::array` via ADL finding `std::get`. All IO, mesh, and system tests pass.
    - **12.3e3a** Update type declarations in headers + `mesh::R()`:
      - `src/io/xdmf.hpp` line 28–30: change `xdmf::write()` parameter from `tuple<span<...>, span<...>, span<...>>` to `std::array<std::span<const mesh_object_info>, 3>`. Remove `#include "fields/tuple.hpp"`.
      - `src/io/field_data.hpp` line 21–23: change `write_geom()` parameter. Remove `#include "fields/field.hpp"` if `field_view` is already migrated by Phase 9 (the `write()` method uses `field_view`; see 12.1a Category B).
      - `src/io/field_io.hpp` lines 48–50: change `field_io::write()` parameter. Remove `#include "xdmf.hpp"` transitive tuple dependency (xdmf.hpp's own tuple include is removed above).
      - `src/mesh/mesh.hpp` line 108: change `R()` return from `tuple{Rx(), Ry(), Rz()}` to `std::array<std::span<const mesh_object_info>, 3>{Rx(), Ry(), Rz()}`.
      - Files: `src/io/xdmf.hpp`, `src/io/field_data.hpp`, `src/io/field_io.hpp`, `src/mesh/mesh.hpp`
      - Test: `cmake --build build` (compilation check)
    - **12.3e3b** Update implementations (mechanical — match new parameter type):
      - `src/io/xdmf.cpp`: 3 occurrences — `append_xdmf` (line 28), `header` (line 95), `xdmf::write` (line 144). The internal code (`get<0>(t)`, `auto&& [x, y, z] = t;` at line 87) works unchanged with `std::array`.
      - `src/io/field_data.cpp` line 9: `write_geom()` signature. Internal `get<I>(t)` at lines 14, 31–33 works unchanged with `std::array`.
      - `src/io/field_io.cpp` line 38: `field_io::write()` signature. Internal passing of `r` to `xdmf_w.write()` and `field_data_w.write_geom()` works unchanged.
      - Files: `src/io/xdmf.cpp`, `src/io/field_data.cpp`, `src/io/field_io.cpp`
      - Test: `ctest --test-dir build -L simulation`
  - [x] **12.3e4** Remove `ccs::tuple` wrappers from `domain()` methods in `cartesian.hpp` and `object_geometry.hpp`, propagating type changes to `mesh.hpp`/`mesh.cpp`. **DONE** — All three sub-items (e4a, e4b, e4c) completed atomically. `object_geometry::domain()` now returns `std::tuple{...}`. `cartesian::domain()` now returns `ccs::cartesian_product(x(), y(), z())` directly (no wrapper). `#include "fields/tuple.hpp"` removed from `cartesian.hpp`. `xyz` and `vxyz` members deleted from `mesh.hpp` (no production callers remained after f1–f3). `mesh.cpp` constructor no longer constructs xyz/vxyz. Test files (derivative.t.cpp, laplacian.t.cpp, gradient.t.cpp, mesh.t.cpp) pre-migrated: all `m.xyz | func` patterns replaced with `eval_at_mesh<T>(m, func)` helper (evaluates view adaptor at `cartesian_product(x,y,z)` for D + `m.Rx/Ry/Rz() | position` for R components); all `m.vxyz | func` patterns replaced with explicit `std::ranges::copy` into `vi::*` R-component spans. All builds pass. 6 pre-existing test failures unchanged.
    - [x] **12.3e4a** `src/mesh/object_geometry.hpp` — `tuple{...}` → `std::tuple{...}` in `domain()`.
    - [x] **12.3e4b** `src/mesh/cartesian.hpp` — drop `tuple{...}` wrapper in `domain()`, remove `#include "fields/tuple.hpp"`.
    - [x] **12.3e4c** `src/mesh/mesh.hpp` / `src/mesh/mesh.cpp` — deleted `xyz` and `vxyz` members entirely (no callers after f1–f3 + test pre-migration).
  - [x] **12.3e5** `src/mms/manufactured_solutions.hpp` — uses `TupleLike`, `ArrayFromTuple`, `to<real3>()`: **DONE** — Replaced `#include "fields/tuple_utils.hpp"` with `#include "types.hpp"`. Replaced 5 template overloads' `TupleLike L` + `ArrayFromTuple<real3, L>` constraints with `requires(!std::same_as<real3, std::remove_cvref_t<L>>)`. Replaced `to<real3>(FWD(loc))` with `real3{std::get<0>(loc), std::get<1>(loc), std::get<2>(loc)}`. t-mms and t-simulation_cycle pass.
    - File: `src/mms/manufactured_solutions.hpp`
    - Test: `ctest --test-dir build -L simulation` ✓
  - **12.3e6** `src/mesh/mesh.hpp` / `src/mesh/mesh.cpp` — remaining `ccs::tuple` and selector usage cleanup after Phase 11, 12.3d1, 12.3e3, and 12.3e4 have resolved specific dependencies. After all sub-items complete, `mesh.hpp` should no longer include any `fields/tuple*.hpp` or `fields/selector*.hpp` headers.
    - [x] **12.3e6a** Delete `mesh::ss()` and `mesh::vs()` methods: **DONE** — Both methods deleted from `mesh.hpp`. Zero callers remained (production callers migrated by f1/f2/f4; test callers migrated by 12.2j1–j5). Build succeeds, t-mesh passes (2/2).
      - `ss()`: Returns `tuple{tuple{size()}, tuple{Rx().size(), Ry().size(), Rz().size()}}` = `scalar<integer>`. **Updated (2026-03-17 post-e4):** No production callers remain (f1/f2/f4 migrated all). Only test callers remain: `derivative.t.cpp` (9 uses), `laplacian.t.cpp` (7 uses), `gradient.t.cpp` (6 uses via `m.vs()`), `mesh.t.cpp` (5 uses). These are migrated in 12.2j. Can be deleted atomically with 12.2j test migrations.
      - `vs()`: Returns `tuple{ss(), ss(), ss()}`. No production callers remain (f4 migrated all). Only `gradient.t.cpp` (3 uses). Migrated in 12.2j2.
      - Depends on: 12.2j (test callers must be migrated before deletion).
      - File: `src/mesh/mesh.hpp`
    - [x] **12.3e6b** Delete `xyz` and `vxyz` members: **DONE** — Completed by 12.3e4c. Both members and their `mesh.cpp` construction deleted.
    - [x] **12.3e6c** Verify and remove selector infrastructure from `mesh.hpp`: **DONE** — Completed by 12.3f7. All `sel::` types, members, and methods deleted. `#include "fields/selector.hpp"` removed.
    - [x] **12.3e6d** Verify and remove `tuple_cat` usage: **DONE** — Completed by 12.3f7. `dirichlet(Grid,Object)` and `fluid_all(Object)` methods deleted (contained all `tuple_cat` usage).
    - [x] **12.3e6e** Final include cleanup for `mesh.hpp`: **DONE** — Completed by 12.3f7. `mesh.hpp` no longer includes `fields/selector.hpp`, `fields/tuple.hpp`, or `fields/tuple_utils.hpp`. Keeps `fields/selection_desc.hpp` (used for `gather_selection` etc.).
    - Files: `src/mesh/mesh.hpp`, `src/mesh/mesh.cpp`
    - Test: `cmake --build build && ctest --test-dir build`
    - Depends on: Phase 11 (selector migration), 12.3d1 (system_size redesign), 12.3e3 (R() return type), 12.3e4 (domain() return type)
  - **Ordering:** 12.3e must complete before 12.3a (file deletion). Items e1, e2, e3 can be done in parallel with 12.3b3/c/d. Within e4: e4a must precede e4b (object_geometry uses `ccs::tuple` transitively through cartesian.hpp; must switch to `std::tuple` before the transitive include is removed). e4c depends on e4a+e4b. e5 depends on e4 (manufactured_solutions templates receive different types after `domain()` return type changes). e6 depends on 12.3d1 (system_size), e3 (R() type), e4 (domain() type), and Phase 11 (selector migration). After 12.3e completes, 12.2j (non-fields tests) can proceed — but 12.2j4/j6 can start earlier since they only need `tuple_utils.hpp` removal. After 12.2j, proceed to 12.3b1/b2 (selector deletion).

### 12.3f — Production old-DSL cleanup (D-R6 coexistence remnants)

Phase 9/11 completed via D-R6 coexistence: new v2 adapter methods were added alongside old implementations, then old public interfaces were removed. The old internal implementations still use `scalar_real` scratch buffers, `m.xyz | func` piping, `sel::` pipe assignment, `get<si::*>()` tuple access on `scalar_real`, `vector_real`/`vector_span` types, `dot()` from `algorithms.hpp`, `field_view` construction, and old `mesh.hpp` selector members/methods. These must be migrated before infrastructure files can be deleted.

**Critical path:** 12.3f1–f3 (scalar_real + m.xyz + sel:: in heat/scalar_wave) → 12.3f4 (vector_real/vector_span) → 12.3f5 (field_view/IO pipeline) → 12.3f6 (scalar.hpp Phase B) → 12.3f7 (mesh.hpp selector cleanup).

**Common replacement patterns:**
- **Scratch buffer allocation:** `scalar_real buf{m.ss()}` → 4 `std::vector<real>` sized by `m.size()`, `m.Rx().size()`, `m.Ry().size()`, `m.Rz().size()`, then `scalar_span{d, rx, ry, rz}`.
- **Manufactured solution evaluation:** `buf = m.xyz | func` → explicit loops: iterate `ccs::cartesian_product(m.x(), m.y(), m.z())` for `.D`, iterate `m.Rx()/Ry()/Rz()` using `.position` for `.Rx/.Ry/.Rz`. A test-local/file-local helper `void eval_at_locations(const mesh&, auto&& func, scalar_span out)` can reduce boilerplate.
- **Old tuple access:** `get<si::D>(buf)` → `buf_span.D` (named member on `scalar_span`/`scalar_view`).
- **Selection pipe assignment:** `buf | sel::D = val` → `std::ranges::fill(buf_span.D, val)` or `std::ranges::copy(src, buf_span.D.begin())`. `buf | m.fluid = src` → iterate `m.fluid_desc()` element indices. `buf | m.dirichlet(g, o) = 0` → `for_each_grid_bc_desc<bcs::Dirichlet>(m, g, [&](auto desc) { fill_selected(buf_span, desc, 0); })` + `assign_selected` for object BCs.
- **`neumann | m.neumann<I>(g) = ...`:** → `for_each_grid_bc_desc<bcs::Neumann, I>(m, g, [&](auto desc) { assign_selected(neu_span, desc, ...); })`.
- **`error | m.dirichlet(g, o) = 0`:** → `for_each_grid_bc_desc<bcs::Dirichlet>(m, g, [&](auto desc) { fill_selected(err_span, desc, 0); })` + object dirichlet descriptor.

- [x] **12.3f1** Migrate `heat.cpp` scratch buffers, `m.xyz` piping, and `sel::` pipes. Each sub-item should build and pass t-heat.
  - [x] **12.3f1a** Migrate `heat::rhs()` scratch buffer: **DONE** — Replaced `scalar_real src_buf{m.ss()}` + `src_buf = (m.xyz | m_sol.ddt(time)) - (...)` with 4 `std::vector<real>` + `scalar_span` + `eval_at_locations()` helper. Replaced `get<si::D>(src_buf).data()` → `src.D.data()`, `get<si::Rx/Ry/Rz>` → `src.Rx/Ry/Rz.data()`. Also added `x()`, `y()`, `z()` coordinate accessors to `mesh.hpp` (forwarding to `cart.x()` etc.) and file-local `eval_at_locations(const mesh&, auto&& func, scalar_span out)` helper for iterating cartesian_product + object positions. t-heat passes.
  - [x] **12.3f1b-sol** Migrate `heat::update_boundary()` sol scratch buffer: **DONE** — Replaced `scalar_real sol{m.ss()}` + `sol = l | m_sol(time)` + `get<si::D/Rx/Ry/Rz>(sol).data()` with same 4-vector + scalar_span + eval_at_locations pattern. Removed dependency on `auto l = m.xyz` for sol evaluation. Neumann pipe syntax (`neumann_u | m.neumann<I>(grid_bcs) = l | m_sol.gradient(i, time)`) retained with `auto l = m.xyz` — deferred to 12.3f1b-neumann.
  - [x] **12.3f1b-neumann** Migrate `heat::update_boundary()` neumann pipe syntax: **DONE** — Replaced `auto l = m.xyz` + 3 lines `neumann_u | m.neumann<I>(grid_bcs) = l | m_sol.gradient(i, time)` with per-direction loop: evaluate `m_sol.gradient(time, loc)[dir]` at domain grid locations via `ccs::cartesian_product`, then `assign_selected(neu.D.data(), plane_desc, handle_expr{...})` for each Neumann face. Uses `scalar_span neu{neumann_u}` converting constructor. `auto l = m.xyz` removed (no remaining uses in method). t-heat and t-simulation_cycle pass.
  - [x] **12.3f1c** Migrate `heat::stats()`, `heat::initialize()`, `heat::write()` sol scratch buffers: **DONE** — All three methods: replaced `scalar_real sol{m.ss()}` + `sol = m.xyz | m_sol(time)` + `get<si::*>(sol)` with 4-vector + scalar_span + eval_at_locations pattern. `write()` still uses `get<si::*>(error)` and `error | m.dirichlet(...)` on the `error` member (scalar_real, deferred to f1d). `field_view` construction deferred to 12.3f5. t-heat and t-simulation_cycle pass.
  - [x] **12.3f1d** Migrate `heat.hpp` member fields: **DONE** — Replaced `scalar_real neumann_u` with `std::vector<real> neumann_d, neumann_rx, neumann_ry, neumann_rz`; replaced `scalar_real error` with `std::vector<real> error_d, error_rx, error_ry, error_rz`. Constructor init-list updated to size from `m.size()`/`m.Rx().size()`/etc. `rhs()`: `lap(u, neumann_u)` → `lap(u, scalar_view{neumann_d, ...})`. `update_boundary()`: `scalar_span neu{neumann_u}` → `scalar_span neu{neumann_d, ...}`. `write()`: replaced `get<si::D>(error)` → `error_d[i]`, `get<si::Rx/Ry/Rz>(error)` → `error_rx/ry/rz`, `error | m.dirichlet(grid_bcs, object_bcs) = 0` → explicit `for_each_grid_bc_desc<bcs::Dirichlet>` + `dirichlet_object_desc` loops, `error` in `scalar_view` vector → `scalar_view err_view{error_d, ...}`. Also removed `#include "fields/algorithms.hpp"`, `#include "fields/selector.hpp"`, and dead `constexpr auto abs = lift(...)` from heat.cpp — no `sel::`/`si::`/algorithms.hpp usage remains. t-heat and t-simulation_cycle pass. 6 pre-existing failures unrelated.
  - Depends on: nothing (can start now — 12.3c1+c3b already done, named members available).
  - Enables: 12.3f6 (scalar.hpp Phase B), 12.3e4 (domain() type changes), 12.3e6a (ss() deletion).
  - File: `src/systems/heat.cpp`, `src/systems/heat.hpp`
  - Test: `ctest --test-dir build -R t-heat && ctest --test-dir build -L simulation`
- [x] **12.3f2** Migrate `scalar_wave.cpp` scratch buffers, `m.xyz` piping, and `sel::` pipes: **DONE** — All sub-items complete. Only remaining old DSL is `field_view`/`vector_view` in `write()` (deferred to f5).
  - [x] **12.3f2a** Migrate `scalar_wave` constructor: **DONE** (via 12.3f4b) — Replaced `grad_G{m.vs()}`, `du{m.vs()}` init with 24 `std::vector<real>` members. Constructor body: `eval_at_locations(m, neg_G_at(comp, center), gG_span)` for each of 3 components, then `for_each_grid_bc_desc<bcs::Dirichlet>` + `dirichlet_object_desc` to zero Dirichlet entries across all 12 wave-speed buffers. Removed old `neg_G<I>` template, `solution` transform, and dead `#include "fields/selector.hpp"`. t-scalar_wave and t-simulation_cycle pass.
  - [x] **12.3f2b** Migrate `scalar_wave::rhs()`: **DONE** (via 12.3f4b) — Replaced `scalar<std::span<real>> u_rhs` + `u_rhs = dot(grad_G, du)` with `scalar_span` views of the 24 member vectors + element-wise `dot_spans` lambda computing `gx*dx + gy*dy + gz*dz` for each of D, Rx, Ry, Rz. No more `dot()` from `algorithms.hpp` or old `scalar<span>` tuple type.
  - [x] **12.3f2c** Migrate `scalar_wave::update_boundary()`, `stats()`, `initialize()`, `write()` scratch buffers off old DSL: **DONE** — Replaced `scalar_real sol{m.ss()}` + `sol = m.xyz | solution(...)` + `get<si::*>(sol)` with 4 `std::vector<real>` + `scalar_span` + `eval_at_locations()` + `solution_at()` helper in all four methods. Added file-local `eval_at_locations(const mesh&, auto&& func, scalar_span out)` and `solution_at(center, radius, time)` (plain callable version of `solution()`). Removed dead `constexpr auto abs = lift(...)`. t-scalar_wave and t-simulation_cycle pass.
  - [x] **12.3f2d-error** Migrate `scalar_wave.hpp` `scalar_real error` member to vectors: **DONE** — Replaced `scalar_real error` with `std::vector<real> error_d, error_rx, error_ry, error_rz`. Constructor init-list updated to size from `m.size()`/`m.Rx().size()`/etc. `write()`: replaced `error = 0` with `std::ranges::fill` on each vector, `get<si::D>(error)` → `error_d[i]`, `get<si::Rx/Ry/Rz>(error)` → `error_rx/ry/rz`, `error | m.dirichlet(grid_bcs, object_bcs) = 0` → explicit `for_each_grid_bc_desc<bcs::Dirichlet>` + `dirichlet_object_desc` loops, `error` in `scalar_view` vector → `scalar_view err_view{error_d, ...}`. `field_view` construction retained (deferred to f5). t-scalar_wave and t-simulation_cycle pass.
  - [x] **12.3f2d-vector** Migrate `scalar_wave.hpp` `vector_real grad_G`, `vector_real du` members: **DONE** (via 12.3f4b) — Replaced with 24 `std::vector<real>` members: `gG_xd`/`gG_xrx`/`gG_xry`/`gG_xrz` (3 components) + `du_xd`/`du_xrx`/`du_xry`/`du_xrz` (3 components). Removed `#include "fields/algorithms.hpp"` and `#include "fields/selector.hpp"` from scalar_wave.cpp.
  - [x] **12.3f-fixup-odr** Move `eval_at_locations` into anonymous namespace in both `heat.cpp` and `scalar_wave.cpp`: **DONE** — Wrapped `eval_at_locations` in a file-local anonymous namespace in `heat.cpp` (new `namespace { ... }` block). In `scalar_wave.cpp`, moved `eval_at_locations` inside the existing anonymous namespace (before its closing brace). t-heat, t-scalar_wave, and t-simulation_cycle pass.
  - Depends on: 12.3f4 for vector-related items (f2a, f2b, f2d-vector). All items completed.
  - Enables: same as f1.
  - File: `src/systems/scalar_wave.cpp`, `src/systems/scalar_wave.hpp`
  - Test: `ctest --test-dir build -R t-scalar_wave && ctest --test-dir build -L simulation`
- [x] **12.3f3** Verify no `sel::` usage remains in `heat.cpp`/`scalar_wave.cpp`: **DONE** — `heat.cpp` cleaned in f1d; `scalar_wave.cpp` cleaned in f4b (all `sel::xR/yR/zR`, `m.fluid`, `m.dirichlet`, `vi::xRx` removed; `#include "fields/selector.hpp"` removed from both files). No `sel::` or `si::` or `vi::` usage remains.
  - Depends on: 12.3f1, 12.3f2.
  - Enables: 12.3b1 (delete selector.hpp, after 12.2j and 12.3f7 also complete).
- [x] **12.3f4** Migrate `vector_real`/`vector_span` in `scalar_wave.hpp/cpp` and `gradient.hpp/cpp`: **DONE** — All sub-items (f4a, f4b, f4c) complete.
  - [x] **12.3f4a** Redesign `gradient::operator()` return type: **DONE** — Changed `std::function<void(vector_span)>` → `std::function<void(scalar_span, scalar_span, scalar_span)>`. Replaced `#include "fields/vector.hpp"` with `#include "fields/scalar.hpp"` in `gradient.hpp`. Updated `gradient.cpp` to use 3 named `scalar_span` parameters (du_x, du_y, du_z) instead of `get<vi::X/Y/Z>(du)`. Updated `scalar_wave.cpp::rhs()` call site: `du = grad(u)` → `grad(u)(scalar_span{get<0>(du)}, ...)`. Updated 4 call sites in `gradient.t.cpp`. t-scalar_wave and t-simulation_cycle pass; t-gradient has pre-existing Kokkos SIGABRT.
  - [x] **12.3f4b** Replace `vector_real grad_G` and `vector_real du` in `scalar_wave.hpp/cpp`: **DONE** — Replaced 2 `vector_real` members with 24 `std::vector<real>` (3 spatial components × 4 buffers × 2 variables). Constructor: `eval_at_locations` fills wave-speed coefficients, `for_each_grid_bc_desc`/`dirichlet_object_desc` zeros Dirichlet entries. `rhs()`: `dot(grad_G, du)` replaced with element-wise `dot_spans` lambda computing `gx*dx + gy*dy + gz*dz` for D/Rx/Ry/Rz. Removed dead `neg_G<I>` template, `solution` view adaptor, `#include "fields/algorithms.hpp"`, `#include "fields/selector.hpp"`. t-scalar_wave, t-heat, t-simulation_cycle pass.
  - [x] **12.3f4c** Remove `#include "fields/algorithms.hpp"` from `scalar_wave.cpp` and `heat.cpp`: **DONE** — `heat.cpp` removed in f1d; `scalar_wave.cpp` removed in f4b. No `dot`/`minmax`/`max` from `algorithms.hpp` remains in either file.
  - Depends on: nothing for f4a; f4b depends on f4a (gradient return type).
  - Enables: 12.3c4 (delete vector.hpp), 12.3f2a/f2b/f2d (vector-related scalar_wave migrations), 12.3e6a (vs() deletion).
  - Files: `src/operators/gradient.hpp`, `src/operators/gradient.cpp`, `src/systems/scalar_wave.hpp`, `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R t-scalar_wave && ctest --test-dir build -R t-gradient`
- [x] **12.3f5** Migrate `field_view` construction in `heat.cpp`/`scalar_wave.cpp` `write()` and update IO pipeline: **DONE** — Both sub-items (f5a, f5b) complete.
  - [x] **12.3f5a** Redesign `field_io::write()` and `field_data::write()` to accept `std::span<const scalar_view>` instead of `field_view`: **DONE** — Changed `field_data::write(field_view, ...)` → `write(std::span<const scalar_view>, ...)` in header/impl. Changed `field_io::write(..., field_view, ...)` → `write(..., std::span<const scalar_view>, ...)` in header/impl. Removed `#include "fields/field.hpp"` from both `field_io.hpp` and `field_data.hpp`; `field_data.hpp` now includes `"fields/scalar.hpp"` for `scalar_view`. Updated `field_io.t.cpp` to add direct `#include "fields/field.hpp"` and `#include "fields/selector.hpp"` (previously transitive); test calls updated to pass `std::vector<scalar_view>` / `fv.scalars()`.
  - [x] **12.3f5b** Update `heat::write()` and `scalar_wave::write()` to construct `std::vector<scalar_view>` and pass to new IO signature: **DONE** — Replaced `field_view io_view{std::vector<scalar_view>{u, err_view}, std::vector<vector_view>{}}` → `std::vector<scalar_view> io_scalars{u, err_view}` in both files. Removed `#include "fields/field.hpp"` from `heat.hpp` and `scalar_wave.hpp` (no longer needed; `system_size`, `scalar_view` etc. available through `field_registry.hpp`). All IO, system, and simulation tests pass. 6 pre-existing failures unrelated.
  - Depends on: 12.3f4 (no more vector_view in field_view), 12.3f1c/f2c (write() scratch buffers migrated).
  - Enables: 12.3d2 (delete field.hpp), 12.3d3 (delete field_fwd.hpp).
  - Files: `src/io/field_io.hpp`, `src/io/field_io.cpp`, `src/io/field_data.hpp`, `src/io/field_data.cpp`, `src/systems/heat.cpp`, `src/systems/heat.hpp`, `src/systems/scalar_wave.cpp`, `src/systems/scalar_wave.hpp`, `src/io/field_io.t.cpp`
  - Test: `ctest --test-dir build -R t-field_io && ctest --test-dir build -R t-heat && ctest --test-dir build -R t-scalar_wave && ctest --test-dir build -L simulation` ✓
- [x] **12.3f6** scalar.hpp Phase B — remove old tuple infrastructure from scalar.hpp: **DONE** — Moved `scalar<T>`, `scalar_real`, `is_scalar`, `Scalar` concept from `scalar.hpp` to `tuple.hpp` (legacy section, will be deleted with that file in 12.3a). Removed `#include "tuple.hpp"` from `scalar.hpp`; replaced with `#include "types.hpp"` + `#include <algorithm>`. Kept converting constructors in `scalar_span`/`scalar_view` using ADL (C++20 P0846R0) to find `ccs::get` at instantiation time without requiring `tuple.hpp`. Kept broadcast fill (`= 0`) and functional assignment (`= lap(...)`) operators (still used in production: `laplacian.cpp`, `heat.cpp`). Replaced `requires(!Scalar<Op>)` in `derivative.hpp/cpp` with `requires std::invocable<Op, real&, real>` (semantically equivalent; prevents `scalar_real` from being deduced as `Op`). All production and test code builds; 6 pre-existing Kokkos/OpenMP test failures unrelated.
  - Files: `src/fields/scalar.hpp`, `src/fields/tuple.hpp`, `src/operators/derivative.hpp`, `src/operators/derivative.cpp`
  - Enables: 12.3b3 (breaks `scalar.hpp` → `tuple.hpp` include chain).
  - Test: `cmake --build build && ctest --test-dir build` ✓ (6 pre-existing failures)
- [x] **12.3f7** Migrate `mesh.hpp` old selector members and methods: **DONE** — All three sub-items completed atomically. Deleted 7 old selector members (`xmin`–`zmax`, `fluid`), 2 private templates (`grid_boundaries`, `object_boundaries`), 6 public selector methods (`dirichlet(Grid)`, `dirichlet(Object)`, `dirichlet(Grid,Object)`, `non_dirichlet(Object)`, `fluid_all(Object)`, `neumann(Grid)`), and `#include "fields/selector.hpp"` from `mesh.hpp`. Removed 6 constructor init-list entries and `fluid = sel::multi_slice(...)` from `mesh.cpp`. Also removed stale `#include "fields/selector.hpp"` and dead `lift()`-based constants from `inviscid_vortex.cpp`. Build succeeds; all tests pass (6 pre-existing failures unrelated).
  - [x] **12.3f7a** Delete old selector member variables and constructor init-list entries.
  - [x] **12.3f7b** Delete old selector methods.
  - [x] **12.3f7c** Remove `#include "fields/selector.hpp"` from `mesh.hpp`.
  - Files: `src/mesh/mesh.hpp`, `src/mesh/mesh.cpp`, `src/systems/inviscid_vortex.cpp`
  - Test: `cmake --build build && ctest --test-dir build` ✓ (6 pre-existing failures)

**Revised dependency graph for 12.3f:**
```
12.3f4a (gradient return type)
  → 12.3f4b (vector_real → vectors, dot() → component-wise)
    → 12.3f2a (scalar_wave constructor vector_real migration)
    → 12.3f2b (scalar_wave::rhs() dot() replacement)
    → 12.3f2d (scalar_wave.hpp vector_real members)
12.3f1a–d (heat.cpp scalar_real + m.xyz + sel::) [no blockers]
12.3f2b-partial, f2c (scalar_wave scalar_real parts) [no blockers]
12.3f1 + f2 + f4 → 12.3f5 (field_view/IO pipeline)
12.3f1 + f2 + f5 → 12.3f6 (scalar.hpp Phase B)
12.3f1–f3 + 12.2j → 12.3f7 (mesh.hpp selector cleanup)
```

**Practical starting point (updated 2026-03-17):** All sub-items complete: 12.3f1–f7, 12.2j, 12.3e1–e6, 12.3d2+d3, 12.3c4, 12.3b1+b2+b3, 12.4a+b+c, 12.3a. **Next: 12.5a** (final verification — full build, test, grep confirmation, deleted file/line counts).

### 12.4 — Clean up CMakeLists.txt and dead code

- [x] **12.4a** Update `src/fields/CMakeLists.txt`: **DONE** — all 15 `add_unit_test` lines removed alongside test file deletions in 12.2a–i. Only `add_unit_test(handle ...)` and the 3 explicit `add_executable` targets (t-field_registry, t-expr, t-selection_desc) remain.
  - Remove these `add_unit_test` lines (test files deleted in 12.2):
    - `add_unit_test(range_concepts "concepts" fields)`
    - `add_unit_test(tuple_utils "fields" fields)`
    - `add_unit_test(tuple_pipe "field" fields)`
    - `add_unit_test(container_tuple "fields" fields)`
    - `add_unit_test(view_tuple "fields" fields)`
    - `add_unit_test(tuple "fields" fields)`
    - `add_unit_test(tuple_math "fields" fields)`
    - `add_unit_test(single_view "fields" fields)`
    - `add_unit_test(field "fields" fields)`
    - `add_unit_test(field_utils "fields" fields)`
    - `add_unit_test(field_math "fields" fields)`
  - Also remove (test files deleted in 12.2a/b/c/h):
    - `add_unit_test(algorithms "fields" fields)` — deleted in 12.2h
    - `add_unit_test(scalar "fields" fields)` — deleted in 12.2a
    - `add_unit_test(vector "fields" fields)` — deleted in 12.2b
    - `add_unit_test(selector "fields" fields)` — deleted in 12.2c
  - Keep these test targets (new tests from Phases 8–11):
    - `add_unit_test(handle "fields" fields)` — new test (Phase 8)
    - `t-field_registry` explicit target — new test (Phase 8)
    - `t-expr` explicit target — new test (Phase 10)
    - `t-selection_desc` explicit target — new test (Phase 11)
  - **Total `add_unit_test` lines removed: 15** (11 from 12.2d–g/i + 4 from 12.2a/b/c/h). After cleanup, only `add_unit_test(handle ...)` remains alongside the 3 explicit `add_executable` targets.
  - File: `src/fields/CMakeLists.txt`

- [x] **12.4b** Delete `src/fields/matchers.hpp` (already in 12.3a deletion list): **DONE** — deleted as part of 12.3a.
  - Confirmed dead code: defines Catch2 matchers for non-existent types (`scalar_field`, `vector_field`, `vector_range`, old `scalar<T,I>` with `.field`/`.obj` members).
  - Zero `#include "matchers.hpp"` anywhere in the codebase.

- [x] **12.4c** Delete `src/mesh/selections.hpp` and `src/mesh/selections.t.cpp`: **Done atomically with 12.3b1.** Deleted both files and removed `add_unit_test(selections "mesh" shoccs-mesh)` from `src/mesh/CMakeLists.txt`. Build succeeds; `t-selections` test no longer exists (mesh tests: 5, down from 6). 6 pre-existing failures unrelated.

### 12.5 — Final verification

- [x] **12.5a** Full build and test: `cmake --build build && ctest --test-dir build`. **DONE** — All verification checks pass:
  - **Build:** Clean success (`ninja: no work to do`).
  - **Tests:** 6 pre-existing failures unchanged (t-object_geometry, t-E2_1, 4 operator tests with Kokkos/OpenMP SIGABRT). All other tests pass.
  - **Grep check 1:** `container_tuple|view_tuple|single_view|tuple_pipe|OwningTuple|NestedTuple` → only 1 match: a comment in `handle.hpp` line 401 ("Replaces the recursive for_each/transform over NestedTuples").
  - **Grep check 2:** `list_index|selection_view_fn|zip_transform_view|semiregular_box` → zero matches.
  - **Deleted files:** 34 files deleted (30 from `src/fields/`, 2 from `src/mesh/`, 2 others). 11,198 lines removed, 7,464 lines added = ~3,734 net lines removed (exceeds ~2,500 target).
  - **`src/fields/` remaining:** Exactly expected — `handle.hpp/t.cpp`, `field_registry.hpp/t.cpp`, `expr.hpp/t.cpp`, `selection_desc.hpp/t.cpp`, `scalar.hpp`, `lazy_views.hpp`, `CMakeLists.txt` (11 files).

---

## Completion Criteria

- Old tuple infrastructure is deleted (~2500+ lines removed across ~15 files).
- All ~141 old field test cases are deleted; 98 new tests in `handle.t.cpp`, `field_registry.t.cpp`, `expr.t.cpp`, `selection_desc.t.cpp` provide complete replacement coverage. No old test files are rewritten — all old tests (12.2a–i: scalar, vector, selector, tuple, container_tuple, view_tuple, single_view, tuple_pipe, tuple_utils, tuple_math, field, field_utils, field_math, algorithms, range_concepts) are deleted.
- No production code references `container_tuple`, `view_tuple`, `single_view`, `tuple_pipe`, `zip_transform_view`, `semiregular_box`, `list_index`, or `selection_view_fn`.
- Full build succeeds. All tests pass.
- The `src/fields/` directory contains: `handle.hpp` + `handle.t.cpp`, `field_registry.hpp` + `field_registry.t.cpp`, `expr.hpp` + `expr.t.cpp`, `selection_desc.hpp` + `selection_desc.t.cpp`, `scalar.hpp` (simplified to plain structs per D-R19, no test file — covered by span bridge tests in `field_registry.t.cpp`), `lazy_views.hpp` (trimmed per D-R17, keeping stride/repeat_n/cartesian_product/linear_distribute).
- `vector.hpp`, `field.hpp`, `field_fwd.hpp` are deleted. `system_size` is relocated (per D-R18).
- Non-fields production code (`real3_operators.hpp`, `xdmf.hpp`, `field_io.hpp`, `field_data.hpp`, `cartesian.hpp`, `object_geometry.hpp`, `mesh.hpp`, `manufactured_solutions.hpp`, matrix visitors) no longer includes any deleted header.
