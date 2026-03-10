# Phase 0: Foundation

**Goal:** Add Kokkos to the build system, establish type aliases and compatibility layers, and replace coroutine-based iteration with Kokkos-compatible patterns.

**Depends on:** Nothing

**Read first:**
- `CMakeLists.txt` (top-level)
- `src/CMakeLists.txt`
- `external/CMakeLists.txt`
- `.devcontainer/spack.yaml`
- `config/shoccsConfig.cmake.in`
- `src/shoccs_config.hpp`
- `src/types.hpp`
- `src/indexing.hpp`
- `src/index_extents.hpp`
- `src/index_view.hpp`
- `src/index_view.t.cpp`
- `src/mesh/mesh_view.hpp`
- `src/mesh/CMakeLists.txt`
- `src/app/shoccs.cpp`
- `src/app/CMakeLists.txt`
- `external/cppcoro/generator.hpp`
- `plans/meta.md`

**Test commands:**
```bash
cmake -S . -B build -G Ninja -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build -R t-indexing
ctest --test-dir build -R t-index_view
ctest --test-dir build -R t-real3_operators
```

---

## Items

### 0.1 — Add Kokkos to the CMake build system

- [x] **0.1a** In `CMakeLists.txt` (top-level): Add `find_package(Kokkos REQUIRED)` after the existing `find_package` block (after line 34, before `include(GNUInstallDirs)`).
  - File: `CMakeLists.txt`
  - Test: `cmake -S . -B build -G Ninja -DBUILD_TESTING=ON` succeeds and prints `-- Enabled Kokkos devices: OPENMP;SERIAL`.

- [x] **0.1b** In `src/CMakeLists.txt`: Add `target_link_libraries(indexing INTERFACE Kokkos::kokkos)` after line 3 (`add_library(indexing INTERFACE)`). This makes Kokkos available to the `indexing` INTERFACE library and its consumers (test targets).
  - File: `src/CMakeLists.txt`
  - Test: `cmake --build build -- t-indexing` succeeds.

- [x] **0.1c** In `config/shoccsConfig.cmake.in`: Add `find_package(Kokkos REQUIRED)` after the existing `find_package` lines (after line 13).
  - File: `config/shoccsConfig.cmake.in`
  - Test: `cmake --build build` succeeds.

- [x] **0.1d** In `.devcontainer/spack.yaml`: Update the comment on line 24 from `# Future: Kokkos for migration from range-v3` to `# Kokkos (parallel execution framework)`. The kokkos spec is already active (not commented out).
  - File: `.devcontainer/spack.yaml`
  - No build test needed (comment-only change).

### 0.2 — Resolve Decision D1 (Host vs. Device strategy)

- [x] **0.2** Update `plans/meta.md` D1 entry.
  - Recommended: **Option (a)** — `Kokkos::DefaultHostExecutionSpace` everywhere initially. The codebase is entirely CPU-only today. Starting host-only minimizes risk and allows validating correctness before adding device portability. GPU migration can be a separate future effort.
  - File: `plans/meta.md`

### 0.3 — Resolve Decision D2 (range-v3 removal strategy)

- [x] **0.3** Update `plans/meta.md` D2 entry.
  - Recommended: **Option (b)** — Keep `range-v3` as a project dependency; remove usage incrementally phase by phase. Rationale: 93 source files use range-v3 (1590 total occurrences of `rs::`/`vs::`). Many uses (`vs::cartesian_product`, `vs::take_exactly`, `vs::stride`, `rs::inner_product`, `rs::to`) have no direct C++20 `std::ranges` equivalents. Each subsequent phase will remove range-v3 from its subsystem. The top-level `find_package(range-v3)` and the `rs`/`vs` namespace aliases in `types.hpp` remain until all phases complete. Phase 0 only removes range-v3 from `index_view.t.cpp`.
  - File: `plans/meta.md`

### 0.4 — Resolve Decision D3 (cppcoro generator replacement)

- [x] **0.4** Update `plans/meta.md` D3 entry.
  - Recommended: **Option (b)** — Replace with plain functions returning `std::vector`. Generators are only used in `index_view.hpp` (2 overloads yielding `int3`) and `mesh_view.hpp` (2 overloads yielding `real3`). Both are simple triple-nested loops. Callers either iterate with range-for or collect to `std::vector` immediately. Returning `std::vector` is the simplest replacement with identical caller semantics. The allocation overhead is negligible for these host-only iteration utilities.
  - File: `plans/meta.md`

### 0.5 — Establish Kokkos type aliases

- [ ] **0.5** Create a new header `src/kokkos_types.hpp` with Kokkos-compatible type aliases.
  - Contents: `#pragma once`, `#include <Kokkos_Core.hpp>`, `#include "shoccs_config.hpp"`, then inside `namespace ccs`: `using execution_space = Kokkos::DefaultHostExecutionSpace;`, `using memory_space = typename execution_space::memory_space;`, `template<typename T> using device_view = Kokkos::View<T, memory_space>;`.
  - **Do NOT add `#include <Kokkos_Core.hpp>` to `types.hpp`** — `types.hpp` is included by 35+ headers across all subsystems; adding Kokkos there would require every CMake library target to link `Kokkos::kokkos`. That cascading change is deferred to Phase 1 (fields), when the `fields` INTERFACE library (which most targets depend on) adds Kokkos.
  - Keep existing `rs`/`vs` namespace aliases in `types.hpp` unchanged (per D2).
  - File: `src/kokkos_types.hpp` (new file)
  - Test: `cmake --build build` succeeds (header is not yet included anywhere, just created).
  - Cross-cutting: See new decision D7 in `plans/meta.md` about Kokkos include propagation strategy.

### 0.6 — Add Kokkos initialization/finalization

- [ ] **0.6a** In `src/app/shoccs.cpp`: Add `#include <Kokkos_Core.hpp>` at the top. Insert `Kokkos::ScopeGuard kokkos(argc, argv);` as the first statement in `main()` (before the `cxxopts::Options` construction on line 14). This ensures Kokkos is initialized before any Kokkos operations and finalized on exit.
  - File: `src/app/shoccs.cpp`

- [ ] **0.6b** In `src/app/CMakeLists.txt`: Add `Kokkos::kokkos` to the `target_link_libraries` for `shoccs-exe` (line 2: append to the existing list `cxxopts::cxxopts shoccs-run_sol spdlog::spdlog`).
  - File: `src/app/CMakeLists.txt`
  - Test: `cmake --build build -- shoccs-exe && ./build/src/app/shoccs --help` starts and stops without error (Kokkos init/finalize runs silently).

### 0.7 — Replace coroutine-based iteration utilities

Depends on: 0.4 (D3 resolved).

- [ ] **0.7a** In `src/index_view.hpp`: Replace cppcoro generators with vector-returning functions.
  - Remove `#include <cppcoro/generator.hpp>` (line 4) and `#include <range/v3/view/take_exactly.hpp>` (line 5). The `take_exactly` include is unused in this file.
  - Add `#include <vector>`.
  - Replace the volume overload `cppcoro::generator<int3> index_view(int3 extents)` (lines 12–32) with a function returning `std::vector<int3>`. Body: reserve `extents[I]*extents[F]*extents[S]` elements, then the same triple nested loop with `result.push_back(ijk)` instead of `co_yield ijk`.
  - Replace the plane overload `cppcoro::generator<int3> index_view(int3 extents, int i)` (lines 36–55) with a function returning `std::vector<int3>`. Body: reserve `extents[F]*extents[S]` elements, same loop with `push_back` instead of `co_yield`.
  - File: `src/index_view.hpp`
  - Test: compiles as part of `t-index_view` (enabled in 0.8b).

- [ ] **0.7b** In `src/mesh/mesh_view.hpp`: Same pattern — replace cppcoro generators with vector-returning functions.
  - Remove `#include <cppcoro/generator.hpp>` (line 6). Add `#include <vector>`.
  - Replace `cppcoro::generator<real3> location_view(const cartesian& m)` (lines 17–37) with a function returning `std::vector<real3>`. Same loop, `push_back` instead of `co_yield`.
  - Replace `cppcoro::generator<real3> location_view(const cartesian& m, int i)` (lines 39–59) with a function returning `std::vector<real3>`. Same loop, `push_back` instead of `co_yield`.
  - Note: `mesh_view.hpp` and its test are currently commented out of `src/mesh/CMakeLists.txt` (lines 7, 10). This item only converts the header. Re-enabling the `mesh_view` test is deferred to a later phase (it depends on mesh library and range-v3 for `vs::zip`, `rs::to`).
  - File: `src/mesh/mesh_view.hpp`
  - Test: `cmake --build build` succeeds (mesh_view.hpp is not compiled by any active target).

### 0.8 — Update index_view test and re-enable it

Depends on: 0.7a.

- [ ] **0.8a** In `src/index_view.t.cpp`: Remove range-v3 usage.
  - Remove `#include <range/v3/algorithm/equal.hpp>` (line 5) and `#include <range/v3/range/conversion.hpp>` (line 6).
  - Line 15: Replace `index_view<0>(extents, 0) | rs::to<std::vector<int3>>()` with just `index_view<0>(extents, 0)` (since `index_view` now returns `std::vector<int3>` directly).
  - Lines 24, 34, 47, 61, 65: Replace `rs::equal(index_view<I>(extents, n), std::vector{...})` with `index_view<I>(extents, n) == std::vector{...}` (vector `==` works directly).
  - File: `src/index_view.t.cpp`

- [ ] **0.8b** In `src/CMakeLists.txt`: Uncomment the index_view test (line 5) and update its dependencies.
  - Change `#add_unit_test(index_view "indexing" indexing cppcoro range-v3::range-v3)` to `add_unit_test(index_view "indexing" indexing)`. The `cppcoro` and `range-v3::range-v3` link dependencies are no longer needed.
  - File: `src/CMakeLists.txt`
  - Test: `ctest --test-dir build -R t-index_view` passes.

### 0.9 — Remove cppcoro vendored dependency

Depends on: 0.7a, 0.7b.

- [ ] **0.9a** Verify no active source files still include `<cppcoro/generator.hpp>`. After 0.7a and 0.7b, the only remaining reference is `src/operators/directional.cpp` which is excluded from scope (commented out of `src/operators/CMakeLists.txt` line 21). That file will have a broken include, which is acceptable since it's dead code.

- [ ] **0.9b** Delete the `external/cppcoro/` directory (contains `generator.hpp` only).

- [ ] **0.9c** In `external/CMakeLists.txt`: Remove both lines (`add_library(cppcoro INTERFACE)` and `target_include_directories(cppcoro INTERFACE ${CMAKE_CURRENT_LIST_DIR})`). If this leaves the file empty, keep it as an empty file (the `add_subdirectory(external)` in the top-level CMakeLists still references it).
  - Files: `external/cppcoro/generator.hpp` (delete), `external/CMakeLists.txt` (edit)
  - Test: `cmake --build build` — full build succeeds with no cppcoro references.

---

## Completion Criteria

- Kokkos is found and linked in the build.
- All decisions D1–D3 are resolved and recorded in `meta.md`.
- `src/kokkos_types.hpp` exists with host execution space aliases.
- `index_view` and `mesh_view` no longer use cppcoro coroutines.
- `t-index_view` test is enabled and passes.
- `external/cppcoro/` directory is removed.
- All existing tests still pass.
