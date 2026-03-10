# Phase 6: Temporal Integration

**Goal:** Migrate time integration subsystem. This is a low-effort phase — no production code uses range-v3.

**Depends on:** Phases 0, 1

**Read first:**
- `src/temporal/rk4.hpp` + `rk4.cpp` (no range-v3)
- `src/temporal/euler.hpp` + `euler.cpp` (no range-v3)
- `src/temporal/integrator.hpp` + `integrator.cpp` (no range-v3)
- `src/temporal/step_controller.hpp` + `step_controller.cpp` (no range-v3)
- `src/temporal/empty_integrator.hpp` + `empty_integrator.cpp` (no range-v3)
- `src/temporal/CMakeLists.txt`
- `plans/meta.md`

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -L temporal
```

---

## Items

- [ ] **6.1** Verify all temporal implementation files have no range-v3 usage. Mark complete if clean.
  - Files to check (all confirmed clean — no `range/v3` or `rs::`/`vs::` usage):
    - `src/temporal/rk4.hpp` + `rk4.cpp`
    - `src/temporal/euler.hpp` + `euler.cpp`
    - `src/temporal/integrator.hpp` + `integrator.cpp`
    - `src/temporal/step_controller.hpp` + `step_controller.cpp`
    - `src/temporal/empty_integrator.hpp` + `empty_integrator.cpp`
  - Action: grep for `range/v3` in all `.hpp`/`.cpp` (non-test) files; confirm zero matches.
  - Test: `cmake --build build` succeeds with no temporal production changes.

- [ ] **6.2** Remove unused `#include <range/v3/all.hpp>` from temporal test files.
  - `src/temporal/euler.t.cpp` line 11: delete `#include <range/v3/all.hpp>`.
    No range-v3 APIs (no `rs::`, `vs::`, or range-v3 types) are used in this file.
  - `src/temporal/rk4.t.cpp` line 11: delete `#include <range/v3/all.hpp>`.
    No range-v3 APIs (no `rs::`, `vs::`, or range-v3 types) are used in this file.
  - `src/temporal/step_controller.t.cpp`: already clean (no range-v3 include).
  - No replacement code needed — the includes are purely unused.
  - Test: `cmake --build build && ctest --test-dir build -L temporal` — all 3 tests pass.

---

## Completion Criteria

- All 3 temporal test files pass (`ctest --test-dir build -L temporal`).
- No `#include <range/v3/...>` remains in any file under `src/temporal/`.
