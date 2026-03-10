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
  - Test: build succeeds.

- [ ] **6.2** Migrate temporal test files — they include `<range/v3/all.hpp>`:
  - `euler.t.cpp`
  - `rk4.t.cpp`
  - `step_controller.t.cpp` (may not use range-v3)
  - Replace with std equivalents for any range-v3 algorithms used in tests.
  - Test: `ctest --test-dir build -L temporal` — all pass.

---

## Completion Criteria

- All 3 temporal test files pass.
- No `#include <range/v3/...>` remains in `src/temporal/`.
