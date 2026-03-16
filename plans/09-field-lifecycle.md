# Phase 9: Field Lifecycle Migration

**Goal:** Migrate `simulation_cycle`, integrators (`rk4`, `euler`), and the `system` interface to use `field_registry` + `field_ref` instead of owning `field` objects and heap-allocating `field_span`/`field_view` on every pass.

**Depends on:** Phase 8

**Read first:**
- `src/fields/field_registry.hpp` (Phase 8 output)
- `src/fields/handle.hpp` (handle types)
- `src/simulation/simulation_cycle.hpp` + `simulation_cycle.cpp`
- `src/temporal/rk4.hpp` + `rk4.cpp`
- `src/temporal/euler.hpp` + `euler.cpp`
- `src/temporal/integrator.hpp` + `integrator.cpp`
- `src/systems/system.hpp` + `system.cpp`
- `src/systems/heat.hpp` + `heat.cpp`
- `src/systems/scalar_wave.hpp` + `scalar_wave.cpp`
- `src/systems/hyperbolic_eigenvalues.hpp` + `hyperbolic_eigenvalues.cpp`
- `src/systems/inviscid_vortex.hpp` + `inviscid_vortex.cpp`
- `src/io/field_io.hpp` + `field_io.cpp`
- `src/io/field_data.hpp` + `field_data.cpp`

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -L simulation
ctest --test-dir build -L temporal
ctest --test-dir build -L systems
ctest --test-dir build
```

---

## Strategy

The migration proceeds bottom-up: concrete systems first, then system wrapper, then integrators, then simulation_cycle. At each level, the old and new interfaces coexist via adapter functions until the full chain is migrated.

The key invariant: **the registry is owned by `simulation_cycle::run()` as a local**. It is passed by reference down the call chain. `field_ref` (a 12-byte trivially-copyable struct) replaces `field`/`field_span`/`field_view` everywhere they are passed by value.

The `std::function<void(field_span)>` return pattern in `system::rhs` and `integrator::operator()` is replaced by direct writes: functions take `field_ref output` as a parameter instead of returning a deferred callable.

---

## Items

### 9.1 — Define the new system interface (TDD)

- [ ] **9.1a** Create `src/systems/system_v2.t.cpp` testing the new interface concept:
  - A `SystemV2` concept requiring:
    - `void rhs(const field_registry&, field_ref input, field_registry&, field_ref output, real time)`
    - `void update_boundary(field_registry&, field_ref, real time)`
    - `system_size size() const`
    - `system_stats stats(const field_registry&, field_ref u0, field_ref u1, const step_controller&) const`
  - Test with a trivial mock system that writes `output[D] = 2 * input[D]`.
  - Verify the mock satisfies the concept.
  - File: `src/systems/system_v2.t.cpp` (new)
  - Test: `ctest --test-dir build -R t-system_v2`

- [ ] **9.1b** Define the `SystemV2` concept in `src/systems/system_v2.hpp`:
  - The concept and any shared type aliases.
  - File: `src/systems/system_v2.hpp` (new)
  - Test: `ctest --test-dir build -R t-system_v2`

### 9.2 — Adapt heat system (TDD)

- [ ] **9.2a** Add registry-based `rhs` and `update_boundary` methods to `heat`:
  - These are new overloads alongside the existing `field_view`/`field_span` ones.
  - They extract `scalar_span`/`scalar_view` via the span bridge and delegate to existing logic.
  - File: `src/systems/heat.hpp`, `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R t-heat`

- [ ] **9.2b** Add registry-based methods to `scalar_wave`, `hyperbolic_eigenvalues`, `inviscid_vortex`, `empty_system`:
  - Same adapter pattern as 9.2a.
  - Files: respective `.hpp`/`.cpp` pairs
  - Test: `ctest --test-dir build -L systems`

### 9.3 — Adapt system variant wrapper

- [ ] **9.3a** Add registry-based dispatch to `system.hpp`/`system.cpp`:
  - New overloads: `rhs(field_registry&, field_ref input, field_ref output, real time)`
  - Uses `std::visit` to dispatch to the concrete system's registry-based method.
  - No `std::function` return — direct write into output slot.
  - Coexists with the old `std::function<void(field_span)> rhs(field_view, real)`.
  - Files: `src/systems/system.hpp`, `src/systems/system.cpp`
  - Test: `ctest --test-dir build -L systems`

### 9.4 — Adapt integrators (TDD)

- [ ] **9.4a** Write tests for a registry-based `rk4` step:
  - Create a registry with 4 slots (u0, u1, rk_rhs, system_rhs).
  - Fill u0 with known data.
  - Call a registry-based `rk4::operator()` that writes into the u1 slot.
  - Verify results match the existing `t-simulation_cycle` expectations.
  - File: `src/temporal/rk4_v2.t.cpp` (new)
  - Test: `ctest --test-dir build -R t-rk4_v2`

- [ ] **9.4b** Implement registry-based `rk4::operator()`:
  - New overload taking `(system&, field_registry&, field_ref u0, field_ref output, const step_controller&, real dt)`.
  - `rk_rhs` and `system_rhs` become **slot indices** in the registry, not owned `field` members.
  - Internally: extracts spans via bridge, calls system's registry-based `rhs`.
  - Files: `src/temporal/rk4.hpp`, `src/temporal/rk4.cpp`
  - Test: `ctest --test-dir build -R t-rk4_v2`

- [ ] **9.4c** Same for `euler`.
  - Files: `src/temporal/euler.hpp`, `src/temporal/euler.cpp`
  - Test: `ctest --test-dir build -L temporal`

- [ ] **9.4d** Adapt `integrator` variant wrapper with registry-based dispatch:
  - New overload: `void operator()(system&, field_registry&, field_ref u0, field_ref output, const step_controller&, real dt)`
  - Files: `src/temporal/integrator.hpp`, `src/temporal/integrator.cpp`
  - Test: `ctest --test-dir build -L temporal`

### 9.5 — Adapt simulation_cycle

- [ ] **9.5a** Rewrite `simulation_cycle::run()` to use the registry:
  - Create `field_registry` as a local in `run()`.
  - Allocate slots for u0, u1 (plus rk4 scratch slots).
  - Replace `field u0{sys(controller)}` with registry allocation + system init.
  - Replace `field u1{u0}` with `registry.deep_copy_slot(slot_u1, slot_u0)`.
  - Replace `swap(u0, u1)` with `std::swap(u0_ref.slot, u1_ref.slot)`.
  - Replace `u1 = integrate(...)` with `integrate(sys, registry, u0_ref, u1_ref, controller, dt)`.
  - Files: `src/simulation/simulation_cycle.cpp`
  - Test: `ctest --test-dir build -R t-simulation_cycle`

### 9.6 — Adapt I/O

- [ ] **9.6a** Add registry-based `write` overloads to `field_io` and `field_data`:
  - Extract `field_view` via span bridge for write operations.
  - Files: `src/io/field_io.hpp`, `src/io/field_io.cpp`, `src/io/field_data.hpp`, `src/io/field_data.cpp`
  - Test: `ctest --test-dir build -R t-field_io -R t-xdmf`

### 9.7 — Remove old field_span/field_view pass-by-value paths

- [ ] **9.7a** Once all callers use registry-based methods, remove the old `std::function<void(field_span)>` return patterns from `system::rhs` and `integrator::operator()`.
  - Remove the old overloads (not the entire methods — just the std::function-returning variants).
  - Remove the invocable constructor/assignment from `field.hpp` if no longer used.
  - Files: `src/systems/system.hpp`, `src/systems/system.cpp`, `src/temporal/integrator.hpp`, `src/temporal/integrator.cpp`, `src/fields/field.hpp`
  - Test: `ctest --test-dir build`

---

## Completion Criteria

- `simulation_cycle::run()` uses `field_registry` + `field_ref` exclusively.
- No `std::function<void(field_span)>` remains in the integrator/system chain.
- `swap(u0, u1)` is an integer swap.
- `system::rhs` writes directly into an output slot (no deferred callable).
- `rk4`/`euler` scratch buffers are registry slots, not owned `field` members.
- All existing tests pass.
