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

### Registry Type

A single concrete registry type is used throughout the simulation chain (see decision D-R10 in `plans/meta.md`):
```cpp
// In src/fields/simulation_registry.hpp (or field_registry.hpp)
using sim_registry = field_registry<8, 8, 4>;
```
- **MaxSlots = 8**: rk4 needs 4 (u0, u1, rk_rhs, system_rhs); 8 provides headroom.
- **MaxS = 8, MaxV = 4**: matches `general_layout`. All current systems fit (heat: 1,0; scalar_wave: 1,0; eigenvalues: 0,0).
- The `system` and `integrator` variant wrappers use `sim_registry` as a concrete type in their new method signatures (no templates needed on the wrapper classes).

### Slot Allocation Convention

Slots are allocated in `simulation_cycle::run()` and passed as `field_ref` tokens:
- **Slot 0**: `u0` — current solution
- **Slot 1**: `u1` — next-step solution
- **Slot 2**: `rk_rhs` — rk4 accumulator (unused by euler)
- **Slot 3**: `system_rhs` — scratch for system RHS evaluation

Integrators receive `field_ref` tokens for their scratch slots (not slot indices) so they don't hard-code the allocation.

### Adapter Pattern for Concrete Systems

Each concrete system gets new overloaded methods that:
1. Accept `sim_registry&` (or `const sim_registry&`) + `field_ref` instead of `field_view`/`field_span`/`const field&`.
2. Extract `scalar_span`/`scalar_view` via the existing `extract_scalar_span()`/`extract_scalar_view()` bridge functions (already in `field_registry.hpp`).
3. Either delegate to the existing implementation or implement standalone (see rules below).

**Delegation rules** (see D-R12 in `plans/meta.md`): The existing methods use two different parameter conventions:
- **`field_view`/`field_span` parameters** (`rhs`, `update_boundary`, `write`): The adapter constructs a temporary `field_view`/`field_span` from extracted `scalar_view`/`scalar_span` and delegates to the existing method. This works because these types are non-owning views.
- **`const field&`/`field&` parameters** (`stats`, `timestep_size`, `operator()`/`initialize`): The adapter **cannot delegate** because `field` (= `detail::field<std::vector<scalar_real>, ...>`) is an owning type — constructing one from spans would copy all data. Instead, the adapter implements the logic directly using extracted `scalar_view`/`scalar_span`. This works because the DSL operators (`|`, `sel::D`, mesh selections, `abs`, `max`, etc.) are templates that work on any `Scalar` type (`scalar_real`, `scalar_view`, or `scalar_span`).

Example **delegating** adapter for `heat::rhs`:
```cpp
void heat::rhs(const sim_registry& reg, field_ref input,
               sim_registry& out_reg, field_ref output, real time) const
{
    constexpr auto sh = scalar_handle{0};  // scalar[0] = u
    auto fv = field_view{{extract_scalar_view(reg, input, sh)}, {}};
    auto fs = field_span{{extract_scalar_span(out_reg, output, sh)}, {}};
    rhs(fv, time, fs);  // delegate to existing
}
```

Example **standalone** adapter for `heat::timestep_size`:
```cpp
real heat::timestep_size(const sim_registry&, field_ref,
                         const step_controller& step) const
{
    // Field data is unused — same body as existing method
    const auto h_min = std::ranges::min(m.h());
    return step.parabolic_cfl() * h_min * h_min / (4 * diffusivity);
}
```

---

## Items

### 9.1 — Registry type alias and new system interface (TDD)

- [x] **9.1a** Add `sim_registry` type alias to `src/fields/field_registry.hpp`:
  - Add `using sim_registry = field_registry<8, 8, 4>;` after the class template definition.
  - This is the single concrete registry type used by the simulation chain.
  - File: `src/fields/field_registry.hpp`
  - Test: `cmake --build build` (compile-only check)

- [x] **9.1b** Create `src/systems/system_v2.hpp` with the `SystemV2` concept:
  - The concept is templated on the system type and checks for these required expressions (using `sim_registry`):
    ```cpp
    template <typename T>
    concept SystemV2 = requires(T& sys, const T& csys,
                                const sim_registry& creg, sim_registry& reg,
                                field_ref input, field_ref output, real time,
                                const step_controller& ctrl) {
        { sys.rhs(creg, input, reg, output, time) } -> std::same_as<void>;
        { sys.update_boundary(reg, input, time) } -> std::same_as<void>;
        { csys.size() } -> std::same_as<system_size>;
        { csys.stats(creg, input, output, ctrl) } -> std::same_as<system_stats>;
        { csys.timestep_size(creg, input, ctrl) } -> std::same_as<real>;
        { sys.initialize(reg, input, ctrl) } -> std::same_as<void>;
    };
    ```
  - Include `field_registry.hpp`, `field_fwd.hpp`, `types.hpp`.
  - File: `src/systems/system_v2.hpp` (new)
  - Test: `cmake --build build`

- [x] **9.1c** Create `src/systems/system_v2.t.cpp` testing the concept:
  - Define a trivial `mock_system` that satisfies `SystemV2`:
    - `rhs`: writes `output[D] = 2 * input[D]` using `extract_scalar_span`/`extract_scalar_view`.
    - Other methods: minimal implementations.
  - Test needs custom `main()` with `Kokkos::ScopeGuard` (per D-R9) since it allocates Views.
  - `static_assert(SystemV2<mock_system>)` to verify concept satisfaction.
  - Runtime test: create `sim_registry`, allocate 2 slots, fill slot 0, call `mock.rhs()`, verify slot 1 = 2× slot 0.
  - Register in `src/systems/CMakeLists.txt` (manual executable, link `Catch2::Catch2` not `Catch2::Catch2WithMain`).
  - File: `src/systems/system_v2.t.cpp` (new), `src/systems/CMakeLists.txt`
  - Test: `ctest --test-dir build -R t-system_v2`

### 9.2 — Adapt concrete systems

Each concrete system gets 6 new adapter methods. Three methods delegate to existing implementations (those already accepting `field_view`/`field_span`); three are standalone implementations (those whose existing versions take `const field&`/`field&` — see "Adapter Delegation Rules" above).

- [x] **9.2a** Add all registry-based adapter methods to `heat`:
  - New methods (all alongside existing overloads), using `scalar_handle{0}` for the single scalar:
  - **Delegating adapters** (construct temporary `field_view`/`field_span`, call existing method):
    - `void rhs(const sim_registry&, field_ref input, sim_registry&, field_ref output, real time) const` — constructs `field_view{{extract_scalar_view(reg, input, sh)}, {}}` and `field_span{{extract_scalar_span(out_reg, output, sh)}, {}}`, delegates to `rhs(field_view, real, field_span)`.
    - `void update_boundary(sim_registry&, field_ref, real time)` — constructs `field_span` from extracted `scalar_span`, delegates to `update_boundary(field_span, real)`.
    - `bool write(field_io&, const sim_registry&, field_ref, const step_controller&, real)` — constructs `field_view` from extracted `scalar_view`, delegates to `write(field_io&, field_view, ...)`.
  - **Standalone adapters** (implement logic directly, cannot delegate to `const field&`/`field&` methods):
    - `real timestep_size(const sim_registry&, field_ref, const step_controller&) const` — field data is unused; implements inline: `return step.parabolic_cfl() * h_min * h_min / (4 * diffusivity)`.
    - `system_stats stats(const sim_registry&, field_ref u0, field_ref u1, const step_controller&) const` — extracts `scalar_view` for u1 via `extract_scalar_view(reg, u1, scalar_handle{0})`, applies the same DSL logic as existing `stats` body directly on the `scalar_view` (`abs(u - sol) | m.fluid_all(...)`, `minmax(...)`, etc. all work generically on `Scalar` types). Note: existing `stats` ignores u0.
    - `void initialize(sim_registry&, field_ref, const step_controller&)` — extracts `scalar_span` via `extract_scalar_span(reg, ref, scalar_handle{0})`, applies the same DSL logic as existing `operator()` body: `u | sel::D = 0; u | m.fluid = sol; u | sel::R = sol;`. Cannot delegate because `operator()` takes `field&` (owning type).
  - `static_assert(SystemV2<heat>)` at bottom of heat.cpp.
  - Files: `src/systems/heat.hpp`, `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R t-heat`

- [x] **9.2b** Add registry-based adapters to `scalar_wave`:
  - Same 6 methods as 9.2a (3 delegating + 3 standalone). Uses `scalar_handle{0}` (ns=1, nv=0 at field level).
  - **Delegating**: `rhs`, `update_boundary`, `write` — same pattern as heat.
  - **Standalone**: `timestep_size` (field unused, inline: `return step.hyperbolic_cfl() * h_min`), `stats` (same DSL pattern on `scalar_view`), `initialize` (same DSL pattern on `scalar_span`).
  - Note: `scalar_wave::rhs` also uses internal `grad_G` and `du` (vector_real members) — these are NOT in the registry, they remain as owned members. Only the field_view/field_span arguments are bridged.
  - `static_assert(SystemV2<scalar_wave>)`.
  - Files: `src/systems/scalar_wave.hpp`, `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -L systems`

- [x] **9.2c** Add registry-based adapters to `hyperbolic_eigenvalues`:
  - All 6 methods are standalone (system has no scalars/vectors in the field: `size()` returns `{0, 0, m.ss()}`).
  - `rhs`, `update_boundary`, `initialize`: no-ops (empty body, matching existing).
  - `timestep_size`: returns `1.0` (matching existing).
  - `stats`: standalone, doesn't read from field arguments (uses `m.Rx()`, `object_bcs`, `grad.visit()`).
  - `write`: returns `true` (matching existing).
  - `static_assert(SystemV2<hyperbolic_eigenvalues>)`.
  - Files: `src/systems/hyperbolic_eigenvalues.hpp`, `src/systems/hyperbolic_eigenvalues.cpp`
  - Test: `ctest --test-dir build -L systems`

- [x] **9.2d** Add registry-based adapters to `inviscid_vortex` and `empty`:
  - Both are stub systems with empty/no-op methods. All 6 adapters are standalone with trivial bodies (empty, return `{}`, return `false`, etc.), matching the existing implementations.
  - `inviscid_vortex::timestep_size` uses internal `U` member (not the field arg), but the adapter can return the same value. `inviscid_vortex::valid()` returns `false`, so its timestep path is never actually called in practice.
  - `static_assert(SystemV2<inviscid_vortex>)` and `static_assert(SystemV2<systems::empty>)`.
  - Files: `src/systems/inviscid_vortex.hpp`, `src/systems/inviscid_vortex.cpp`, `src/systems/empty_system.hpp`, `src/systems/empty_system.cpp`
  - Test: `ctest --test-dir build -L systems`

### 9.3 — Adapt system variant wrapper

- [x] **9.3a** Add registry-based dispatch methods to `system`:
  - New methods that use `std::visit` to dispatch to concrete systems' registry-based methods:
    - `void rhs(const sim_registry&, field_ref input, sim_registry&, field_ref output, real time)` — replaces `std::function<void(field_span)> rhs(field_view, real)`.
    - `void update_boundary(sim_registry&, field_ref, real time)` — replaces `void update_boundary(field_span, real)`.
    - `system_stats stats(const sim_registry&, field_ref u0, field_ref u1, const step_controller&) const`
    - `void initialize(sim_registry&, field_ref, const step_controller&)` — replaces `std::function<void(field&)> operator()(const step_controller&)`.
    - `bool write(field_io&, const sim_registry&, field_ref, const step_controller&, real)`
    - **Note**: `timestep_size` is NOT included here — it is handled separately in 9.3b because the `system` wrapper adds `step_controller` check wrapping (returning `std::optional<real>`) rather than just dispatching.
  - Each method follows the pattern:
    ```cpp
    void system::rhs(const sim_registry& creg, field_ref input,
                     sim_registry& reg, field_ref output, real time) {
        std::visit([&](auto&& s) { s.rhs(creg, input, reg, output, time); }, v);
    }
    ```
  - The `sim_registry` type is used directly (not templated) since it's a fixed alias.
  - `system.hpp` includes `fields/field_registry.hpp` for the `sim_registry` type.
  - Old methods remain until 9.7 cleanup.
  - Files: `src/systems/system.hpp`, `src/systems/system.cpp`
  - Test: `ctest --test-dir build -L systems`
  - Must come after: 9.2a–9.2d (all concrete systems must have registry methods first).

- [x] **9.3b** Add registry-based `timestep_size` wrapper with controller check:
  - The current `system::timestep_size` calls `controller.check_timestep_size(predicted_dt)`. In the new path, keep this wrapping in the system class:
    ```cpp
    std::optional<real> system::timestep_size(const sim_registry& reg, field_ref u,
                                              const step_controller& ctrl) const {
        real predicted = std::visit([&](auto&& s) {
            return s.timestep_size(reg, u, ctrl);
        }, v);
        return ctrl.check_timestep_size(predicted);
    }
    ```
  - This overload returns `std::optional<real>` (different from the concept's `real` return).
  - Files: `src/systems/system.hpp`, `src/systems/system.cpp`
  - Test: `ctest --test-dir build -L systems`

### 9.4 — Adapt integrators (TDD)

- [x] **9.4a** Write tests for a registry-based `rk4` step:
  - Create a `sim_registry`, allocate 4 slots (0=u0, 1=u1, 2=rk_rhs, 3=system_rhs) with matching scalar layout (1 scalar, 0 vectors) matching the heat system.
  - Use the same Lua-based heat system setup as `src/temporal/rk4.t.cpp`:
    - Build `system` from Lua, build `step_controller`, get dt.
    - Fill slot 0 with initial data via `sys.initialize(reg, u0_ref, step)`.
    - Call `sys.update_boundary(reg, u0_ref, step)`.
  - Call the new `rk4::operator()` overload:
    ```cpp
    rk4_integrator(sys, reg, u0_ref, u1_ref, rk_ref, srhs_ref, step, dt);
    ```
  - Verify: `sys.stats(reg, u0_ref, u1_ref, step).stats[0]` ≈ 0 (within 1e-13).
  - Needs custom `main()` with `Kokkos::ScopeGuard` (per D-R9).
  - Register in `src/temporal/CMakeLists.txt` with manual executable target.
  - File: `src/temporal/rk4_v2.t.cpp` (new), `src/temporal/CMakeLists.txt`
  - Test: `ctest --test-dir build -R t-rk4_v2`
  - Must come after: 9.3a (system wrapper must have registry methods for the test to call).

- [x] **9.4b** Implement registry-based `rk4::operator()`:
  - New overload signature:
    ```cpp
    void operator()(system& sys, sim_registry& reg,
                    field_ref u0, field_ref output,
                    field_ref rk_rhs, field_ref system_rhs,
                    const step_controller& ctrl, real dt);
    ```
  - The scratch slots (`rk_rhs`, `system_rhs`) are passed as `field_ref` tokens — the integrator does NOT own them.
  - Implementation mirrors existing `rk4.cpp` logic but uses registry bulk ops and shared slot helpers (see D-R13 in `plans/meta.md`):
    1. `slot_zero(reg, rk_rhs)` and `slot_zero(reg, system_rhs)`.
    2. `reg.deep_copy_slot(output.slot, u0.slot)` — copy u0 → output.
    3. RK4 loop (`i = 0..3`):
       - If `i > 0`: `slot_assign_lc(reg, output, u0, dt * rki[i], system_rhs)`.
       - If `i > 0`: `sys.update_boundary(reg, output, time + dt * rki[i])`.
       - `sys.rhs(reg, output, reg, system_rhs, time + dt * rki[i])` — writes into system_rhs slot. Note: `reg` is passed twice (const ref for input, mutable ref for output); this is safe because input/output are different slots.
       - `slot_accumulate(reg, rk_rhs, dt * rkf[i], system_rhs)`.
    4. `slot_assign_lc(reg, output, u0, 1.0, rk_rhs)` — final: output = u0 + rk_rhs.
    5. `sys.update_boundary(reg, output, time + dt)`.
  - **Slot helpers** — create `src/temporal/slot_ops.hpp` (header-only, shared by rk4 and euler):
    ```cpp
    #pragma once
    #include "fields/field_registry.hpp"
    namespace ccs {

    // Zero all allocated buffers in a slot.
    inline void slot_zero(sim_registry& reg, field_ref ref) {
        assert(ref.n_vectors == 0 && "slot_ops: vector support not yet implemented");
        for (int s = 0; s < ref.n_scalars; ++s) {
            scalar_handle sh{s * sim_registry::layout_type::scalar_stride};
            for (auto bh : sh.all())
                Kokkos::deep_copy(reg.view(ref, bh), 0.0);
        }
    }

    // dst[i] = src[i] + coeff * rhs[i]  for all allocated buffers.
    inline void slot_assign_lc(sim_registry& reg, field_ref dst,
                                field_ref src, real coeff, field_ref rhs) {
        assert(dst.n_vectors == 0 && "slot_ops: vector support not yet implemented");
        for (int s = 0; s < dst.n_scalars; ++s) {
            scalar_handle sh{s * sim_registry::layout_type::scalar_stride};
            for (auto bh : sh.all()) {
                int n = reg.size(dst, bh);
                real* d = reg.data(dst, bh);
                const real* s0 = reg.data(src, bh);
                const real* r = reg.data(rhs, bh);
                for (int i = 0; i < n; ++i) d[i] = s0[i] + coeff * r[i];
            }
        }
    }

    // dst[i] += coeff * src[i]  for all allocated buffers.
    inline void slot_accumulate(sim_registry& reg, field_ref dst,
                                 real coeff, field_ref src) {
        assert(dst.n_vectors == 0 && "slot_ops: vector support not yet implemented");
        for (int s = 0; s < dst.n_scalars; ++s) {
            scalar_handle sh{s * sim_registry::layout_type::scalar_stride};
            for (auto bh : sh.all()) {
                int n = reg.size(dst, bh);
                real* d = reg.data(dst, bh);
                const real* r = reg.data(src, bh);
                for (int i = 0; i < n; ++i) d[i] += coeff * r[i];
            }
        }
    }

    } // namespace ccs
    ```
    All three iterate only over allocated scalars (via `n_scalars`) and assert `n_vectors == 0`. All current systems have `nv=0` in the main field. If vectors are needed later, add analogous loops using `vector_handle::components()` and remove the assertions.
  - Does NOT remove old `rk4::operator()` or the owned `field rk_rhs`/`field system_rhs` members yet.
  - Files: `src/temporal/slot_ops.hpp` (new), `src/temporal/rk4.hpp`, `src/temporal/rk4.cpp`
  - Test: `ctest --test-dir build -R t-rk4_v2`

- [x] **9.4b-fix** Add missing `n_vectors == 0` assertions to `slot_ops.hpp`:
  - The plan specifies that all three helpers (`slot_zero`, `slot_assign_lc`, `slot_accumulate`) must `assert(ref.n_vectors == 0 && "slot_ops: vector support not yet implemented")` as a safety guard. The 9.4b implementation omitted these assertions. Without them, if a system with vectors is used before vector support is added, the helpers would silently skip vector data, producing incorrect numerical results.
  - Add `assert(ref.n_vectors == 0 && "slot_ops: vector support not yet implemented");` as the first line in each of the three functions (using `dst.n_vectors` for the two-ref variants).
  - File: `src/temporal/slot_ops.hpp`
  - Test: `cmake --build build && ctest --test-dir build -R t-rk4_v2`

- [x] **9.4c** Implement registry-based `euler::operator()`:
  - New overload signature:
    ```cpp
    void operator()(system& sys, sim_registry& reg,
                    field_ref u0, field_ref output,
                    field_ref system_rhs,
                    const step_controller& ctrl, real dt);
    ```
  - Simpler than rk4: uses shared helpers from `slot_ops.hpp`:
    1. `slot_zero(reg, system_rhs)`.
    2. `sys.rhs(reg, u0, reg, system_rhs, time)` — evaluate RHS at current time.
    3. `slot_assign_lc(reg, output, u0, dt, system_rhs)` — output = u0 + dt * system_rhs.
    4. `sys.update_boundary(reg, output, time + dt)`.
  - Files: `src/temporal/euler.hpp`, `src/temporal/euler.cpp`
  - Test: `ctest --test-dir build -R t-euler`
  - Must come after: 9.4b (`slot_ops.hpp` is created in that item).

- [x] **9.4d** Adapt `integrator` variant wrapper:
  - New overload:
    ```cpp
    void operator()(system& sys, sim_registry& reg,
                    field_ref u0, field_ref output,
                    const step_controller& ctrl, real dt);
    ```
  - The wrapper must provide scratch `field_ref` tokens to the concrete integrators. Strategy: the `integrator` class allocates additional scratch slots in the registry during the first call (lazy allocation):
    - `rk4` needs 2 scratch slots (rk_rhs + system_rhs).
    - `euler` needs 1 scratch slot (system_rhs).
    - `empty` needs 0.
  - Alternative (simpler): the caller (`simulation_cycle`) pre-allocates all scratch slots and passes them. But then the integrator wrapper's signature must include scratch refs...
  - **Decision**: Use the simpler approach — the integrator wrapper takes additional scratch `field_ref` args:
    ```cpp
    void operator()(system& sys, sim_registry& reg,
                    field_ref u0, field_ref output,
                    field_ref scratch1, field_ref scratch2,
                    const step_controller& ctrl, real dt);
    ```
    - `rk4` uses scratch1 as rk_rhs and scratch2 as system_rhs.
    - `euler` uses scratch1 as system_rhs, ignores scratch2.
    - `empty` ignores both.
  - Dispatch via `std::visit`:
    ```cpp
    std::visit([&](auto&& integ) {
        if constexpr (std::is_same_v<std::decay_t<decltype(integ)>, integrators::rk4>) {
            integ(sys, reg, u0, output, scratch1, scratch2, ctrl, dt);
        } else if constexpr (...euler...) {
            integ(sys, reg, u0, output, scratch1, ctrl, dt);
        } else {
            // empty: no-op
        }
    }, v);
    ```
  - Old `std::function<void(field_span)> operator()(...)` remains until 9.7.
  - Files: `src/temporal/integrator.hpp`, `src/temporal/integrator.cpp`
  - Test: `ctest --test-dir build -L temporal`

### 9.5 — Adapt simulation_cycle

- [x] **9.5a** Add registry setup and initialization to `simulation_cycle::run()`:
  - At the top of `run()`, replace `field u0{sys(controller)}; field u1{u0};` with:
    ```cpp
    sim_registry reg;
    auto sz = sys.size();
    auto& ss = sz.scalar_size;
    int d_sz  = get<si::D>(ss);
    int rx_sz = get<si::Rx>(ss);
    int ry_sz = get<si::Ry>(ss);
    int rz_sz = get<si::Rz>(ss);

    // Initialize with valid slot indices — critical for zero-scalar systems
    // (e.g., hyperbolic_eigenvalues with nscalars=0) where the loop below
    // never runs. bulk ops (deep_copy_slot, swap_slots) assert slot >= 0.
    field_ref u0_ref{0}, u1_ref{1}, rk_ref{2}, srhs_ref{3};
    for (int s = 0; s < sz.nscalars; ++s) {
        u0_ref   = reg.allocate_scalar(0, s, d_sz, rx_sz, ry_sz, rz_sz);
        u1_ref   = reg.allocate_scalar(1, s, d_sz, rx_sz, ry_sz, rz_sz);
        rk_ref   = reg.allocate_scalar(2, s, d_sz, rx_sz, ry_sz, rz_sz);
        srhs_ref = reg.allocate_scalar(3, s, d_sz, rx_sz, ry_sz, rz_sz);
    }
    for (int v = 0; v < sz.nvectors; ++v) {
        u0_ref   = reg.allocate_vector(0, v, d_sz, rx_sz, ry_sz, rz_sz);
        u1_ref   = reg.allocate_vector(1, v, d_sz, rx_sz, ry_sz, rz_sz);
        rk_ref   = reg.allocate_vector(2, v, d_sz, rx_sz, ry_sz, rz_sz);
        srhs_ref = reg.allocate_vector(3, v, d_sz, rx_sz, ry_sz, rz_sz);
    }
    sys.initialize(reg, u0_ref, controller);
    reg.deep_copy_slot(u1_ref.slot, u0_ref.slot);  // u1 = u0
    ```
  - **Allocation semantics**: Each `allocate_scalar` returns `metadata_[slot]` with an incremented `n_scalars`; each `allocate_vector` increments `n_vectors`. After the loops, `u0_ref = {.slot=0, .n_scalars=ns, .n_vectors=nv}`, `u1_ref = {.slot=1, ...}`, etc. For single-scalar systems (heat, scalar_wave), the scalar loop runs once and the vector loop is skipped. For zero-scalar systems (e.g., `hyperbolic_eigenvalues`), neither loop body executes; the refs keep their initial slot indices with `n_scalars=0, n_vectors=0`. Bulk operations (`deep_copy_slot`, `swap_slots`) still receive valid slot indices and gracefully skip zero-extent Views.
  - **Note**: All current systems have `nvectors=0` in their main field (heat: `{1,0}`, scalar_wave: `{1,0}`, eigenvalues: `{0,0}`). The vector allocation loop is included for forward-compatibility but is currently a no-op.
  - **Size extraction**: `system_size::scalar_size` is `scalar<integer>` = `tuple<tuple<int>, tuple<int,int,int>>`. The `si::D`, `si::Rx`, `si::Ry`, `si::Rz` indices from `selector_fwd.hpp` extract the sizes for domain and boundary buffers respectively. This is validated by existing tests (e.g., `src/systems/heat.t.cpp:220`).
  - `simulation_cycle.cpp` needs `#include "fields/field_registry.hpp"` and `#include "fields/selector_fwd.hpp"`.
  - File: `src/simulation/simulation_cycle.cpp`
  - Test: `cmake --build build` (compile check; full test in 9.5b)
  - Must come after: 9.3a, 9.4d.

- [x] **9.5b** Rewrite `simulation_cycle::run()` to use registry for all field operations:
  - **Pre-loop calls** (between initialization and `while`): replace field-based calls with registry equivalents:
    ```cpp
    // OLD: sys.update_boundary(u0, controller);
    // NEW:
    sys.update_boundary(reg, u0_ref, controller);

    // OLD: system_stats stats = sys.stats(u0, u1, controller);
    // NEW:
    system_stats stats = sys.stats(reg, u0_ref, u1_ref, controller);

    // OLD: sys.write(io, u0, controller, .0);
    // NEW:
    sys.write(io, reg, u0_ref, controller, 0.0);
    ```
  - **Inside the `while` loop body**:
    ```cpp
    // OLD: auto dt = sys.timestep_size(u0, controller);
    // NEW:
    auto dt = sys.timestep_size(reg, u0_ref, controller);

    // OLD: u1 = integrate(sys, u0, controller, *dt);
    // NEW:
    integrate(sys, reg, u0_ref, u1_ref, rk_ref, srhs_ref, controller, *dt);

    // OLD: stats = sys.stats(u0, u1, controller);
    // NEW:
    stats = sys.stats(reg, u0_ref, u1_ref, controller);

    // OLD: sys.write(io, u1, controller, *dt);
    // NEW:
    sys.write(io, reg, u1_ref, controller, *dt);
    ```
  - **Replace swap**:
    ```cpp
    // OLD: using std::swap; swap(u0, u1);
    // NEW:
    reg.swap_slots(u0_ref.slot, u1_ref.slot);
    // Note: field_ref tokens still point to slots 0 and 1, but the
    // underlying Views have been swapped. The ref.slot values stay the same.
    ```
  - Post-loop summary remains unchanged (`sys.summary(stats)` doesn't touch fields).
  - `sys.log(stats, controller)` calls are unchanged (no field arguments).
  - File: `src/simulation/simulation_cycle.cpp`
  - Test: `ctest --test-dir build -R t-simulation_cycle`

- [x] **9.5c** Remove `field u0`/`field u1` locals from `simulation_cycle::run()`:
  - After 9.5b, the old `field` locals should be completely unused. Delete them.
  - Verify no `field` or `field_span` or `field_view` types remain in `run()`.
  - File: `src/simulation/simulation_cycle.cpp`
  - Test: `ctest --test-dir build -R t-simulation_cycle`

### 9.6 — Adapt I/O

- [x] **9.6a** Add registry-based `write` method to `system` wrapper:
  - This was already included in 9.3a's list. This item covers the concrete systems' `write` adapters (already in 9.2a-d) and the `field_io`/`field_data` path.
  - The concrete system `write` adapters construct a temporary `field_view` from the registry for passing to `field_io::write()`. No changes needed to `field_io` or `field_data` themselves — they continue to accept `field_view`.
  - Verify: `ctest --test-dir build -R t-simulation_cycle` exercises the full write path.
  - **No changes to `field_io.hpp/cpp` or `field_data.hpp/cpp`** — the bridge happens in the system's `write` adapter.
  - Test: `ctest --test-dir build -R t-simulation_cycle`

### 9.7 — Remove old field_span/field_view pass-by-value paths

**Dependency note**: Old system wrapper methods are still called by old integrator overloads (`rk4.cpp`, `euler.cpp`) and by test files (`rk4.t.cpp`, `euler.t.cpp`, `heat.t.cpp`, `hyperbolic_eigenvalues.t.cpp`). Callers must be removed/migrated before the old methods can be deleted. Items below are ordered to maintain compilation at each step.

- [x] **9.7a** Remove old integrator overloads and update integrator tests:
  - Delete old `void rk4::operator()(system&, const field&, field_span, const step_controller&, real)` from `rk4.hpp`/`rk4.cpp`.
  - Delete `rk4::ensure_size()` and owned `field rk_rhs`/`field system_rhs` members.
  - Delete old `void euler::operator()(system&, const field&, field_span, const step_controller&, real)` from `euler.hpp`/`euler.cpp`.
  - Delete `euler::ensure_size()` and owned `field system_rhs` member.
  - Delete old `void integrators::empty::operator()(system&, const field&, field_span, const step_controller&, real)` and `ensure_size` from `empty_integrator.hpp`/`empty_integrator.cpp`.
  - Delete old `std::function<void(field_span)> integrator::operator()(system&, const field&, const step_controller&, real)` from `integrator.hpp`/`integrator.cpp`.
  - Remove `#include "fields/field.hpp"` from `rk4.hpp`, `euler.hpp`, `empty_integrator.hpp` if no longer needed (they may still need `types.hpp` for `system_size` etc.).
  - **Test files**: `rk4.t.cpp` uses the old overload — delete it (superseded by `rk4_v2.t.cpp`). `euler.t.cpp` uses the old overload — either migrate it to registry-based or delete if covered by simulation_cycle test. Update `src/temporal/CMakeLists.txt`.
  - Files: `src/temporal/rk4.hpp`, `src/temporal/rk4.cpp`, `src/temporal/euler.hpp`, `src/temporal/euler.cpp`, `src/temporal/empty_integrator.hpp`, `src/temporal/empty_integrator.cpp`, `src/temporal/integrator.hpp`, `src/temporal/integrator.cpp`, `src/temporal/rk4.t.cpp`, `src/temporal/euler.t.cpp`, `src/temporal/CMakeLists.txt`
  - Test: `ctest --test-dir build -L temporal`

- [x] **9.7b** Migrate system test files to registry-based methods:
  - `heat.t.cpp`: 3 test cases migrated from old system methods to `sim_registry` + `field_ref` + registry-based methods. Custom `main()` with `Kokkos::ScopeGuard` added.
  - `hyperbolic_eigenvalues.t.cpp`: Migrated from `field f{sys(step)}` and `sys.stats(f,f,step)` to registry-based methods. Custom `main()` with `Kokkos::ScopeGuard` added.
  - `src/systems/CMakeLists.txt`: Replaced `add_unit_test` macro with manual executable definitions using `Catch2::Catch2` (not WithMain) and `Kokkos::kokkos` link.
  - Files: `src/systems/heat.t.cpp`, `src/systems/hyperbolic_eigenvalues.t.cpp`, `src/systems/CMakeLists.txt`
  - Test: `ctest --test-dir build -L systems` — all 3 tests pass

- [x] **9.7c** Remove old methods from `system` wrapper:
  - Delete `std::function<void(field&)> operator()(const step_controller&)`.
  - Delete `std::function<void(field_span)> rhs(field_view, real)`.
  - Delete old `void update_boundary(field_span, real)` overload.
  - Delete old `system_stats stats(const field&, const field&, const step_controller&) const`.
  - Delete old `std::optional<real> timestep_size(const field&, const step_controller&) const`.
  - Delete old `bool write(field_io&, field_view, const step_controller&, real)`.
  - All callers have been removed/migrated in 9.7a and 9.7b.
  - Files: `src/systems/system.hpp`, `src/systems/system.cpp`
  - Test: `ctest --test-dir build`

- [x] **9.7d-i** Remove old methods from stub systems (hyperbolic_eigenvalues, inviscid_vortex, empty):
  - These systems' registry adapters are fully standalone — they do NOT delegate to old methods.
  - Delete 6 old field-based methods from each (operator(), stats, timestep_size, rhs, update_boundary, write) in both .hpp and .cpp.
  - Remove `static_assert(SystemV2<...>)` from their .cpp files.
  - Remove `#include "system_v2.hpp"` from their .cpp files.
  - Remove `#include "fields/field.hpp"` from hyperbolic_eigenvalues.hpp and empty_system.hpp (no longer needed). Keep it in inviscid_vortex.hpp (has `field U` member).
  - Files: `src/systems/hyperbolic_eigenvalues.hpp`, `src/systems/hyperbolic_eigenvalues.cpp`, `src/systems/inviscid_vortex.hpp`, `src/systems/inviscid_vortex.cpp`, `src/systems/empty_system.hpp`, `src/systems/empty_system.cpp`
  - Test: `cmake --build build && ctest --test-dir build -L systems`

- [x] **9.7d-ii** Inline delegation and remove old methods from heat:
  - The registry `rhs` and `update_boundary` adapters currently construct temp `field_view`/`field_span` and delegate to old methods. Inline the old method bodies:
    - `rhs`: extract `scalar_view u` and `scalar_span u_rhs` from registry, then apply: `u_rhs = lap(u, neumann_u); u_rhs *= diffusivity;` plus the manufactured solution source term.
    - `update_boundary`: extract `scalar_span u` from registry, then apply: `u | m.dirichlet(...) = ...;` and neumann BCs.
  - The registry `write` adapter delegates to old `write`. Inline: extract `scalar_view u`, compute error, build `field_view io_view{...}`, call `io.write(...)`.
  - Delete all 6 old field-based method declarations from heat.hpp and implementations from heat.cpp.
  - Remove `static_assert(SystemV2<heat>)` and `#include "system_v2.hpp"` from heat.cpp.
  - Keep `#include "fields/field.hpp"` in heat.hpp — still needed for `field_view` in `write` (constructs `field_view` for `io.write()`).
  - Remove "Registry-based adapter methods (SystemV2)" comments — these are now the primary implementations.
  - Files: `src/systems/heat.hpp`, `src/systems/heat.cpp`
  - Test: `cmake --build build && ctest --test-dir build -R t-heat`

- [x] **9.7d-iii** Inline delegation and remove old methods from scalar_wave:
  - Same pattern as 9.7d-ii. The registry `rhs` and `update_boundary` adapters delegate to old methods.
    - `rhs`: extract `scalar_view u` and `scalar_span u_rhs`, then apply: `du = grad(u); u_rhs = dot(grad_G, du);`.
    - `update_boundary`: extract `scalar_span u`, then apply: `u | m.dirichlet(...) = sol;`.
  - The registry `write` adapter delegates to old `write`. Inline same pattern as heat.
  - Delete all 6 old field-based method declarations from scalar_wave.hpp and implementations from scalar_wave.cpp.
  - Remove `static_assert(SystemV2<scalar_wave>)` and `#include "system_v2.hpp"` from scalar_wave.cpp.
  - Keep `#include "fields/field.hpp"` in scalar_wave.hpp — still needed for `field_view` in `write`.
  - Remove "Registry-based adapter methods (SystemV2)" comments.
  - Files: `src/systems/scalar_wave.hpp`, `src/systems/scalar_wave.cpp`
  - Test: `cmake --build build && ctest --test-dir build -L systems`

- [x] **9.7d-iv** Clean up system_v2.hpp and remaining references:
  - Deleted `system_v2.hpp` and `system_v2.t.cpp` — no production code references the `SystemV2` concept anymore.
  - Removed t-system_v2 test target from `src/systems/CMakeLists.txt`.
  - Files: `src/systems/system_v2.hpp`, `src/systems/system_v2.t.cpp`, `src/systems/CMakeLists.txt`
  - Test: `cmake --build build && ctest --test-dir build` — all relevant tests pass

- [x] **9.7e** Remove invocable constructor/assignment from `field.hpp` if unused:
  - Verified no `std::function<void(field...>` callers remain in source code (only in plan files).
  - Removed `template <std::invocable<field&> F> field(F&&)` constructor and `operator=(F&&)` from `field.hpp`.
  - Removed the invocable assignment test from `field.t.cpp` (lines 163-170 in the "assignment" TEST_CASE).
  - Files: `src/fields/field.hpp`, `src/fields/field.t.cpp`
  - Test: `ctest --test-dir build` — all relevant tests pass (3 pre-existing failures in unrelated mesh/stencils/operators)

---

## Completion Criteria

- `simulation_cycle::run()` uses `field_registry` + `field_ref` exclusively.
- No `std::function<void(field_span)>` remains in the integrator/system chain.
- `swap(u0, u1)` is a View-pointer swap via `registry.swap_slots()`.
- `system::rhs` writes directly into an output slot (no deferred callable).
- `rk4`/`euler` scratch buffers are registry slots, not owned `field` members.
- All existing tests pass.
