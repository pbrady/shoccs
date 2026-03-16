# Phase 14: Kokkos GPU Execution

**Goal:** Switch from `DefaultHostExecutionSpace` to `DefaultExecutionSpace` (CUDA/HIP/SYCL), move field data to device memory, and ensure the full simulation loop runs on GPU.

**Depends on:** Phases 8–13 (all infrastructure on host-parallel Kokkos)

**Read first:**
- `src/kokkos_types.hpp` (execution/memory space aliases)
- `src/fields/field_registry.hpp` (registry using `Kokkos::View<real*>`)
- `src/fields/expr.hpp` (expression templates with `KOKKOS_LAMBDA`)
- `src/fields/selection_desc.hpp` (selection descriptors)
- `src/matrices/block.hpp` (parallel_for over lines)
- `src/matrices/circulant.cpp` (parallel_for over rows)
- `src/matrices/dense.cpp` (NBS stencil)
- `src/matrices/csr.hpp` + `csr.cpp` (sparse boundary)
- `src/io/field_io.cpp` (writes field data — needs host access)

**Test commands:**
```bash
cmake -S . -B build-gpu -G Ninja -DKokkos_ENABLE_CUDA=ON  # or HIP/SYCL
cmake --build build-gpu
ctest --test-dir build-gpu
```

---

## Strategy

This phase is only viable on a machine with GPU hardware and a Kokkos build with device support. The key changes:

1. **Switch execution space**: Change `kokkos_types.hpp` to use `Kokkos::DefaultExecutionSpace` instead of `DefaultHostExecutionSpace`.
2. **Field data on device**: Registry Views are allocated in device memory by default. Host access requires explicit `Kokkos::create_mirror_view` + `Kokkos::deep_copy`.
3. **Matrix coefficients on device**: `dense::v`, `circulant::v`, `csr::w/v/u` must become `Kokkos::View` (currently `std::vector`).
4. **I/O requires host mirrors**: Before writing field data, `deep_copy` from device to host.
5. **Fencing**: Insert `Kokkos::fence()` at synchronization points (after parallel kernels, before host reads).

---

## Items

### 14.1 — Switch execution space

- [ ] **14.1a** Change `src/kokkos_types.hpp`:
  - `using execution_space = Kokkos::DefaultExecutionSpace;` (was `DefaultHostExecutionSpace`).
  - `using memory_space = typename execution_space::memory_space;`
  - `using host_mirror_space = typename Kokkos::View<real*>::host_mirror_space;`
  - File: `src/kokkos_types.hpp`
  - Test: `cmake --build build-gpu` (compile only; tests may fail until remaining items complete)

### 14.2 — Matrix coefficient storage migration

- [ ] **14.2a** Replace `std::vector<real> v` in `dense` with `Kokkos::View<real*>`:
  - Constructor: allocate View, `deep_copy` from host data.
  - `operator()`: access via `v(index)` instead of `v[index]`.
  - File: `src/matrices/dense.hpp`, `src/matrices/dense.cpp`

- [ ] **14.2b** Same for `circulant::v`:
  - File: `src/matrices/circulant.hpp`, `src/matrices/circulant.cpp`

- [ ] **14.2c** Same for `csr::w`, `csr::v`, `csr::u`:
  - File: `src/matrices/csr.hpp`, `src/matrices/csr.cpp`

- [ ] **14.2d** Test: `ctest --test-dir build-gpu -L matrices`

### 14.3 — Host mirrors for I/O

- [ ] **14.3a** Add `Kokkos::deep_copy` from device to host mirror before I/O writes:
  - In `field_data::write`: create host mirror, deep_copy, write from mirror.
  - In `xdmf::write`: same pattern.
  - Add `Kokkos::fence()` before the deep_copy.
  - Files: `src/io/field_data.cpp`, `src/io/xdmf.cpp`
  - Test: `ctest --test-dir build-gpu -R t-field_io -R t-xdmf`

### 14.4 — Host mirrors for MMS/initialization

- [ ] **14.4a** MMS functions (`manufactured_solutions.hpp`) evaluate on the host.
  - After computing MMS values on host, `deep_copy` to device before the time loop.
  - Files: `src/mms/manufactured_solutions.hpp`, `src/systems/heat.cpp`, `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build-gpu -L systems`

### 14.5 — Remove the span bridge (device-only)

- [ ] **14.5a** The span bridge (`extract_scalar_span`) from Phase 8 uses `view.data()` which returns a device pointer on GPU. `std::span` wrapping a device pointer is UB when accessed from host code.
  - Replace span-based operator signatures with `Kokkos::View`-based signatures.
  - `derivative::operator()` takes `Kokkos::View<const real*>` and `Kokkos::View<real*>` instead of `scalar_view`/`scalar_span`.
  - Or: keep `scalar_view`/`scalar_span` as type aliases for `Kokkos::View<const real*>`/`Kokkos::View<real*>`.
  - Files: `src/operators/derivative.hpp`, `src/operators/gradient.hpp`, `src/operators/laplacian.hpp`, `src/fields/scalar.hpp`
  - Test: `ctest --test-dir build-gpu -L operators`

### 14.6 — Full GPU test

- [ ] **14.6a** Run the full test suite on GPU build:
  - `ctest --test-dir build-gpu`
  - All tests pass (with pre-existing FP tolerance failures excluded).
  - The 4 Lua config files run successfully on GPU.

---

## Completion Criteria

- `DefaultExecutionSpace` is GPU (CUDA/HIP/SYCL).
- All field data resides in device memory.
- Matrix coefficients are `Kokkos::View` in device memory.
- I/O uses host mirrors with explicit `deep_copy`.
- All expression templates dispatch `KOKKOS_LAMBDA` on the device.
- `block::operator()` and `circulant::operator()` run parallel on GPU.
- Full test suite passes on GPU build.
