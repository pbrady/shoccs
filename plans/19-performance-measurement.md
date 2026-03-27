# Phase 19: Performance Measurement Infrastructure

**Goal:** Add zero-overhead profiling annotations, per-step wall-clock timing, a Google Benchmark suite for critical kernels, and kokkos-tools integration for runtime profiling.

**Depends on:** Phase 17 (Kokkos 5.0), Phase 18 (cleanup)

**Priority:** Backlog — implement before any performance-focused optimization work.

**Read first:**
- `src/simulation/simulation_cycle.cpp` (main time loop — no timing exists)
- `src/temporal/rk4.cpp` (RK4 stages — no profiling regions)
- `src/systems/heat.cpp` (RHS evaluation — no profiling regions)
- `src/operators/derivative.cpp` (matrix chain — no profiling regions)
- `src/matrices/block.hpp` (TeamPolicy kernel — no profiling regions)
- `src/kokkos_types.hpp` (execution space aliases)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build
./build/benchmarks/bench-stencil  # when benchmark suite exists
```

---

## Current State

- **Zero** wall-clock timing anywhere in the codebase
- **Zero** `Kokkos::Profiling::ScopedRegion` annotations
- **Zero** benchmark files
- `Kokkos::fence()` calls exist in 14 files — natural insertion points for timing
- `step_controller` tracks PDE time only, not wall time
- `system_stats` contains error norms only, no timing fields
- Google Benchmark is not in spack.yaml or CMakeLists.txt
- kokkos-tools is not in spack.yaml

---

## Items

### 19.1 — Kokkos profiling region annotations (zero overhead)

Kokkos profiling hooks are always compiled in. When no tool is loaded, the cost is a single null-pointer check per callback site. These annotations enable runtime profiling via `KOKKOS_TOOLS_LIBS=...` without recompilation.

- [x] **19.1a** Add `ScopedRegion` to `simulation_cycle::run()`:
  - `"simulation_cycle::run"` around the entire time loop
  - `"simulation_cycle::integrate"` around the `integrate(...)` call
  - `"simulation_cycle::stats"` around `sys.stats(...)`
  - `"simulation_cycle::write"` around `sys.write(...)`
  - File: `src/simulation/simulation_cycle.cpp`
  - Test: `ctest --test-dir build -R t-simulation_cycle`

- [x] **19.1b** Add `ScopedRegion` to RK4/Euler stages:
  - `"rk4::stage_N"` (N=0..3) around each stage in the RK4 loop
  - `"rk4::rhs"` around `sys.submit_rhs_graph(...)`
  - `"rk4::accumulate"` around `slot_accumulate(...)`
  - `"euler::step"` around the euler body
  - Files: `src/temporal/rk4.cpp`, `src/temporal/euler.cpp`
  - Test: `ctest --test-dir build -L temporal`

- [x] **19.1c** Add `ScopedRegion` to system RHS methods:
  - `"heat::rhs"`, `"heat::update_boundary"`, `"heat::stats"` in heat.cpp
  - Same pattern for scalar_wave.cpp
  - Files: `src/systems/heat.cpp`, `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R "t-heat|t-scalar_wave"`

- [x] **19.1d** Add `ScopedRegion` to operator methods:
  - `"derivative::operator()"` in derivative.cpp
  - `"laplacian::operator()"` in laplacian.cpp
  - `"gradient::operator()"` in gradient.cpp
  - `"block::operator()"` in block.hpp
  - Files: `src/operators/derivative.cpp`, `src/operators/laplacian.cpp`, `src/operators/gradient.cpp`, `src/matrices/block.hpp`
  - Test: `ctest --test-dir build -L operators`

### 19.2 — Per-step wall-clock timing

- [x] **19.2a** Add `Kokkos::Timer` to `simulation_cycle::run()`:
  - Measure wall time per step (around the integrate + stats + write block).
  - Log alongside existing step output: `"time={} step={} dt={} s0={} wall={:.3f}ms"`.
  - Add cumulative wall time at the end of the loop.
  - File: `src/simulation/simulation_cycle.cpp`
  - Test: `ctest --test-dir build -R t-simulation_cycle`

- [x] **19.2b** Optionally add wall time to `system_stats`:
  - Add `real wall_time_s = 0.0;` field to `system_stats` (in `types.hpp`).
  - Populate in `simulation_cycle::run()` after each step.
  - This lets the existing `sys.log(stats, controller)` pipeline report timing.
  - Files: `src/types.hpp`, `src/simulation/simulation_cycle.cpp`
  - Test: `ctest --test-dir build`

### 19.3 — kokkos-tools integration

- [x] **19.3a** Add `kokkos-tools` to spack.yaml:
  - Add `- kokkos-tools` to the specs list.
  - Rebuild devcontainer to install.
  - File: `.devcontainer/spack.yaml`

- [x] **19.3b** Add a convenience script for profiling:
  - Create `scripts/profile.sh` that runs the application with SimpleKernelTimer loaded:
    ```bash
    #!/bin/bash
    KOKKOS_TOOLS_LIBS=$(spack location -i kokkos-tools)/lib/libkp_kernel_timer.so \
        ./build/src/app/shoccs "$@"
    kp_reader *.dat
    ```
  - File: `scripts/profile.sh` (new)

### 19.3.1 — Follow-up fixes from review

- [x] **19.3.1a** Fix hot-loop string allocation in `rk4::stage_N` region name:
  - `"rk4::stage_" + std::to_string(i)` (line 28-29 of `src/temporal/rk4.cpp`) performs a heap allocation every RK4 stage on every time step, violating the zero-overhead goal when no profiling tool is loaded.
  - Replace with a `static constexpr` array of string literals: `{"rk4::stage_0", "rk4::stage_1", "rk4::stage_2", "rk4::stage_3"}` indexed by `i`.
  - Also remove the now-unnecessary `#include <string>`.
  - File: `src/temporal/rk4.cpp`
  - Test: `ctest --test-dir build -L temporal`

### 19.4 — Google Benchmark suite

- [x] **19.4a** Add `google-benchmark` to spack.yaml and CMake:
  - spack.yaml: added `- benchmark` (spack package name for Google Benchmark)
  - Top-level CMakeLists.txt: `option(BUILD_BENCHMARKS)` + `find_package(benchmark REQUIRED)` when ON
  - Created `benchmarks/CMakeLists.txt` with `add_bench()` helper function
  - Files: `.devcontainer/spack.yaml`, `CMakeLists.txt`, `benchmarks/CMakeLists.txt` (new)
  - Note: `benchmark` spack package needs `spack install` (devcontainer rebuild) before `-DBUILD_BENCHMARKS=ON` works

- [x] **19.4a-fix** Fix `add_bench()` to forward extra link libraries via `${ARGN}`:
  - Current helper ignores additional arguments, so `add_bench(bench_stencil shoccs-stencils)` silently drops `shoccs-stencils`.
  - Add `${ARGN}` to the `target_link_libraries` call in `benchmarks/CMakeLists.txt`, matching the pattern used by `add_unit_test` in the top-level `CMakeLists.txt`.
  - File: `benchmarks/CMakeLists.txt`
  - Test: `cmake --build build` (configure with `-DBUILD_BENCHMARKS=ON` once `benchmark` spack package is available)

- [x] **19.4b** Benchmark: stencil apply (circulant convolution):
  - Parameterized by grid size (64, 128, 256, 512, 1024 points per line).
  - Reports: time/iteration, FLOP/s (2*stencil_width FLOPs per output point).
  - Uses 4th-order (p=2, 5-point) centered first-derivative coefficients with `matrix::circulant`.
  - Custom `main()` with `Kokkos::ScopeGuard`; removed `benchmark_main` from `add_bench()` helper.
  - File: `benchmarks/bench_stencil.cpp` (new)
  - Test: `./build/benchmarks/bench_stencil --benchmark_out=results.json`

- [x] **19.4c** Benchmark: block matvec (TeamPolicy over lines):
  - Parameterized by mesh size (16³, 32³, 64³).
  - Reports: time/iteration, effective bandwidth (GB/s), point/line counts.
  - Builds a realistic block matrix: N² teams, N points/line, stride=N², 4th-order stencil with 2-row dense boundaries.
  - 128³ omitted from defaults (too memory-intensive for CI); users can add via `--benchmark_arg`.
  - File: `benchmarks/bench_block.cpp` (new)
  - Test: `./build/benchmarks/bench_block`

- [x] **19.4d** Benchmark: derivative operator (full chain):
  - Full `derivative::operator()` on a cubic N³ mesh with Floating BCs (no embedded objects).
  - Parameterized by mesh size (16³, 32³, 64³) and stencil order (E2=2nd, E4=4th).
  - Reports time/iteration and effective memory bandwidth (GB/s).
  - Links `shoccs-operators` and `shoccs-stencils`.
  - File: `benchmarks/bench_derivative.cpp` (new)
  - Test: `./build/benchmarks/bench_derivative`

- [x] **19.4e** Benchmark: expression template assign/compound-assign:
  - `assign(dst, N, a + b * c)` for various N (1K, 16K, 128K, 1M elements).
  - `plus_assign(dst, N, a * b)` for various N (same sizes).
  - Reports effective memory bandwidth (GB/s).
  - `reduce_max`/`reduce_sum` were removed in Phase 18 as dead code — not benchmarked.
  - File: `benchmarks/bench_expr.cpp` (new)
  - Test: `./build/benchmarks/bench_expr`

- [x] **19.4f** Benchmark: selection descriptor scatter:
  - `assign_selected` with contiguous, strided, and gather descriptors.
  - Parameterized by selection size (1K, 16K, 128K, 1M elements) and pattern.
  - Reports effective memory bandwidth (GB/s).
  - Contiguous: dense range in buffer middle; strided: y-plane-like blocked pattern; gather: strided-sampling index list.
  - File: `benchmarks/bench_selection.cpp` (new)
  - Test: `./build/benchmarks/bench_selection`

- [x] **19.4f-fix** Fix strided selection bandwidth counter for non-perfect-square sizes:
  - `BM_assign_selected_strided` computes `nz = sqrt(n); nx = n/nz; nz = n/nx`, but `nz * nx` can be less than `n` when `n` is not a perfect square (e.g. 128K=131072 yields `nz*nx=131044`).
  - The bandwidth counter uses `n` instead of `desc.count()`, overstating the work by 28 elements for 128K.
  - Fix: use `const int actual = desc.count();` for the `BW(GB/s)` and `points` counters instead of `n`.
  - Also fix the `points` counter in all three benchmarks to use the actual element count from the descriptor for consistency.
  - File: `benchmarks/bench_selection.cpp`

- [x] **19.4g** Benchmark: full RHS evaluation:
  - Heat RHS (laplacian + scale + source + BC) via graph submit.
  - Parameterized by mesh size (16³, 32³, 64³) with E2 stencil.
  - Uses Gaussian MMS with Dirichlet BCs on xmin/xmax, exercising full graph path.
  - Constructs heat system via Lua, builds graph once, benchmarks `submit_rhs_graph()`.
  - Reports time/iteration and effective memory bandwidth (GB/s).
  - Links `shoccs-system` (provides heat, laplacian, derivative, mesh, MMS, BCs).
  - File: `benchmarks/bench_rhs.cpp` (new)
  - Test: `./build/benchmarks/bench_rhs`

### 19.5 — Regression tracking

- [x] **19.5a** Add benchmark comparison script:
  - `scripts/bench_compare.sh`: discovers and runs all bench_* executables, merges JSON output, compares against baseline.
    Supports `--save` to store a new baseline (with timestamped file + `baseline-latest.json` symlink)
    and direct comparison against a stored or specified baseline file.
  - `scripts/bench_compare.py`: standalone comparison tool that matches benchmarks by name,
    reports % change in `real_time`, lists added/removed benchmarks, and exits non-zero on regressions.
    Threshold configurable via `--threshold` (default 10%).
  - Files: `scripts/bench_compare.sh` (new), `scripts/bench_compare.py` (new)

- [x] **19.5b** Store baseline benchmark results:
  - Ran full benchmark suite (37 benchmarks across 6 executables), saved JSON to `benchmarks/baselines/`.
  - `baseline-latest.json` symlink points to most recent snapshot.
  - `README.md` documents usage, comparison workflow, and notes on machine-specificity.
  - Files: `benchmarks/baselines/README.md`, `benchmarks/baselines/baseline-*.json`

---

## Usage Patterns

### Quick profiling run (no recompilation):
```bash
KOKKOS_TOOLS_LIBS=/path/to/libkp_kernel_timer.so ./build/src/app/shoccs config.lua
kp_reader *.dat
```

### Hierarchical call-tree profiling:
```bash
KOKKOS_TOOLS_LIBS=/path/to/libkp_space_time_stack.so ./build/src/app/shoccs config.lua
```

### GPU timeline (NVIDIA):
```bash
KOKKOS_TOOLS_LIBS=/path/to/libkp_nvtx_connector.so nsys profile ./build/src/app/shoccs config.lua
```

### Regression check:
```bash
./build/benchmarks/bench_rhs --benchmark_out=results.json
python3 compare.py benchmarks baseline.json results.json
```

---

## Completion Criteria

- `ScopedRegion` annotations in simulation_cycle, rk4, euler, heat, scalar_wave, derivative, laplacian, gradient, block.
- Per-step wall-clock timing logged in simulation_cycle.
- kokkos-tools loadable at runtime via `KOKKOS_TOOLS_LIBS`.
- Google Benchmark suite with 6 benchmark cases covering stencil, block, derivative, expr, selection, and full RHS.
- Baseline results stored for regression comparison.
- All existing tests pass (profiling annotations add zero overhead).
