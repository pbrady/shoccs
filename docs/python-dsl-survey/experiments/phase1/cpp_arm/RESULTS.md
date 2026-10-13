# Phase 1, C++ fused-stage arm: compact-table kernels against the live heat RHS graph

Measured 2026-10-13 on the native macOS build (`build-host`: Release, Apple clang 21, Kokkos 5.1.1 OpenMP via llvm-openmp) on the owner's M4, inside the Bash sandbox. Google Benchmark, `--benchmark_min_time=0.3s`, microseconds unless stated. Thread settings: 1 thread (`--kokkos-num-threads=1`), 4 threads (`--kokkos-num-threads=4`, libomp defaults), 4 threads spin (the same with `KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active`). The 4-thread default setting was also run with three repetitions (median reported in its own section) because macOS has no thread affinity and single runs at that setting can land threads on efficiency cores (an earlier single run put the 65^3 graph submit at 1502 us; the repeats and the final runs give 937 to 944 us).

Code: `src/operators/tables_kernels.hpp` (kernels), `src/operators/tables_kernels.t.cpp` (test `t-tables_kernels`), `benchmarks/bench_tables.cpp` (this benchmark). Raw data in this directory: `bench_tables_{t1,t4,t4_spin}.json`, `bench_tables_t4_rep3.json`, `bench_tables_t10_triad.json`, `bench_rhs_cutcell_{t1,t4,t4_spin}.json` (the shipped benchmark's `BM_heat_cut_rhs`, run in the same session); `summarize.py` renders the tables below.

## What was measured

All kernels run over the packed vector V = [D | Rx | Ry | Rz] on the compact tables of `docs/reference/pyshoccs-export.md` Section 4, built from the live laplacian's three derivatives (`build_compact_tables`) and copied to the device by `make_device_tables`, which folds the diffusivity into the taps and the exception values, adds a per-row source flag (fluid D rows away from Dirichlet faces, non-Dirichlet R rows) and drops exception rows on Dirichlet rows (none exist in these cases). The heat source and the Dirichlet boundary values are held fixed for the step benchmarks, so they time kernels and not the manufactured solution.

- graph submit: `heat::submit_rhs_graph` on the shipped graph (the fixture of `bench_rhs_cutcell`), both standalone (`bench_rhs_cutcell`) and in the `bench_tables` process.
- structured RHS: `out = k lap(u) + src` from the tables. Launch forms: two-launch (RangePolicy over D, then RangePolicy over exception rows plus the R rows without one), two-launch MDRange (MDRangePolicy<Rank<3>> over D with tile {2, 4, 64} in 3D and {2, 64, 1} in 2D), one-launch (both index sets in one RangePolicy; no atomics because every row is owned by exactly one thread), pencil two-launch (RangePolicy over the fast-axis lines of D, each walked in runs of equal role bytes so the inner loop has no branch and vectorises), pencil one-launch (pencils and rows in one RangePolicy).
- graph step: one RK4 step built exactly as `integrators::rk4` builds it (slot_zero, deep copy, slot_assign_lc, graph submit, slot_accumulate) with `update_boundary` replaced by a scatter of the fixed boundary vector, 4 submits plus 13 slot launches plus 4 refreshes.
- fused step: four `fused_rk_stage` launches (each stage computes the RHS and performs `acc (+)= w r; u_next = u0 + c r`, the fourth writes `u0 + acc`) plus four `assign_at` Dirichlet refreshes on ping-pong buffers; 8 launches in the one-launch forms, 12 in the two-launch forms.
- STREAM triad: `a = b + s c` over three 134 MB buffers (16.8M doubles each), median of 10 per iteration.

Every table benchmark checks its output once at setup: the structured RHS against the live graph on a random vector (`rhs_check`, max |diff| / max |rhs|: 1.9e-16 to 3.7e-16 over the five cases) and the fused step against the graph step from the same state (`step_check`: 1.1e-16).

## Tables

### 1 thread: RHS (us)

| case | graph (bench_rhs_cutcell) | graph (in process) | two-launch | two-launch MDRange | one-launch | pencil two-launch | pencil one-launch |
|---|---|---|---|---|---|---|---|
| 2D 51^2 | 21.4 | 21.0 | 4.67 | 5.56 | 4.59 | 2.83 | 2.81 |
| 2D 71^2 | 34.7 | 33.3 | 8.95 | 10.3 | 9.23 | 4.88 | 4.84 |
| 2D 201^2 | 223 | 218 | 66.6 | 79.0 | 70.4 | 30.3 | 30.8 |
| 2D 401^2 | 851 | 827 | 257 | 306 | 273 | 113 | 114 |
| 3D 65^3 | 2107 | 2052 | 635 | 554 | 673 | 340 | 350 |

### 1 thread: RK4 step (us)

| case | graph step | fused two-launch | fused one-launch | fused pencil two-launch | fused pencil one-launch |
|---|---|---|---|---|---|
| 2D 51^2 | 93.7 | 27.4 | 28.5 | 20.2 | 20.1 |
| 2D 71^2 | 149 | 53.8 | 56.9 | 38.3 | 38.6 |
| 2D 201^2 | 948 | 399 | 428 | 282 | 282 |
| 2D 401^2 | 3614 | 1565 | 1679 | 1073 | 1073 |
| 3D 65^3 | 8782 | 3624 | 3821 | 2798 | 2842 |

### 1 thread: ratios against the standalone graph submit

| case | best structured RHS | pencil one-launch RHS | flat one-launch RHS | graph step / 4 submits | best fused step / graph step | best fused step / 4 submits |
|---|---|---|---|---|---|---|
| 2D 51^2 | 0.13x (pencil one-launch) | 0.13x | 0.21x | 1.10x | 0.21x (fused pencil one-launch) | 0.23x |
| 2D 71^2 | 0.14x (pencil one-launch) | 0.14x | 0.27x | 1.07x | 0.26x (fused pencil two-launch) | 0.28x |
| 2D 201^2 | 0.14x (pencil two-launch) | 0.14x | 0.32x | 1.06x | 0.30x (fused pencil two-launch) | 0.32x |
| 2D 401^2 | 0.13x (pencil two-launch) | 0.13x | 0.32x | 1.06x | 0.30x (fused pencil one-launch) | 0.32x |
| 3D 65^3 | 0.16x (pencil two-launch) | 0.17x | 0.32x | 1.04x | 0.32x (fused pencil two-launch) | 0.33x |

### 4 threads: RHS (us)

| case | graph (bench_rhs_cutcell) | graph (in process) | two-launch | two-launch MDRange | one-launch | pencil two-launch | pencil one-launch |
|---|---|---|---|---|---|---|---|
| 2D 51^2 | 304 | 304 | 18.5 | 18.8 | 10.8 | 17.4 | 9.92 |
| 2D 71^2 | 315 | 314 | 21.3 | 22.8 | 12.3 | 19.7 | 11.0 |
| 2D 201^2 | 354 | 352 | 38.1 | 46.5 | 30.3 | 28.7 | 18.7 |
| 2D 401^2 | 566 | 564 | 96.4 | 107 | 90.3 | 53.0 | 41.5 |
| 3D 65^3 | 943 | 937 | 198 | 298 | 194 | 119 | 114 |

### 4 threads: RK4 step (us)

| case | graph step | fused two-launch | fused one-launch | fused pencil two-launch | fused pencil one-launch |
|---|---|---|---|---|---|
| 2D 51^2 | 1486 | 116 | 79.5 | 107 | 75.6 |
| 2D 71^2 | 1541 | 125 | 91.7 | 119 | 88.5 |
| 2D 201^2 | 1731 | 227 | 192 | 196 | 147 |
| 2D 401^2 | 2701 | 564 | 554 | 410 | 372 |
| 3D 65^3 | 7412 | 1110 | 1122 | 908 | 889 |

### 4 threads: ratios against the standalone graph submit

| case | best structured RHS | pencil one-launch RHS | flat one-launch RHS | graph step / 4 submits | best fused step / graph step | best fused step / 4 submits |
|---|---|---|---|---|---|---|
| 2D 51^2 | 0.03x (pencil one-launch) | 0.03x | 0.04x | 1.22x | 0.05x (fused pencil one-launch) | 0.06x |
| 2D 71^2 | 0.03x (pencil one-launch) | 0.03x | 0.04x | 1.22x | 0.06x (fused pencil one-launch) | 0.07x |
| 2D 201^2 | 0.05x (pencil one-launch) | 0.05x | 0.09x | 1.22x | 0.08x (fused pencil one-launch) | 0.10x |
| 2D 401^2 | 0.07x (pencil one-launch) | 0.07x | 0.16x | 1.19x | 0.14x (fused pencil one-launch) | 0.16x |
| 3D 65^3 | 0.12x (pencil one-launch) | 0.12x | 0.21x | 1.96x | 0.12x (fused pencil one-launch) | 0.24x |

(The 65^3 graph step in this single run, 7412 us, is a scheduling outlier; the three-repetition median below is 4449 us, 1.18x of four submits, and the fused step ratio against it is 0.20x.)

### 4 threads spin: RHS (us)

| case | graph (bench_rhs_cutcell) | graph (in process) | two-launch | two-launch MDRange | one-launch | pencil two-launch | pencil one-launch |
|---|---|---|---|---|---|---|---|
| 2D 51^2 | 51.6 | 51.5 | 6.35 | 6.18 | 5.24 | 4.33 | 3.16 |
| 2D 71^2 | 60.9 | 64.6 | 8.71 | 12.1 | 7.62 | 5.74 | 4.81 |
| 2D 201^2 | 220 | 238 | 54.1 | 58.3 | 53.1 | 24.3 | 22.5 |
| 2D 401^2 | 810 | 827 | 194 | 191 | 197 | 76.3 | 72.8 |
| 3D 65^3 | 1647 | 1637 | 468 | 543 | 448 | 266 | 264 |

### 4 threads spin: RK4 step (us)

| case | graph step | fused two-launch | fused one-launch | fused pencil two-launch | fused pencil one-launch |
|---|---|---|---|---|---|
| 2D 51^2 | 266 | 37.8 | 33.4 | 33.7 | 26.7 |
| 2D 71^2 | 321 | 56.5 | 58.1 | 48.1 | 41.2 |
| 2D 201^2 | 1049 | 313 | 316 | 226 | 186 |
| 2D 401^2 | 3394 | 1085 | 1105 | 696 | 694 |
| 3D 65^3 | 7224 | 2509 | 2510 | 1966 | 1996 |

### 4 threads spin: ratios against the standalone graph submit

| case | best structured RHS | pencil one-launch RHS | flat one-launch RHS | graph step / 4 submits | best fused step / graph step | best fused step / 4 submits |
|---|---|---|---|---|---|---|
| 2D 51^2 | 0.06x (pencil one-launch) | 0.06x | 0.10x | 1.29x | 0.10x (fused pencil one-launch) | 0.13x |
| 2D 71^2 | 0.08x (pencil one-launch) | 0.08x | 0.13x | 1.32x | 0.13x (fused pencil one-launch) | 0.17x |
| 2D 201^2 | 0.10x (pencil one-launch) | 0.10x | 0.24x | 1.19x | 0.18x (fused pencil one-launch) | 0.21x |
| 2D 401^2 | 0.09x (pencil one-launch) | 0.09x | 0.24x | 1.05x | 0.20x (fused pencil one-launch) | 0.21x |
| 3D 65^3 | 0.16x (pencil one-launch) | 0.16x | 0.27x | 1.10x | 0.27x (fused pencil two-launch) | 0.30x |

### 4 threads, median of 3 repetitions: RHS (us)

| case | graph (bench_rhs_cutcell) | graph (in process) | two-launch | two-launch MDRange | one-launch | pencil two-launch | pencil one-launch |
|---|---|---|---|---|---|---|---|
| 2D 51^2 | 304 | 306 | 18.8 | 18.8 | 10.7 | 17.2 | 9.92 |
| 2D 71^2 | 315 | 314 | 21.2 | 22.6 | 12.3 | 19.5 | 10.8 |
| 2D 201^2 | 354 | 355 | 38.1 | 45.9 | 30.3 | 28.1 | 18.3 |
| 2D 401^2 | 566 | 571 | 97.9 | 109 | 91.3 | 53.3 | 41.3 |
| 3D 65^3 | 943 | 944 | 198 | 300 | 196 | 120 | 116 |

### 4 threads, median of 3 repetitions: RK4 step (us)

| case | graph step | fused two-launch | fused one-launch | fused pencil two-launch | fused pencil one-launch |
|---|---|---|---|---|---|
| 2D 51^2 | 1495 | 109 | 79.9 | 106 | 75.9 |
| 2D 71^2 | 1539 | 120 | 89.9 | 115 | 88.0 |
| 2D 201^2 | 1727 | 218 | 192 | 185 | 148 |
| 2D 401^2 | 2715 | 565 | 554 | 413 | 376 |
| 3D 65^3 | 4449 | 1107 | 1122 | 914 | 894 |

### STREAM triad, a = b + s c, three buffers of 16.8M doubles (134 MB each)

| threads | median of 10 (ms) | GiB/s (median iteration) | GiB/s (best iteration) |
|---|---|---|---|
| 1 thread | 4.16 | 90.3 | 90.6 |
| 4 threads | 4.14 | 90.7 | 90.8 |
| 4 threads spin | 5.42 | 68.9 | 82.2 |
| 10 threads | 4.24 | 88.6 | 88.9 |

One performance core already saturates the memory system at about 90 GiB/s (3 x 8 B per element, write-allocate traffic not counted); 4 and 10 threads add nothing, and spinning threads cost 10 to 25 percent, presumably through placement on efficiency cores.

## Gate (2) of docs/PYTHON_DSL_SURVEY.md Section 4.2 (RHS against the native RHS graph)

Thresholds: at most 1.0x at 51x51 and at most 0.5x at 65^3 with the sphere, at the thread setting fixed in phase 1(a): 1 thread for the shipped 2D cases (bar 21.0 us in Part IV E, 21.4 us in this session), 4 threads with libomp defaults for the 65^3 RHS submit (bar 961 us in Part IV E, 943 us in this session). Also shown at the other settings. The gate text names the fixed-up C++ RHS graph of phase 1(b) as its bar; that graph does not exist yet, so the shipped graph submit stands in for it here. Changes (1), (2) and (4) can only make the C++ graph faster, so every "met" below is provisional against a bar that will move down.

| threshold | bar (this session) | two-launch | one-launch | pencil (default) | pencil one-launch | met |
|---|---|---|---|---|---|---|
| 51x51, 1 thread, at most 1.0x | 21.4 us | 4.67 us (0.22x) | 4.59 us (0.21x) | 2.83 us (0.13x) | 2.81 us (0.13x) | yes, every form |
| 51x51, 4 threads spin, at most 1.0x | 51.6 us | 6.35 us (0.12x) | 5.24 us (0.10x) | 4.33 us (0.08x) | 3.16 us (0.06x) | yes, every form |
| 65^3, 4 threads, at most 0.5x | 943 us | 198 us (0.21x) | 194 us (0.21x) | 119 us (0.13x) | 114 us (0.12x) | yes, every form |
| 65^3, 1 thread, at most 0.5x | 2107 us | 635 us (0.30x) | 673 us (0.32x) | 340 us (0.16x) | 350 us (0.17x) | yes, every form |
| 65^3, 4 threads spin, at most 0.5x | 1647 us | 468 us (0.28x) | 448 us (0.27x) | 266 us (0.16x) | 264 us (0.16x) | yes, every form |

Both gate (2) thresholds are met by every launch form, including the plain two-launch RangePolicy form the task asked for first; the pencil form roughly halves the time again. The MDRange form of launch 1 (tile {2, 4, 64}) is slower than the flat RangePolicy everywhere except 65^3 at 1 thread (554 against 635 us) and much slower at 4 threads there (298 against 198 us), so the flat form was kept and the pencil form made the default (`launch_config::kind = pencil`). The one-launch forms pay one wake-up instead of two and win at 4 threads with sleeping libomp threads (9.9 against 17.4 us at 51x51); at 1 thread and with spinning threads they are within a few percent of their two-launch twins.

## Gate (3) (fused B11 step against the fixed-up C++ step on heat.lua)

The gate quotes a native fixed-up step estimate of 131 us at 1 thread and between 131 us and the measured 0.266 ms at 4 threads spin, with a 0.5x threshold. The graph step measured here (the shipped RK4 sequence with the source and boundary vectors held fixed, so the MMS evaluation of fill_source and update_boundary is removed, which is the substance of change (3) of the fixed-up list) is 93.7 us at 1 thread and 266 us at 4 threads spin at 51x51, 1.04x to 1.3x of four graph submits.

| setting | graph step (fixed source and boundary) | fused step (pencil one-launch) | ratio | against the 131 us estimate |
|---|---|---|---|---|
| 51x51, 1 thread | 93.7 us | 20.1 us | 0.21x | 0.15x |
| 51x51, 4 threads spin | 266 us | 26.7 us | 0.10x | 0.20x |
| 65^3, 4 threads (median of 3) | 4449 us | 894 us | 0.20x | (estimate 6.0 ms: 0.15x) |
| 65^3, 1 thread | 8782 us | 2842 us (2798 two-launch) | 0.32x | (estimate 12.4 ms: 0.23x) |

The fused step is under 0.5x of both the measured graph-built step and the survey's estimate at every setting. The "against the 131 us estimate" column compares a step without stats against an estimate that includes 26.9 us of stats; against the estimate's 104 us of submits and residuals the 1-thread ratio is 0.19x, which changes no verdict. The gate itself is defined against the fixed-up C++ of phase 1(b), which is not yet built; the graph step here stands in for it and removes only the host MMS work, not the gather-list and full-height row-pointer costs of changes (1), (2) and (4).

## Gate (4) (bandwidth) is not measured here

The 128^3 sphere case is not in this benchmark's case list. For orientation, with the survey's byte count (49 B per D point plus 20 B per exception nonzero: 15.6 MB per stage at 65^3), the fused pencil stage at 4 threads (894 / 4 = 224 us including its Dirichlet refresh) moves 65 GiB/s, 72 percent of the 90.7 GiB/s triad, but at 65^3 the working set is L2-resident, so this is not the DRAM-bound measurement the gate asks for.

## Test outcomes

- `t-tables_kernels` (new, 4 test cases, 188 assertions): passes at 1 and 4 threads. (a) On the two floating-object setups of `tables.t.cpp`, every launch form of `structured_rhs` equals the serial `apply_structured` times k plus the source rule (0 on Dirichlet faces, src on fluid D rows and non-Dirichlet R rows) with worst |diff| / (k |row|_1 max|u| + |src|) at most 1e-14 and exact agreement on rows without entries or source; `fused_rk_stage` equals a hand-written stage (first, two middle and last stage parameters, every launch form, and the last stage without a next buffer) to the same tolerance on acc and u_next. (b) On heat.lua 2D 51x51 and the 3D 33^3 version of the same geometry, every launch form of `structured_rhs` equals the live `heat` RHS graph on a random vector with worst row ratio at most 1e-13 (observed about 2e-16).
- `t-tables`: passes. `t-laplacian`: fails as before this change (pre-existing, untouched).
- In-process checks in the benchmark: `rhs_check` 1.9e-16 to 3.7e-16, `step_check` 1.1e-16 on all five cases.

## Reproduce

```
cmake --build build-host --target bench_tables t-tables_kernels
ctest --test-dir build-host -R t-tables_kernels
build-host/benchmarks/bench_tables --kokkos-num-threads=1 --benchmark_min_time=0.3s \
    --benchmark_out=docs/python-dsl-survey/experiments/phase1/cpp_arm/bench_tables_t1.json --benchmark_out_format=json
KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active build-host/benchmarks/bench_tables --kokkos-num-threads=4 ...
build-host/benchmarks/bench_rhs_cutcell --kokkos-num-threads=1 --benchmark_filter=BM_heat_cut_rhs ...
uv run --no-project python -I docs/python-dsl-survey/experiments/phase1/cpp_arm/summarize.py
```
