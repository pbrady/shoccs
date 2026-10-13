# Phase 1, Python array arm: pyshoccs on the real exports against the native C++ bar

Measured 2026-10-13 on the owner's Apple M4 (hw.ncpu 10, 4 performance cores, 16 GiB), macOS 26.2, the machine of Part IV E, against the native C++ bar of `docs/python-dsl-survey/05-measurements.md` Part IV E. Code: `python/pyshoccs/bench/bench_heat.py` on the real exports `.scratch/export/heat51`, `heat201` and `heat3d65` (the exports the parity protocol passed on, byte-fresh against the exporter). Protocol: per backend and metric the median of 100 timed calls at 51x51, 60 at 201x201 and 20 at 65^3 after 10 (5 at 65^3) warm-ups; every cell was run twice in separate processes (run 1 of every cell first, then run 2) and the lower of the two medians is kept, metric by metric. The four CPU fp64 backends (scipy CSR, structured NumPy, torch eager, torch.compile) ran inside the Bash sandbox at three thread settings: 1 thread (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`), 4 threads (libomp defaults) and 4 threads spin (`KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active`). The MLX CPU backend and the two fp32 GPU backends (torch MPS eager, MLX GPU compiled) need a Metal device, which the sandbox hides, so they ran in one unsandboxed call (`run_matrix.py`, setting `metal`) with one host thread. Load average at the start of every process was 1.0 to 1.7 and nothing else was benchmarking.

A step here is one RK4 step from `u0_V` with the export's `dt`, including the three stage boundary updates, the final `update_boundary(t0 + dt)` and the stats of the new state (what `simulation_cycle` does once per step), so it is comparable with the shipped C++ step. The RHS is `k A u + src(t)` with the Dirichlet rows zeroed, applied to `rand_V`. The correctness columns are the stepped state against the exporter's `u1_V` (relative to max|u1|) and the state after 10 chained steps against the fp64 SciPy path. Raw data: `<case>_<setting>_run<n>.json` and `.log` in this directory; `summarize.py` renders this file.

Python 3.14.3, numpy 2.5.4, scipy 1.18.1, torch 2.14.1, mlx 0.32.3; macOS-26.2-arm64-arm-64bit-Mach-O.
Runs per cell: heat201/1t: [1, 2], heat201/4t: [1, 2], heat201/4t_spin: [1, 2], heat201/metal: [1, 2], heat3d65/1t: [1, 2], heat3d65/4t: [1, 2], heat3d65/4t_spin: [1, 2], heat3d65/metal: [1, 2], heat51/1t: [1, 2], heat51/4t: [1, 2], heat51/4t_spin: [1, 2], heat51/metal: [1, 2].

## Per case: ms per RHS and per step, every backend and thread setting

`RHS / C++ RHS submit` and `step / C++ shipped step` use the bar at the same thread setting; `step / 4 submits` divides the step by four RHS graph submits, the discretisation-only bar. The Metal rows carry two ratios, against the 1-thread and the 4-thread-spin bars, because MLX exposes no CPU thread count and the GPU rows have no matching CPU setting. `ns per D point per step` divides the step by n_D (grid points, solid ones included), as the Part IV E table counts D points. The thread setting does not move the NumPy backends (scipy's CSR matvec and NumPy's ufuncs are single-threaded; the BLAS thread count only touches the two-row dgemv of the hoisted source), so their 4-thread rows repeat the 1-thread numbers to within noise. torch at 4 threads with sleeping libomp threads shows the same wake-up pathology as the Kokkos 4t bar at 51x51 (torch.compile RHS 100 us against 24 us at 1 thread and 20 us with spinning threads); at 65^3 four threads halve the torch.compile RHS (1.44 ms to 0.745 ms default, 0.667 ms spin).

### heat51 (heat.lua 51x51, n_D 2601, n_V 2633, 250 exception rows)

Native C++ bar (Part IV E): RHS graph submit 21 us 1t / 309 us 4t / 54 us 4t_spin; shipped step 0.856 ms 1t / 2.26 ms 4t / 0.266 ms 4t_spin.

| setting | backend | operator ms | RHS ms | step ms | RHS / C++ RHS submit | step / 4 submits | step / C++ shipped step | ns per D point per step | u1 vs C++ | drift 10 steps vs fp64 NumPy |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 thread | scipy CSR (fp64) | 0.0050 | 0.016 | 0.099 | 0.76x | 1.18x | 0.12x | 38 | 1.1e-16 | 0.0e+00 |
| 1 thread | structured NumPy (fp64) | 0.015 | 0.030 | 0.146 | 1.41x | 1.74x | 0.17x | 56 | 1.1e-16 | 1.1e-15 |
| 1 thread | torch eager CPU (fp64) | 0.031 | 0.057 | 0.272 | 2.70x | 3.24x | 0.32x | 105 | 1.1e-16 | 1.4e-15 |
| 1 thread | torch.compile CPU (fp64) | 0.019 | 0.024 | 0.089 | 1.13x | 1.06x | 0.10x | 34 | 1.1e-16 | 1.4e-15 |
| 4 threads | scipy CSR (fp64) | 0.0050 | 0.016 | 0.101 | 0.051x | 0.082x | 0.045x | 39 | 1.1e-16 | 0.0e+00 |
| 4 threads | structured NumPy (fp64) | 0.016 | 0.030 | 0.146 | 0.098x | 0.12x | 0.065x | 56 | 1.1e-16 | 1.1e-15 |
| 4 threads | torch eager CPU (fp64) | 0.033 | 0.058 | 0.277 | 0.19x | 0.22x | 0.12x | 106 | 1.1e-16 | 1.4e-15 |
| 4 threads | torch.compile CPU (fp64) | 0.065 | 0.101 | 0.329 | 0.33x | 0.27x | 0.15x | 127 | 1.1e-16 | 1.4e-15 |
| 4 threads spin | scipy CSR (fp64) | 0.0050 | 0.016 | 0.102 | 0.29x | 0.47x | 0.38x | 39 | 1.1e-16 | 0.0e+00 |
| 4 threads spin | structured NumPy (fp64) | 0.016 | 0.029 | 0.146 | 0.54x | 0.68x | 0.55x | 56 | 1.1e-16 | 1.1e-15 |
| 4 threads spin | torch eager CPU (fp64) | 0.033 | 0.057 | 0.269 | 1.06x | 1.24x | 1.01x | 104 | 1.1e-16 | 1.4e-15 |
| 4 threads spin | torch.compile CPU (fp64) | 0.015 | 0.020 | 0.061 | 0.37x | 0.28x | 0.23x | 23 | 1.1e-16 | 1.4e-15 |
| Metal, 1 host thread (vs 1t / 4t spin) | MLX CPU compiled (fp64 arrays, cos/sin in fp32) | 0.043 | 0.093 | 0.367 | 4.41x / 1.71x | 4.37x / 1.69x | 0.43x / 1.38x | 141 | 1.8e-08 | 6.5e-09 |
| Metal, 1 host thread (vs 1t / 4t spin) | torch MPS eager (fp32) | 0.341 | 0.359 | 1.19 | 17x / 6.63x | 14x / 5.50x | 1.39x / 4.47x | 457 | 7.5e-08 | 6.4e-07 |
| Metal, 1 host thread (vs 1t / 4t spin) | MLX GPU compiled (fp32) | 0.213 | 0.251 | 0.292 | 12x / 4.65x | 3.48x / 1.35x | 0.34x / 1.10x | 112 | 5.9e-08 | 7.6e-07 |

### heat201 (201x201, n_D 40401, n_V 40539, 1030 exception rows)

Native C++ bar (Part IV E): RHS graph submit 217 us 1t / 358 us 4t / 228 us 4t_spin; shipped step 6.82 ms 1t / 4.05 ms 4t / 2.17 ms 4t_spin.

| setting | backend | operator ms | RHS ms | step ms | RHS / C++ RHS submit | step / 4 submits | step / C++ shipped step | ns per D point per step | u1 vs C++ | drift 10 steps vs fp64 NumPy |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 thread | scipy CSR (fp64) | 0.066 | 0.133 | 0.804 | 0.61x | 0.93x | 0.12x | 20 | 1.1e-16 | 0.0e+00 |
| 1 thread | structured NumPy (fp64) | 0.120 | 0.185 | 1.02 | 0.85x | 1.18x | 0.15x | 25 | 1.1e-16 | 2.7e-15 |
| 1 thread | torch eager CPU (fp64) | 0.152 | 0.264 | 1.36 | 1.22x | 1.57x | 0.20x | 34 | 1.1e-16 | 2.1e-15 |
| 1 thread | torch.compile CPU (fp64) | 0.136 | 0.169 | 0.780 | 0.78x | 0.90x | 0.11x | 19 | 1.1e-16 | 2.9e-15 |
| 4 threads | scipy CSR (fp64) | 0.072 | 0.136 | 0.803 | 0.38x | 0.56x | 0.20x | 20 | 1.1e-16 | 0.0e+00 |
| 4 threads | structured NumPy (fp64) | 0.128 | 0.188 | 1.03 | 0.53x | 0.72x | 0.25x | 25 | 1.1e-16 | 2.7e-15 |
| 4 threads | torch eager CPU (fp64) | 0.300 | 0.449 | 2.20 | 1.25x | 1.53x | 0.54x | 54 | 1.1e-16 | 2.1e-15 |
| 4 threads | torch.compile CPU (fp64) | 0.127 | 0.164 | 0.639 | 0.46x | 0.45x | 0.16x | 16 | 1.1e-16 | 2.9e-15 |
| 4 threads spin | scipy CSR (fp64) | 0.072 | 0.136 | 0.801 | 0.60x | 0.88x | 0.37x | 20 | 1.1e-16 | 0.0e+00 |
| 4 threads spin | structured NumPy (fp64) | 0.127 | 0.189 | 1.02 | 0.83x | 1.12x | 0.47x | 25 | 1.1e-16 | 2.7e-15 |
| 4 threads spin | torch eager CPU (fp64) | 0.111 | 0.225 | 1.09 | 0.99x | 1.20x | 0.50x | 27 | 1.1e-16 | 2.1e-15 |
| 4 threads spin | torch.compile CPU (fp64) | 0.054 | 0.076 | 0.338 | 0.34x | 0.37x | 0.16x | 8 | 1.1e-16 | 2.9e-15 |
| Metal, 1 host thread (vs 1t / 4t spin) | MLX CPU compiled (fp64 arrays, cos/sin in fp32) | 0.149 | 0.615 | 2.71 | 2.83x / 2.70x | 3.13x / 2.98x | 0.40x / 1.25x | 67 | 1.1e-08 | 1.8e-08 |
| Metal, 1 host thread (vs 1t / 4t spin) | torch MPS eager (fp32) | 0.541 | 0.449 | 1.46 | 2.07x / 1.97x | 1.68x / 1.60x | 0.21x / 0.67x | 36 | 8.1e-08 | 2.3e-06 |
| Metal, 1 host thread (vs 1t / 4t spin) | MLX GPU compiled (fp32) | 0.246 | 0.270 | 0.441 | 1.24x / 1.18x | 0.51x / 0.48x | 0.065x / 0.20x | 11 | 8.9e-08 | 2.1e-06 |

### heat3d65 (65^3 sphere, n_D 274625, n_V 275211, 24012 exception rows)

Native C++ bar (Part IV E): RHS graph submit 2065 us 1t / 961 us 4t / 1810 us 4t_spin; shipped step 58.4 ms 1t / 20 ms 4t / 17.5 ms 4t_spin.

| setting | backend | operator ms | RHS ms | step ms | RHS / C++ RHS submit | step / 4 submits | step / C++ shipped step | ns per D point per step | u1 vs C++ | drift 10 steps vs fp64 NumPy |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 thread | scipy CSR (fp64) | 0.624 | 1.03 | 6.09 | 0.50x | 0.74x | 0.10x | 22 | 1.1e-16 | 0.0e+00 |
| 1 thread | structured NumPy (fp64) | 1.28 | 1.74 | 8.99 | 0.84x | 1.09x | 0.15x | 33 | 1.1e-16 | 1.0e-14 |
| 1 thread | torch eager CPU (fp64) | 1.99 | 2.29 | 12.11 | 1.11x | 1.47x | 0.21x | 44 | 1.1e-16 | 1.1e-14 |
| 1 thread | torch.compile CPU (fp64) | 1.21 | 1.44 | 7.14 | 0.70x | 0.86x | 0.12x | 26 | 1.1e-16 | 1.1e-14 |
| 4 threads | scipy CSR (fp64) | 0.640 | 1.06 | 6.04 | 1.10x | 1.57x | 0.30x | 22 | 1.1e-16 | 0.0e+00 |
| 4 threads | structured NumPy (fp64) | 1.31 | 1.77 | 8.88 | 1.84x | 2.31x | 0.44x | 32 | 1.1e-16 | 1.0e-14 |
| 4 threads | torch eager CPU (fp64) | 1.23 | 1.96 | 9.80 | 2.04x | 2.55x | 0.49x | 36 | 1.1e-16 | 1.1e-14 |
| 4 threads | torch.compile CPU (fp64) | 0.540 | 0.745 | 3.74 | 0.78x | 0.97x | 0.19x | 14 | 1.1e-16 | 1.1e-14 |
| 4 threads spin | scipy CSR (fp64) | 0.646 | 1.06 | 6.13 | 0.59x | 0.85x | 0.35x | 22 | 1.1e-16 | 0.0e+00 |
| 4 threads spin | structured NumPy (fp64) | 1.32 | 1.78 | 9.16 | 0.98x | 1.26x | 0.52x | 33 | 1.1e-16 | 1.0e-14 |
| 4 threads spin | torch eager CPU (fp64) | 0.942 | 1.64 | 8.40 | 0.90x | 1.16x | 0.48x | 31 | 1.1e-16 | 1.1e-14 |
| 4 threads spin | torch.compile CPU (fp64) | 0.485 | 0.667 | 3.54 | 0.37x | 0.49x | 0.20x | 13 | 1.1e-16 | 1.1e-14 |
| Metal, 1 host thread (vs 1t / 4t spin) | MLX CPU compiled (fp64 arrays, cos/sin in fp32) | 2.94 | 6.20 | 28.29 | 3.00x / 3.43x | 3.42x / 3.91x | 0.48x / 1.62x | 103 | 4.5e-09 | 1.5e-08 |
| Metal, 1 host thread (vs 1t / 4t spin) | torch MPS eager (fp32) | 1.37 | 1.76 | 6.98 | 0.85x / 0.97x | 0.85x / 0.96x | 0.12x / 0.40x | 25 | 1.0e-07 | 6.9e-06 |
| Metal, 1 host thread (vs 1t / 4t spin) | MLX GPU compiled (fp32) | 0.496 | 0.667 | 2.09 | 0.32x / 0.37x | 0.25x / 0.29x | 0.036x / 0.12x | 8 | 1.1e-07 | 6.4e-06 |

## Gate (2): RHS against the native RHS graph submit

Thresholds from `docs/PYTHON_DSL_SURVEY.md` Section 4.2: the Python RHS at most 1.0x of the native RHS graph submit at 51x51 and at most 0.5x at 65^3 with the sphere, at the thread setting fixed in phase 1(a): 1 thread for the shipped 2D case (bar 21.0 us) and 4 threads with libomp defaults for 65^3 (bar 961 us). The other settings are shown for orientation. The gate text names the fixed-up C++ RHS graph of phase 1(b) as its bar; that graph does not exist yet, so the shipped graph submit stands in for it. Changes (1), (2) and (4) can only make the C++ graph faster, so a "met" below is provisional against a bar that will move down, while a "not met" stands.

Verdict: met at 51x51 (scipy CSR 15.9 us, 0.76x; it is the only backend under 1.0x at 1 thread, torch.compile is 1.13x and structured NumPy 1.41x); not met at 65^3 as the gate is defined (best 4-thread-default RHS torch.compile 745 us, 0.78x of 961 us; scipy CSR 1.10x). At 65^3 the 0.5x threshold is reached only against the other two bars: scipy CSR at 1 thread sits exactly on it (1032 us, 0.50x of 2065 us; the second run gave 1078 us, 0.52x, so this cell is inside run-to-run noise and is not a pass) and torch.compile with spinning threads is under it (667 us, 0.37x of 1810 us). The C++ fused-stage arm (`docs/python-dsl-survey/experiments/phase1/cpp_arm/RESULTS.md`) is the other reference point: its pencil one-launch RHS is 2.81 us at 51x51 1t and 114 us at 65^3 4t, so the best Python RHS is 5.7x and 6.5x slower than the compact-table C++ kernel at the gate settings (3.0x at 65^3 1t: 1032 against 340 us), and the best Python step is 4.4x slower than the fused pencil step at 51x51 1t (89 against 20.1 us), 2.3x at 51x51 4t spin (61 against 26.7 us), 4.2x at 65^3 4t (3.74 against 0.894 ms) and 2.2x at 65^3 1t (6.09 against 2.80 ms).

| threshold | bar | scipy CSR | structured NumPy | torch eager | torch.compile | best | met |
|---|---|---|---|---|---|---|---|
| 51x51, 1 thread, at most 1.0x | 21 us | 15.9 us (0.76x) | 29.7 us (1.41x) | 56.8 us (2.70x) | 23.8 us (1.13x) | scipy CSR (fp64) 0.76x | yes |
| 51x51, 4 threads spin, at most 1.0x | 54 us | 15.8 us (0.29x) | 29.3 us (0.54x) | 57.2 us (1.06x) | 19.8 us (0.37x) | scipy CSR (fp64) 0.29x | yes |
| 51x51, 4 threads default, at most 1.0x | 309 us | 15.8 us (0.051x) | 30.4 us (0.098x) | 57.8 us (0.19x) | 100.5 us (0.33x) | scipy CSR (fp64) 0.051x | yes |
| 65^3, 4 threads default, at most 0.5x | 961 us | 1058.5 us (1.10x) | 1769.6 us (1.84x) | 1959.4 us (2.04x) | 745.0 us (0.78x) | torch.compile CPU (fp64) 0.78x | no |
| 65^3, 1 thread, at most 0.5x | 2065 us | 1031.9 us (0.50x) | 1738.5 us (0.84x) | 2287.0 us (1.11x) | 1436.6 us (0.70x) | scipy CSR (fp64) 0.50x | yes |
| 65^3, 4 threads spin, at most 0.5x | 1810 us | 1062.2 us (0.59x) | 1778.7 us (0.98x) | 1636.4 us (0.90x) | 667.0 us (0.37x) | torch.compile CPU (fp64) 0.37x | yes |

## Best Python step, ns per point

- heat51 1 thread: best step 0.089 ms (torch.compile CPU (fp64)), 34 ns per D point per step; C++ shipped step 0.856 ms, 329 ns per D point (0.10x).
- heat51 4 threads: best step 0.101 ms (scipy CSR (fp64)), 39 ns per D point per step; C++ shipped step 2.26 ms, 869 ns per D point (0.045x).
- heat51 4 threads spin: best step 0.061 ms (torch.compile CPU (fp64)), 23 ns per D point per step; C++ shipped step 0.266 ms, 102 ns per D point (0.23x).
- heat201 1 thread: best step 0.780 ms (torch.compile CPU (fp64)), 19 ns per D point per step; C++ shipped step 6.82 ms, 169 ns per D point (0.11x).
- heat201 4 threads: best step 0.639 ms (torch.compile CPU (fp64)), 16 ns per D point per step; C++ shipped step 4.05 ms, 100 ns per D point (0.16x).
- heat201 4 threads spin: best step 0.338 ms (torch.compile CPU (fp64)), 8 ns per D point per step; C++ shipped step 2.17 ms, 54 ns per D point (0.16x).
- heat3d65 1 thread: best step 6.09 ms (scipy CSR (fp64)), 22 ns per D point per step; C++ shipped step 58.4 ms, 213 ns per D point (0.10x).
- heat3d65 4 threads: best step 3.74 ms (torch.compile CPU (fp64)), 14 ns per D point per step; C++ shipped step 19.96 ms, 73 ns per D point (0.19x).
- heat3d65 4 threads spin: best step 3.54 ms (torch.compile CPU (fp64)), 13 ns per D point per step; C++ shipped step 17.49 ms, 64 ns per D point (0.20x).

## Step decomposition of the structured NumPy backend beside the C++ decomposition

The structured NumPy backend is the one whose step can be taken apart on the same buffers: operator (4x), source (4x, the hoisted manufactured solution: one small dgemv per call), RHS glue (4x: the scale by k, the scatter-add of the source onto the fluid and R rows and the zero fill of the Dirichlet rows), boundary update (4x), stats (1x) and the stage arithmetic measured on its own (`stage_axpy` in `bench_heat.py`: the two fills, the copy and the seven multiply or add passes of `rk4_step_generic`). The C++ column is Part IV E's decomposition (4 x submit, 4 x fill_source, 4 x update_boundary, stats) with its remainder (slot ops, graph launch, logging). The Python remainder is a few percent at 201x201 and 65^3 and within noise at 51x51 (the parts are timed separately, so cache state differs). Reading the 65^3 column: the operator is 57 percent of the step and the RHS glue 19 percent, so the two passes that a fused kernel removes (the separate scale and the scatter of the source) cost as much as the stats and the stage arithmetic together; the source and boundary evaluations, which dominate the shipped C++ step (fill_source is 55 percent of it at 1 thread), are 1 to 2 percent here because the Gaussian's spatial factors are hoisted. The scipy CSR backend has no such split (one matvec over V), which is why its RHS is the cheapest fp64 one at every size at 1 thread: a single pass over 1.5 to 3.4 million stored entries against the structured form's five passes over the padded D block plus the exception CSR.

### heat51, 1 thread

| component | Python structured 1 thread ms | share | C++ 1t ms (Part IV E) | Python / C++ |
|---|---|---|---|---|
| operator (4x) | 0.059 | 41% | 0.084 | 0.71x |
| source (4x) | 0.014 | 9% | 0.239 | 0.058x |
| RHS glue: scale, scatter-add, zero fill (4x) | 0.046 | 31% | (in the step remainder) |  |
| boundary update (4x) | 0.0033 | 2% | 0.098 | 0.034x |
| stats (1x) | 0.019 | 13% | 0.027 | 0.70x |
| stage axpy and fills (1x) | 0.0075 | 5% | (in the step remainder) |  |
| sum of parts | 0.148 | 102% | 0.448 (RHS, source, boundary, stats) | 0.33x |
| measured step (with stats) | 0.146 | 100% | 0.856 (shipped step) | 0.17x |
| remainder (Python call overhead, ufunc dispatch) | -0.0022 | -2% | 0.408 (slot ops, graph launch, logging) |  |
| step without stats | 0.132 | 90% |  |  |

### heat51, 4 threads

| component | Python structured 4 threads ms | share | C++ 4t ms (Part IV E) | Python / C++ |
|---|---|---|---|---|
| operator (4x) | 0.063 | 43% | 1.24 | 0.051x |
| source (4x) | 0.014 | 9% | 0.182 | 0.075x |
| RHS glue: scale, scatter-add, zero fill (4x) | 0.045 | 31% | (in the step remainder) |  |
| boundary update (4x) | 0.0033 | 2% | 0.359 | 0.009x |
| stats (1x) | 0.019 | 13% | 0.086 | 0.22x |
| stage axpy and fills (1x) | 0.0078 | 5% | (in the step remainder) |  |
| sum of parts | 0.152 | 103% | 1.86 (RHS, source, boundary, stats) | 0.081x |
| measured step (with stats) | 0.146 | 100% | 2.26 (shipped step) | 0.065x |
| remainder (Python call overhead, ufunc dispatch) | -0.0050 | -3% | 0.397 (slot ops, graph launch, logging) |  |
| step without stats | 0.131 | 90% |  |  |

### heat201, 1 thread

| component | Python structured 1 thread ms | share | C++ 1t ms (Part IV E) | Python / C++ |
|---|---|---|---|---|
| operator (4x) | 0.479 | 47% | 0.868 | 0.55x |
| source (4x) | 0.027 | 3% | 3.69 | 0.007x |
| RHS glue: scale, scatter-add, zero fill (4x) | 0.234 | 23% | (in the step remainder) |  |
| boundary update (4x) | 0.0095 | 1% | 1.40 | 0.007x |
| stats (1x) | 0.131 | 13% | 0.377 | 0.35x |
| stage axpy and fills (1x) | 0.092 | 9% | (in the step remainder) |  |
| sum of parts | 0.973 | 95% | 6.34 (RHS, source, boundary, stats) | 0.15x |
| measured step (with stats) | 1.02 | 100% | 6.82 (shipped step) | 0.15x |
| remainder (Python call overhead, ufunc dispatch) | 0.049 | 5% | 0.483 (slot ops, graph launch, logging) |  |
| step without stats | 0.903 | 88% |  |  |

### heat201, 4 threads

| component | Python structured 4 threads ms | share | C++ 4t ms (Part IV E) | Python / C++ |
|---|---|---|---|---|
| operator (4x) | 0.511 | 50% | 1.43 | 0.36x |
| source (4x) | 0.027 | 3% | 1.18 | 0.023x |
| RHS glue: scale, scatter-add, zero fill (4x) | 0.215 | 21% | (in the step remainder) |  |
| boundary update (4x) | 0.0095 | 1% | 0.736 | 0.013x |
| stats (1x) | 0.132 | 13% | 0.191 | 0.69x |
| stage axpy and fills (1x) | 0.098 | 10% | (in the step remainder) |  |
| sum of parts | 0.993 | 97% | 3.54 (RHS, source, boundary, stats) | 0.28x |
| measured step (with stats) | 1.03 | 100% | 4.05 (shipped step) | 0.25x |
| remainder (Python call overhead, ufunc dispatch) | 0.033 | 3% | 0.507 (slot ops, graph launch, logging) |  |
| step without stats | 0.907 | 88% |  |  |

### heat3d65, 1 thread

| component | Python structured 1 thread ms | share | C++ 1t ms (Part IV E) | Python / C++ |
|---|---|---|---|---|
| operator (4x) | 5.10 | 57% | 8.26 | 0.62x |
| source (4x) | 0.105 | 1% | 31.94 | 0.003x |
| RHS glue: scale, scatter-add, zero fill (4x) | 1.75 | 19% | (in the step remainder) |  |
| boundary update (4x) | 0.106 | 1% | 11.25 | 0.009x |
| stats (1x) | 0.783 | 9% | 2.96 | 0.26x |
| stage axpy and fills (1x) | 0.671 | 7% | (in the step remainder) |  |
| sum of parts | 8.51 | 95% | 54.41 (RHS, source, boundary, stats) | 0.16x |
| measured step (with stats) | 8.99 | 100% | 58.40 (shipped step) | 0.15x |
| remainder (Python call overhead, ufunc dispatch) | 0.476 | 5% | 3.99 (slot ops, graph launch, logging) |  |
| step without stats | 7.91 | 88% |  |  |

### heat3d65, 4 threads

| component | Python structured 4 threads ms | share | C++ 4t ms (Part IV E) | Python / C++ |
|---|---|---|---|---|
| operator (4x) | 5.23 | 59% | 3.84 | 1.36x |
| source (4x) | 0.104 | 1% | 9.72 | 0.011x |
| RHS glue: scale, scatter-add, zero fill (4x) | 1.75 | 20% | (in the step remainder) |  |
| boundary update (4x) | 0.107 | 1% | 3.62 | 0.030x |
| stats (1x) | 0.795 | 9% | 0.955 | 0.83x |
| stage axpy and fills (1x) | 0.694 | 8% | (in the step remainder) |  |
| sum of parts | 8.67 | 98% | 18.14 (RHS, source, boundary, stats) | 0.48x |
| measured step (with stats) | 8.88 | 100% | 19.96 (shipped step) | 0.44x |
| remainder (Python call overhead, ufunc dispatch) | 0.208 | 2% | 1.82 (slot ops, graph launch, logging) |  |
| step without stats | 7.94 | 89% |  |  |

## GPU (fp32) rows

Single precision on the Apple GPU, same structured tables, run outside the sandbox (the Metal rows of the tables above). MLX GPU compiled: step 0.292 ms at 51x51 (112 ns per D point; 1.10x of the 4-thread-spin shipped step, 0.34x of the 1-thread one), 0.441 ms at 201x201 (11 ns per point, 0.20x of 2.17 ms) and 2.09 ms at 65^3 (8 ns per point, 0.12x of 17.49 ms and 0.036x of 58.4 ms): the fastest step measured at 65^3 in either precision, 0.59x of the best CPU fp64 step (torch.compile 3.54 ms at 4 threads spin). Its RHS at 65^3 is 667 us, 0.69x of the 4-thread-default bar, 0.37x of 4t spin and 0.32x of 1t; at 51x51 the step costs little more than one RHS (0.29 against 0.25 ms) because a fixed per-eval floor of about 0.2 ms dominates. torch MPS eager is launch-bound: 1.19 / 1.46 / 6.98 ms per step at the three sizes, no better than the CPU paths. Precision: the stepped state differs from the C++ fp64 `u1_V` by 6e-8 to 1.1e-7 (relative to max|u1|) and after 10 chained steps from the fp64 NumPy run by 7.6e-7 (51x51), 2.1e-6 (201x201) and 6.4e-6 (65^3) for MLX GPU and 6.4e-7 / 2.3e-6 / 6.9e-6 for torch MPS, where every fp64 CPU path is at 1e-15 to 1e-14. The stats of each step are computed on the host from two device-to-host copies of the source-point values, as the C++ step computes them each step.

## Backend failures and skips

none

No backend failed on any case. Inside the Bash sandbox MLX and torch MPS are unavailable (`import mlx.core` raises `[metal::load_device] No Metal device available`, even for the CPU stream; `torch.backends.mps.is_available()` is False), so the script records them as skipped there (`smoke_heat51.json`) and the Metal cells were run outside it. MLX's CPU `cos`, `sin` and `exp` on float64 arrays are evaluated in single precision (`mlx_f64_probe.py`: relative error 7e-8 to 1.2e-7, every other op of the step exact), which is why the MLX CPU row agrees with `u1_V` only to 1.8e-8 although its arrays are float64; only the MMS time factors go through those functions here. The MLX CPU stream is also slow on this workload (28 ms per step at 65^3, 4.7x the scipy step), so it was not pursued further. torch.compile's first step costs 0.5 s per process with a warm inductor cache (2.2 to 2.5 s when the 4-thread variant compiles afresh); MLX compile 0.8 to 6 s in the first process of a case.

## Reproduce

```
cd python/pyshoccs && uv sync --extra bench
python3 -I docs/python-dsl-survey/experiments/phase1/py_arm/run_matrix.py cpu                                   # sandboxed: 3 cases x (1t, 4t, 4t_spin) x 2 runs
python3 -I docs/python-dsl-survey/experiments/phase1/py_arm/run_matrix.py mlx,gpu heat51,heat201,heat3d65 metal 1,2   # outside the sandbox
uv run --project python/pyshoccs python docs/python-dsl-survey/experiments/phase1/py_arm/mlx_f64_probe.py       # outside the sandbox
python3 -I docs/python-dsl-survey/experiments/phase1/py_arm/summarize.py > docs/python-dsl-survey/experiments/phase1/py_arm/RESULTS.md
```
