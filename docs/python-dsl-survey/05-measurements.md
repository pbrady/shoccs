# Part IV: Fresh C++ measurements (2026-10-12)

This is Part IV of the report. It holds the measurements of the current C++ solver made for this study on 2026-10-12. Where a matching case exists they replace the March 2026 baseline that Part I Section 6 used before. The C++ was measured inside the project devcontainer because no native macOS build exists: the C++ dependency stack (Kokkos 5.1.1, sol2, spdlog, lapackpp and the rest) lives only in the container's Spack view. A Linux container cannot reach the Apple GPU, so every Apple GPU (MPS or Metal) measurement has to run natively on the macOS host. Those runs go in Section D. This study added one new file to the code base: `benchmarks/bench_rhs_cutcell.cpp`, with a one-line `add_bench(bench_rhs_cutcell shoccs-system)` entry in `benchmarks/CMakeLists.txt`. It is the only benchmark that exercises cut cells and the per-step host work. Raw results and run scripts are in `docs/python-dsl-survey/experiments/container/`:

| File | Content |
|---|---|
| `run_bench.sh` | Runs the five original micro-benchmarks (8 threads, OMP_PROC_BIND unset) |
| `run_cutcell.sh` | Builds and runs `bench_rhs_cutcell` at 1 and 8 threads |
| `run_e2e.sh` | End-to-end runs of the shipped configs and larger grids, io off, 1 and 8 threads |
| `bench_expr.json`, `bench_selection.json`, `bench_block.json`, `bench_derivative.json`, `bench_rhs.json` | Section B, 8 threads |
| `micro-t1-linux.log` | Section B, 1-thread column (bench_expr, bench_selection and bench_rhs with OMP_NUM_THREADS=1) |
| `bench_rhs_cutcell_t1.json`, `bench_rhs_cutcell_t8.json` | Section C, 1 and 8 threads |
| `e2e_steps_summary.json` | Section A (median, mean, min and p90 per case) |

The 1-thread figures have their sources under `experiments/container/`: `bench_rhs_cutcell_t1.json` for Section C, and `micro-t1-linux.log` plus the JSON forms `bench_expr_t1.json`, `bench_selection_t1.json` and `bench_rhs_t1.json` for Section B. The host-side Python runs of Section D live one level up, in `experiments/host_primitives_results.md`, `host_primitives_results.json`, `host_primitives_run1.json`, `host_primitives_run2.json` and the script `host_primitives_bench.py`.

Provenance: the solver and benchmarks were built from this worktree inside the project devcontainer image (shoccs-devcontainer:latest, 2026-07-30, Debian trixie, gcc 14, Kokkos 5.1.1 with Serial+OpenMP, Release, -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=ON). The host was Docker Desktop on the owner's Apple-silicon Mac, with 8 vCPU and 12 GB. The binaries link libkokkoscore.so.5.1 and libgomp.so.1, so the default host execution space is Kokkos::OpenMP. Other raw logs: `.scratch/bench-linux-default.log`, `.scratch/e2e-linux.log`, `.scratch/e2e/*/logs/system.csv`, `.scratch/cutcell-linux.log`.

## A. End-to-end RK4 step, shipped configs and larger grids

Each case ran as `shoccs <config> --script '...'` with `simulation.io = nil`. The figure is the Wall_ms column of logs/system.csv, the median over all steps after step 0. Wall_ms is timed around integrate, stats and write (`src/simulation/simulation_cycle.cpp:87-110`). With io off, `heat::write` still evaluates the manufactured solution at every point and computes the error field (`src/systems/heat.cpp:404-424`); only the file output is skipped. 1 thread means OMP_NUM_THREADS=1. 8 threads means OMP_NUM_THREADS=8 with OMP_PROC_BIND=spread.

| case | config | D points | steps | 1 thread median ms | 8 threads median ms |
|---|---|---|---|---|---|
| heat51 | heat.lua as shipped (51x51, one floating sphere, E2, RK4, Gaussian MMS) | 2601 | 50 | 0.900 | 0.935 |
| wave71 | scalar_wave.lua as shipped (71x71, two spheres, E2-poly, RK4), max_step 60 | 5041 | 60 | 0.434 | 0.675 |
| heat201 | heat.lua with index_extents {201,201}, max_time 0.03 | 40401 | 80 | 12.18 | 4.63 |
| heat401 | heat.lua with index_extents {401,401}, max_time 0.006 | 160801 | 64 | 47.99 | 13.69 |
| heat3d65 | heat.lua in 3D: {65,65,65}, domain {2,2,2}, sphere centre (16/17, 25/22, 1), max_time 0.04 | 274625 | 11 | 148.9 | 37.9 |

Derived: heat51 costs about 0.9 ms per step at either thread count. It is bound by launches and host work, so extra threads do not help, and they slow wave71 down. At 201x201 and above, 8 threads are 2.6x to 3.9x faster than 1. Per point per step at 8 threads: heat201 115 ns, heat401 85 ns, heat3d65 138 ns. Each step is 4 RHS evaluations, 4 fill_source and 4 update_boundary host MMS evaluations, stats, write (error field only) and slot copies.

## B. Micro-benchmarks (Google Benchmark, --benchmark_min_time=0.3s)

These are the same programs as the March 2026 baseline. The 8-thread column ran with OMP_PROC_BIND unset. The 1-thread column ran with OMP_NUM_THREADS=1. Small sizes on 8 unpinned threads are slower than the baseline figures the catalog quoted (0.73 us per 1024-element kernel, 32 us per 16^3 RHS). The 1-thread run gives the launch floor: 0.55 us per tiny kernel.

| benchmark | case | 8 threads | 1 thread | notes |
|---|---|---|---|---|
| bench_expr BM_assign_fma | 1024 / 16384 / 131072 / 1048576 | 1.94 us / 3.57 us / 20.0 us / 385 us | 0.551 us / 6.73 us / 67.9 us / 463 us | 1M: 0.37 ns per element and 97 GiB/s effective at 8 threads; 0.44 ns per element and 67 GiB/s at 1 thread |
| bench_expr BM_plus_assign_mul | 1024 / 1048576 | 2.25 us / 296 us | 0.545 us / 399 us | |
| bench_selection contiguous | 1024 / 1048576 | 1.93 us / 205 us | 0.506 us / 265 us | |
| bench_selection strided | 1024 / 1048576 | 2.31 us / 432 us | 0.777 us / 536 us | |
| bench_selection gather | 1024 / 16384 / 131072 / 1048576 | 8.84 us / 46.3 us / 534 us / 508 us | 29.1 us / 130 us / 2905 us / 440 us | gather is 4 to 25x the contiguous cost; 131072 is anomalous in both runs |
| bench_block BM_block_matvec | 16^3 / 32^3 / 64^3 | 4 us / 109 us / 2.32 ms | not run | 64^3: 8.9 ns per point, 6.2 GiB/s effective; 21x the 32^3 time for 8x the points |
| bench_derivative E2 | 16^3 / 32^3 / 64^3 | 56 us / 101 us / 1.74 ms | not run | |
| bench_derivative E4 | 16^3 / 32^3 / 64^3 | 20 us / 132 us / 1.81 ms | not run | |
| bench_rhs BM_heat_rhs (no shapes, graph submit) | 16^3 / 32^3 / 64^3 | 84 us / 169 us / 1.16 ms | 40 us / 418 us / 4.51 ms | 64^3: 4.4 ns per point at 8 threads, 17.2 at 1 thread |

Cross-check against the March baseline (Part I Section 6.2): the baseline 16^3 RHS (32 us) sits between the fresh 1-thread (40 us) and 8-thread (84 us) figures. The baseline 64^3 RHS (1646 us) sits between the fresh 8-thread (1164 us) and 1-thread (4513 us) figures. That fits a multi-threaded baseline on 4 CPUs. The baseline 1M axpby (409.5 us, about 61 GB/s) sits between the fresh 8-thread (385 us) and 1-thread (463 us) figures.

## C. Cut-cell RHS and per-step host work (new benchmark benchmarks/bench_rhs_cutcell.cpp)

Google Benchmark, --benchmark_min_time=0.3s. "1t" = OMP_NUM_THREADS=1; "8t" = 8 threads, OMP_PROC_BIND=spread OMP_PLACES=threads. R = total Rx+Ry+Rz points. heat cases use the heat.lua geometry (one floating sphere, E2 order 2, Gaussian MMS); wave cases the scalar_wave.lua geometry (two spheres, E2-poly order 1).

| benchmark | case | D points | R | 1t | 8t |
|---|---|---|---|---|---|
| heat RHS graph submit | 2D 51x51 | 2601 | 32 | 22.7 us | 87.8 us |
| heat RHS graph submit | 2D 71x71 | 5041 | 50 | 34.5 us | 92.0 us |
| heat RHS graph submit | 2D 201x201 | 40401 | 138 | 198 us | 152 us |
| heat RHS graph submit | 2D 401x401 | 160801 | 276 | 772 us | 422 us |
| heat RHS graph submit | 3D 65^3 | 274625 | 586 | 1867 us | 798 us |
| heat fill_source (host MMS: ddt - k lap at every D and R point, per stage) | 2D 51x51 | 2601 | 32 | 132 us | 60.3 us |
| heat fill_source | 2D 201x201 | 40401 | 138 | 2034 us | 588 us |
| heat fill_source | 3D 65^3 | 274625 | 586 | 28011 us | 6838 us |
| heat update_boundary (host MMS at every point + face/object assign, per stage) | 2D 51x51 | 2601 | 32 | 33.9 us | 36.1 us |
| heat update_boundary | 2D 201x201 | 40401 | 138 | 462 us | 246 us |
| heat update_boundary | 3D 65^3 | 274625 | 586 | 3917 us | 1119 us |
| heat stats (host MMS at every point + masked reductions, per step) | 2D 51x51 | 2601 | 32 | 31.9 us | 32.8 us |
| heat stats | 2D 201x201 | 40401 | 138 | 438 us | 249 us |
| heat stats | 3D 65^3 | 274625 | 586 | 3769 us | 1078 us |
| wave RHS graph submit | 2D 71x71 | 5041 | 44 | 30.7 us | 86.1 us |
| wave RHS graph submit | 2D 201x201 | 40401 | 128 | 177 us | 123 us |
| wave update_boundary | 2D 71x71 | 5041 | 44 | 41.1 us | 32.4 us |
| wave stats | 2D 71x71 | 5041 | 44 | 46.0 us | 37.9 us |

Decomposition check (1 thread, heat51): 4 x (22.7 + 132 + 33.9) + 31.9 = 786 us, against the measured 900 us step. The 114 us remainder is slot copies, timestep logic, logging and the write call, which still evaluates the MMS and the error field with io off (Section A). So about 59% of the shipped heat.lua step is fill_source, 15% update_boundary, 10% the four RHS graph submits and 4% stats. At 65^3 on 8 threads: 4 x (0.80 + 6.84 + 1.12) + 1.08 = 36.1 ms, against the measured 37.9 ms; fill_source is 72%. The RHS graph itself costs 8.7 ns per point at 51x51 (1t) and 2.9 ns per point at 65^3 (8t).

Consequences for the bar. There are three bars, and a replacement should be judged against all three:

- **Discretization bar** (what a replacement's kernels must match): the RHS graph submit column. 22.7 us at 51x51 (1t; 87.8 us at 8t), 34.5 us at 71x71 heat and 30.7 us for scalar_wave at 71x71 (1t), 198 us (1t) / 152 us (8t) at 201x201, 772 / 422 us at 401x401, 1867 / 798 us at 65^3.
- **Shipped step bar** (today's C++ as it runs): heat 51x51 0.90 ms at either thread count, scalar_wave 71x71 0.43 ms (1t), heat 201x201 4.6 ms (8t), 401x401 13.7 ms (8t), 65^3 37.9 ms (8t). This bar is dominated by host-side manufactured-solution evaluation through virtual calls.
- **Fixed-up step bar** (estimate). The "fixed-up C++" is a frozen list of exactly four changes: (1) hoist the object-BC gather lists out of the per-call path; (2) compact B and N tables with no full-height row pointers; (3) manufactured-solution evaluation once per step and only where needed, with time factors hoisted (fill_source and update_boundary); (4) packed [D | Rx | Ry | Rz] buffers. The estimate models change (3) only. It keeps the measured RHS and stats times and replaces fill_source plus update_boundary with an assumed residual of about 5 us per stage at 51x51 and about 0.3 ms per stage at 65^3. Those residuals are assumptions, not measurements. The result is roughly 4 x (23 + 5) + 32 = 150 us (about 0.15 ms) at 51x51 and roughly 4 x (0.8 + 0.3) + 1.1 = 5.5 ms at 65^3 on 8 threads. Changes (1), (2) and (4) would lower these further; that saving is not estimated. The estimate also leaves out the 114 us remainder of the measured 51x51 step. Part of that remainder is the write call, which evaluates the MMS for an error field that nobody reads with io off; applying change (3) there too would remove it. Even if the whole remainder stayed, the estimate would rise to at most about 0.26 ms. Any Python replacement should be compared against these fixed-up figures, not against today's step times.
- **Thread count.** With 8 threads (unpinned, or spread), runs below about 40k points are slower than with one thread, because the fork and join per Kokkos node costs more than the work. A serial path for small cases is part of the bar.

## D. Host-side Python measurements (numpy, torch CPU fp64, torch MPS fp32, MLX CPU fp64 and GPU fp32)

Source: `docs/python-dsl-survey/experiments/host_primitives_results.md` (write-up), `host_primitives_results.json` (merged), `host_primitives_run1.json` and `host_primitives_run2.json` (per run) and the script `host_primitives_bench.py`. Every number in this section is taken from the write-up. The runs were made natively on the macOS host on 2026-10-12, outside the Claude Code sandbox, because the sandbox hides the Metal device.

**Machine.** Apple M4 (hw.ncpu 10, 4 performance cores, 16 GiB), macOS 26.2, Python 3.14.3, numpy 2.4.4, torch 2.11.0 with MPS (4 intra-op threads), mlx 0.32.3. Run 1 started at 19:08:55 (wall 141 s, load average 1.36 at start); run 2 at 19:29:52 (wall 98 s, load average 1.13). Each cell below is the lower of the two run medians.

**What is timed.** One RK4 step of a synthetic SHOCCS-shaped heat problem built from the Part I primitives P0 to P6: a square or cubic grid with h = 1/(n-1); a sphere of solid points (about 6% of the grid); closure rows on the 2.4% of non-face fluid points nearest the sphere; Dirichlet faces; a packed [D | R] state with |R| = 0.5% of |D|. One RHS is the E2 3-point second difference per axis over shifted slices times 1/h^2, an interior mask, a width-7 ELL gather-dot over the packed vector for the closure and R rows (random positive off-diagonals with a diagonal of -1.05 times their sum, size about 1/h^2), a precomputed source times 1 + 0.5 sin(2 pi t), and a face zero. The step is classical RK4 with dt = 0.1 h^2, followed by one masked Linf reduction read back to the host, which forces the synchronisation. Medians over 100, 40 or 20 timed steps after 10 check steps and 5 warm-up steps. The GPU runs fp32 because MPS rejects float64 and MLX has no GPU float64.

**Milliseconds per RK4 step (median, lower of two runs).** The C++ columns restate Sections C and A for the matching case: four RHS graph submits (4 x 22.7 us = 91 us at 51x51 on 1 thread; 4 x 798 us = 3.2 ms at 65^3 on 8 threads; 4 x 1867 us = 7.5 ms at 65^3 on 1 thread) and the shipped step.

| Case | points | numpy CPU f64 | torch CPU f64 eager | torch CPU f64 compile | torch MPS f32 eager | torch MPS f32 compile | MLX GPU f32 compile | MLX CPU f64 compile | C++ four RHS submits | C++ shipped step |
|---|---|---|---|---|---|---|---|---|---|---|
| 2D 51x51 | 2601 | 0.103 | 0.149 | 0.168 | 2.171 | 0.501 | 0.353 | 0.085 | 0.091 (1t) | 0.90 |
| 2D 71x71 | 5041 | 0.178 | 0.219 | 0.278 | 1.697 | 0.505 | 0.351 | 0.115 | 0.138 (1t) | heat not run |
| 2D 201x201 | 40401 | 0.674 | 1.406 | 0.523 | 1.696 | 0.530 | 0.376 | 0.622 | 0.61 (8t), 0.79 (1t) | 4.6 (8t) |
| 2D 401x401 | 160801 | 2.409 | 2.210 | 1.486 | 1.454 | 0.670 | 0.573 | 2.350 | 1.69 (8t), 3.09 (1t) | 13.7 (8t) |
| 3D 65^3 | 274625 | 5.331 | 4.605 | 4.485 | 3.993 | 1.094 | 1.014 | 4.243 | 3.2 (8t), 7.5 (1t) | 37.9 (8t) |
| 3D 129^3 | 2146689 | 54.4 | 44.5 | 33.6 | 23.2 | 5.991 | 7.377 | 36.2 | not measured | not measured |
| 3D 257^3 | 16974593 | 448 | 340 | 272 | 180 | 43.6 | 57.2 | 300 | not measured | not measured |

**Nanoseconds per point per step.** Best CPU fp64 path: 32.8 ns (MLX CPU, 51x51), 22.8 (71x71), 12.9 (torch.compile, 201x201), 9.24 (401x401), 15.5 (MLX CPU, 65^3), 15.6 and 16.0 (torch.compile, 129^3 and 257^3); the CPU fp64 paths as a group sit at 9 to 27 ns from 201x201 up. Best GPU fp32 path: 136 ns (51x51), 69.7 (71x71), 9.30 (201x201), 3.56 (401x401), 3.69 (65^3), 2.79 (129^3), 2.57 (257^3). For comparison the C++ RHS graph alone is 8.7 ns per point at 51x51 (1 thread) and 2.9 ns at 65^3 (8 threads), so four submits are 35 and 11.6 ns per point per step.

**Dispatch and sync floors** (median of 200 calls of `x + 1` on 1024 elements plus synchronisation): numpy 0.46 us; torch CPU eager 0.75 us; torch.compile CPU 6.37 us; MLX CPU compiled 8.71 us; torch MPS eager 175 us; torch MPS compiled 181 us; MLX GPU compiled 167 us. The C++ launch floor is 0.55 us per tiny kernel (Section B). The 167 to 181 us figure is the measured replacement for the earlier unverified forum figure of about 120 us for a Metal round trip (`00-apple-gpu-facts.md`).

**fp32 error datapoint.** After 10 steps from the same initial state, max |u - u_ref| / max |u_ref| against the numpy fp64 run over the packed vector: the fp64 backends agree to between 0 and 2.3e-15; the three fp32 GPU backends disagree by 1.1e-7 to 2.4e-7 on every case (1.1e-7 to 1.7e-7 in 2D, 2.1e-7 to 2.4e-7 in 3D), a few fp32 ulps. These are E2 interior taps with random dissipative closure coefficients; the datapoint says nothing about the conditioning of real small-psi closures.

**First-step (compile) cost.** torch.compile CPU 0.75 s at 51x51 and 3.86 s at 65^3 (0.09 to 0.68 s at the other sizes); torch.compile MPS 0.77 to 2.18 s; MLX GPU 0.75 to 0.77 s where a compile happened and under 0.1 s otherwise; MLX CPU 0.00 to 1.23 s; the eager backends under 0.5 s everywhere.

**Two backend findings.** (1) torch.compile on MPS emitted invalid Metal for an in-place face-slice fill (InductorError: SyntaxError: failed to compile); the benchmark fell back to a static face-mask multiply for that backend. This is a measured instance of Inductor-MPS immaturity to set beside the open correctness issue cited in `frameworks/pytorch.md`. (2) `mx.array` silently converts a numpy float64 array to float32 unless the dtype is passed explicitly; the MLX CPU fp64 backend passes the dtype.

**Readings.**

- Small 2D is dispatch-bound everywhere. A fused synthetic step in fp64 on the CPU from Python takes 0.085 ms (compiled MLX CPU), 0.103 ms (eager numpy) and 0.149 ms (torch eager) at 51x51: 0.9x to 1.6x the four C++ RHS submits (91 us at 1 thread) and 6 to 11x under the shipped 0.90 ms step, which is 59% host MMS evaluation (Section C). torch.compile on the CPU is slower than eager at this size (6.4 us per compiled call against under 1 us per eager op).
- At 65^3 the best Python CPU fp64 paths (4.2 ms compiled MLX CPU, 4.5 ms torch.compile CPU with 4 intra-op threads) are within 1.4x of the eight-thread Kokkos RHS submits (3.2 ms) and under the one-thread figure (7.5 ms). CPU parity is plausible but not shown: the coefficients are synthetic, the step has no host MMS or update_boundary, and the C++ ran in the container. (The write-up itself compares against the shape-free 64^3 RHS of Section B, 4 x 1.16 ms = 4.6 ms; the cut-cell 65^3 case is the matching one and the stricter bar.)
- Every GPU path pays one synchronisation per step and the floor is 167 to 181 us. With the whole step compiled, MLX GPU and torch.compile MPS sit at 0.35 to 0.5 ms per step from 51x51 to 201x201 (two to three floors), so the GPU loses to the CPU at the shipped 2D sizes (0.353 ms against 0.085 ms at 51x51). The best GPU path draws level with the best CPU fp64 path between 71x71 and 201x201 (0.376 against 0.523 ms at 201x201) and is clearly ahead from 401x401 (0.573 against 1.486 ms). From 401x401 and 65^3 up the compiled GPU paths are bandwidth-bound at 2.6 to 4.2 ns per point per step in fp32, and are 4 to 6x faster than the best CPU fp64 path at 129^3 (5.99 against 33.6 ms) and 257^3 (43.6 against 272 ms). Eager MPS costs 1.5 to 2.2 ms at the small sizes because it issues on the order of a hundred kernels per step.
- For phase 4 of the roadmap this gives a measured floor to amortise: batching N small cases per dispatch must beat N CPU runs given the 167 to 181 us sync floor, which alone is twice the best CPU step at 51x51.

**Caveats.** Synthetic coefficients (E2 interior plus random closure and R-row coefficients; one merged ELL table; no cross-axis interpolation rows; no B/N split). No host MMS evaluation, no update_boundary, no stats beyond one Linf, no I/O; the C++ step includes all of these, so step-to-step ratios favour Python and the fair bar is the four RHS submits. The C++ bar is a container measurement (8 vCPU Linux VM, Kokkos OpenMP), not a native one. A concurrent container job may have used CPU during run 1 (load average 1.36 at its start against 1.13 for run 2); most best cells come from run 2. The functional backends (torch.compile, MLX) allocate new buffers per step; eager numpy and torch use preallocated buffers for the stage updates.

## E. Native macOS build (2026-10-13)

Source: `experiments/host/` (`bench_rhs_cutcell_t1.json`, `_t4.json`, `_t4_spin.json`; `bench_rhs_t1.json`, `_t4.json`, `_t10.json`, `_t4_spin.json`; `bench_expr_t1.json`; `e2e_steps_summary.json` produced by `run_e2e.py` and `summarize_e2e.py`). The build is the one documented in CLAUDE.md ("Native macOS build") and added on branch `host-build` (lanl PR #36): spack environment `tools/spack/macos`, Apple clang 21.0.0, Kokkos 5.1.1 +openmp +serial with llvm-openmp 20.1.8, OpenBLAS 0.3.33 without Fortran, Release, `-march=armv8.4-a` propagated by Kokkos. Same Apple M4 as Section D (4 performance and 6 efficiency cores, 16 GiB), run inside the Bash sandbox (no GPU involved). 47 of 48 tests pass; `t-laplacian` fails as in the container. Thread settings: "1t" is OMP_NUM_THREADS=1 (end-to-end) or `--kokkos-num-threads=1` (micro-benchmarks); "4t" is four threads with libomp defaults; "4t spin" is four threads with `KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active`. macOS libomp has no thread affinity, so OMP_PROC_BIND is unset; four is the performance-core count, and ten threads was slower in every case tried. Single runs, not the two-run minimum of Section D.

**End-to-end RK4 step (median Wall_ms, the cases and scripts of Section A).**

| case | D points | steps | native 1t | native 4t | native 4t spin | container 1t | container 8t |
|---|---|---|---|---|---|---|---|
| heat51 | 2601 | 50 | 0.856 (min 0.648, p90 1.35) | 2.26 | 0.266 (min 0.265, p90 0.271) | 0.900 | 0.935 |
| wave71 | 5041 | 60 | 0.278 | 1.84 | 0.198 | 0.434 | 0.675 |
| heat201 | 40401 | 80 | 6.82 | 4.05 | 2.17 | 12.18 | 4.63 |
| heat401 | 160801 | 64 | 26.96 | 10.09 | 8.18 | 47.99 | 13.69 |
| heat3d65 | 274625 | 11 | 58.4 | 19.96 | 17.49 | 148.9 | 37.9 |

**Cut-cell stages (the Section C benchmark, --benchmark_min_time=0.3s, microseconds).**

| benchmark | case | native 1t | native 4t | native 4t spin | container 1t | container 8t |
|---|---|---|---|---|---|---|
| heat RHS graph submit | 2D 51x51 | 21.0 | 309 | 54.1 | 22.7 | 87.8 |
| | 2D 71x71 | 33.3 | 318 | 67.6 | 34.5 | 92.0 |
| | 2D 201x201 | 217 | 358 | 228 | 198 | 152 |
| | 2D 401x401 | 827 | 575 | 797 | 772 | 422 |
| | 3D 65^3 | 2065 | 961 | 1810 | 1867 | 798 |
| heat fill_source | 2D 51x51 | 59.8 | 45.6 | 47.4 | 132 | 60.3 |
| | 2D 201x201 | 922 | 296 | 682 | 2034 | 588 |
| | 3D 65^3 | 7985 | 2429 | 5250 | 28011 | 6838 |
| heat update_boundary | 2D 51x51 | 24.4 | 89.7 | 32.7 | 33.9 | 36.1 |
| | 2D 201x201 | 351 | 184 | 312 | 462 | 246 |
| | 3D 65^3 | 2813 | 906 | 2076 | 3917 | 1119 |
| heat stats | 2D 51x51 | 26.9 | 86.0 | 33.0 | 31.9 | 32.8 |
| | 2D 201x201 | 377 | 191 | 319 | 438 | 249 |
| | 3D 65^3 | 2960 | 955 | 2045 | 3769 | 1078 |
| wave RHS graph submit | 2D 71x71 | 25.2 | 307 | 54.6 | 30.7 | 86.1 |
| | 2D 201x201 | 157 | 347 | 178 | 177 | 123 |
| wave update_boundary | 2D 71x71 | 27.8 | 76.8 | 36.1 | 41.1 | 32.4 |
| wave stats | 2D 71x71 | 31.5 | 89.7 | 42.0 | 46.0 | 37.9 |

**Micro-benchmarks.** bench_rhs BM_heat_rhs (no shapes) at 16^3 / 32^3 / 64^3: 1t 0.041 / 0.354 / 3.48 ms (container 1t 0.040 / 0.418 / 4.51); 4t 0.318 / 0.427 / 1.36 ms; 4t spin 0.063 / 0.313 / 2.25 ms; 10t 1.06 / 1.14 / 1.88 ms (container 8t 0.084 / 0.169 / 1.16). bench_expr at 1t: BM_assign_fma 1024 / 16384 / 131072 / 1048576 = 0.276 / 4.49 / 34.8 / 289 us (container 1t 0.551 / 6.73 / 67.9 / 463 us), so the launch floor is 0.28 us per tiny kernel natively and the 1M streaming rate at one thread is 108 GiB/s (container 67 GiB/s); BM_plus_assign_mul 1M 249 us (container 399 us).

**Readings.**

- One thread: the RHS graph submit is within 10% of the container at every size (faster below 201x201, 5 to 10% slower above; Kokkos builds with `-march=armv8.4-a` here and with gcc's defaults there). The host work is much cheaper natively: fill_source 2.2x at 51x51 and 3.5x at 65^3, update_boundary and stats 1.2 to 1.4x. The one-thread step follows the host work: 0.86 ms at heat51 (noisy, min 0.65 and p90 1.35 ms; a lone busy thread is not pinned to a performance core), 0.28 ms at wave71, 6.8 ms at heat201, 27 ms at heat401 and 58 ms at heat3d65, against 0.90, 0.43, 12.2, 48.0 and 149 ms in the container.
- Four threads with libomp defaults: every small case is slower than one thread, by more than in the container, because idle libomp threads sleep between parallel regions and every Kokkos node pays a wake-up (heat RHS 51x51: 309 us against 21 us). Large kernels do scale: the 65^3 RHS takes 961 us (2.1x over one thread) and fill_source 2429 us (3.3x).
- Spin-waiting removes the wake-up cost: heat RHS 51x51 54 us, and the whole shipped heat51 step 0.266 ms with a tight distribution. It is the best end-to-end setting at every size tried (heat3d65 17.5 ms against 20.0 ms by default and 37.9 ms in the container). In the micro-benchmarks, though, spinning slows the large kernels (65^3 RHS 1810 against 961 us; shape-free 64^3 RHS 2.25 against 1.36 ms), so the two settings are not interchangeable: the bar for a replacement runtime is the better of the two per case.
- Decomposition. At 1t, heat51: 4 x (21.0 + 59.8 + 24.4) + 26.9 = 448 us of stage work against a 0.65 to 0.86 ms step, a remainder of 0.2 to 0.4 ms where the container had 0.11 ms. At 4t spin the stage figures (4 x (54.1 + 47.4 + 32.7) + 33.0 = 570 us) exceed the 0.266 ms step, so under spinning the micro-benchmark overstates the stage costs and the step figure is the one to use. With fill_source 2 to 3.5x cheaper natively, its share of the one-thread heat51 step falls from 59% to roughly 30 to 40%, and the native 4t-spin shipped step, 0.27 ms, already sits in the range of the container-based "fixed-up C++" estimate of Section C (0.15 to 0.26 ms) before any of the four changes. That estimate, and the stop rule built on it, have to be restated from native figures before phase 2.
- What this changes for the bar. The one-thread discretization bar stands (21 to 23 us at 51x51; 2.0 to 2.1 ms at 65^3). The multi-thread discretization bar on this machine is 54 us at 51x51 (4t spin) and 961 us at 65^3 (4t default). The shipped-step bar is now native: 0.27 ms at heat51 and 17.5 ms at heat3d65 (4t spin). The Python figures of Section D (0.085 ms synthetic CPU step at 51x51, 0.35 ms GPU) are now on the same machine as the C++ figures: the synthetic CPU step is 3x under the native shipped step and about 1.6x over four native RHS submits at 4t spin (4 x 54 = 216 us) or 4x over them at one thread (84 us). Regime C (257^3) remains unmeasured in C++.

**Caveats.** Single runs. The container figures come from a Linux VM with 8 vCPU on the same hardware, so "container 8t" and "native 4t" are different core budgets. Apple clang contracts multiply-adds by default (`-ffp-contract=on`), gcc on the container does not; no result differences were checked beyond the test suite. The one-thread end-to-end medians carry scheduling noise (heat51 p90 1.35 ms against a 0.65 ms minimum); the micro-benchmarks, which average over many iterations, do not show it.
