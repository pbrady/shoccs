# Host primitive benchmark: SHOCCS-shaped RK4 step in Python array frameworks

This is the host-side measurement the survey lacked. It times one RK4 step of a synthetic, SHOCCS-shaped heat problem built from the runtime primitives P0 to P6 of `01-codebase-idioms.md` Section 4, on the CPU in fp64 and on the Apple GPU in fp32, and compares it with the C++ bar measured in the devcontainer. Script: `host_primitives_bench.py`; raw data: `host_primitives_results.json` (merged) and the per-run files.

## Machine

- CPU: Apple M4; hw.ncpu 10, performance cores 4; memory 16 GiB
- uname -m arm64, macOS 26.2, Python 3.14.3, numpy 2.4.4, torch 2.11.0 (intra-op threads 4), mlx 0.32.3
- host_primitives_run1.json: started 2026-10-12 19:08:55, wall 141 s, load average at start 1.36, 1.34, 1.34
- host_primitives_run2.json: started 2026-10-12 19:29:52, wall 98 s, load average at start 1.13, 1.17, 1.28

## Observations

Best backend per case (lower-of-two-runs medians):

| Case | best CPU fp64 | ms | x C++ | best GPU fp32 | ms | x C++ |
|---|---|---|---|---|---|---|
| 2D 51x51 | mlx-cpu-f64-compile | 0.085 | 0.09 | mlx-gpu-f32-compile | 0.353 | 0.39 |
| 2D 71x71 | mlx-cpu-f64-compile | 0.115 | - | mlx-gpu-f32-compile | 0.351 | - |
| 2D 201x201 | torch-cpu-f64-compile | 0.523 | 0.11 | mlx-gpu-f32-compile | 0.376 | 0.08 |
| 2D 401x401 | torch-cpu-f64-compile | 1.486 | 0.11 | mlx-gpu-f32-compile | 0.573 | 0.04 |
| 3D 65^3 | mlx-cpu-f64-compile | 4.243 | 0.11 | mlx-gpu-f32-compile | 1.014 | 0.03 |
| 3D 129^3 | torch-cpu-f64-compile | 33.6 | - | torch-mps-f32-compile | 5.991 | - |
| 3D 257^3 | torch-cpu-f64-compile | 272 | - | torch-mps-f32-compile | 43.6 | - |

- Small 2D cases (51x51, 71x71) are dispatch-bound everywhere. Eager numpy and compiled MLX on the CPU run a whole RK4 step in about 0.1 to 0.3 ms, several times under the 0.90 ms C++ bar, mainly because the synthetic step has no host MMS evaluation, no update_boundary and far fewer launches than the C++ step (see Caveats). torch.compile on the CPU is slower than eager here (its per-call guard and wrapper overhead is about 6 us, against under 1 us for an eager op).
- Every GPU path pays one CPU-GPU synchronisation per step, and the measured floor for a trivial op plus sync is about 140 to 190 us on both MPS and MLX. With the whole step compiled, MLX and torch.compile on MPS sit at about 0.37 and 0.5 ms per step up to 201x201, i.e. two to three sync floors; eager MPS is 1.5 to 2.2 ms because it issues on the order of a hundred separate small kernels per step (about 20 ops per RHS times four, plus stage updates).
- From 401x401 and 65^3 upwards the compiled GPU paths are bandwidth-bound and reach 2.6 to 4 ns per point per RK4 step in fp32; CPU fp64 paths sit at 9 to 27 ns per point (torch.compile CPU best at about 16 ns at 3D). The GPU paths are 4 to 6x faster than the best CPU fp64 path at 129^3 and 257^3.
- The ratio column overstates the Python advantage: the C++ step also does host MMS evaluation, update_boundary, full stats and many more launches, none of which the synthetic step has. A fairer reference is the C++ RHS-only graph submit, 1.16 ms per RHS at 64^3 (8 threads), about 4.6 ms for the four RHS of one RK4 step before any stage updates. The best CPU fp64 Python step at 65^3 (4.2 to 4.5 ms, MLX CPU or torch.compile CPU, including stage updates and the reduction) is at parity with that, and the compiled GPU fp32 paths (about 1.0 to 1.1 ms) are about 4x under it.
- The fp32 error after 10 steps is 1e-7 to 2.4e-7 relative on every case and backend, i.e. a few fp32 ulps; this says nothing about real small-psi closures.

## Milliseconds per RK4 step (median)

Each cell is the median wall time of one full RK4 step (four RHS evaluations, stage updates, one masked Linf reduction and its host readback), the lower of the two runs; the run it came from is in brackets. `skip` and `fail` reasons are listed below the table.

| Case | grid points | numpy-cpu-f64 | torch-cpu-f64-eager | torch-cpu-f64-compile | torch-mps-f32-eager | torch-mps-f32-compile | mlx-gpu-f32-compile | mlx-cpu-f64-compile | C++ bar |
|---|---|---|---|---|---|---|---|---|---|
| 2D 51x51 | 2601 | 0.103 [r2] | 0.149 [r2] | 0.168 [r2] | 2.171 [r1] | 0.501 [r2] | 0.353 [r2] | 0.085 [r2] | 0.9 |
| 2D 71x71 | 5041 | 0.178 [r1] | 0.219 [r1] | 0.278 [r2] | 1.697 [r2] | 0.505 [r2] | 0.351 [r2] | 0.115 [r2] | - |
| 2D 201x201 | 40401 | 0.674 [r2] | 1.406 [r2] | 0.523 [r2] | 1.696 [r1] | 0.530 [r2] | 0.376 [r1] | 0.622 [r2] | 4.6 |
| 2D 401x401 | 160801 | 2.409 [r2] | 2.210 [r2] | 1.486 [r2] | 1.454 [r2] | 0.670 [r1] | 0.573 [r2] | 2.350 [r2] | 13.7 |
| 3D 65^3 | 274625 | 5.331 [r2] | 4.605 [r1] | 4.485 [r1] | 3.993 [r2] | 1.094 [r2] | 1.014 [r1] | 4.243 [r2] | 38.0 |
| 3D 129^3 | 2146689 | 54.4 [r2] | 44.5 [r1] | 33.6 [r2] | 23.2 [r2] | 5.991 [r2] | 7.377 [r2] | 36.2 [r2] | - |
| 3D 257^3 | 16974593 | 448 [r2] | 340 [r2] | 272 [r2] | 180 [r2] | 43.6 [r1] | 57.2 [r2] | 300 [r2] | - |

## Nanoseconds per grid point and ratio to the C++ bar

ns/pt is the step median divided by |D|. The ratio is Python time over C++ time (below 1 means faster than C++); it is given only where the C++ bar has a matching case: 51x51 against 0.90 ms, 201x201 against 4.6 ms, 401x401 against 13.7 ms, 65^3 against 38 ms. The C++ 71x71 figure (0.43 ms) is for scalar_wave, not heat, so it is not used as a ratio.

| Case | numpy-cpu-f64 | torch-cpu-f64-eager | torch-cpu-f64-compile | torch-mps-f32-eager | torch-mps-f32-compile | mlx-gpu-f32-compile | mlx-cpu-f64-compile |
|---|---|---|---|---|---|---|---|
| 2D 51x51 | 39.6 ns / 0.11x | 57.4 ns / 0.17x | 64.5 ns / 0.19x | 834 ns / 2.41x | 193 ns / 0.56x | 136 ns / 0.39x | 32.8 ns / 0.09x |
| 2D 71x71 | 35.3 ns | 43.5 ns | 55.1 ns | 337 ns | 100 ns | 69.7 ns | 22.8 ns |
| 2D 201x201 | 16.7 ns / 0.15x | 34.8 ns / 0.31x | 12.9 ns / 0.11x | 42.0 ns / 0.37x | 13.1 ns / 0.12x | 9.30 ns / 0.08x | 15.4 ns / 0.14x |
| 2D 401x401 | 15.0 ns / 0.18x | 13.7 ns / 0.16x | 9.24 ns / 0.11x | 9.05 ns / 0.11x | 4.17 ns / 0.05x | 3.56 ns / 0.04x | 14.6 ns / 0.17x |
| 3D 65^3 | 19.4 ns / 0.14x | 16.8 ns / 0.12x | 16.3 ns / 0.12x | 14.5 ns / 0.11x | 3.98 ns / 0.03x | 3.69 ns / 0.03x | 15.5 ns / 0.11x |
| 3D 129^3 | 25.4 ns | 20.7 ns | 15.6 ns | 10.8 ns | 2.79 ns | 3.44 ns | 16.8 ns |
| 3D 257^3 | 26.4 ns | 20.0 ns | 16.0 ns | 10.6 ns | 2.57 ns | 3.37 ns | 17.7 ns |

## Dispatch floors

Median of 200 calls of a trivial elementwise op (`x + 1` on 1024 elements) plus synchronisation (`torch.mps.synchronize`, `mx.eval`); compiled backends call a compiled function.

| Backend | floor (us) |
|---|---|
| numpy-cpu-f64 | 0.46 |
| torch-cpu-f64-eager | 0.75 |
| torch-cpu-f64-compile | 6.37 |
| torch-mps-f32-eager | 175 |
| torch-mps-f32-compile | 181 |
| mlx-gpu-f32-compile | 167 |
| mlx-cpu-f64-compile | 8.71 |

## Correctness and the fp32 error datapoint

After 10 steps from the same initial state every backend is compared with the numpy fp64 run: max |u - u_ref| / max |u_ref| over the packed [D | R] vector. Tolerance 1e-12 for fp64, 1e-4 for fp32. This is E2 with random (dissipative, diagonally dominant) closure coefficients of size about 1/h^2, not the real psi-dependent closures, so the fp32 numbers say nothing about small-psi conditioning.

| Case | numpy-cpu-f64 | torch-cpu-f64-eager | torch-cpu-f64-compile | torch-mps-f32-eager | torch-mps-f32-compile | mlx-gpu-f32-compile | mlx-cpu-f64-compile |
|---|---|---|---|---|---|---|---|
| 2D 51x51 | 0.0e+00 | 1.4e-16 | 0.0e+00 | 1.1e-07 | 1.4e-07 | 1.4e-07 | 1.9e-15 |
| 2D 71x71 | 0.0e+00 | 1.4e-16 | 0.0e+00 | 1.2e-07 | 1.3e-07 | 1.3e-07 | 9.5e-16 |
| 2D 201x201 | 0.0e+00 | 1.3e-16 | 0.0e+00 | 1.5e-07 | 1.5e-07 | 1.5e-07 | 1.3e-16 |
| 2D 401x401 | 0.0e+00 | 2.6e-16 | 0.0e+00 | 1.6e-07 | 1.7e-07 | 1.7e-07 | 1.3e-16 |
| 3D 65^3 | 0.0e+00 | 1.8e-16 | 0.0e+00 | 2.4e-07 | 2.4e-07 | 2.4e-07 | 2.3e-15 |
| 3D 129^3 | 0.0e+00 | 3.5e-16 | 0.0e+00 | 2.1e-07 | 2.2e-07 | 2.2e-07 | 5.2e-16 |
| 3D 257^3 | 0.0e+00 | 3.4e-16 | 0.0e+00 | 2.4e-07 | 2.4e-07 | 2.4e-07 | 3.4e-16 |

## First-step (compile) cost

Wall time of the first step, which includes tracing and compilation for compiled backends (seconds).

| Case | numpy-cpu-f64 | torch-cpu-f64-eager | torch-cpu-f64-compile | torch-mps-f32-eager | torch-mps-f32-compile | mlx-gpu-f32-compile | mlx-cpu-f64-compile |
|---|---|---|---|---|---|---|---|
| 2D 51x51 | 0.00 | 0.00 | 0.75 | 0.02 | 0.91 | 0.01 | 0.02 |
| 2D 71x71 | 0.00 | 0.00 | 0.09 | 0.01 | 0.77 | 0.00 | 0.00 |
| 2D 201x201 | 0.00 | 0.00 | 0.09 | 0.01 | 0.81 | 0.75 | 0.01 |
| 2D 401x401 | 0.00 | 0.00 | 0.10 | 0.01 | 1.30 | 0.00 | 0.01 |
| 3D 65^3 | 0.01 | 0.00 | 3.86 | 0.01 | 1.53 | 0.77 | 0.01 |
| 3D 129^3 | 0.05 | 0.05 | 0.16 | 0.04 | 1.51 | 0.01 | 0.05 |
| 3D 257^3 | 0.46 | 0.35 | 0.68 | 0.30 | 2.18 | 0.07 | 1.23 |

## Backend notes

- numpy-cpu-f64: eager, out= buffers for stage axpby (two ops: multiply then add), ping-pong u
- torch-cpu-f64-eager: eager; torch.add(alpha=, out=) fused axpby into preallocated buffers, ping-pong u
- torch-cpu-f64-compile: torch.compile(default mode) over the whole functional RK4 step incl. reduction; t as 0-dim tensor
- torch-mps-f32-eager: eager; torch.add(alpha=, out=) fused axpby into preallocated buffers, ping-pong u
- torch-mps-f32-compile: torch.compile(default mode) over the whole functional RK4 step incl. reduction; t as 0-dim tensor; FALLBACK 1: whole-step compile with in-place face-slice fill failed (InductorError: SyntaxError: failed to compile ); face fill done as a static face-mask multiply
- mlx-gpu-f32-compile: mx.compile over the whole functional RK4 step incl. reduction; face fill as static mask; scatter via .at[].add
- mlx-cpu-f64-compile: mx.compile over the whole functional RK4 step incl. reduction; face fill as static mask; scatter via .at[].add

## Methods

The problem is a heat-like system on a cubic or square grid with h = 1/(n-1). A sphere of solid points (about 6% of the grid) sits in the middle; the 2.4% of non-face fluid points nearest to it are closure rows; the 2*dim boundary planes are Dirichlet faces. Role bits combine into an interior mask. The packed state is [D | R] with |R| = 0.5% of |D|. One RHS is: zero-pad D, sum over axes of the E2 3-point second difference times 1/h^2 using shifted slices, multiply by the interior mask, append zeros for R, then a w = 7 ELL gather-dot over the packed vector for the closure rows and all R rows (int32 indices, columns are the row itself, axis neighbours and random R points; random positive off-diagonals with a diagonal of -1.05 times their sum), scattered to unique rows; scale by kappa = 1, add a precomputed source times 1 + 0.5 sin(2 pi t); zero the faces. The RK4 step is the classical four-stage scheme with dt = 0.1 h^2 (u1 = u0 + c k, rk accumulation, u_new = u0 + rk), followed by max |u| over fluid points (all of R plus non-solid D) read back to the host once.

Timing: 10 check steps (which include compilation), 5 more warm-up steps, then the median of 100 (|D| < 1e5), 40 (|D| < 1e6) or 20 timed steps; each timed step ends with the host readback of the reduction, which forces synchronisation (MLX additionally evaluates the new state with `mx.eval`). The 257^3 case runs only on backends whose 129^3 median was under 2 s. The full suite ran twice; each cell keeps the lower median. Metal is not visible inside the agent's Bash sandbox, so the runs that include MPS and MLX GPU were made outside it.

## Caveats

- Synthetic coefficients: E2 interior plus random closure and R-row coefficients. No real psi-dependent closures, no cross-axis interpolation rows, no B/N split into separate tables: one merged ELL table.
- No host MMS evaluation (the C++ step evaluates the manufactured solution on the host several times per step), no update_boundary, no stats beyond one Linf, no I/O. The C++ bar includes stats and its host MMS work, so the comparison favours Python somewhat at small sizes.
- The C++ bar was measured in a Linux devcontainer (8 vCPU aarch64 VM on this Mac, Kokkos OpenMP), not natively; native C++ may be faster, particularly at large sizes where VM memory bandwidth matters.
- A concurrent container job may have been using CPU during the first run; the per-run load averages are listed under Machine (1.36 at the start of run 1, 1.13 at run 2), and each cell reports which run it came from. Most best cells come from run 2.
- The functional backends (torch.compile, MLX) allocate new buffers per step rather than ping-ponging fixed ones; eager numpy and torch use preallocated buffers for the stage updates.
- fp32 is used on the GPU because MPS rejects float64 and MLX has no GPU float64.
