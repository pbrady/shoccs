# Part V: Phase-1 results (2026-10-13)

Status: measured 2026-10-13 on branch `pyshoccs-phase1`, on the machine and build of Part IV E (`05-measurements.md`): the owner's Apple M4 (4 performance and 6 efficiency cores, 16 GiB), macOS 26.2, the native build in `build-host` (Release, Apple clang 21, Kokkos 5.1.1 OpenMP via llvm-openmp, the user-managed `shoccs-host` spack environment); Python 3.14.3, numpy 2.5.4, scipy 1.18.1, torch 2.14.1, mlx 0.32.3. The C++ and the CPU Python measurements ran inside the Bash sandbox; the Metal rows ran outside it. Evidence: `experiments/phase1/` (Section 8), copied from the working directories of the three measurement passes (parity, C++ arm, Python arm) and reviewed by an independent critic, whose caveats bind the wording here.

Phase 1 was asked one question (editor's report Section 4.3): does either arm, the C++ compact-table kernels or a Python array runtime on the same tables, match or beat the fixed-up Kokkos RHS and step on this machine, natively, under the five gates of Section 4.2 and at the thread settings fixed there (1 thread for the shipped 2D cases, 4 threads with libomp defaults for the 65^3 RHS submit). Three things were built on the branch: the exporter `shoccs-export` (`src/app/export_tables.cpp`, the table code in `src/operators/tables.hpp` and `tables.cpp`, the npy writer `src/io/npy.hpp`), which writes the exact operators and the compact tables of one case under the data contract `docs/reference/pyshoccs-export.md`; the Python array arm `python/pyshoccs` (loader, structured decomposition, manufactured solution, heat system with RK4, monomial oracle, parity protocol, `bench/bench_heat.py`); and the C++ fused-stage arm (`src/operators/tables_kernels.hpp`, `benchmarks/bench_tables.cpp`, tests `src/operators/tables.t.cpp` and `tables_kernels.t.cpp`). Not done in phase 1: the fixed-up C++ of phase 1(b) (the four frozen changes on a tagged branch), the fp32/df64 study, the 128^3 sphere case and the nanobind binding. Every gate verdict below is therefore against the shipped graph, and every "met" is provisional (Section 4).

## 1. What was built and how it is checked

The data contract (`docs/reference/pyshoccs-export.md`) fixes one packed vector `V = [D | Rx | Ry | Rz]` per scalar field, an export directory of NumPy `.npy` arrays plus `meta.json` (mesh and R records, stencil taps and wall slabs, every stored entry of the three second-derivative operators as COO over V with a part code, the oracle dumps `u0_V`, `lap_rand_V`, `rhs_rand_V`, `u1_V`, `stats_u1.json`, and the exporter's own compact tables `role`, `exc_rows`, `exc_rowptr`, `exc_col`, `exc_val`), the five-check parity protocol of its Section 3 and the structured decomposition of its Section 4 that both arms derive independently (Python from the COO, C++ from the block metadata) and cross-check bitwise.

Checks that exist and their counts on 2026-10-13:

| check | what it covers | count | result |
|---|---|---|---|
| `t-tables` (`src/operators/tables.t.cpp`) | `extract_coo`, `build_compact_tables`, `apply_structured` against the eager Laplacian on the two floating-object setups of `laplacian.t.cpp` | 2 test cases, 82,081 assertions | passes at 1 and 4 threads |
| `t-tables_kernels` (`src/operators/tables_kernels.t.cpp`) | every launch form of `structured_rhs` and `fused_rk_stage` against the serial `apply_structured` plus the source rule on the floating setups, and against the live `heat` RHS graph on heat.lua 51x51 and its 33^3 version (worst row ratio at most 1e-13, observed about 2e-16) | 4 test cases, 228 assertions (run for this write-up; `cpp_arm/RESULTS.md` quotes 188 from an earlier build of the same test) | passes at 1 and 4 threads |
| `pytest` in `python/pyshoccs` (6 test modules) | loader and dtype kinds, structured decomposition on hand-built tables, manufactured solution against finite differences, RK4 on one-point ODEs, the oracle on a hand-built transverse row, the full parity protocol on a synthetic export and on the real one | 39 passed with `PYSHOCCS_EXPORT_DIR` set to each of the three exports; 31 passed, 8 skipped without an export | passes |
| in-process checks of `bench_tables` | structured RHS against the live graph on a random vector, fused step against the graph-built step | max diff / max value 1.9e-16 to 3.7e-16 (RHS), 1.1e-16 (step) on five cases | passes |
| `t-laplacian` | the pre-existing failure ("E2 with Floating Objects", 3D, rx_vec off by 2 to 3 percent) | unchanged | fails as before; Section 3 |

The three exports (`.scratch/export/heat51`, `heat201`, `heat3d65`; the critic re-exported all three with the rebuilt binary and found them byte-identical with `meta.json`, which carries the commit hash, excluded from the comparison):

| case | config | h | n_D | n_R (x, y, z) | n_V | stored entries per direction (A0, A1, A2) | exception rows (D + R) | exception nnz | files | size | export wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| heat51 | heat.lua as shipped | 0.04 | 2,601 | 16, 16, 0 | 2,633 | 7,333, 7,333, 0 | 250 (218 + 32) | 1,238 | 99 (97 .npy, meta.json, stats_u1.json) | 0.69 MB | 0.039 s |
| heat201 | heat.lua, 201x201 | 0.01 | 40,401 | 68, 70, 0 | 40,539 | 117,262, 117,246, 0 | 1,030 (892 + 138) | 5,164 | 99 | 10.4 MB | 0.044 s |
| heat3d65 | heat.lua in 3D, 65^3, sphere at (16/17, 25/22, 1) | 1/32 | 274,625 | 198, 194, 194 | 275,211 | 759,764, 759,773, 759,716 | 24,012 (23,426 + 586) | 106,337 | 107 (105 .npy) | 92.2 MB | 0.169 s |

Sources: `meta.json` of each export (n_D, n_R, n_V, h); `experiments/phase1/parity/REPORT.md` Section 2 (exception rows and nnz); `experiments/phase1/gate5_memory.py` run on the exports (entries per direction, the `nnz` column of its per-direction table); `experiments/phase1/export_timings.json` (process wall of `shoccs-export` from start to exit, re-export byte-identical to the previous one; file count there excludes `meta.json`); sizes are the byte totals of the directories. All three cases are scheme E2, order 2, `interp {2, 3}`, diffusivity 1/30, Gaussian MMS, floating object, Dirichlet walls.

Two documentation points the critic raised and this write-up records rather than fixes: `python/pyshoccs/README.md` says every array's dtype is checked against the contract, while `tables.py` checks the dtype kind (any integer width passes); and the exporter's `extract_coo` materialises every entry as a 32-byte record, so a 256^3 export (of order 50M entries per direction) needs a streaming writer (already stated in the contract, Section 2.3).

## 2. Parity (gate 1)

From `experiments/phase1/parity/REPORT.md` (the protocol run by `python -m pyshoccs.parity` on each export, JSON beside it) and the independent re-derivation in `analyze.py` (`analysis_<case>.json`). All gated checks passed on all three cases.

Row classes derived from the export alone:

| case | n_V | solid D | Dirichlet D | fluid D | clamp-adjacent D | R | R Dirichlet |
|---|---|---|---|---|---|---|---|
| heat51 | 2,633 | 56 | 200 | 2,345 | 0 | 32 | 0 |
| heat201 | 40,539 | 940 | 800 | 38,661 | 0 | 138 | 0 |
| heat3d65 | 275,211 | 717 | 24,578 | 249,330 | 0 | 586 | 0 |

Rows with any entry are exactly the fluid D and R rows (2,377 / 38,799 / 249,916); no solid, Dirichlet-face or Dirichlet-R row has an entry and `lap_rand_V` is exactly 0 on all of them; no fluid or R row lacks entries. The Dirichlet face lists have corner duplicates (204 / 804 / 25,350 entries for 200 / 800 / 24,578 unique points), as `heat::build_rhs_graph` builds them.

| check | heat51 | heat201 | heat3d65 |
|---|---|---|---|
| 1 operator identity, max abs / rel to max abs(lap_rand) | 9.1e-13 / 2.0e-16 | 1.5e-11 / 1.9e-16 | 3.6e-12 / 3.3e-16 |
| 1 worst row (class) | D(22,45) fluid | D(1,34) fluid | D(6,6,4) fluid |
| 2 rhs(rand_V), max abs / rel | 2.8e-14 / 1.9e-16 | 6.8e-13 / 2.7e-16 | 1.7e-13 / 4.7e-16 |
| 2 rhs(u0_V), max abs / rel to max abs(rhs_u0) | 1.6e-14 / 1.4e-12 | 4.9e-13 / 3.1e-11 | 9.1e-14 / 3.1e-12 |
| 3 R rows, axial k <= 2: fail / rows, worst ratio to tolerance | 0 / 192, 0.052 | 0 / 828, 0.047 | 0 / 5,274, 0.034 |
| 4 step identity, max abs / rel | 2.2e-16 / 1.1e-16 | 2.2e-16 / 1.1e-16 | 2.2e-16 / 1.1e-16 |
| 4b stats (11 numbers, 4 indices) | 11/11 exact | 11/11 exact | 11/11 exact |
| 5 structural identity, max abs / rel | 9.1e-13 / 2.0e-16 | 1.5e-11 / 1.9e-16 | 3.6e-12 / 3.3e-16 |

The larger relative figure of check 2 on `u0_V` is against max abs(rhs_u0), which is small because the MMS time derivative vanishes at t0; against the contract's per-row tolerance the worst ratio is about 0.05, like the other checks.

Structural cross-check (check 5 taken further, bitwise): `analyze.py` rebuilds the exporter's compact CSR from `exc_rows`, `exc_rowptr`, `exc_col`, `exc_val` and compares it with `pyshoccs.structure.derive_structure` from the COO alone. `role.npy` is equal as bytes on all three cases; `exc_rows` equal (sorted, unique); `exc_rowptr` equal; the exporter's columns are already sorted per row and equal the Python ones; values bitwise equal on heat51 and heat3d65, and on heat201 equal except 2 entries where the exporter holds +0.0 and the Python duplicate sum -0.0 (stored zeros of psi = 1 closures, kept on purpose: 260 / 1,082 / 2,334 zero entries). `interior_rows[d] x width` equals the number of part-1 entries in every direction and every part-1 coefficient equals `taps.npy` to 4 eps.

| case | interior rows per direction | exception rows (D + R) | exception nnz | max nnz per exception row | taps |
|---|---|---|---|---|---|
| heat51 | 2,231, 2,231, 0 | 250 (218 + 32) | 1,238 | 12 | 625, -1250, 625 |
| heat201 | 38,195, 38,193, 0 | 1,030 (892 + 138) | 5,164 | 13 | 1e4, -2e4, 1e4 |
| heat3d65 | 241,194, 241,198, 241,198 | 24,012 (23,426 + 586) | 106,337 | 20 | 1024, -2048, 1024 |

fp64 sum-order spread on `rand_V`, max difference as a fraction of max abs(lap_rand) (global) / of |row|_1 max|u| (per row):

| pair | heat51 | heat201 | heat3d65 |
|---|---|---|---|
| `exact_apply` (stored order) vs `to_scipy` | 2.0e-16 / 1.8e-16 | 2.8e-16 / 2.7e-16 | 3.3e-16 / 3.0e-16 |
| `exact_apply` vs `apply_structured` (Python) | 2.0e-16 / 1.8e-16 | 1.9e-16 / 1.8e-16 | 3.3e-16 / 3.0e-16 |
| `exact_apply` vs C++ eager `lap_rand_V` | 2.0e-16 / 1.8e-16 | 1.9e-16 / 1.8e-16 | 3.3e-16 / 3.0e-16 |
| `to_scipy` vs `apply_structured` | 2.0e-16 / 1.8e-16 | 1.9e-16 / 1.8e-16 | 3.3e-16 / 3.0e-16 |
| `to_scipy` vs C++ eager | 2.0e-16 / 1.8e-16 | 1.9e-16 / 1.8e-16 | 3.3e-16 / 3.0e-16 |
| `apply_structured` vs C++ eager | 2.0e-16 / 1.8e-16 | 1.9e-16 / 1.8e-16 | 1.7e-16 / 1.5e-16 |

Every pairing is within 2 ulp of the row scale. Two branches of the contract have no real-data coverage in these three cases: the Dirichlet-wins rule (a solid point on a Dirichlet face) and the clamp-adjacent tolerance (no closure row has psi outside [1e-4, 1 - 1e-4]: psi runs from 1.1e-3 at heat51 to 0.999 at heat201 across the three exports). The Dirichlet-wins rule is exercised by the synthetic export of `tests/synthetic.py` (a solid slab under Dirichlet walls); the clamp-adjacent tolerance is exercised by neither the real exports nor the synthetic one (psi 0.4 throughout).

## 3. The monomial oracle: transverse R rows of the E2 order-2 export are inconsistent at the row level

Scope first, because the finding is easy to overstate. What follows is a row-level truncation statement: it says what the exported Bf/Br row of a transverse R point returns on a polynomial, read as an approximation of the second derivative at that point. It is not a measured solution error and not a convergence-order result; the R set is codimension one and stiffly coupled, and the global effect needs an MMS refinement series that phase 1 did not run. Only E2 order 2 exports were examined (all three cases use scheme E2, order 2, `interp {2, 3}`). The gated check (axial monomials, k <= 2, R rows) passes on every case as the contract specifies; the k = 3 and transverse results below are reported, not gated. The export reproduces the legacy C++ rows faithfully (check 1 agrees with the eager C++ to 2e-16 on these rows), so this is a property of the Bf/Br formulation, not of the export or of pyshoccs.

Axial monomials x_d^k (the contract's check 3). All R rows and all fluid D closure rows are exact for k = 0, 1, 2 on every case (worst residual 0.052 of the 16 eps tolerance). At k = 3:

| rows | heat51 | heat201 | heat3d65 | residual / h |
|---|---|---|---|---|
| R rows along their ray (Rx in A0, Ry in A1, Rz in A2) | 32/32 fail | 138/138 | 586/586 | 6.00 to 8.11, a function of psi (6.0 as psi tends to 0, 8.11 at psi 0.57, 6.9 at psi 0.90) |
| R rows transverse to their ray (Rx in A1 or A2, and so on) | 32/32 fail | 138/138 | 1,128/1,172 (44 pass) | exactly 6.000 |
| D closure rows, wall kind (psi = 1 Dirichlet, 4 taps) | 0/196 | 0/796 | 0/23,814 | exact through k = 3 |
| D closure rows, object kind (first fluid point of a line ending on the body) | 32/32 fail | 138/138 | 586/586 | 6.0 as psi tends to 0, 0.96 at psi 0.90, 0.07 at psi 0.99 |

The along-ray k = 3 residual c(psi) h is ordinary second-order behaviour for a closure exact through quadratics (max 0.324 / 0.081 / 0.253 at h = 0.04 / 0.01 / 0.03125, scaling with h), not a defect. The transverse rows return exactly 6h because their stencil is the psi = 1 R-point row of the floating block, (1, -2, 1, 0)/h^2, anchored at the R point and pointing away from the body (`interp_deriv_coefficients`, `src/operators/derivative.cpp:22`): a one-sided three-point second difference. `verify_claims.py` confirms on every row that a transverse row passes k = 3 if and only if its record's normal has no component in the derivative direction (the 44 rows of heat3d65, which get the symmetric interior stencil).

Transverse monomials x_c^m, c != d (added to the oracle in this pass, `pyshoccs.oracle.transverse_oracle`; reported, not gated). x_d^k is constant across the grid lines a transverse R row interpolates between, so the contract's oracle cannot see that interpolation; these monomials can. The exact second x_d-derivative of x_c^m is 0.

| case | x_c^1 fail / records | x_c^2 fail / records | max residual on x_c^2 (row, psi) | x_c^3 max |
|---|---|---|---|---|
| heat51 (h 0.04) | 0 / 64 | 32 / 64 | 0.2500 (Rx[10] in A1, psi 0.502) | 0.97 |
| heat201 (h 0.01) | 0 / 276 | 138 / 276 | 0.4836 (Rx[33] in A1, psi 0.574) | 1.63 |
| heat3d65 (h 0.03125) | 0 / 3,516 | 1,172 / 3,516 | 0.5692 (Rz[92] in A0, psi 0.629) | 2.03 |

x_c^1 is exact on every row (residual at most 5e-12), so the interpolation point and the weights are right; it is the degree that is short. On x_c^2 every transverse row fails, and the failing records are exactly the transverse rows applied to the square of the coordinate they interpolate along. On the rows that use the interior interpolant the residual equals psi(1 - psi) exactly (heat51 20/20, heat201 84/84, heat3d65 820/820 rows with a non-flat normal; the 6 flat-normal rows with the symmetric stencil give 2 psi(1 - psi)); on the rows that use the wall interpolant through a neighbouring R point (12 / 54 / 346 rows) it lies between 0.04 and 0.57. The residual does not shrink with h: 0.25, 0.48, 0.57 at h = 0.04, 0.01, 0.031. The transverse second derivative at an R point therefore carries a zeroth-order error of about psi(1 - psi) u_cc / 2.

Mechanism, read off the exported rows and `src/stencils/E2_2.cpp`: `E2_2::interp_interior` (line 33) returns the linear weights (1 - y, y); `query_interp()` returns {2, 3}, a width, not a degree. A transverse row combines the exact R value (coefficient 1/h^2) with two cross-line values on the next two grid lines, each of which carries the linear-interpolation error psi(1 - psi) h^2 u_cc / 2; under the (1, -2, 1)/h^2 difference the exact centre contributes no error and the two interpolated values contribute (-2 + 1) e / h^2 = -psi(1 - psi) u_cc / 2, independent of h. The critic re-derived both mechanisms (6h on x^3 for the one-sided difference; psi(1 - psi) for the interpolated x_c^2, 2 psi(1 - psi) for the symmetric form) and `tests/test_oracle.py` encodes them on a hand-built row.

Consequences visible elsewhere, stated at the strength the evidence supports:

- Consistent with the `t-laplacian` failure (`docs/CLEANUP_PLAN.md` Section 0a), not demonstrated. "E2 with Floating Objects" (3D) fails on rx_vec by 2 to 3 percent (for example 9.66 expected against 9.44) with f2 = x^2(y+z) + y^2(x+z) + z^2(x+y) + 3xyz + x + y + z; the x^2(y+z) term is quadratic across the lines the Rx rows interpolate between, the x_c^2 case above with error psi(1 - psi)(y+z), and the 2D case of the same test passes because 3xy + x + y + 1 is linear in each coordinate. Nobody changed the interpolation and re-ran the test, so "explains" is not claimed.
- The one-step error of every export peaks on an R point: heat51 Linf 7.61e-5 at Rx[10] (psi 0.502, the largest psi(1 - psi) in the case) against 2.5e-5 on D; heat3d65 1.14e-4 at Rz[140] against 2.2e-5 on D. With dt k = 2e-4 in heat51 that is an RHS error of about 0.4 at that point, which psi(1 - psi) |u_xx| / 2 plus the along-ray first-order term predicts for the MMS Gaussian.

What this asks of the maintainer is a numerics decision outside phase 1, and one that Part III Section 10 already names as a trigger: raise the interpolation degree, either in `E2_2::interp_interior` (the `interp_wall` branch at `E2_2.cpp:46` already uses three points) or in the R-row assembly of `interp_deriv_coefficients`, re-export, and re-run `t-laplacian` and the transverse oracle (`python -m pyshoccs.parity <export>`). The oracle now gives a row-level pass/fail for that change before any convergence study. Until the decision is made, porting work on R rows ports this row faithfully, which is what the gated protocol asks for and what both arms do.

The maintainer's response (2026-10-13): the proper way of interpolating for the second-derivative operator at cut points was never worked out, and the project's tests have focused on hyperbolic problems, where the first-derivative operators and their interpolants (polyE2_1 with `interp {2, 4}`) are the ones exercised. The finding is therefore recorded as an open numerics item for the future (`docs/CLEANUP_PLAN.md` Section 0a), not as a defect to be fixed inside the DSL work, and no tolerance is loosened to accommodate it.

## 4. The C++ fused-stage arm

From `experiments/phase1/cpp_arm/RESULTS.md` (Google Benchmark, `--benchmark_min_time=0.3s`, microseconds; raw JSON beside it; `summarize.py` renders the tables). All kernels run over the packed vector on the compact tables, built from the live Laplacian's three derivatives by `build_compact_tables` and copied to the device by `make_device_tables`, which folds the diffusivity into the taps and the exception values, adds a per-row source flag and drops exception rows on Dirichlet rows. The heat source and the Dirichlet values are held fixed in the step benchmarks, so they time kernels and not the manufactured solution. Definitions:

- graph submit: `heat::submit_rhs_graph` on the shipped graph, standalone (`bench_rhs_cutcell`, the bar) and in the `bench_tables` process.
- structured RHS, `out = k lap(u) + src`: two-launch (RangePolicy over D, then over the exception rows plus the R rows without one), two-launch MDRange (MDRangePolicy rank 3 over D, tile {2, 4, 64} in 3D and {2, 64, 1} in 2D), one-launch (both index sets in one RangePolicy, no atomics since every row has one owner), pencil two-launch (fast-axis lines of D walked in runs of equal role bytes so the inner loop has no branch and vectorises), pencil one-launch.
- graph step: one RK4 step built as `integrators::rk4` builds it (slot_zero, deep copy, slot_assign_lc, graph submit, slot_accumulate) with `update_boundary` replaced by a scatter of the fixed boundary vector: 4 submits, 13 slot launches, 4 refreshes. No stats.
- fused step: four `fused_rk_stage` launches (each computes the RHS and does `acc += w r; u_next = u0 + c r`, the fourth writes `u0 + acc`) plus four `assign_at` Dirichlet refreshes on ping-pong buffers: 8 launches in the one-launch forms, 12 in the two-launch forms. No stats.

The 4-thread default setting was run three times because macOS has no thread affinity and a single run can land threads on efficiency cores (one run put the 65^3 graph step at 7,412 us; the median of three is 4,449 us); the median-of-3 tables are the ones shown for that setting, the single run is in `bench_tables_t4.json`.

### 4.1 RHS, microseconds

1 thread:

| case | graph (bench_rhs_cutcell) | graph (in process) | two-launch | two-launch MDRange | one-launch | pencil two-launch | pencil one-launch |
|---|---|---|---|---|---|---|---|
| 2D 51^2 | 21.4 | 21.0 | 4.67 | 5.56 | 4.59 | 2.83 | 2.81 |
| 2D 71^2 | 34.7 | 33.3 | 8.95 | 10.3 | 9.23 | 4.88 | 4.84 |
| 2D 201^2 | 223 | 218 | 66.6 | 79.0 | 70.4 | 30.3 | 30.8 |
| 2D 401^2 | 851 | 827 | 257 | 306 | 273 | 113 | 114 |
| 3D 65^3 | 2,107 | 2,052 | 635 | 554 | 673 | 340 | 350 |

4 threads, libomp defaults, median of 3 repetitions:

| case | graph (bench_rhs_cutcell) | graph (in process) | two-launch | two-launch MDRange | one-launch | pencil two-launch | pencil one-launch |
|---|---|---|---|---|---|---|---|
| 2D 51^2 | 304 | 306 | 18.8 | 18.8 | 10.7 | 17.2 | 9.92 |
| 2D 71^2 | 315 | 314 | 21.2 | 22.6 | 12.3 | 19.5 | 10.8 |
| 2D 201^2 | 354 | 355 | 38.1 | 45.9 | 30.3 | 28.1 | 18.3 |
| 2D 401^2 | 566 | 571 | 97.9 | 109 | 91.3 | 53.3 | 41.3 |
| 3D 65^3 | 943 | 944 | 198 | 300 | 196 | 120 | 116 |

4 threads spin (`KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active`):

| case | graph (bench_rhs_cutcell) | graph (in process) | two-launch | two-launch MDRange | one-launch | pencil two-launch | pencil one-launch |
|---|---|---|---|---|---|---|---|
| 2D 51^2 | 51.6 | 51.5 | 6.35 | 6.18 | 5.24 | 4.33 | 3.16 |
| 2D 71^2 | 60.9 | 64.6 | 8.71 | 12.1 | 7.62 | 5.74 | 4.81 |
| 2D 201^2 | 220 | 238 | 54.1 | 58.3 | 53.1 | 24.3 | 22.5 |
| 2D 401^2 | 810 | 827 | 194 | 191 | 197 | 76.3 | 72.8 |
| 3D 65^3 | 1,647 | 1,637 | 468 | 543 | 448 | 266 | 264 |

### 4.2 RK4 step, microseconds

1 thread:

| case | graph step | fused two-launch | fused one-launch | fused pencil two-launch | fused pencil one-launch |
|---|---|---|---|---|---|
| 2D 51^2 | 93.7 | 27.4 | 28.5 | 20.2 | 20.1 |
| 2D 71^2 | 149 | 53.8 | 56.9 | 38.3 | 38.6 |
| 2D 201^2 | 948 | 399 | 428 | 282 | 282 |
| 2D 401^2 | 3,614 | 1,565 | 1,679 | 1,073 | 1,073 |
| 3D 65^3 | 8,782 | 3,624 | 3,821 | 2,798 | 2,842 |

4 threads, libomp defaults, median of 3 repetitions:

| case | graph step | fused two-launch | fused one-launch | fused pencil two-launch | fused pencil one-launch |
|---|---|---|---|---|---|
| 2D 51^2 | 1,495 | 109 | 79.9 | 106 | 75.9 |
| 2D 71^2 | 1,539 | 120 | 89.9 | 115 | 88.0 |
| 2D 201^2 | 1,727 | 218 | 192 | 185 | 148 |
| 2D 401^2 | 2,715 | 565 | 554 | 413 | 376 |
| 3D 65^3 | 4,449 | 1,107 | 1,122 | 914 | 894 |

4 threads spin:

| case | graph step | fused two-launch | fused one-launch | fused pencil two-launch | fused pencil one-launch |
|---|---|---|---|---|---|
| 2D 51^2 | 266 | 37.8 | 33.4 | 33.7 | 26.7 |
| 2D 71^2 | 321 | 56.5 | 58.1 | 48.1 | 41.2 |
| 2D 201^2 | 1,049 | 313 | 316 | 226 | 186 |
| 2D 401^2 | 3,394 | 1,085 | 1,105 | 696 | 694 |
| 3D 65^3 | 7,224 | 2,509 | 2,510 | 1,966 | 1,996 |

Ratios computed from the tables above (best structured RHS over the standalone graph submit; best fused step over the graph step at the same setting):

| case | 1 thread RHS | 1 thread step | 4 threads RHS | 4 threads step | 4 threads spin RHS | 4 threads spin step |
|---|---|---|---|---|---|---|
| 2D 51^2 | 0.13x | 0.21x | 0.033x | 0.051x | 0.061x | 0.10x |
| 2D 71^2 | 0.14x | 0.26x | 0.034x | 0.057x | 0.079x | 0.13x |
| 2D 201^2 | 0.14x | 0.30x | 0.052x | 0.086x | 0.10x | 0.18x |
| 2D 401^2 | 0.13x | 0.30x | 0.073x | 0.14x | 0.090x | 0.20x |
| 3D 65^3 | 0.16x | 0.32x | 0.12x | 0.20x | 0.16x | 0.27x |

Launch-form decision. Every form, including the plain two-launch RangePolicy the task asked for first, meets gate (2); the pencil form roughly halves the time again. The MDRange form of the D launch is slower than the flat RangePolicy everywhere except 65^3 at 1 thread (554 against 635 us) and much slower at 4 threads there (300 against 198 us), so the flat form was kept and the pencil form made the default (`launch_config::kind = pencil`). The one-launch forms pay one wake-up instead of two and win at 4 threads with sleeping libomp threads (9.9 against 17.2 us at 51x51); at 1 thread and with spinning threads they are within a few percent of their two-launch twins. The graph-built step costs 1.04x to 1.10x of four graph submits at 1 thread, 1.05x to 1.32x at 4 threads spin and 1.18x to 1.23x at 4 threads default (median of 3).

### 4.3 STREAM triad, a = b + s c, three buffers of 16.8M doubles (134 MB each)

| threads | median of 10 (ms) | GiB/s (median iteration) | GiB/s (best iteration) |
|---|---|---|---|
| 1 | 4.16 | 90.3 | 90.6 |
| 4 | 4.14 | 90.7 | 90.8 |
| 4 spin | 5.42 | 68.9 | 82.2 |
| 10 | 4.24 | 88.6 | 88.9 |

One performance core saturates the memory system at about 90 GiB/s (3 x 8 B per element, write-allocate traffic not counted); 4 and 10 threads add nothing, and spinning threads cost 10 to 25 percent, presumably through placement on efficiency cores. This answers the survey's risk row "the 96 GB/s single-thread rate is cache-assisted" (editor's report Section 5): the 1M-element figure of Part IV E (108 GiB/s on about 25 MB) was about 20 percent cache-assisted, and 90 GiB/s is the DRAM figure to project with.

### 4.4 Gate (2), RHS against the native RHS graph

Thresholds: at most 1.0x at 51x51 and at most 0.5x at 65^3 with the sphere, at the thread setting fixed in phase 1(a): 1 thread for the shipped 2D case (bar 21.0 us in Part IV E, 21.4 us in this session) and 4 threads with libomp defaults for the 65^3 submit (961 us in Part IV E, 943 us here). The other settings are shown for orientation. Condition (the critic's): the gate names the fixed-up C++ RHS graph of phase 1(b) as its bar; that graph does not exist yet, and changes (1), (2) and (4) can only make it faster, so every "met" below is provisional against a bar that will move down. The margins (0.12x to 0.22x) leave room, but the verdict is not final until phase 1(b) is measured.

| threshold | bar (this session) | two-launch | one-launch | pencil two-launch (default) | pencil one-launch | met |
|---|---|---|---|---|---|---|
| 51x51, 1 thread, at most 1.0x | 21.4 us | 4.67 us (0.22x) | 4.59 us (0.21x) | 2.83 us (0.13x) | 2.81 us (0.13x) | yes, every form, provisional |
| 51x51, 4 threads spin, at most 1.0x | 51.6 us | 6.35 us (0.12x) | 5.24 us (0.10x) | 4.33 us (0.08x) | 3.16 us (0.06x) | yes, every form, provisional |
| 65^3, 4 threads, at most 0.5x | 943 us | 198 us (0.21x) | 194 us (0.21x) | 119 us (0.13x) | 114 us (0.12x) | yes, every form, provisional |
| 65^3, 1 thread, at most 0.5x | 2,107 us | 635 us (0.30x) | 673 us (0.32x) | 340 us (0.16x) | 350 us (0.17x) | yes, every form, provisional |
| 65^3, 4 threads spin, at most 0.5x | 1,647 us | 468 us (0.28x) | 448 us (0.27x) | 266 us (0.16x) | 264 us (0.16x) | yes, every form, provisional |

The 65^3 4-thread row quotes the single run (`bench_tables_t4.json`), as `cpp_arm/RESULTS.md` does; the median of 3 shown in Section 4.1 gives 198, 196, 120 and 116 us, the same ratios to two digits (0.21x, 0.21x, 0.13x, 0.12x).

### 4.5 Gate (3), fused step against the fixed-up C++ step

The gate quotes a fixed-up step estimate of 131 us at 1 thread and between 131 us and the measured 0.266 ms at 4 threads spin, with a 0.5x threshold, against a fixed-up C++ that does not exist yet. What stands in for it is the graph-built step with the source and boundary vectors held fixed: it removes the host MMS evaluation of `fill_source` and `update_boundary`, which is the substance of change (3), and keeps the gather lists and full-height row pointers of changes (1), (2) and (4), so it is an upper bound on the fixed-up step. It measures 93.7 us at 1 thread and 266 us at 4 threads spin at 51x51, and 4,449 us at 65^3 and 4 threads (median of 3).

| setting | graph step (fixed source and boundary) | fused step (pencil one-launch) | ratio | against the survey's estimate |
|---|---|---|---|---|
| 51x51, 1 thread | 93.7 us | 20.1 us | 0.21x | 0.15x of 131 us |
| 51x51, 4 threads spin | 266 us | 26.7 us | 0.10x | 0.20x of 131 us |
| 65^3, 4 threads (median of 3) | 4,449 us | 894 us | 0.20x | 0.15x of 6.0 ms |
| 65^3, 1 thread | 8,782 us | 2,842 us (2,798 two-launch) | 0.32x | 0.23x of 12.4 ms |

The fused step is under 0.5x of both the graph-built proxy and the survey's estimate at every setting, provisionally for the reason given under gate (2). The "against the estimate" column compares a step without stats to an estimate that includes 26.9 us of stats; against the estimate's 104 us of submits and residuals the 1-thread ratio is 0.19x, which changes no verdict. Two further readings: at 4 threads spin the graph-built step (266 us) equals the shipped step of Part IV E (0.266 ms), so at that setting change (3) alone buys nothing and the fused step (26.7 us) is where the gain is; and at 1 thread the graph-built step plus stats is 93.7 + 26.9 = 120.6 us, which Section 6 uses as the stop-rule proxy.

### 4.6 Gate (4), bandwidth: not measured

The 128^3 sphere case is not in the benchmark's case list, so the gate is not judged. For orientation only: with the survey's byte count (49 B per D point plus 20 B per exception nonzero, 15.6 MB per stage at 65^3) the fused pencil stage at 4 threads (894 / 4 = 224 us including its Dirichlet refresh) moves 65 GiB/s, 72 percent of the 90.7 GiB/s triad; but at 65^3 the working set is L2-resident, so this is not the DRAM-bound measurement the gate asks for. What the triad does settle is the ceiling the gate is a fraction of: one performance core already reaches 90.3 GiB/s on 134 MB buffers (Section 4.3).

### 4.7 Gate (5), table memory for the 64^3 sphere: computed, not measured

Computed by `experiments/phase1/gate5_memory.py` from the heat3d65 export (65^3 points, the 64-cell grid of the gate), with the role bits, the exception CSR, the taps and the per-direction COO part codes and line tables as inputs. The compact footprint is n_D bytes of role bits, 4 B per exception row (`exc_rows`, int32), 8 B per row pointer (`exc_rowptr`, int64, n_exc + 1 entries), 12 B per exception nonzero (8 B value, 4 B int32 column) and 8 B per tap. The legacy footprint per direction is `meta_d`, 13 ints x 4 B per kept line (`src/matrices/inner_block_meta.hpp`), plus `coeffs_d`, 8 B x (part-0 entries + part-2 entries + 3 x kept lines) (every left and right dense coefficient and one 3-tap interior per line), plus 16 B per CSR nonzero over parts 3 and 5 to 10 (8 B value, 8 B int64 column: B, Bf_x, Br_x, Bf_y, Br_y, Bf_z, Br_z), not counting the full-height row pointers; kept lines are the rows of `lines{d}` minus those with `lines{d}_skipped` set.

| compact part | bytes |
|---|---|
| role bits (n_D = 274,625) | 274,625 |
| exception rows, 24,012 x 4 B | 96,048 |
| row pointers, 24,013 x 8 B | 192,104 |
| exception nonzeros, 106,337 x 12 B | 1,276,044 |
| taps, 9 x 8 B | 72 |
| compact total | 1,838,893 |

| direction | lines | skipped | kept lines | part 0 | part 1 | part 2 | CSR entries (parts 3, 5 to 10) | meta_d B | coeffs_d B | CSR value + column B |
|---|---|---|---|---|---|---|---|---|---|---|
| x | 4,324 | 256 | 4,068 | 16,173 | 723,582 | 16,173 | 3,836 | 211,536 | 356,400 | 61,376 |
| y | 4,322 | 256 | 4,066 | 16,167 | 723,594 | 16,167 | 3,845 | 211,432 | 356,256 | 61,520 |
| z | 4,322 | 256 | 4,066 | 16,167 | 723,594 | 16,167 | 3,788 | 211,432 | 356,256 | 60,608 |
| legacy total | | | | | | | | | | 1,886,816 |

Compact 1,838,893 B (1.84 MB) against legacy 1,886,816 B (1.89 MB): ratio 0.975, so the gate holds as computed, by 2.5 percent. The margin is thin because the compact exception table carries a 4-byte column with every dense closure coefficient that the legacy `coeffs_d` stores bare: the 106,337 exception nonzeros, 97k of them left and right dense closure entries, are 69 percent of the compact bytes, while the legacy per-line metadata the compact form removes is 634 KB here. For orientation, the same formula gives 0.93x at heat51 and 1.25x at heat201, where one role byte per D point outweighs a thin legacy; the gate names only the 64^3 sphere. The legacy full-height row pointers of B and N (editor's report Section 2.4), excluded by the gate's definition, would add 8 B x (n_D + 1) = 2.2 MB per CSR to the legacy side, more than the whole compact table.

## 5. The Python array arm

From `experiments/phase1/py_arm/RESULTS.md` (`python/pyshoccs/bench/bench_heat.py` on the three exports; raw JSON per case, setting and run beside it; `summarize.py` renders the tables). Protocol: per backend and metric the median of 100 timed calls at 51x51, 60 at 201x201 and 20 at 65^3 after 10 (5 at 65^3) warm-ups; every cell run twice in separate processes and the lower median kept, metric by metric (disclosed; it biases down by a few percent). The four CPU fp64 backends (scipy CSR over V, structured NumPy, torch eager, torch.compile) ran in the sandbox at 1 thread, 4 threads and 4 threads spin; the MLX CPU backend and the two fp32 GPU backends (torch MPS eager, MLX GPU compiled) need a Metal device and ran outside the sandbox with one host thread (`run_matrix.py`, setting `metal`). A step is one RK4 step from `u0_V` with the export's `dt`, including the three stage boundary updates, the final `update_boundary(t0 + dt)` and the stats of the new state, so it is comparable with the shipped C++ step; the RHS is `k A u + src(t)` with the Dirichlet rows zeroed, applied to `rand_V`. The correctness columns are the stepped state against the exporter's `u1_V` (relative to max|u1|) and the state after 10 chained steps against the fp64 SciPy path. The thread setting does not move the NumPy backends (scipy's CSR matvec and NumPy's ufuncs are single-threaded), so their 4-thread rows repeat the 1-thread numbers to within noise.

### 5.1 heat51 (51x51, n_D 2,601, n_V 2,633, 250 exception rows)

Bar (Part IV E): RHS graph submit 21 us 1t / 309 us 4t / 54 us 4t spin; shipped step 0.856 ms 1t / 2.26 ms 4t / 0.266 ms 4t spin.

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

### 5.2 heat201 (201x201, n_D 40,401, n_V 40,539, 1,030 exception rows)

Bar (Part IV E): RHS graph submit 217 us 1t / 358 us 4t / 228 us 4t spin; shipped step 6.82 ms 1t / 4.05 ms 4t / 2.17 ms 4t spin.

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

### 5.3 heat3d65 (65^3 sphere, n_D 274,625, n_V 275,211, 24,012 exception rows)

Bar (Part IV E): RHS graph submit 2,065 us 1t / 961 us 4t / 1,810 us 4t spin; shipped step 58.4 ms 1t / 20 ms 4t / 17.5 ms 4t spin.

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

### 5.4 Gate (2) for the array arm

Same thresholds and settings as Section 4.4, same condition: the bar is the shipped graph submit standing in for the fixed-up one, so a "met" is provisional while a "not met" stands.

| threshold | bar | scipy CSR | structured NumPy | torch eager | torch.compile | best | met |
|---|---|---|---|---|---|---|---|
| 51x51, 1 thread, at most 1.0x | 21 us | 15.9 us (0.76x) | 29.7 us (1.41x) | 56.8 us (2.70x) | 23.8 us (1.13x) | scipy CSR 0.76x | yes, provisional |
| 51x51, 4 threads spin, at most 1.0x | 54 us | 15.8 us (0.29x) | 29.3 us (0.54x) | 57.2 us (1.06x) | 19.8 us (0.37x) | scipy CSR 0.29x | yes, provisional (orientation) |
| 51x51, 4 threads default, at most 1.0x | 309 us | 15.8 us (0.051x) | 30.4 us (0.098x) | 57.8 us (0.19x) | 100.5 us (0.33x) | scipy CSR 0.051x | yes, provisional (orientation) |
| 65^3, 4 threads default, at most 0.5x | 961 us | 1,058.5 us (1.10x) | 1,769.6 us (1.84x) | 1,959.4 us (2.04x) | 745.0 us (0.78x) | torch.compile 0.78x | no |
| 65^3, 1 thread, at most 0.5x | 2,065 us | 1,031.9 us (0.50x) | 1,738.5 us (0.84x) | 2,287.0 us (1.11x) | 1,436.6 us (0.70x) | scipy CSR 0.4997x | no: inside run-to-run noise (second run 1,078 us, 0.52x), not a pass |
| 65^3, 4 threads spin, at most 0.5x | 1,810 us | 1,062.2 us (0.59x) | 1,778.7 us (0.98x) | 1,636.4 us (0.90x) | 667.0 us (0.37x) | torch.compile 0.37x | yes, but not the gate's setting for the 65^3 submit |

Verdict: met at 51x51 with scipy CSR at 0.76x (the only backend under 1.0x at 1 thread; torch.compile 1.13x, structured NumPy 1.41x); not met at 65^3 at the fixed 4-thread setting, where the best RHS is torch.compile at 745 us, 0.78x of 961 us against a 0.5x threshold (scipy CSR 1.10x). The 1-thread 65^3 cell at 0.4997x is inside run-to-run noise and is not a pass. The 0.37x cell with spinning threads is a real measurement, but the gate fixed libomp defaults for the 65^3 submit because that is the setting at which the C++ bar is fastest (961 against 1,810 us); choosing the slower bar for the comparison is not the gate.

Comparison with the C++ arm at the gate settings (Section 4): the pencil one-launch RHS is 2.81 us at 51x51 and 1 thread and 114 us at 65^3 and 4 threads (the single run, as in both RESULTS files; the median of 3 gives 116 us and 6.4x), so the best Python RHS is 5.7x and 6.5x slower than the compact-table C++ kernel (3.0x at 65^3 and 1 thread: 1,032 against 340 us); the best Python step is 4.4x slower than the fused pencil step at 51x51 and 1 thread (89 against 20.1 us), 2.3x at 51x51 and 4 threads spin (61 against 26.7 us), 4.2x at 65^3 and 4 threads (3.74 against 0.894 ms) and 2.2x at 65^3 and 1 thread (6.09 against 2.80 ms).

Best Python step per setting, nanoseconds per D point per step (the shipped C++ step in parentheses): heat51 34 (329) at 1 thread, 39 (869) at 4 threads, 23 (102) at 4 threads spin; heat201 19 (169), 16 (100), 8 (54); heat3d65 22 (213), 14 (73), 13 (64). The best fp64 Python step per setting is 4 to 22x under the shipped step; against the discretization bar of four submits it sits at 0.7x to 1.1x at 1 thread and 0.3x to 0.5x with spinning threads, which is where the Part IV D synthetic result had placed it.

### 5.5 Step decomposition

The structured NumPy backend is the one whose step can be taken apart on the same buffers: operator (4x), source (4x, the hoisted manufactured solution, one small dgemv per call), RHS glue (4x: the scale by k, the scatter-add of the source onto the fluid and R rows, the zero fill of the Dirichlet rows), boundary update (4x), stats (1x) and the stage arithmetic measured on its own. The C++ column is Part IV E's decomposition with its remainder (slot ops, graph launch, logging). The two gate settings follow; the other four tables are in `py_arm/RESULTS.md`.

heat51, 1 thread:

| component | Python structured ms | share | C++ ms (Part IV E) | Python / C++ |
|---|---|---|---|---|
| operator (4x) | 0.059 | 41% | 0.084 | 0.71x |
| source (4x) | 0.014 | 9% | 0.239 | 0.058x |
| RHS glue: scale, scatter-add, zero fill (4x) | 0.046 | 31% | (in the step remainder) | |
| boundary update (4x) | 0.0033 | 2% | 0.098 | 0.034x |
| stats (1x) | 0.019 | 13% | 0.027 | 0.70x |
| stage axpy and fills (1x) | 0.0075 | 5% | (in the step remainder) | |
| sum of parts | 0.148 | 102% | 0.448 (RHS, source, boundary, stats) | 0.33x |
| measured step (with stats) | 0.146 | 100% | 0.856 (shipped step) | 0.17x |
| remainder | -0.0022 | -2% | 0.408 (slot ops, graph launch, logging) | |

heat3d65, 4 threads:

| component | Python structured ms | share | C++ ms (Part IV E) | Python / C++ |
|---|---|---|---|---|
| operator (4x) | 5.23 | 59% | 3.84 | 1.36x |
| source (4x) | 0.104 | 1% | 9.72 | 0.011x |
| RHS glue: scale, scatter-add, zero fill (4x) | 1.75 | 20% | (in the step remainder) | |
| boundary update (4x) | 0.107 | 1% | 3.62 | 0.030x |
| stats (1x) | 0.795 | 9% | 0.955 | 0.83x |
| stage axpy and fills (1x) | 0.694 | 8% | (in the step remainder) | |
| sum of parts | 8.67 | 98% | 18.14 (RHS, source, boundary, stats) | 0.48x |
| measured step (with stats) | 8.88 | 100% | 19.96 (shipped step) | 0.44x |
| remainder | 0.208 | 2% | 1.82 (slot ops, graph launch, logging) | |

Reading the 65^3 column: the operator is 57 to 59 percent of the Python step and the RHS glue about 20 percent, so the two passes a fused kernel removes (the separate scale and the scatter of the source) cost as much as the stats and the stage arithmetic together; the source and boundary evaluations, which dominate the shipped C++ step (fill_source is 49 percent of it at 4 threads), are 1 to 2 percent here because the Gaussian's spatial factors are hoisted. The scipy CSR backend has no such split (one matvec over V), which is why its RHS is the cheapest fp64 one at every size at 1 thread: one pass over 1.5 to 3.4 million stored entries against the structured form's five passes over the padded D block plus the exception CSR. The Python remainder (call overhead, ufunc dispatch) is 2 to 5 percent at 201x201 and 65^3 and within noise at 51x51.

### 5.6 GPU fp32 rows and the MLX fp64 finding

| backend | case | RHS ms | step ms | step vs shipped C++ step (1t / 4t spin) | u1 vs C++ fp64 | drift after 10 steps vs fp64 NumPy |
|---|---|---|---|---|---|---|
| MLX GPU compiled (fp32) | heat51 | 0.251 | 0.292 | 0.34x / 1.10x | 5.9e-08 | 7.6e-07 |
| | heat201 | 0.270 | 0.441 | 0.065x / 0.20x | 8.9e-08 | 2.1e-06 |
| | heat3d65 | 0.667 | 2.09 | 0.036x / 0.12x | 1.1e-07 | 6.4e-06 |
| torch MPS eager (fp32) | heat51 | 0.359 | 1.19 | 1.39x / 4.47x | 7.5e-08 | 6.4e-07 |
| | heat201 | 0.449 | 1.46 | 0.21x / 0.67x | 8.1e-08 | 2.3e-06 |
| | heat3d65 | 1.76 | 6.98 | 0.12x / 0.40x | 1.0e-07 | 6.9e-06 |

MLX GPU compiled at 65^3 is the fastest step measured in either precision (2.09 ms; 0.59x of the best CPU fp64 step, torch.compile at 3.54 ms with spinning threads); at 51x51 its step costs little more than one RHS (0.29 against 0.25 ms) because a fixed per-evaluation floor of about 0.2 ms dominates, and it loses to the CPU there. torch MPS eager is launch-bound and no better than the CPU paths. The fp32 drift after 10 steps (7.6e-7 to 6.9e-6 relative) is where every fp64 CPU path sits at 1e-15 to 1e-14; these runs say nothing about small-psi closures (the smallest psi in the three cases is about 1e-3) and are not the fp32/df64 study. The stats of each GPU step are computed on the host from two device-to-host copies, as the C++ step computes them each step.

MLX's CPU `cos`, `sin` and `exp` on float64 arrays are evaluated in single precision (`py_arm/mlx_f64_probe.py`: relative error 7e-8 to 1.2e-7, every other op of the step exact), which is why the MLX CPU row agrees with `u1_V` only to 1.8e-8 although its arrays are float64; only the MMS time factors go through those functions here. The MLX CPU stream is also slow on this workload (28 ms per step at 65^3, 4.7x the scipy step), so it was not pursued further. Inside the Bash sandbox `import mlx.core` fails even for the CPU stream and torch reports no MPS device (`smoke_heat51.json` records them as skipped), which is why the Metal rows were run outside it. torch.compile's first step costs 0.5 s per process with a warm inductor cache (2.2 to 2.5 s when the 4-thread variant compiles afresh); MLX compile 0.8 to 6 s in the first process of a case.

### 5.7 What this means for piece 3

The array arm is a correct runtime: every fp64 backend reproduces the C++ one-step state to 1.1e-16 and the stats exactly on all three cases, and its best backend per setting is 4 to 22x under the shipped C++ step. It is also slow against the discretization: it meets gate (2) at 51x51 only through scipy's CSR matvec at 1 thread (0.76x) and misses it at 65^3 (0.78x against 0.5x at the fixed setting), and at every setting it is 2 to 7x behind the compact-table C++ kernels on the same tables. Its fastest forms are scipy CSR at one thread and torch.compile at four. On this evidence the Kokkos CPU backend (piece 3 of the plan, editor's report Section 4.1) is not optional: the condition under which it would become so, the array arm alone passing gate (2) with real tables, did not occur at 65^3, and the gate's bar can only move down once phase 1(b) exists. The array arm's place is the one Part III gave the NumPy/SciPy role: front end, `to_scipy()` provider and parity oracle, not the hot loop.

## 6. What this does to the roadmap

Findings and options; no gate's definition changes here.

The stop rule on measured proxies. The rule (editor's report Section 4.3, Part III Section 10) stops at phase 2 if the native fixed-up heat.lua step is within 1.5x of four native RHS graph submits plus stats, and is to be judged after phase 1(b). Phase 1(b) was not built, but the graph-built step with fixed source and boundary vectors (Section 4.5) is a measured proxy for change (3) alone: 93.7 us at 51x51 and 1 thread, plus Part IV E's 26.9 us of stats, is 120.6 us, 1.09x of the 111 us base (4 x 21.0 + 26.9) and under the 166 us threshold, where the survey's estimate said 1.18x. At 65^3 and 4 threads the same proxy is 4,449 + 955 = 5,404 us, 1.13x of 4,799 us (threshold 7.2 ms; the estimate said 1.25x). So the rule fires on the proxy at both sizes: once the host MMS work is removed, the shipped C++ step is almost entirely its discretization, and the remaining changes (1), (2) and (4) can only make that more true. The one route found below the discretization bar is the fused compact-table C++ step: 20.1 us at 51x51 and 1 thread, 0.18x of the 111 us base (47 us, 0.42x, if the shipped stats are added back), 894 us at 65^3 and 4 threads against 4,799 us. Phases 2 and later therefore rest where Section 4.3 said they would if the rule fired: on gate (2), which the C++ arm meets with a factor of 4 to 8 in hand and the array arm does not at 65^3; on the Python front end and `to_scipy()`; and on the ensemble GPU question of phase 4, which the fp32 rows of Section 5.6 inform but do not decide.

The array arm's role. Correct, slow, and not the hot loop (Section 5.7): it is the driver and the oracle. Its scipy CSR path over V is the `to_scipy()` the stability cascade and the BO loop need, and its parity protocol is the test every later backend runs against.

What phase 2 should now contain. The nanobind module over the C++ tables and kernels (`build_compact_tables`, `make_device_tables`, `structured_rhs`, `fused_rk_stage`, `assign_at`, the exporter's arrays as DLPack views), so that the Python package drives the measured C++ kernels instead of re-implementing them; the Python package `pyshoccs` as the driver, `to_scipy()` provider and parity oracle, with the step loop calling the fused stage; the R-ordering checksum; and the phase-2 gate as written (`to_scipy()` against the legacy matvec on 100 random vectors, of which the parity protocol runs one per export, three in all). The schedule runner and the remaining primitives of phase 3 are unchanged; the fused stage covers the heat step's axis stencil, exception rows, RK axpy pair, source add and Dirichlet fill, and lacks the stats partials that B11's definition includes (the benchmarks time the step without stats).

Still owed from phase 1, in the order the gates need them:

1. The four frozen fixes on a tagged branch (phase 1(b)), so gates (2) and (3) can be judged against the fixed-up graph and lose their "provisional". The C++ arm's margins (0.12x to 0.22x on the RHS, 0.10x to 0.32x on the step) make a reversal unlikely, but the gate says fixed-up and the measurement says shipped.
2. The 128^3 sphere case and gate (4). The triad ceiling is now measured (90.3 GiB/s at one thread, Section 4.3); the stage at 128^3 is not.
3. The fp32/df64 study on the dumped operators, which decides whether phase 4 is fp32-first, df64-only or deleted. The GPU rows of Section 5.6 are the only fp32 datapoints and do not cover small psi.
4. The numerics item on the transverse R rows (Section 3), noted for the future at the maintainer's request rather than scheduled: when it is taken up, raise the interpolation degree, re-export, re-run `t-laplacian` and the transverse oracle. Every arm ports the current row faithfully until then.

## 7. Reproduce

From the repository root with the native build in `build-host` (CLAUDE.md, "Native macOS build"). The scripts under `experiments/phase1/` keep their original output locations (`.scratch/phase1/...`, gitignored) in their constants; `run_matrix.py` and the exports' paths are absolute to the worktree they were run in and have to be edited for another checkout.

```
# exports (one per case; the 3D script is the one in heat3d65/meta.json)
cmake --build build-host --target shoccs-export t-tables t-tables_kernels bench_tables bench_rhs_cutcell
build-host/src/app/shoccs-export heat.lua --out .scratch/export/heat51
build-host/src/app/shoccs-export heat.lua --out .scratch/export/heat201 --script 'simulation.mesh.index_extents = {201, 201}'
build-host/src/app/shoccs-export heat.lua --out .scratch/export/heat3d65 --script '<script field of heat3d65/meta.json>'

# C++ tests
ctest --test-dir build-host -R t-tables

# parity (gate 1) and the oracle
cd python/pyshoccs && uv sync && uv run pytest -q
PYSHOCCS_EXPORT_DIR=../../.scratch/export/heat3d65 uv run pytest -q
uv run python -m pyshoccs.parity ../../.scratch/export/heat51 --json out/heat51.json
uv run python ../../docs/python-dsl-survey/experiments/phase1/parity/analyze.py ../../.scratch/export/heat51 ../../.scratch/export/heat201 ../../.scratch/export/heat3d65
# mixed_monomial.py, inspect_rows.py, verify_claims.py, deviating_rows.py take one export directory each

# C++ fused-stage arm
build-host/benchmarks/bench_tables --kokkos-num-threads=1 --benchmark_min_time=0.3s --benchmark_out=bench_tables_t1.json --benchmark_out_format=json
build-host/benchmarks/bench_tables --kokkos-num-threads=4 --benchmark_min_time=0.3s --benchmark_repetitions=3 --benchmark_out=bench_tables_t4_rep3.json --benchmark_out_format=json
KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active build-host/benchmarks/bench_tables --kokkos-num-threads=4 --benchmark_min_time=0.3s --benchmark_out=bench_tables_t4_spin.json --benchmark_out_format=json
build-host/benchmarks/bench_rhs_cutcell --kokkos-num-threads=1 --benchmark_filter=BM_heat_cut_rhs --benchmark_min_time=0.3s --benchmark_out=bench_rhs_cutcell_t1.json --benchmark_out_format=json
uv run --no-project python -I docs/python-dsl-survey/experiments/phase1/cpp_arm/summarize.py

# Python array arm
cd python/pyshoccs && uv sync --extra bench
python3 -I docs/python-dsl-survey/experiments/phase1/py_arm/run_matrix.py cpu                                    # sandboxed: 3 cases x (1t, 4t, 4t_spin) x 2 runs
python3 -I docs/python-dsl-survey/experiments/phase1/py_arm/run_matrix.py mlx,gpu heat51,heat201,heat3d65 metal 1,2   # outside the sandbox (Metal)
uv run --project python/pyshoccs python docs/python-dsl-survey/experiments/phase1/py_arm/mlx_f64_probe.py        # outside the sandbox
python3 -I docs/python-dsl-survey/experiments/phase1/py_arm/summarize.py

# gate (5)
uv run --no-project --with numpy python -I docs/python-dsl-survey/experiments/phase1/gate5_memory.py .scratch/export/heat3d65
```

## 8. Files

Evidence under `docs/python-dsl-survey/experiments/phase1/`:

| path | content |
|---|---|
| `cpp_arm/RESULTS.md`, `cpp_arm/summarize.py` | the C++ arm's report and the script that renders it from the JSON |
| `cpp_arm/bench_tables_t1.json`, `bench_tables_t4.json`, `bench_tables_t4_rep3.json`, `bench_tables_t4_spin.json`, `bench_tables_t10_triad.json` | `bench_tables` at 1 thread, 4 threads (single run and three repetitions), 4 threads spin, and the 10-thread triad |
| `cpp_arm/bench_rhs_cutcell_t1.json`, `_t4.json`, `_t4_spin.json` | the shipped `BM_heat_cut_rhs` bar, same session |
| `py_arm/RESULTS.md`, `py_arm/summarize.py`, `py_arm/run_matrix.py`, `py_arm/mlx_f64_probe.py` | the Python arm's report, its renderer, the run matrix driver and the MLX float64 probe |
| `py_arm/<case>_<setting>_run<n>.json` (24 files), `py_arm/smoke_heat51.json` | one `bench_heat.py` output per case (heat51, heat201, heat3d65), setting (1t, 4t, 4t_spin, metal) and run (1, 2); the sandboxed smoke run that records the Metal backends as skipped |
| `parity/REPORT.md` | the parity protocol report and the monomial-oracle finding |
| `parity/heat51.json`, `heat201.json`, `heat3d65.json` | `python -m pyshoccs.parity` output per case |
| `parity/analysis_heat51.json`, `analysis_heat201.json`, `analysis_heat3d65.json`, `parity/analyze.py` | the independent re-derivation (row accounting, bitwise structural cross-check, sum-order spread) |
| `parity/mixed_monomial.py`, `inspect_rows.py`, `verify_claims.py`, `deviating_rows.py` | the oracle investigation scripts (transverse monomials, row dumps, the normal-component claim, the wall-interpolant rows) |
| `export_timings.json` | process wall of the three exports and the byte-identity check of the re-export |
| `gate5_memory.py` | the gate (5) computation of Section 4.7 |

Not copied: the exports themselves (`.scratch/export/{heat51,heat201,heat3d65}`, 103 MB, regenerated by the commands above), the run logs, and the critic's working directory (`CRITIQUE.md`, `spot_check.py`, `warn_check.py`, the re-exports), whose findings are folded into the wording of Sections 1, 3, 4 and 5. Code: `src/app/export_tables.cpp`, `src/io/npy.hpp`, `src/operators/tables.hpp`, `tables.cpp`, `tables.t.cpp`, `tables_kernels.hpp`, `tables_kernels.t.cpp`, `benchmarks/bench_tables.cpp`, `python/pyshoccs/`; contract and protocol: `docs/reference/pyshoccs-export.md`.
