# Phase-1 parity protocol on the real exports (2026-10-13, native macOS build)

Exports: .scratch/export/heat51 (heat.lua as shipped, h 0.04), heat201 (201x201, h 0.01), heat3d65 (65^3 sphere, h 1/32). All three are byte-fresh (mtime: tables.cpp/export_tables.cpp < build-host/src/app/shoccs-export < meta.json), so no re-export was needed. Parity JSON: docs/python-dsl-survey/experiments/phase1/parity/{heat51,heat201,heat3d65}.json (from `python -m pyshoccs.parity`); independent cross-checks: docs/python-dsl-survey/experiments/phase1/parity/analysis_<case>.json (from analyze.py).

## 1. Gated checks: ALL PASSED on all three cases

Row classes (from the export alone):

| case | n_V | solid_D | dirichlet_D | fluid_D | clamp_D | R | R_dirichlet |
|---|---|---|---|---|---|---|---|
| heat51 | 2633 | 56 | 200 | 2345 | 0 | 32 | 0 |
| heat201 | 40539 | 940 | 800 | 38661 | 0 | 138 | 0 |
| heat3d65 | 275211 | 717 | 24578 | 249330 | 0 | 586 | 0 |

Row accounting (analyze.py): rows with any entry are exactly the fluid_D and R rows (2377 / 38799 / 249916); no solid, Dirichlet-face or Dirichlet-R row has an entry and lap_rand_V is exactly 0 on all of them; no fluid or R row lacks entries. Dirichlet face lists: 204 / 804 / 25350 with corner duplicates (200 / 800 / 24578 unique). No solid point lies on a Dirichlet face in these cases, and no clamp-adjacent rows exist, so those two contract branches remain untested on real data.

| check | heat51 | heat201 | heat3d65 |
|---|---|---|---|
| 1 operator identity, max abs / rel to max\|lap_rand\| | 9.1e-13 / 2.0e-16 | 1.5e-11 / 1.9e-16 | 3.6e-12 / 3.3e-16 |
| 1 worst row (class) | D(22,45) fluid | D(1,34) fluid | D(6,6,4) fluid |
| 2 rhs(rand_V) max abs / rel | 2.8e-14 / 1.9e-16 | 6.8e-13 / 2.7e-16 | 1.7e-13 / 4.7e-16 |
| 2 rhs(u0_V) max abs / rel to max\|rhs_u0\| | 1.6e-14 / 1.4e-12 | 4.9e-13 / 3.1e-11 | 9.1e-14 / 3.1e-12 |
| 3 R rows k <= 2: fail / rows, worst ratio to tol | 0 / 192, 0.052 | 0 / 828, 0.047 | 0 / 5274, 0.034 |
| 4 step identity, max abs / rel | 2.2e-16 / 1.1e-16 | 2.2e-16 / 1.1e-16 | 2.2e-16 / 1.1e-16 |
| 4b stats (11 numbers, 4 indices) | 11/11 exact | 11/11 exact | 11/11 exact |
| 5 structural identity, max abs / rel | 9.1e-13 / 2.0e-16 | 1.5e-11 / 1.9e-16 | 3.6e-12 / 3.3e-16 |

(The larger "rel" of check 2 on u0_V is against max|rhs_u0|, which is small because the MMS time derivative vanishes at t0; against the contract's per-row tolerance k 1e-13 |row|_1 max|u| + 1e-13 |src| the worst ratio is about 0.05, like the other checks.)

Tests: `uv run pytest -q` in python/pyshoccs: 39 passed with PYSHOCCS_EXPORT_DIR set to each of the three exports and with the default lookup (.scratch/export/heat51); 31 passed, 8 skipped with PYSHOCCS_EXPORT_DIR pointing at a non-existent directory. `ctest --test-dir build-host -R t-tables`: t-tables Passed, t-tables_kernels Passed (the latter is a concurrent session's test; C++ was not touched in this pass). t-laplacian still fails as before (see Section 4).

## 2. Structural cross-check (step 4), bitwise

analyze.py rebuilds the exporter's compact CSR from exc_rows / exc_rowptr (int64) / exc_col (int32) / exc_val and compares it with pyshoccs.structure.derive_structure (from the COO alone): role.npy equal as bytes on all three; exc_rows equal (sorted, unique); exc_rowptr equal; the exporter's columns are already sorted per row and equal the Python ones; values bitwise equal on heat51 and heat3d65, and on heat201 equal except 2 entries where the exporter holds +0.0 and the Python duplicate sum -0.0 (stored zeros of psi = 1 closures, kept on purpose per Section 2.3: 260 / 1082 / 2334 zero entries). Max relative value difference 0.0 everywhere.

| case | interior rows per direction | exception rows (D + R) | exception nnz | max nnz per exc row | taps |
|---|---|---|---|---|---|
| heat51 | 2231, 2231, 0 | 250 (218 + 32) | 1238 | 12 | 625, -1250, 625 |
| heat201 | 38195, 38193, 0 | 1030 (892 + 138) | 5164 | 13 | 1e4, -2e4, 1e4 |
| heat3d65 | 241194, 241198, 241198 | 24012 (23426 + 586) | 106337 | 20 | 1024, -2048, 1024 |

interior_rows[d] * width == number of part-1 entries of A_d holds in every direction; every part-1 coefficient equals taps.npy to 4 eps.

## 3. fp64 sum-order spread on rand_V (step 5)

Max |difference| as a fraction of max|lap_rand| (global) / of |row|_1 max|u| (per row):

| pair | heat51 | heat201 | heat3d65 |
|---|---|---|---|
| exact_apply (stored order) vs to_scipy | 2.0e-16 / 1.8e-16 | 2.8e-16 / 2.7e-16 | 3.3e-16 / 3.0e-16 |
| exact_apply vs apply_structured (Python) | 2.0e-16 / 1.8e-16 | 1.9e-16 / 1.8e-16 | 3.3e-16 / 3.0e-16 |
| exact_apply vs C++ eager lap_rand_V | 2.0e-16 / 1.8e-16 | 1.9e-16 / 1.8e-16 | 3.3e-16 / 3.0e-16 |
| to_scipy vs apply_structured | 2.0e-16 / 1.8e-16 | 1.9e-16 / 1.8e-16 | 3.3e-16 / 3.0e-16 |
| to_scipy vs C++ eager | 2.0e-16 / 1.8e-16 | 1.9e-16 / 1.8e-16 | 3.3e-16 / 3.0e-16 |
| apply_structured vs C++ eager | 2.0e-16 / 1.8e-16 | 1.9e-16 / 1.8e-16 | 1.7e-16 / 1.5e-16 |

Every pairing is within 2 ulp of the row scale.

## 4. Check 3, the monomial oracle: the finding

### 4.1 Axial monomials x_d^k (the contract's check)

All R rows and all fluid D closure rows are exact for k = 0, 1, 2 on every case (worst residual 0.052 of the 16 eps tolerance; absolute 7.7e-13 / 6.3e-12 / 5.4e-13). At k = 3:

| rows | heat51 | heat201 | heat3d65 | residual / h |
|---|---|---|---|---|
| R rows along their ray (Rx in A0, Ry in A1, Rz in A2) | 32/32 fail | 138/138 | 586/586 | 6.00 to 8.11, a function of psi: 6.0 at psi -> 0, 8.11 at psi 0.57, 6.9 at psi 0.90 |
| R rows transverse to their ray (Rx in A1/A2, ...) | 32/32 fail | 138/138 | 1128/1172 (44 pass) | exactly 6.000 (median = max) |
| D closure rows, wall kind (psi = 1 Dirichlet, 4 taps) | 0/196 | 0/796 | 0/23814 | exact through k = 3 |
| D closure rows, object kind (first fluid point of a line ending on the body; one per record) | 32/32 fail | 138/138 | 586/586 | 6.0 at psi -> 0, decreasing to 0.96 at psi 0.90 and 0.07 at psi 0.99 |

So the D closure rows stop being exact at k = 3, and only the object closures do: the Dirichlet wall closure is a four-point formula exact for cubics; the E2 floating closure (the 2x4 nbs block: row 0 the derivative at the R point, row 1 at the first fluid point, columns R, x1, x2, x3) is exact for quadratics only, with a first-order k = 3 residual that scales exactly with h (max 0.324 / 0.081 / 0.253 at h = 0.04 / 0.01 / 0.03125).

Worst k = 3 rows (psi, direction, ray):

| case | row | ray, record | dir | psi | position | residual | res / h |
|---|---|---|---|---|---|---|---|
| heat51 | V[2622] | Ry[5] | 1 (along) | 0.540 | (0.88, 1.298) | 0.324 | 8.10 |
| heat51 | V[2611] | Rx[10] | 0 (along) | 0.502 | (0.780, 1.2) | 0.323 | 8.07 |
| heat51 | V[1408] = D(27,31) | object closure of Rx[13] | 0 | 0.0011 | (1.08, 1.24) | 0.240 | 6.00 |
| heat201 | V[40490] | Ry[21] | 1 (along) | 0.573 | (0.87, 1.294) | 0.0811 | 8.11 |
| heat201 | V[21832] = D(108,124) | object closure of Rx[55] | 0 | 0.0045 | (1.08, 1.24) | 0.0599 | 5.99 |
| heat3d65 | V[274750] | Rx[125] | 0 (along) | 0.577 | (1.013, 1.156, 1.156) | 0.253 | 8.11 |
| heat3d65 | V[150442] = D(35,39,32) | object closure of Rx[157] | 0 | 0.0069 | (1.094, 1.219, 1.0) | 0.187 | 5.99 |

Every transverse R row has residual exactly 6h because its stencil is the psi = 1 R-point row of the floating block, (1, -2, 1, 0)/h^2, anchored at the R point and pointing away from the body (interp_deriv_coefficients in src/operators/derivative.cpp): a one-sided three-point second difference centred one cell from the R point, which returns 6(x +- h) on x^3. Verified on every row (verify_claims.py): a transverse row passes k = 3 if and only if its record's normal has no component in the derivative direction (the 44 passing rows of heat3d65, which get the symmetric interior stencil centred on the point).

### 4.2 Transverse monomials x_c^m, c != d (added to the oracle in this pass; reported, not gated)

x_d^k is constant across the grid lines a transverse R row interpolates between, so the contract's oracle cannot see that interpolation. Applying the R rows of A_d to x_c^m (exact second x_d-derivative 0):

| case | x_c^1 fail / records | x_c^2 fail / records | max residual on x_c^2 (row, psi) | x_c^3 max |
|---|---|---|---|---|
| heat51 | 0 / 64 | 32 / 64 (every transverse row) | 0.2500 (Rx[10] in A1, psi 0.502) | 0.97 |
| heat201 | 0 / 276 | 138 / 276 | 0.4836 (Rx[33] in A1, psi 0.574) | 1.63 |
| heat3d65 | 0 / 3516 | 1172 / 3516 | 0.5692 (Rz[92] in A0, psi 0.629) | 2.03 |

The failing records are exactly the transverse rows applied to the square of the coordinate they interpolate along (their own ray coordinate r); the third coordinate in 3D gives 0 because a row's columns lie in one plane of it. On the rows that use the interior interpolant the residual equals psi(1-psi) exactly (heat51 20/20, heat201 84/84, heat3d65 820/820 rows with a non-flat normal; the 6 flat-normal rows with the symmetric stencil give 2 psi(1-psi)); on the rows that use the wall interpolant through a neighbouring R point (12 / 54 / 346 rows) it is between 0.04 and 0.57. It does not shrink with h (0.25, 0.48, 0.57 at h = 0.04, 0.01, 0.031): the transverse second derivative at an R point is inconsistent, with a zeroth-order error of about psi(1-psi) u_cc / 2.

Mechanism, read off the exported rows and src/stencils/E2_2.cpp: interp_interior returns the linear weights (1 - y, y) (query_interp {2, 3} is a width, not a degree); the row combines the exact R value (coefficient 1/h^2) with two cross-line values on the next grid lines that carry the interpolation error psi(1-psi) h^2 u_cc / 2, so the (1, -2, 1)/h^2 difference leaves (-2 + 1) e / h^2 = -psi(1-psi) u_cc / 2, independent of h. Linear functions of the transverse coordinate are reproduced (x_c^1 passes on every row, residual <= 5e-12), so the interpolation point and weights are right; it is the degree that is short.

Consequences visible elsewhere:
- t-laplacian "E2 with Floating Objects" (3D) fails on rx_vec by 2 to 3 percent (e.g. 9.66 expected vs 9.44) with f2 = x^2(y+z) + y^2(x+z) + z^2(x+y) + 3xyz + x + y + z: the x^2(y+z) term is quadratic across the lines the Rx rows interpolate between, i.e. the x_c^2 case above with error psi(1-psi)(y+z). The 2D case of the same test passes because 3xy + x + y + 1 is linear in each coordinate.
- The one-step stats of every export peak on an R point: heat51 Linf 7.61e-5 at Rx[10] (psi 0.502, the largest psi(1-psi)) against 2.5e-5 on D; heat3d65 1.14e-4 at Rz[140] against 2.2e-5 on D. With dt k = 2e-4 in heat51 that is an RHS error of about 0.4 at that point, which psi(1-psi)|u_xx|/2 plus the along-ray first-order term predicts for the MMS Gaussian.

This is a property of the legacy Bf/Br formulation that the export reproduces faithfully (check 1 equals the C++ apply to 2e-16 on those rows), not of the export or of pyshoccs; the gate (axial k <= 2) passes as the contract specifies. Whether to raise the interpolation degree (the interp_wall branch already uses three points) is a numerics decision outside this phase.

## 5. Fix log

No gated check failed on any case, so no correctness fix was needed on either side. Changes made, all Python/docs:
1. python/pyshoccs/pyshoccs/oracle.py: added transverse_oracle (R rows of A_d on x_c^m, c != d, m = 1, 2), the `coord` and `psi` (row's own record) columns of the result table, OracleResult.combined, and a summary that iterates over the k present; the axial oracle is unchanged.
2. python/pyshoccs/pyshoccs/parity.py: check_oracle runs the transverse oracle, folds it into the reported summary (oracle_summary in the JSON) and adds two notes naming the max-residual row; the gate is unchanged.
3. python/pyshoccs/tests/test_oracle.py (new, 3 tests): a hand-built transverse R row in the legacy formulation gives residual psi(1-psi) on x_c^2 and 0 on x_c^1, the axial k = 3 residual exactly 6h, and the synthetic export's along-ray rows pass the transverse oracle (its first version wrongly expected an empty table; corrected).
4. docs/reference/pyshoccs-export.md Section 3 check 3: one sentence naming the transverse diagnostic; python/pyshoccs/README.md: the oracle row of the module table.

Observations needing no change: the two signed-zero exc_val differences in heat201; the Dirichlet-wins and clamp-adjacent contract branches have no real-data coverage in these three cases.

## 6. Reproduce
cd python/pyshoccs; uv run pytest -q; for each case: PYSHOCCS_EXPORT_DIR=../../.scratch/export/<case> uv run pytest -q; uv run python -m pyshoccs.parity ../../.scratch/export/<case> --json ../../docs/python-dsl-survey/experiments/phase1/parity/<case>.json; uv run python ../../docs/python-dsl-survey/experiments/phase1/parity/analyze.py <export dirs>; mixed_monomial.py, inspect_rows.py, verify_claims.py, deviating_rows.py take one export dir each.
