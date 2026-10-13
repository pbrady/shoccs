# Operator table export (`shoccs-export`) and the phase-1 protocol

Status: phase 1 of the thin-waist plan (`docs/PYTHON_DSL_SURVEY.md` Section 4). This page fixes the data contract between the C++ assembly and the Python runtime so the two can be developed and tested independently. Everything here is over one packed vector per scalar field and is produced once per (geometry, boundary conditions, scheme, alpha).

## 1. Packed vector V

A scalar field's four buffers are concatenated into one vector V of length `n_V`:

| segment | offset | length | meaning |
|---|---|---|---|
| D | `0` | `n_D = nx*ny*nz` | every grid point, C order, z fastest (`ic = i*ny*nz + j*nz + k`), solid points included |
| Rx | `off_Rx = n_D` | `n_Rx` | x-ray intersection records in generation order |
| Ry | `off_Ry = n_D + n_Rx` | `n_Ry` | y-ray records |
| Rz | `off_Rz = n_D + n_Rx + n_Ry` | `n_Rz` | z-ray records (empty in 2D) |

The R ordering is the one `object_geometry` generates (`src/mesh/object_geometry.cpp`, slow index outer, fast index inner, then increasing distance along the ray). It is an implicit contract shared by field buffers, B columns and Bf/Br rows, so the export carries the records themselves (psi, position, solid coordinate) and a consumer must not re-sort them.

No padding is applied by the exporter; a backend that wants a padded length pads itself.

## 2. The export directory

`shoccs-export <config.lua> --out DIR [--script '...'] [--seed N]` (target `shoccs-export`, source `src/app/export_tables.cpp`; the default seed is 20261013) builds the mesh, boundary conditions, stencil, manufactured solution and the three second-derivative operators exactly as `systems::heat` does (same `from_lua` factories, same `laplacian` construction), then writes the files below. Arrays are NumPy `.npy` version 1.0 files (little-endian, C order, written by `src/io/npy.hpp`); scalars and strings go in `meta.json`. All indices are 0-based. Index arrays are `int64` unless stated. The table code itself lives in `src/operators/tables.hpp` (`packed_layout`, `extract_coo`, `build_compact_tables`, `apply_structured`) and is checked by `t-tables`.

### 2.1 Mesh and geometry

| file | dtype, shape | content |
|---|---|---|
| `meta.json` | | `export_version`, `config`, `script`, `dims`, `extents [nx,ny,nz]`, `domain_min [3]`, `domain_max [3]`, `h [3]` (0 for a direction with extent 1), `n_D`, `n_R [3]`, `offsets {D, Rx, Ry, Rz}`, `n_V`, `grid_bcs` (3 pairs of `"dirichlet" | "floating" | "neumann"`), `object_bcs` (one string per shape), `scheme {type, order, p, interp {p, t}, query {dirichlet: {r, t, nextra}, floating: ..., neumann: ...}}`, `system {type, diffusivity}`, `mms` (for the gaussian type: `type`, `center [[...]]`, `variance [[...]]`, `amplitude [...]`, `frequency [...]`, read from the Lua table as `manufactured_solution::from_lua` reads it, so 2D entries carry a third 0), `t0`, `dt` (the shipped parabolic rule `cfl * h_min^2 / (4 k)` with the config's cfl, from `step_controller::from_lua`), `seed`, `git_commit` (omitted when `git rev-parse HEAD` is unavailable) |
| `x.npy`, `y.npy`, `z.npy` | f64 `[nx]`, `[ny]`, `[nz]` | grid coordinates |
| `R{d}_psi.npy` | f64 `[n_R[d]]` | psi of each record, d in 0,1,2 |
| `R{d}_position.npy` | f64 `[n_R[d], 3]` | intersection position |
| `R{d}_normal.npy` | f64 `[n_R[d], 3]` | outward normal |
| `R{d}_ray_outside.npy` | u8 `[n_R[d]]` | 1 if the ray leaves the body at this record |
| `R{d}_solid_coord.npy` | i32 `[n_R[d], 3]` | first solid grid point beyond the intersection |
| `R{d}_solid_ic.npy` | i64 `[n_R[d]]` | the same as a flat D index |
| `R{d}_shape_id.npy` | i32 `[n_R[d]]` | shape index |
| `lines{d}.npy` | i64 `[n_lines, 7]` | per line fragment: `stride, start_ic, end_ic, start_R (record index or -1), start_shape (-1), end_R (-1), end_shape (-1)` |
| `lines{d}_psi.npy` | f64 `[n_lines, 2]` | psi at the start and end object (NaN when the end is a wall) |
| `lines{d}_skipped.npy` | u8 `[n_lines]` | 1 if `mesh::dirichlet_line` skips this line (it lies in a Dirichlet wall of another axis), so it has no rows in O |
| `fluid_idx.npy` | i32 `[n_fluid]` | `mesh::fluid_desc()`, ascending D indices |
| `dirichlet_face_idx.npy` | i32 | the flattened Dirichlet grid-face list exactly as `heat::build_rhs_graph` builds it (duplicates at edges included, same order) |
| `R{d}_dirichlet_idx.npy`, `R{d}_non_dirichlet_idx.npy` | i32 | positions in R(d) whose shape has (not) a Dirichlet condition |

### 2.2 Stencil data

| file | dtype, shape | content |
|---|---|---|
| `taps.npy` | f64 `[3, 2p+1]` | interior coefficients per direction from `stencil::interior(h(d), ...)`; rows for directions with extent 1 are zero |
| `wall_{d}_{L|R}_{D|F|N}.npy` | f64 `[r, t]` | the psi = 1 closure block from `stencil::nbs(h(d), bc, 1.0, right, ...)` for each direction, side and BC kind the stencil supports (a kind with `r = 0` is omitted) |
| `wall_{d}_{L|R}_N_extra.npy` | f64 `[nextra]` | the Neumann extras, when present |

These are the inputs of a structural decomposition; the operator itself is in Section 2.3.

### 2.3 The exact operators

For each direction d the export holds every stored entry of the second-derivative operator `derivative{d, ...}` as raw COO entries over V, in the order the matrices store them (O line by line as left, interior, right; then B, Bf_x, Br_x, Bf_y, Br_y, Bf_z, Br_z), duplicates not summed and explicit zeros kept (a psi = 1 Dirichlet closure stores a zero coefficient, for instance):

| file | dtype, shape | content |
|---|---|---|
| `A{d}_row.npy`, `A{d}_col.npy` | i64 `[nnz]` | packed row and column |
| `A{d}_val.npy` | f64 `[nnz]` | coefficient |
| `A{d}_part.npy` | u8 `[nnz]` | 0 O left dense, 1 O interior circulant, 2 O right dense, 3 B, 5 + 2r Bf_r, 6 + 2r Br_r (r = 0, 1, 2) |
| `A{d}_line.npy` | i32 `[nnz]` | line index into `lines{d}` for O entries, -1 otherwise |
| `N{d}_row.npy`, `N{d}_col.npy`, `N{d}_val.npy` | i64, i64, f64 | the Neumann matrix N: rows in D (packed), columns in D of the separate Neumann data buffer `nu`; empty without Neumann walls (N has no part code; `coo_part::N = 4` is internal to the C++) |

Mapping rules, from `src/operators/derivative.hpp` and `derivative.cpp`:

- O: `D -> D`. Row of a left dense entry `(line, r, j)`: `row_offset + r*stride`, column `col_offset + j*stride`. Interior row r: `row_offset + (left_rows + r)*stride`, column `row + (j - w/2)*stride`. Right dense: row `row_offset + (left_rows + interior_rows + r)*stride`, column `right_col_offset + j*stride`. Coefficients come from `block::metadata_view()` and `block::coefficients_view()` (`src/matrices/block.hpp`, `inner_block_meta.hpp`).
- B: `R(d) -> D`: packed column `off_R(d) + object_coordinate`.
- Bf_r: `D -> R(r)`: packed row `off_R(r) + i`. Br_r: `R(r) -> R(r)`: packed row and column both offset by `off_R(r)`.
- The Laplacian is `A = A0 + A1 + A2` (plus `N0 + N1 + N2` applied to `nu`); a direction with extent 1 contributes nothing.

Rows that the C++ never writes (solid points, Dirichlet grid faces, the dropped Dirichlet-wall rows, Dirichlet-object R points) have no entries; the C++ zero-fills the output before accumulating, so their output is 0.

Size limit of the phase-1 exporter: `extract_coo` materialises every entry as a 32-byte record before writing, about 2.3M entries (73 MB) per direction at 65^3 and of order 50M entries (1.6 GB) per direction at 256^3 with a sphere, so a 256^3 export needs a streaming writer; columns and exception rows are `int32` and the exporter refuses an `n_V` above 2^31 - 1.

### 2.4 Oracle dumps

All over V. The MMS is the config's manufactured solution; `k` is the diffusivity.

| file | content |
|---|---|
| `u0_V.npy` | the state after `heat::initialize` and `heat::update_boundary(t0)`: MMS on fluid D points and all R points, MMS on Dirichlet grid faces, 0 on solid points |
| `rhs_u0_V.npy` | the live `heat` RHS graph (`build_rhs_graph`, `fill_source(t0)`, `submit_rhs_graph`) applied to `u0_V` |
| `rand_V.npy` | uniform random values in [-1, 1] on all of V, fixed seed |
| `lap_rand_V.npy` | the eager Laplacian (`laplacian::operator()(u, nu)` with `nu = 0`) applied to `rand_V` |
| `rhs_rand_V.npy` | the live RHS graph applied to `rand_V` (source included), at `t0` |
| `sol_t0_V.npy`, `src_t0_V.npy` | MMS value and source `ddt - k lap` at every V location at `t0` |
| `sol_t1_V.npy`, `src_t1_V.npy` | the same at `t0 + dt/2` |
| `u1_V.npy` | the state after exactly one RK4 step from `u0_V` with `dt`, including the final `update_boundary(t0 + dt)` that `integrators::rk4` performs |
| `stats_u1.json` | `{"time": t0 + dt, "stats": [...]}`: `heat::stats` of `u1_V` at `t0 + dt`, the 11 numbers of `system_stats` in order (Linf, min, max, domain Linf and index, Rx Linf and index, Ry, Rz) |

The exporter reproduces the RK4 step inline with the `slot_ops.hpp` helpers (the same sequence as `integrators::rk4::operator()`, with `fill_source` before every graph submit as `system::submit_rhs_graph` does) rather than through the `system` wrapper, so that the same `heat` object also serves the other dumps.

### 2.5 Compact tables (the C++ derivation of Section 4)

The exporter also writes its own structured decomposition so the Python derivation can be cross-checked against it:

| file | dtype, shape | content |
|---|---|---|
| `role.npy` | u8 `[n_D]` | role bits as defined in Section 4 |
| `exc_rows.npy` | i32 `[n_exc]` | the exception rows over V, sorted and unique |
| `exc_rowptr.npy` | i64 `[n_exc + 1]` | CSR row pointers into the two arrays below |
| `exc_col.npy` | i32 `[nnz_exc]` | packed columns |
| `exc_val.npy` | f64 `[nnz_exc]` | coefficients, duplicates summed, no system constant folded in |

## 3. Parity protocol (phase-1 gate 1)

Row classes over V, derived from the export alone:

- solid D points: not in `fluid_idx`;
- Dirichlet D points: in `dirichlet_face_idx` (and, for a Dirichlet wall, the wall row the scheme drops); a solid point on a Dirichlet face counts as Dirichlet, because `update_boundary` writes the whole face;
- fluid D rows away from the clamps: fluid, not Dirichlet, and every closure row they touch has psi in `[1e-4, 1 - 1e-4]`;
- clamp-adjacent D rows: fluid rows that are closure rows of a line whose object end has psi outside `[1e-4, 1 - 1e-4]`;
- R rows: the Rx, Ry, Rz segments, split into Dirichlet-object (no rows) and non-Dirichlet.

Checks:

1. Operator identity. `to_scipy()` (duplicates summed) applied to `rand_V` equals `lap_rand_V` on every row with |difference| <= 1e-13 * (|row|_1 * max|rand|) on fluid D rows away from the clamps and, on clamp-adjacent rows, the larger of that and 64 eps * max|coef| * h (both sides sum the same coefficients, so a row's summation-order error is bounded by nnz * eps * |row|_1 * max|rand| whatever its psi; the h-scaled form alone is tighter than that bound once h < 0.1). Solid, Dirichlet-face and Dirichlet-object rows must be exactly 0 on both sides. R rows are compared too, but only as a check that the export is faithful; their correctness is judged by check 3.
2. RHS identity. The Python RHS (`k * A u + src(t) on fluid and non-Dirichlet R, 0 on Dirichlet faces and Dirichlet R points`) applied to `rand_V` at `t0` equals `rhs_rand_V` under the same tolerances; applied to `u0_V` equals `rhs_u0_V`.
3. Monomial oracle for R rows. For each R row of `A_d`, apply the row to `x_d^k` evaluated at the column locations (grid coordinate for D columns, intersection position for R columns) and compare with `k (k-1) x_R^(k-2)` at the row's intersection position, for k = 0, 1, 2, with residual <= 16 eps * sum |coef * x^k|. Report k = 3 separately (not a gate). Fluid D closure rows get the same test as a diagnostic. A second diagnostic, also reported and not gated, applies the R rows of `A_d` to the transverse monomials `x_c^m` (c != d, m = 1, 2; exact value 0): `x_d^k` is constant across the grid lines a transverse R row interpolates between, so only these monomials exercise the cross-line interpolation of the Bf/Br formulation.
4. Step identity. One Python RK4 step from `u0_V` equals `u1_V` to 1e-12 relative (sum-order differences expected at 1e-15), and the Python stats equal `stats_u1.json`.
5. Structural identity. The structured form (per-direction axis stencil on rows flagged interior for that direction, plus one exception CSR for every remaining entry) applied to `rand_V` equals `to_scipy()` applied to `rand_V` to 1e-14 relative.

## 4. The structured tables

Both arms derive the same decomposition from Section 2.3 (Python in NumPy from the COO, C++ from the block metadata directly) and cross-check it:

- `role` (u8 per D point): bit d set iff the point is an interior-circulant row of a line in direction d (`A{d}_part == 1` on that row); bit 3 fluid (`fluid_idx`); bit 4 Dirichlet grid face (`dirichlet_face_idx`).
- `taps[d]` (f64 `[2p+1]`), scaled by the system constant when fused.
- `exc`: a CSR over V whose rows are every V row that has at least one entry not covered by an interior tap (left and right dense rows, rows receiving B entries, every Bf and Br row), duplicates summed, int32 columns, rows sorted and unique. Interior taps of a row that is interior in one direction and dense in another stay in the axis stencil for the interior direction; only the dense direction's entries go to `exc`.

A total output is then: `out[i] = sum_{d in role[i]} taps[d] . u[i + (j - w/2) stride_d]` for every D point (0 when no bit is set, so solid points, Dirichlet faces and dropped rows come out 0 without a separate fill), followed by `out[row] += exc[row] . u` over the exception rows. R rows have no bits, so they receive exactly their exception row.

The C++ side of this decomposition is written as the files of Section 2.5; its taps are read from the block metadata and checked against `taps.npy`. `apply_structured` in `src/operators/tables.hpp` is the serial reference of the total output above, and `t-tables` checks it against the eager Laplacian on the floating-object setups of `laplacian.t.cpp`.

## 5. Phase-1 measurements

Both arms are timed on the same machine and settings as Part IV E of the survey (`docs/python-dsl-survey/05-measurements.md`): 1 thread for the shipped 2D cases, 4 threads (spin-wait for small cases, default for 65^3) where the arm is multi-threaded, reported beside the native RHS graph submit and shipped step from Part IV E.

- Python array arm: `python/pyshoccs` (NumPy and SciPy, optional torch and MLX) on the real exported tables for heat.lua 51x51, 201x201 and the 3D 65^3 sphere: one RHS, one RK4 step with its stats, with the whole-step cost split into operator, source, RHS glue, boundary, stats and stage arithmetic (`bench/bench_heat.py`); results in `.scratch/phase1/py_arm/RESULTS.md`. The MLX backends and the fp32 GPU rows need a Metal device, which the Bash sandbox hides, so they are measured in one unsandboxed call per case.
- C++ fused-stage arm: `benchmarks/bench_tables.cpp` on the same tables: the structured RHS (axis stencil plus exception rows, as two launches or one) and the fused RK stage against the live graph submit and a graph-built step with the source and boundary vectors held fixed. The kernels are in `src/operators/tables_kernels.hpp` (`make_device_tables` folds k into the taps and exception values and drops exception rows on Dirichlet rows; `structured_rhs`, `fused_rk_stage`, `assign_at`), checked by `t-tables_kernels` against `apply_structured` plus the source rule and against the live heat RHS graph; results in `.scratch/phase1/cpp_arm/RESULTS.md`.
