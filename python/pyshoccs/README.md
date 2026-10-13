# pyshoccs

The Python array arm of the SHOCCS phase-1 thin waist (`docs/PYTHON_DSL_SURVEY.md`
Section 4). It consumes the operator tables that `shoccs-export` writes for one
case (geometry, boundary conditions, scheme) and runs the heat system on the
packed vector `V = [D | Rx | Ry | Rz]` in NumPy and SciPy, with the parity checks
that gate the phase. The data contract is `docs/reference/pyshoccs-export.md`;
everything here follows it.

## What is in the package

| module | content |
|---|---|
| `pyshoccs.tables` | `load_tables(dir)`: meta, layout (offsets, segment slices), coordinates, R records, lines, index lists, taps, raw COO per direction, Neumann COO, the exporter's `role`/`exc_*` tables and the oracle vectors. `Tables.to_scipy(dirs)` is the CSR over V with duplicates summed; `Tables.exact_apply(u)` sums the raw entries in stored order. Every array's dtype kind (integer, float, unsigned) and shape are checked against the contract and a mismatch names the file. |
| `pyshoccs.structure` | `derive_structure`: role bits (bit d interior in direction d, bit 3 fluid, bit 4 Dirichlet face), the interior mask per direction and the exception CSR (int32 columns, rows sorted and unique), derived from the COO alone and cross-checked against the taps. `StructuredOperator` / `apply_structured`: axis stencils on a zero-padded copy of D times the masks, plus the exception rows. `compare_with_export` diffs the derivation against `role.npy` and `exc_*.npy` when the exporter wrote them. |
| `pyshoccs.mms` | `Gaussian`: value, ddt, gradient, laplacian of the gaussian manufactured solution (`src/mms/gauss2d.cpp`, `gauss3d.cpp`), vectorised over points, built from `meta['mms']` and `meta['dims']`. `HoistedGaussian` precomputes the spatial factors at a fixed point set so value and source at time t are one small matrix-vector product. |
| `pyshoccs.heat` | `HeatSystem(tables, backend='scipy' or 'structured')`: `source`, `rhs`, `update_boundary`, `stats` (the eleven numbers of `compute_scalar_stats`) and `rk4_step` exactly as `src/temporal/rk4.cpp`, on preallocated work vectors. Neumann walls raise. |
| `pyshoccs.oracle` | the monomial oracle: rows of `A_d` applied to `x_d^k`, k = 0..3, on every R row and (as a diagnostic) every fluid D closure row, with per-row residual and tolerance `16 eps sum |coef x^k|`; and the transverse oracle, the R rows of `A_d` applied to `x_c^m` (c != d, m = 1, 2, exact value 0), which is the only one of the two that sees the cross-line interpolation of a transverse R row. Both are in `check_oracle`; only the axial k <= 2 R rows gate. |
| `pyshoccs.parity` | the five checks of the contract's Section 3 as functions returning a `CheckResult` (worst row, its class, max abs, max rel, per-class maxima), and `run_all(export_dir, json_path)` which prints a table and writes JSON. `python -m pyshoccs.parity DIR --json out.json` is the command-line form. |
| `bench/bench_heat.py` | timings of the operator apply, one RHS, one RK4 step and its split per backend, with the native C++ bar of Part IV E for the matching case. |

Row classes used by the checks: `solid_D`, `dirichlet_D`, `fluid_D` (away from
the clamps), `clamp_D` (closure rows of a line whose object end has psi outside
`[1e-4, 1 - 1e-4]`), `R_dirichlet` (no rows) and `R`.

## Producing an export

From the repository root with the native build configured in `build-host`
(see CLAUDE.md), after the exporter target exists:

```bash
cmake --build build-host --target shoccs-export
build-host/src/app/shoccs-export heat.lua --out .scratch/export/heat51
```

The tests and tools look for `PYSHOCCS_EXPORT_DIR`, then
`.scratch/export/heat51` under the repository root.

## Running the tests

```bash
cd python/pyshoccs
uv sync
uv run pytest -q
```

Without an export the parity tests skip; the rest (manufactured solution
against finite differences, the structured decomposition on hand-built tables,
RK4 on one-point ODEs, the npy loader, and a complete parity run on a synthetic
export written by `tests/synthetic.py`) needs nothing. With an export:

```bash
PYSHOCCS_EXPORT_DIR=../../.scratch/export/heat51 uv run pytest -q
uv run python -m pyshoccs.parity ../../.scratch/export/heat51 --json ../../.scratch/phase1/parity_heat51.json
```

A failing check prints the worst row as `V[row] = D[ic] (i,j,k)=(...)` or
`V[row] = Rx[n]` with its class, the difference and the tolerance, and the JSON
carries the per-class maxima. `pyshoccs.oracle.describe_row_entries(tables, d, row)`
lists the raw entries of one row for a closer look.

## Running the bench

```bash
uv run python bench/bench_heat.py --export ../../.scratch/export/heat51 --out ../../.scratch/phase1/bench_heat51.json
uv sync --extra bench
uv run python bench/bench_heat.py --export ../../.scratch/export/heat51 --backends all --out ../../.scratch/phase1/bench_heat51_all.json
```

NumPy and BLAS run on one thread unless `--threads N` is given (the variables
are set before numpy is imported). The `bench` extra adds torch (CPU wheel,
float64, eager and `torch.compile`) and MLX (CPU stream, float64, `mx.compile`)
implementations of the structured RHS and step, and the `gpu` alias selects the
same structured form in float32 on the Apple GPU (`torch-mps` eager, `mlx-gpu`
compiled). Every MLX and GPU backend needs a Metal device, which the Bash
sandbox hides: `import mlx.core` fails there even for the CPU stream and torch
reports no MPS device, so those backends are recorded as skipped inside the
sandbox and have to be run outside it. Each record holds the median, minimum
and p90 of `--reps` timed calls after `--warm` warm-ups for the operator, the
RHS, the source, the boundary update, the stats, the step with its stats (what
the shipped C++ step does once per step) and, for the NumPy backends, the step
without stats and the stage arithmetic alone; the step split (4 x operator,
4 x source, 4 x RHS glue, 4 x boundary, 1 x stats, 1 x axpy, remainder); the
relative error of the stepped state against `u1_V` and against the SciPy path
and after `--steps` chained steps against the SciPy path; and the ratios to the
C++ bar at the 1-thread, 4-thread and 4-thread spin-wait settings. `--runs N`
repeats the measurement N times in one process and writes one JSON per run.
Never compare against the stored Linux baseline; the bar constants in the script
carry their provenance string. The phase-1 numbers on the real exports are in
`.scratch/phase1/py_arm/RESULTS.md`, rendered from the JSONs by `summarize.py`
beside it.
