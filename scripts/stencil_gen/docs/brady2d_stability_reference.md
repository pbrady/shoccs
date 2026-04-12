# Brady-Livescu 2D Analytical Stability Reference

## Problem statement

The benchmark is the two-dimensional varying-coefficient scalar advection
problem from Brady & Livescu 2019 &sect;4.3 (pp. 92&ndash;94):

```
u_t + grad(psi) . grad(u) = 0    on [0, sqrt(2)]^2,  t in [0, 1000]

psi(x, y) = sqrt((x + 0.25)^2 + (y + 0.25)^2)
c_x = d(psi)/dx = (x + 0.25) / psi
c_y = d(psi)/dy = (y + 0.25) / psi

u(x, y, 0)   = sin(2*pi*psi)
u(0, y, t)   = sin(2*pi*(psi(0, y) - t))    (inflow, Dirichlet)
u(x, 0, t)   = sin(2*pi*(psi(x, 0) - t))    (inflow, Dirichlet)

Exact: u(x, y, t) = sin(2*pi*(psi - t))
```

The radial velocity field `(c_x, c_y)` is unit-magnitude everywhere. Inflow
enters at `x = 0` and `y = 0`; outflow exits at `x = sqrt(2)` and
`y = sqrt(2)`.

## Layered pipeline

Each layer is strictly cheaper and more informative than running the full
C++ simulation. The pipeline short-circuits on the first failure, so
expensive layers only run for candidates that pass the cheap checks.

| Layer | Metric | What it catches | Approx. cost (E4, N=81) |
|-------|--------|-----------------|-------------------------|
| L1 | Interior + boundary group velocity error (1D) | Dispersion-quality mismatch at boundary | sub-ms |
| L2 | Rigorous GKS Kreiss determinant test | Boundary-closure instability (necessary **and** sufficient for the 1D reduction) | ~100 ms |
| L3 | 1D eigenvalue `max Re(lambda)` at multiple N | Semi-discrete asymptotic stability, constant coefficient | ~30 ms |
| L4 | Per-point local GV error across the 2D grid | Local dispersion error from varying `(c_x, c_y)` | ~tens ms |
| L5 | 2D anisotropy `max angle_error` over propagation angles | Grid anisotropy interacting with the radial flow | ~100 ms |
| L6 | Non-normality on 1D operator: spectral + numerical abscissa, Henrici departure, Kreiss constant, transient-growth bound | Transient growth from non-normal 1D spatial operator | ~seconds |
| L7 | Sparse 2D Arnoldi `max Re(lambda)` on the full varying-coefficient operator | True 2D semi-discrete asymptotic stability | few seconds per N |
| L8 | (Plan 42) C++ simulation via Lua bridge | Actual long-time L-infinity bound | minutes |

## Failure thresholds

| Layer | Metric | Threshold constant | Value | Rationale |
|-------|--------|--------------------|-------|-----------|
| L1 | `boundary_gv_err` | `L1_TOL` | 0.05 (5%) | Boundary dispersion error > 5% indicates poor closure quality |
| L2 | `KreissResult.is_stable` | `sigma_tol` | 1e-8 | GKS determinant condition; `sigma_min(M(s)) < tol` indicates instability |
| L3 | `max_stab_eig` | `STABILITY_TOL` | 1e-10 | 1D eigenvalue in right half-plane → unstable semi-discrete scheme |
| L4 | `max_local_gv_error` | `L4_TOL` | 0.1 (10%) | Looser than L1 because varying-coefficient scaling amplifies baseline error |
| L5 | `max_aligned_error` | `L5_TOL` | 0.05 (5%) | Grid anisotropy projected onto the local propagation direction |
| L6 | `spectral_abscissa` | `STABILITY_TOL` | 1e-10 | 1D operator eigenvalue check |
| L6 | `transient_growth_bound` | `L6_TRANSIENT_GROWTH_TOL` | 50.0 | Kreiss constant bound `e*K > 50` indicates dangerous transient growth |
| L7 | `max_spectral_abscissa` | `L7_TOL` | 5e-3 | 2D varying-coefficient operator; stable schemes show max Re ~ O(1e-3), unstable ~ O(0.1) |
| L7+ | `transient_growth_bound` | `L7_TRANSIENT_GROWTH_TOL` | 50.0 | Same as L6 but on the 2D operator |

## API reference

### `brady2d_stability_score`

Main entry point in `stencil_gen/brady2d_stability.py`.

```python
def brady2d_stability_score(
    scheme: str,           # "E2" or "E4"
    kernel: str,           # "classical" | "tension" | "gaussian" | "multiquadric"
    params: dict,          # {"alpha": [...]}, {"sigma": float}, or {"epsilon": float}
    *,
    max_layer: int = 7,    # highest layer to run (1-7)
    short_circuit: bool = True,  # stop at first failure
) -> StabilityReport
```

### `StabilityReport`

Dataclass returned by `brady2d_stability_score`.

| Field | Type | Description |
|-------|------|-------------|
| `layer1` .. `layer7` | `dict \| None` | Per-layer metrics (populated when that layer runs) |
| `kreiss` | `KreissResult \| None` | Alias for `layer2` (the Kreiss determinant result) |
| `non_normality` | `NonNormalityReport \| None` | Populated by L6 (1D) or L7 (2D) |
| `overall_verdict` | `"pass" \| "fail" \| "unknown"` | Final verdict |
| `failed_layer` | `int \| None` | First layer that failed |
| `failed_reason` | `str` | Human-readable failure description |
| `compute_time` | `float` | Total wall-clock seconds |

The `__str__` method produces a compact per-layer summary table.

### `kreiss_stability_check`

Rigorous GKS determinant test in `stencil_gen/gks_kreiss.py`. Implements
Trefethen 1983 (pp. 206-207).

```python
def kreiss_stability_check(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    *,
    s_grid_params: dict | None = None,
    sigma_tol: float = 1e-8,
    refine: bool = True,
    min_s_magnitude: float = 0.1,
) -> KreissResult
```

`KreissResult` fields: `is_stable`, `witness_s`, `witness_sigma_min`,
`imaginary_axis_perturbation_verdict`, `defective_kappa_detected`,
`s_grid_shape`, `compute_time`, `sigma_min_field`, `s_grid`,
`n_admissible_roots`.

### `compute_non_normality`

Non-normality diagnostics in `stencil_gen/non_normality.py`. Calibration
bands from Trefethen & Embree 2005, ch. 14.

```python
def compute_non_normality(
    L,                           # sparse or dense matrix
    *,
    small_dense_threshold: int = 900,
    epsilon_values: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
    s_grid_params: dict | None = None,
) -> NonNormalityReport
```

`NonNormalityReport` fields: `spectral_abscissa`, `numerical_abscissa`,
`henrici_departure`, `eigenvector_condition`, `pseudospectral_abscissae`
(dict mapping epsilon to abscissa), `kreiss_constant`,
`transient_growth_bound` (`= e * kreiss_constant`), `n`, `compute_time`,
`notes`.

## CLI usage

```bash
# Score a single scheme at max_layer=6
cd scripts/stencil_gen
uv run python -m stencil_gen.brady2d --scheme E4 --kernel tension --sigma 3.0 --max-layer 6

# Run calibration across all families
uv run python -m stencil_gen.brady2d --run-calibration --max-layer 6

# Run calibration and persist to known_values.json
uv run python -m stencil_gen.brady2d --run-calibration --max-layer 7 --update-known-values
```

## Calibration results

Results stored in `sweeps/known_values.json` under the `"brady2d_calibration"` key.
Regression tests in `tests/test_phs.py::TestRegressionBrady2DCalibration` verify
that re-running at `max_layer=3` reproduces the stored verdicts.

At `max_layer=6` (2026-04-12):
- **Pass (6/9):** E4_classical, E2_phs_k2, E4_phs_k2, E4_tension_3, E4_gaussian_09, E4_multiquadric_1
- **Fail at L1 (3/9):** E2_tension_6, E2_gaussian_2, E2_multiquadric_1

E2 families (except PHS k=2) fail at L1 because 2nd-order boundary closures
have inherently larger dispersion error. Only E2_phs_k2 (sigma=0, fully
determined weights) passes.

## References

- Brady & Livescu 2019: "High-order multiblock/multiresolution finite
  difference methods for the compressible Navier-Stokes equations", &sect;4.3 pp. 92-94
- Trefethen 1983: "Group velocity in finite difference schemes", pp. 204-210
  (GKS determinant condition)
- Trefethen & Embree 2005: *Spectra and Pseudospectra*, ch. 14 (Kreiss constant
  calibration bands)
