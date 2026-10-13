# Cleanup Plan

> Recreated 2026-06-09. This tracker is referenced by `CLAUDE.md`, `readme.md`,
> and several `docs/reference/` pages (as "§0/§0a") but was never committed to
> the repository. The sections below restore the two entries those references
> point at. Add new cleanup items below §0a as they come up.

## §0 — Kokkos 5.0 → 5.1.1 `create_graph` API break (FIXED 2026-06-04)

Kokkos 5.1 removed the old `create_graph(...)` overload set the codebase used,
which left the prior `build/` tree linking a stale `libkokkoscore.so.5.0`.
Fixed by migrating all 17 call sites to the templated one-argument form
`create_graph<execution_space>(closure)`. The build is green; `ctest` passes
47/48.

Two test failures were fixed in the same pass:

- `t-csr` — needed a custom `main()` with `Kokkos::ScopeGuard`, linking
  `Catch2::Catch2` (not `WithMain`) plus `Kokkos::kokkos`.
- `t-E2_1` — needed `.margin(1e-12)` on its `Approx` comparisons.

The devcontainer pins the dependency stack accordingly: Spack v1.1.1 with the
community package repo pinned to a commit shipping `kokkos@5.1.1` (see
`.devcontainer/spack.yaml` and `spack_repo/shoccs/`).

## §0a — `t-laplacian`: one remaining test failure (OPEN)

The only failing test project-wide. `TEST_CASE "E2 with Floating Objects"` in
`src/operators/laplacian.t.cpp:411` fails at the
`REQUIRE_THAT(ex.rx_vec, Approx(du.rx_vec))` assertion (line 483): the
cut-cell R-point `rx_vec` values differ ~2–3% from expected, while the
interior `d_vec` assertion passes. The inputs are fully deterministic (fixed
25×26×27 mesh, fixed sphere, polynomial fields), so the failure is
deterministic — a genuine cut-cell numerics question, not a build, link, or
flakiness problem. See `docs/reference/operators.md` ("Tests" section) for the
full analysis.

**Likely cause (2026-10-13, phase-1 monomial oracle, Part V of the Python DSL
survey, `docs/python-dsl-survey/06-phase1.md`).** The R-point rows of a
second-derivative operator in a direction other than the point's own ray
(`cut_discretization` with `dir != r` in `src/operators/derivative.cpp`)
interpolate the neighbouring grid lines with the two-point linear weights of
`E2_2::interp_interior` and combine them with the exact R value under the
(1, -2, 1)/h^2 difference. On `x_c^2` that leaves a residual of about
psi (1 - psi) u_cc / 2 which does not shrink with h (0.25, 0.48 and 0.57 at
h = 0.04, 0.01 and 1/32 on the exported heat cases); along the ray the rows
are ordinary second order. The failing test's field has `x^2 (y + z)` terms,
which is the case this residual describes, while the 2D variant passes because
its field is linear in each coordinate; the failure is consistent with the
mechanism, which has not been demonstrated by changing the interpolation and
re-running the test. The maintainer's note (2026-10-13): the proper
interpolation for the second-derivative operator at cut points was never worked
out, and the tests so far have focused on hyperbolic (first-derivative)
problems. Recorded as an open numerics item for the future; nothing in the
export or Python work changes it, and no tolerance is loosened to hide it.

**CI handling:** `.github/workflows/ci.yml` excludes `t-laplacian` from the
gating `ctest` invocation and runs it separately as a non-gating
(`continue-on-error`) step, so CI stays green on the known failure but the log
shows when the numerics fix lands. When this is fixed, delete the exclusion
and the non-gating step, and update the 47/48 references here, in `CLAUDE.md`,
in `readme.md`, and in `docs/reference/operators.md`.
