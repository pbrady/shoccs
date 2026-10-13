# loopy (Kloeckner)

> **Verdict:** tier2. Useful as a design reference and possible C code generator, not as a runtime substrate.
> **Fit for SHOCCS:** moderate as a codegen backend or IR model for D1/D4; low as a runtime.
> **CPU fp64 parity:** plausible (score 6/10) for the affine axis stencil if one fused step function is generated; not benchmarked.
> **Apple GPU:** none (score 1/10). No Metal/MSL target found (unverified absence); OpenCL on Apple is deprecated and Apple GPUs lack fp64 (unverified here).
> **Risk:** medium-high (score 4/10). Slow release cadence, near-single maintainer, heavy dependency stack (islpy, pymbolic, pytools, genpy, cgen).

Verifier-adjusted overall score: 4.5 / 10.

## Status as of 2026-10-12

| Item | Finding | Verification |
|---|---|---|
| Latest release | 2025.2, PyPI, 2025-07-30 | confirmed |
| Earlier releases | 2025.1 (2025-04-02), 2024.1 (2024-02-16), then 2020.2.1 (2020-10-29) | confirmed (2020.2 and 2020.2.1 share a date) |
| Newer release | none found | confirmed |
| License, Python | MIT, Python 3.10+ | confirmed |
| Repo size | about 6.5k commits (6,534), 140 open issues, 49 open PRs | confirmed |
| Last commit | 2026-09-27 per GitHub API | corrected: the original assessment could not read this date; the repo is actively committed to |
| Cadence | slow releases, active main branch | release gap confirmed; maintainer base very small |

## Backends and fp64

| Backend | Status | fp64 | Notes |
|---|---|---|---|
| C (CPU) | ExecutableCTarget reported by assessment | yes | Unverified in docs fetched. Single-thread by default; multicore needs OpenCL/pocl or ISPC. |
| ISPC | listed in repo topics | yes | CPU SIMD. Confirmed only via topics. |
| OpenCL | listed in repo topics | yes on pocl/CPU | Primary execution path through PyOpenCL. |
| CUDA | listed in repo topics | yes | Code generation; run path is less used. |
| Numba, Fortran frontend | reported by assessment | yes | Unverified: not seen in the docs fetched. |
| Metal / MSL | none found | n/a | Absence of evidence, not proof. A new TargetBase/ASTBuilderBase subclass would be required; no precedent found. |
| Apple GPU via OpenCL | deprecated since macOS 10.14 | no | Plausible but unverified here. Apple GPUs have no cl_khr_fp64 and MSL has no double (taken from 00-apple-gpu-facts.md, not re-verified). No df64 support. |

Dtypes are first class per array, so precision-generic kernels are easy.

## Mapping of the hard runtime shapes

| Idiom | How in loopy | Support |
|---|---|---|
| Packed field vector [D\|Rx\|Ry\|Rz] | Flat array with static offsets or sub-array views; block iname covers all four blocks | native |
| Masked axis stencil | Affine loop nest, unrolled tap iname, mask as instruction predicate or clamped domain; split_iname, tags, add_prefetch apply. Domains must be convex (confirmed), so an arbitrary 0/1 mask must be a predicate or data array | native |
| ELL gather-dot-scatter | Indirect subscripts allowed. Footprint analysis errors on data-dependent indices by default (confirmed), so add_prefetch of the gathered array is not usable; the claim that bounds checks are skipped is inferred, not documented. Scatter is safe only with unique rows; atomics are recognised only on exact LHS/RHS match (confirmed) | workaround |
| Face-plane assignment | 2D iname domain with a fixed third index | native |
| Masked reduction | Reduction over an iname with predicate or mask multiply; the reductions docs section is a TODO (confirmed) | native |
| Whole-step compilation (fused RK step) | Kernel fusion exists but there is no graph capture or cross-kernel scheduler; one C function must be written and scheduled by hand | workaround |
| I37/I38/I46 RK slot arithmetic, fused epilogues | Elementwise statement fused into the stencil kernel via dependencies | native |
| I19/I23/I24 closure B/N tables | Padded 2-D coefficient plus index arrays, same as ELL | workaround |
| I16 per-direction team stencil | g/l tag hierarchy with local-memory prefetch | native |
| I02-I09 geometry, ray casting, merge-join | Data-dependent control flow, ragged results, sorting; stay on host | unsupported |

## Code sketch

Illustrative only. The verifier notes that it mixes an indirect gather and scatter into one kernel, and the closure rows would in practice need a second kernel because write order matters. It is not a working kernel.

```python
import loopy as lp, numpy as np
# masked x-axis 5-tap stencil + ELL closure + fused RK epilogue (one kernel)
knl = lp.make_kernel(
  ["{[i,j,k]: 2<=i<nx-2 and 0<=j<ny and 0<=k<nz}",
   "{[r,m]: 0<=r<nc and 0<=m<w}"],
  """
  <> acc = w0*u[i,j,k] + w1*(u[i+1,j,k]-u[i-1,j,k]) + w2*(u[i+2,j,k]-u[i-2,j,k]) {id=st}
  du[i,j,k] = acc if mask[i,j,k] else du[i,j,k] {dep=st,id=wr}
  for r
    <> s = sum(m, cval[r,m]*u[cidx[r,m]]) {id=ell}  # indirect gather
    du[crow[r]] = s {dep=ell,id=sc}                  # scatter (rows unique)
  end
  y[i,j,k] = y0[i,j,k] + dt*b*du[i,j,k] {dep=wr}     # fused RK epilogue
  """,
  [lp.GlobalArg("u,du,y,y0", np.float64, shape="nx,ny,nz"),
   lp.GlobalArg("cidx", np.int32, shape="nc,w"), ...],
  target=lp.ExecutableCTarget())
knl = lp.split_iname(knl, "i", 16, inner_tag="l.0", outer_tag="g.0")
knl = lp.add_prefetch(knl, "u", "i_inner,j")  # only for the affine stencil read, not the gather
```

## Performance assessment

No benchmark was run; everything below is reasoned judgment, not measurement.

| Case | Assessment |
|---|---|
| Large 3D, 64^3 (about 1.7 ms per RHS in C++, bandwidth-bound) | Generated C or OpenCL on pocl can plausibly match for the affine axis stencil, where loopy tiling and prefetch are strongest. No automatic multicore on CPU without OpenCL/pocl or ISPC. |
| ELL closure rows (0.5-2.5% of points) | Cheap, but cannot be prefetched; expect a small penalty. |
| Small 2D, 51x51 (22.7 us per RHS measured at 1 thread, `05-measurements.md` Section C; the file was written against an earlier 30-40 us estimate) | Per-call Python dispatch (executor or pyopencl, tens of microseconds, unmeasured) is comparable to or larger than the whole C++ RHS. To reach the bar, compile the whole RK step into one C function and bypass the Python executor. |
| Apple GPU | No fp64 route. fp32 via deprecated OpenCL 1.2 is possible in principle but not a sound base. |

## What the DSL designs should copy or avoid

| Copy | Avoid |
|---|---|
| Separate the algorithm (domains, instructions, dependencies) from the schedule (split_iname, tag_inames, add_prefetch), so one spec lowers to C, ISPC or GPU | Per-kernel dispatch through a Python-level executor; generate one monolithic step function |
| First-class dtype per array for precision-generic fp64/fp32 kernels | Convexity restrictions: data-driven closure loops fall outside ISL analysis and lose prefetch and bounds checking |
| Per-instruction predicates for the interior mask | Heavy dependency stack (isl, pymbolic) and single-maintainer exposure |
| Multi-target codegen via a CFamily target class, the model for adding an MSL target | |

## Deciding fact

There is no Metal target (absence not proven) and the Apple OpenCL route is deprecated with no fp64, so loopy offers no Apple-GPU path. On CPU it can emit competitive C, but Python-side dispatch overhead per kernel call is comparable to or larger than the whole C++ RHS at 51x51, measured at 22.7 us on 1 thread (`05-measurements.md` Section C; this file originally quoted a 30-40 us estimate), so loopy is viable only as an ahead-of-time or JIT generator for a single fused step function with its own executor bypassed.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| PyPI releases 2024.1, 2025.1, 2025.2; earlier 2020.2.1; nothing newer | confirmed | PyPI JSON; GitHub tags show v2025.2 newest; MIT, Python >=3.10 |
| README/repo: CPU/GPU codegen, OpenCL/CUDA/ISPC, 140 issues, 49 PRs, about 6.5k commits | confirmed | Repo page; the OpenCL/CUDA/ISPC list comes from repo topics, not README text |
| Last commit date unreadable | refuted | GitHub API shows latest commit 2026-09-27 |
| No Metal/MSL target | uncertain | Search found none; absence of evidence only |
| Convex domains; atomics need exact LHS/RHS; Reductions TODO | confirmed | Kernel reference quotes |
| Footprint analysis errors on data-dependent indices | confirmed | ref_other docs; prefetch inference is reasonable but not stated explicitly |
| ExecutableCTarget, ISPC, Numba, CudaTarget, Fortran frontend | uncertain | Only repo topics confirm ISPC, OpenCL, CUDA; others not seen in docs fetched |
| Apple OpenCL deprecation, pocl advice, no Apple GPU fp64 | uncertain | Not re-fetched; assessment marks it inferred from local survey file |
| Per-call dispatch overhead, no whole-step fusion, tier2 judgment | uncertain | Reasoned, not measured |
| Bounds checks skipped for gathers | uncertain | Inferred, not documented |

Counts: 4 confirmed, 1 refuted, 3 uncertain (plus the unmeasured judgments above, treated as unverified).

## Sources

- https://pypi.org/project/loopy/ and https://pypi.org/pypi/loopy/json
- https://github.com/inducer/loopy (and /releases, API tags and commits endpoints)
- https://documen.tician.de/loopy/ref_kernel.html
- https://documen.tician.de/loopy/ref_other.html
- https://developer.apple.com/opencl (not verified)
- https://gitlab.tiker.net/inducer/pyopencl/-/commit/32cdd0b51fb528c93e7344a4626f6847d883d0bc (not verified)
- https://gitlab.tiker.net/inducer/loopy/-/merge_requests/25
- docs/python-dsl-survey/00-apple-gpu-facts.md (survey claim, not re-verified)
