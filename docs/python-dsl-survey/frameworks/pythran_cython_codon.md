# Low-risk CPU emitters: Pythran, Cython, Codon

| | |
|---|---|
| Verdict | tier2: CPU-only emitters, no GPU path on Apple silicon |
| Fit for SHOCCS | Good for the host fp64 half of the goal; Pythran is the most natural target for a NumPy-subset DSL, Cython the safe fallback, Codon a larger bet |
| CPU fp64 parity | Expected near parity with C++/Kokkos host (unmeasured estimate, no benchmark run) |
| Apple GPU | None. No Metal backend in any of the three; Codon GPU is NVIDIA-only |
| Risk | Low for CPU. macOS OpenMP toolchain setup (unverified), per-module compile latency, unmeasured performance, and defects in the code sketch below |

## Status as of 2026-10-12

| Tool | Version | Notes |
|---|---|---|
| Pythran | 0.19.0 (PyPI, 2026-08-17) | BSD-style license; PyPI classifiers list macOS and Linux, no Windows |
| Cython | 3.3.0 | Python 3.9 to 3.14. The 2026-08-22 date comes from the changelog heading; the PyPI upload time was not read, so the date is weakly confirmed |
| Codon | v0.20.3 (2026-09-28) | Release assets include darwin-arm64, darwin-x86_64, linux-aarch64, linux-x86_64 and manylinux2014. LICENSE on develop is Apache 2.0 with no change-date clause. The claim that older releases were BSL was not checked, so verify the license at any pinned tag |

## Backends and fp64

| Backend | Pythran | Cython | Codon |
|---|---|---|---|
| CPU | C++ via the system compiler; `#omp` comments become OpenMP directives under `-fopenmp`; `-Ofast -march=native` forwarded; xsimd via `-DUSE_XSIMD` | C code, `prange`/`nogil` with OpenMP, typed memoryviews | Native LLVM, `@par` multithreading |
| CUDA | None | None (hand-written CUDA C only) | `@gpu.kernel`, `@par(gpu=True)`; docs say "Currently, only Nvidia devices are supported." |
| Metal/MPS | None | None | None |
| fp64 | Native on CPU | Native on CPU | Native on CPU; CUDA fp64 on NVIDIA only |

No GPU fp64 story on Apple silicon for any of them. Pythran's manual documents no compilation cache (this shows only absence from that page).

## Mapping of the hard runtime shapes

| Idiom | Approach | Support | Status |
|---|---|---|---|
| Packed field vector [D\|Rx\|Ry\|Rz] | One flat float64 array with slice views or offsets per segment; Pythran array argument, Cython `double[::1]` memoryview, Codon numpy array or `Ptr[float]` | native | unverified, plausible |
| Masked axis stencil (constant coefficients, 3D) | Explicit loops with taps unrolled; mask encoded as per-line [lo,hi) ranges rather than boolean arrays; OpenMP on the outer loop | native | unverified, plausible |
| ELL gather-dot with scatter | Fixed-width loop, unique row targets so the scatter is race-free; use explicit loops rather than fancy indexing | native | unverified; Pythran fancy-indexing limits not checked |
| Face-plane assignment | Strided slice assignment or a 2-loop nest | native | unverified, plausible |
| Masked reduction | Pythran `reduction(+:s)`; Cython `prange` infers only in-place operators such as `+=`, so max needs a manual per-thread approach; Codon `@par` | native for sums, workaround for max | max-inference claim corrected |
| Whole-step (RK4) compilation | Export the step as one function so per-step overhead is one Python-to-native call; no graph capture or cross-op fusion beyond the C++ compiler | native | callee inlining and ~1 us call cost unverified |
| Runtime kernel generation | DSL emits specialised source; it must hash source and cache built modules itself. Compile latency per module (Pythran especially) is a cost | workaround | latency magnitude unverified |
| Dynamic dispatch / variant systems | Specialise per case at generation time | not applicable | |
| Apple GPU execution | No path | unsupported | confirmed |

## Code sketch

This is a design sketch, not tested code. Verification found likely defects, listed below the block.

```python
# Pythran: masked x-axis stencil + ELL gather-dot (emitted by the DSL)
#pythran export step_rhs(float64[:,:,:], float64[:,:,:], int32[:,:], float64[:], int32[:,:], float64[:,:], int32[:], float64)
import numpy as np
def step_rhs(u, out, rng, c, col, val, row, inv_h2):
    # rng[j*nz+k] = (lo, hi) interior range for line (j,k); c = 3 symmetric taps
    nx, ny, nz = u.shape
    #omp parallel for collapse(2)
    for j in range(ny):
        for k in range(nz):
            lo, hi = rng[j*nz+k, 0], rng[j*nz+k, 1]
            for i in range(lo, hi):
                out[i,j,k] = inv_h2*(c[0]*u[i-1,j,k] + c[1]*u[i,j,k] + c[2]*u[i+1,j,k])
    uf = u.ravel(); of = out.ravel()
    #omp parallel for
    for r in range(row.shape[0]):             # ELL: fixed width W = col.shape[1]
        s = 0.0
        for w in range(col.shape[1]):
            s += val[r,w] * uf[col[r,w]]
        of[row[r]] = inv_h2 * s               # unique targets: race-free scatter
# build: pythran -Ofast -march=native -DUSE_XSIMD -fopenmp step.py
# macOS: use Homebrew clang/gcc plus libomp
```

Suspected defects (unverified, need a compile-and-run check):

| Issue | Why it matters |
|---|---|
| `ravel()` may return a copy | Writes through `of` would then not reach `out`, silently dropping the ELL scatter. Pass flat 1-D arguments instead |
| Export types use int32 for `rng`, `col`, `row` | NumPy index arrays default to int64; the call would be rejected on dtype mismatch |
| `collapse(2)` over loops with statements between them | `collapse` needs perfectly nested loops; the `lo, hi = ...` line breaks that |
| Loop order | With C layout `u[i,j,k]` makes k contiguous, so the inner `i` loop is strided unless the layout is changed (the original comment assumed x last) |

## Performance assessment

No benchmarks were run. Every number here is an unmeasured estimate from how the loops would compile.

| Case | Estimate | Status |
|---|---|---|
| 51x51 2D (about 30 to 40 us per RHS, 0.3 to 1 ms per RK4 step) | Whole-step compiled function pays about one Python call per step; keep it serial or use a size threshold because OpenMP fork/join costs microseconds | unverified |
| 64^3 3D (about 1.7 ms per RHS, bandwidth-bound) | OpenMP over outer loops with a fused one-pass stencil should reach memory bandwidth | unverified |
| Parity with C++/Kokkos host | Roughly 1.0 to 1.5x | unverified |
| Apple GPU | No gain; the no-regression bar is met only on CPU | confirmed (no Metal path) |

Risks: Pythran array expressions allocate temporaries unless fused, so write loops (unverified). Apple clang lacks `omp.h`, so OpenMP on macOS likely needs Homebrew libomp or gcc/LLVM (widely known, not confirmed from a source here). Mixing libomp copies is a reported crash source (unverified).

## What the DSL designs should copy or avoid

Copy: emit loop-form, constant-folded kernels with per-line [lo,hi) ranges instead of boolean masks; export the whole RK step as one native function; keep ELL arrays fixed-width with unique scatter targets; hash generated source and cache built modules in the DSL. Avoid: NumPy fancy indexing and whole-array temporaries in hot paths; depending on one C++ toolchain on macOS; expecting any GPU path from these tools. The Apple GPU wish needs a separate backend (Metal or MLX style, fp32 or emulated fp64).

## Deciding fact

None of the three has an Apple-GPU path (Codon's GPU backend is NVIDIA-only, confirmed in its docs), so they qualify only as CPU emitters. Pythran fits a DSL that generates NumPy-subset source best, and Cython is the safe fallback.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| Pythran 0.19.0 latest (2026-08-17), BSD-style, Linux and macOS | confirmed | PyPI JSON: upload date, license, classifiers |
| Cython 3.3.0 latest, Python 3.9 to 3.14 | confirmed (date weak) | requires_python and classifiers; 2026-08-22 from changelog heading only |
| Codon v0.20.3 (2026-09-28) with macOS arm64 and Linux tarballs | confirmed | GitHub latest release assets |
| Codon GPU NVIDIA-only, no Metal | confirmed | gpu.md text |
| Codon develop LICENSE is Apache 2.0 | confirmed | LICENSE file; older BSL status not checked |
| Pythran `#omp`, `-Ofast -march=native`, xsimd, no documented cache | confirmed | Pythran manual; absence of cache is page-limited |
| Apple clang lacks omp.h/libomp | uncertain | Not fetched; inferred from general guidance |
| Parity 1.0 to 1.5x, Pythran whole-step inlining, partial fancy indexing, libomp crashes | uncertain | No benchmark or source checked |
| Cython `prange` infers max reductions | refuted (dropped) | Only in-place operators such as `+=` are inferred |
| Code sketch correct as written | refuted | Probable ravel copy, int32 vs int64, collapse(2) nesting |
| Timing figures (30 to 40 us, 1.7 ms, ~1 us) | uncertain | No source, unmeasured |

Verifier-adjusted overall score: 6 (assessment 6.5, lowered 0.5 for unmeasured performance and sketch defects). Counts: 6 confirmed, 2 refuted (max-reduction inference, sketch correctness), 3 uncertain.

## Sources

- https://pypi.org/pypi/pythran/json
- https://pypi.org/pypi/Cython/json
- https://api.github.com/repos/exaloop/codon/releases/latest
- https://raw.githubusercontent.com/exaloop/codon/develop/README.md
- https://raw.githubusercontent.com/exaloop/codon/develop/LICENSE
- https://raw.githubusercontent.com/exaloop/codon/develop/docs/parallel/gpu.md
- https://pythran.readthedocs.io/en/latest/MANUAL.html
- https://mac.r-project.org/openmp/
