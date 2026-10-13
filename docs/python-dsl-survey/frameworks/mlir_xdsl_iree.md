# MLIR substrates: xDSL stencil dialect, MLIR Python bindings, IREE

> **Verdict:** reference only (overall 3.5 / 10).
> **Fit for SHOCCS:** moderate in principle, low in practice. The stencil dialect fits the constant-offset axis stencil, but the irregular closure rows and masks fall outside it.
> **CPU fp64 parity:** plausible via LLVM (f64 is first class), but unmeasured and SHOCCS would own all tiling, vectorization and threading work.
> **Apple GPU:** none usable. xDSL has no Metal path. IREE's Metal HAL has no documented fp64 and no command-buffer replay. Apple GPUs have no fp64 hardware (unverified in this pass, see log).
> **Risk:** high. Owning an MLIR/LLVM toolchain (or third-party wheels), 0.x APIs that change every few weeks, and a weak GPU story.

## Status as of 2026-10-12

| Item | State | Confidence |
|---|---|---|
| xDSL | v0.73.0, released 9 Oct (release page does not show the year). Recent cadence: 0.72.0 on 2 Oct, 0.71.0 on 11 Sep, 0.70.0 on 28 Aug. Still 0.x. | Confirmed |
| xDSL stencil dialect | Still maintained. v0.70.0 notes include "(stencil) remove deprecated get helpers". | Confirmed |
| IREE | Only automatic daily pre-releases seen. Latest is 3.13.0rc20261012 (12 Oct). Only the first releases page was checked, so a stable tag on a later page is not excluded. | Confirmed (first page) |
| MLIR Python bindings | No official PyPI release found. Installs go through community wheels (makslevental/mlir-wheels, conda-forge, third-party repackagings such as xtc-mlir-python-bindings) or a source LLVM build. MLIR docs still say "under development". Absence rests on a single search. | Confirmed |
| Upstream wheel packaging | PR #160388 was merged on 24 Sep 2025 and reverted the same day (#160481) after breaking a buildbot. PR #177143 was closed unmerged on 28 Feb 2026 with "we're not gonna support this upstream". Whether #160388 was later relanded was not checked. The original assessment said both were open; that was wrong. | Corrected |
| Research use | ASPLOS'26 wafer-scale stencil paper, Devito and PSyclone via the shared stack. | Unverified |

## Backends and fp64

| Backend | Route | fp64 | Notes |
|---|---|---|---|
| CPU (xDSL) | stencil to scf/memref/llvm, then LLVM JIT or AOT; OpenMP path exists | Yes (f64 first-class type in MLIR) | OpenMP lowering existence is unverified. Needs an emitter that matches Kokkos; untested for SHOCCS. |
| CPU (IREE) | llvm-cpu backend | Yes (not checked against IREE docs in this pass) | Adds VM and HAL dispatch overhead. |
| CUDA | MLIR gpu to nvvm; IREE CUDA backend | Yes | Stencil dialect GPU lowering is described as research-grade (unverified). |
| Apple Metal (xDSL) | None. Would need gpu dialect to SPIR-V to SPIRV-Cross to MSL. | No | Nobody ships this for xDSL. |
| Apple Metal (IREE) | Metal HAL driver: Metal 3, SPIR-V cross-compiled to MSL via SPIRV-Cross, one queue, one-shot command buffers only | No f64 mention in design doc | The design doc does not call the driver "experimental"; that label comes from source paths under experimental/metal/. User guide page still reads "Documentation coming soon". |
| Vulkan (IREE) | Requires Vulkan 1.3 with timelineSemaphore, scalarBlockLayout, synchronization2. Page presents Android, Linux and Windows tabs and calls Vulkan cross-platform. | Page never mentions f64, shaderFloat64, MoltenVK or macOS | Not a documented route on macOS. |
| Other | IREE VMVX, ROCm; xDSL Cerebras CSL and FPGA research paths | ROCm yes in principle | Not relevant to the Apple goal. |

Apple GPUs have no fp64 and MSL has no usable double, so any Metal route is fp32 at best, or an emulated df64 that someone would have to write as a custom pass. Nobody ships one for MLIR. The no-fp64 fact was taken from the repo's own `00-apple-gpu-facts.md` and was not re-verified externally.

## Mapping of the hard runtime shapes

| Idiom | MLIR / xDSL expression | Support |
|---|---|---|
| Packed field vector [D\|Rx\|Ry\|Rz] | One 1-D memref with static offsets and `memref.subview` per part. xDSL stencil fields are typed 3-D, so a packed buffer needs casts or separate fields per component. | Native |
| Masked axis stencil | `stencil.apply` with constant-offset `stencil.access`; interior mask as restricted bounds, or `arith.select` on a mask field (computes both branches). Irregular masks are not first-class. | Native |
| ELL gather-dot-scatter | No indirect access in the stencil dialect (inferred from design, unverified). Use `scf.parallel` or `linalg.generic` with `memref.load` of index arrays, fixed-width inner `scf.for`, then `memref.store`. Disjoint rows, or `memref.atomic_rmw`. Needs a custom op or lowering. | Workaround |
| Face-plane assignment | `stencil.apply` over a one-cell-thick bounds box, or `memref.subview` plus `linalg.fill` or copy. | Native |
| Masked reduction | `linalg.reduce` or `scf.parallel` with `scf.reduce` and `arith.select`. GPU needs `gpu.all_reduce` or subgroup ops. | Workaround |
| Whole-step compilation | Whole RK step in one `func.func`; one native call per step on CPU. IREE compiles to one VM function that dispatches many kernels. Cross-kernel fusion on GPU is limited. | Native |
| I32 graph replay / whole-step launch | Compiled function on CPU, zero Python overhead per step. IREE Metal has only one-shot command buffers, so no replay there. | Workaround |
| I19/I23/I24 CSR/ELL boundary coupling tables | Same as ELL row: runtime memrefs of index type, static width. | Workaround |
| I34 MMS closed forms, I35/I36 source callbacks | `math` dialect (sin, exp) lowers to libm or LLVM intrinsics. SymPy expressions emitted as arith/math ops. No Lua callbacks at runtime. | Native |
| I50 sparse eigensolver, I44/I48/I49 diagnostics | Stay in SciPy on the host. | Not applicable |

## Code sketch

Hand-written MLIR, not compiled or tested. The `stencil.store` syntax and `!stencil.field<...>` types should not be trusted, and mixing the stencil dialect with memref index arrays in one function was not validated.

```mlir
// Axis-x second-order interior stencil with an ELL closure fix-up.
func.func @dx(%u: !stencil.field<[-1,65]x[-1,65]x[-1,65]xf64>,
              %du: !stencil.field<[0,64]x[0,64]x[0,64]xf64>,
              %rows: memref<?xindex>,     // closure row linear ids
              %cols: memref<?x4xindex>,   // ELL column ids, fixed width 4
              %w: memref<?x4xf64>,        // ELL weights
              %u1d: memref<?xf64>, %du1d: memref<?xf64>, %nrows: index) {
  %ut = stencil.load %u : !stencil.field<...> -> !stencil.temp<?x?x?xf64>
  %r = stencil.apply(%a = %ut : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %m = stencil.access %a[-1, 0, 0] : !stencil.temp<?x?x?xf64>
    %p = stencil.access %a[ 1, 0, 0] : !stencil.temp<?x?x?xf64>
    %d = arith.subf %p, %m : f64
    %s = arith.mulf %d, %half_over_h : f64
    stencil.return %s : f64
  }
  stencil.store %r to %du ([0,0,0] : [64,64,64]) : ...
  // closure rows (0.5 to 2.5 percent of points): gather-dot then scatter
  scf.parallel (%i) = (%c0) to (%nrows) step (%c1) {
    %row = memref.load %rows[%i] : memref<?xindex>
    %acc = scf.for %k = %c0 to %c4 step %c1 iter_args(%s = %zero) -> f64 {
      %c = memref.load %cols[%i, %k] : memref<?x4xindex>
      %v = memref.load %u1d[%c] : memref<?xf64>
      %wk = memref.load %w[%i, %k] : memref<?x4xf64>
      %t = math.fma %wk, %v, %s : f64
      scf.yield %t : f64
    }
    memref.store %acc, %du1d[%row] : memref<?xf64>   // rows disjoint
  }
  return
}
```

## Performance assessment

All figures below are reasoned estimates; none were measured for this survey.

- **Small 2D (51x51, bar of about 30 to 40 us per RHS, 0.3 to 1 ms per RK4 step, launch-bound):** a whole-step `func.func` compiled by LLVM to a native function could match the C++ bar, since no Python dispatch occurs inside the step and per-call overhead is a few microseconds. Threading would add fork/join cost at this size, so serial LLVM code is the realistic target. Vectorization of constant-offset stencils is plausible but unmeasured.
- **IREE on small grids:** adds VM and HAL overhead per dispatch (tens of microseconds is typical on its CPU task system; unmeasured), which is a real risk at 51x51.
- **Large 3D (64^3, about 1.7 ms per RHS, bandwidth-bound):** a fused tiled stencil through LLVM could approach bandwidth, but only if SHOCCS adds tiling, OpenMP and vectorization passes itself. Parity is not automatic.
- **Apple GPU:** fp32 only. IREE Metal has one-shot command buffers and a SPIR-V to MSL path, so no replay and a long dispatch path. xDSL has no Metal path.

## What the DSL designs should copy or avoid

Copy:

1. A typed stencil-apply op with constant offsets and explicit bounds, so interior versus boundary regions are expressed as bounds, not masks.
2. A separate sparse or ELL op for irregular closure rows, instead of forcing them into the stencil abstraction.
3. Whole step as one function so no host dispatch occurs inside the step.
4. Dtype as a pass-level parameter so a df64 lowering can be added later.

Avoid:

- Owning an MLIR/LLVM toolchain with pinned versions, or depending on third-party wheels (upstream wheel packaging is not offered: one PR was reverted, another closed as unsupported).
- Relying on dialect APIs that change every release.
- Assuming a GPU backend exists because the IR is portable. The Metal route is the weakest link.

A purpose-built DSL can emit C++/Kokkos or LLVM IR as plain Python-generated text and keep the same four ideas at a fraction of the maintenance cost.

## Deciding fact

There is no maintained Apple-GPU path for xDSL, and IREE's Metal driver is a SPIR-V-to-MSL route with no documented fp64 and no command-buffer replay. An MLIR toolchain therefore adds heavy maintenance cost without delivering either an fp64 CPU speedup over a template emitter or the Apple GPU goal.

## Verification log

Verifier-adjusted overall score: 3.5 (unchanged). Counts: 6 confirmed, 1 refuted, 3 uncertain.

| Claim | Verdict | Evidence |
|---|---|---|
| xDSL v0.73.0 latest (9 Oct); stencil dialect maintained (v0.70.0 removed deprecated getters) | Confirmed | Releases page marks v0.73.0 Latest, 09 Oct; v0.70.0 notes list "(stencil) remove deprecated get helpers" (#6294). Year not shown on page. Last 20k chars of page unread. |
| IREE latest is nightly 3.13.0rc20261012; no stable release seen | Confirmed | Top entry on 12 Oct; all ten listed are "Automatic candidate release" pre-releases. Only first page checked. |
| IREE Metal user guide says "Documentation coming soon" | Confirmed | Page content is only that placeholder; no f64 mention. |
| Metal HAL: Metal 3, SPIRV-Cross to MSL, one-shot command buffers, one queue, no fp64 mention | Confirmed | Design doc text matches. It does not use the word "experimental"; that wording is inferred from source paths. |
| Vulkan guide: Vulkan 1.3 plus timelineSemaphore, scalarBlockLayout, synchronization2; no fp64 or MoltenVK mention | Confirmed | Page matches; no shaderFloat64, MoltenVK or macOS. The "Android, Linux, Windows only" phrasing is slightly stronger than the page, which calls Vulkan cross-platform and lists those three as tabs. |
| No official PyPI release of MLIR Python bindings; community wheels or source build | Confirmed | Search found no official package; docs say under development; community and third-party packages exist. Single search, so absence is not proof. |
| Upstream wheel PRs #160388 and #177143 were open | Refuted | #160388 merged 24 Sep 2025 and reverted by #160481 the same day; #177143 closed unmerged 28 Feb 2026. Later reland of #160388 not checked. |
| No Apple GPU supports fp64 | Uncertain | Not re-verified externally; sourced from the repo's own survey. Widely held. |
| Stencil dialect has no indirect gather/scatter access | Uncertain | Inference from dialect design. The xDSL stencil docs page returned 404 and the paper was not re-read. |
| Devito/PSyclone lower to shared stencil/dmp/mpi dialects; stencil dialect has OpenMP and research-grade GPU lowering | Uncertain | Not checked in this pass. |

## Sources

- https://github.com/xdslproject/xdsl/releases
- https://github.com/iree-org/iree/releases
- https://iree.dev/developers/design-docs/metal-hal-driver/
- https://iree.dev/guides/deployment-configurations/gpu-metal/
- https://iree.dev/guides/deployment-configurations/gpu-vulkan/
- https://mlir.llvm.org/docs/Bindings/Python
- https://pypi.org/project/xtc-mlir-python-bindings/22.1.8.1/
- https://www.stephendiehl.com/posts/mlir_install/
- https://github.com/llvm/llvm-project/pull/160388
- https://github.com/llvm/llvm-project/pull/177143
- https://arxiv.org/pdf/2404.02218 (stencil dialect design; claims from it unverified)
- https://arxiv.org/pdf/2601.17754 and https://www.xdsl.dev/appdev (cited by the assessment; not verified in this pass)
- docs/python-dsl-survey/00-apple-gpu-facts.md (repo)
