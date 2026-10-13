# Triton (triton-lang/triton) as a SHOCCS DSL substrate

| Verdict | |
|---|---|
| Fit for SHOCCS | Plausible CUDA/ROCm code emitter under a multi-backend DSL; not sufficient alone. Overall score 4/10. |
| CPU parity | Not achievable today: no mainline CPU backend, experimental Linux-only fork (triton-cpu), unmeasured. |
| Apple GPU | Out-of-tree only (triton-ext PR #126 open, triton-msl 0.3.0 alpha). fp32 only, no graph capture, unaudited for gather/scatter. |
| fp64 | CUDA yes for loads, stores and elementwise math (fp64 tl.dot and atomics unverified). Apple GPUs none. |
| Risk | High: unmerged community Apple work, 10-30 us Python launch cost (estimate), JIT cold start, fast release churn. |

## Status as of 2026-10-12

- Latest release: Triton 3.8.0, marked Latest on the GitHub releases page and dated 28 Aug. The page shows no year; 2026 is inferred from the page footer. Prior releases: 3.7.1 (18 Jun), 3.7.0 (07 May), 3.6.0 (21 Jan).
- Mainline has no Apple/Metal backend and no production CPU backend.
- The survey's claims about gfx1250 and Rubin SM107 support in 3.8, and about Gluon being available in 3.x, were not verified against release notes (unverified).
- Triton targets dense tile-level AI kernels (GEMM, attention, norms). Unstructured cut-cell work is possible with pointer arithmetic and masks but is not a design goal.

## Backends and fp64

| Backend | Status | fp64 | Notes |
|---|---|---|---|
| CUDA | First-class (Ampere through newer) | Loads, stores, elementwise: supported. fp64 tl.dot on Ampere+ and fp64 atomics: unverified. | Excellent for bandwidth-bound strided stencils with coalesced tiles. |
| ROCm (AMD) | Supported upstream | Likely on CDNA, unverified | Version coverage (gfx90a to gfx1250) unverified. |
| CPU, triton-cpu | Experimental long-lived fork. README: "still work in progress"; CPU listed as "Under development". Linux only. Select with TRITON_DEFAULT_BACKEND=cpu. | Expected via LLVM, unverified | The same fork also lists NVIDIA and AMD GPU support, so it is not CPU-only. Benchmarks, op coverage and upstream lag were not checked. Unusable on the macOS dev host; only in the Linux devcontainer. |
| CPU, TRITON_INTERPRET=1 | NumPy interpreter | Yes (NumPy) | Believed to be a debugging tool, orders of magnitude slower than C++ (unverified; not a performance path). |
| Apple, triton-ext PR #126 | Open, unmerged out-of-tree plugin by imperatormk, opened 2026-08-31, 186 commits, latest 2026-10-04 building against Triton 3.8 and 3.9. TTIR to TTGIR to MSL to metallib, dispatch on the torch MPS stream, needs the Xcode Metal toolchain, adds an 8x8 simdgroup MMA encoding. | None | Testers report Liger RMSNorm passing and a 3x3 conv regression (partly recovered). No maintainer answered a merge-plans question on 2 Oct. |
| Apple, triton-msl 0.3.0 | Alpha, third party (Janssena) | README: "FP64 GPU arithmetic is not generally available" | Targets Triton 3.7.0 and PyTorch 2.12, not 3.8 or 3.9. M1 not validated. 9.48 s cold first add versus 0.63 s. 13 native-wheel and 14 pure-wheel warm workloads reported more than 10% slower than previous main. |
| Apple, triton-metal (akashmadhu4) | Experimental fork | None | Link not checked; masked-store correctness issues are survey-reported only (unverified). |
| Others | Intel XPU, Ascend, forks such as flagtree | n/a | Not relevant to SHOCCS. |

Apple GPUs have no fp64 hardware. An Apple path would need fp32 or a user-written double-float emulation (several times the flops, no built-in support). Whether fp32 is accurate enough for small-psi closures is the open catalog question. PyTorch itself is believed to emit MSL directly for Inductor-MPS rather than using Triton (unverified); triton-msl advertises its own separate Inductor integration.

## Idiom mapping

Support values: native, workaround, unsupported. Performance is expectation, not measurement.

| Idiom | Support | How | Expected perf |
|---|---|---|---|
| I01 | native | Dict or dataclass of torch tensors, or one packed D\|Rx\|Ry\|Rz tensor with static offsets; kernels take raw pointers plus constexpr sizes. | Not hot; packing cuts launches 4x. |
| I02 | workaround | Host NumPy ray casting (vectorised quadratics, lexsort); Triton adds nothing. | Setup; faster than the C++ virtual-call loop if vectorised. |
| I03 | workaround | Host NumPy/Numba fragment merge-join producing SoA start/len/stride/kind arrays. | Setup, equal or better. |
| I04 | native | NumPy-built bool/uint8 masks or int32 index arrays uploaded as tensors; prefer masks for D. | Setup. |
| I05 | native | Host Python constants; interior taps as tl.constexpr args for unrolled FMAs. | Setup; fully unrolled taps. |
| I06 | workaround | Evaluate closures on host (sympy.lambdify/CSE on a psi array in fp64, cast on upload); optional Triton elementwise kernel if psi moves, pole guards need care. | Setup. |
| I07 | workaround | Host Numba/NumPy searchsorted per pencil emitting padded (idx, coef) ELL tables. | Setup; far better than the O(R) scan. |
| I08 | workaround | Host NumPy grouping lines by (left_kind, right_kind, bc) into closure, line and ELL tables. | Setup. |
| I09 | workaround | Host NumPy batched over intersections, emitting ELL tables. | Setup. |
| I10 | native | np.lexsort or scipy COO to CSR on host; keep compact COO/ELL on device. | Setup. |
| I11 | native | np.cumsum/np.concatenate with closure dedup, upload once. | Setup; removes the 700k allocations. |
| I12 | native | Python config front end replacing Lua. | Setup. |
| I13 | native | Flat pointer arithmetic offs = i*sx + j*sy + k*sz with strides as arguments, coordinates from tl.arange. | Equal to C++ if coalesced on the fast axis. |
| I14 | native | Keep R in (slow, fast, t) order from a lexsort; index R tables by position; assert on host. | No runtime cost. |
| I15 | native | Work buffers are torch tensors allocated once; kernels take pointers; on MPS they must be device tensors. | Equal. |
| I16 | workaround | Decompose per direction into interior stencil tile, closure-row kernel (I18/I19) and R terms. A literal one-program-per-line ragged kernel is possible but strided gathers coalesce poorly, as in the C++ baseline. | Fused tile form may match or beat C++ on CUDA; ragged form about parity at best; triton-cpu unproven. |
| I17 | native | 2D tile (BLOCK_S x BLOCK_F), tl.load at shifted offsets per tap with constexpr coefficients, masked at interior bounds; x-axis vectorised across the contiguous index. | Bandwidth-bound; survey expects 1.0-1.3x faster than C++ on CUDA (unmeasured). |
| I18 | native | Closure-plane branch or tiny kernel over first/last T planes with unrolled static_range matvec. | Negligible; launch-bound when small. |
| I19 | workaround | ELL gather-dot kernel: load idx and C rows, gather x, tl.sum(axis=1), store to out_idx; unique rows so no atomics. | Gather-bound, about C++ parity; launch latency dominates when small. |
| I20 | workaround | Replace CSR by ELL/COO; a CSR row-parallel kernel with data-dependent loops is possible but ELL is simpler. | Parity or better when compacted. |
| I21 | native | Compact COO kernel out[row[e]] += w[e]*xR[col[e]]; plain store if rows unique, tl.atomic_add otherwise. | Much faster than C++ (no 16.8M row-pointer sweep). |
| I22 | native | Slab kernel adding extra[k]*g_face onto X planes, skipped when no Neumann BC. | Free. |
| I23 | native | ELL gather-dot over source D into output R(r); r != dir rows padded to 16-25 nnz. | Parity; launch-bound when small. |
| I24 | native | Merge into I23 via concatenated [D;R] source, or a second ELL kernel. | Parity. |
| I25 | native | Eliminate: the fused kernel writes the full output, no pre-zero pass. | Faster than C++ (one pass fewer). |
| I26 | native | Fold scalar k into the final fused elementwise expression. | Free. |
| I27 | workaround | Single 3D-halo-tile kernel for tile-local stencils, or three kernels with the first writing and later ones accumulating. | Fused can hit roofline (1 read + 1 write); per-axis costs 3x traffic like C++. |
| I28 | native | Program axis over the 3 directions, or 3 launches into a stacked tensor; fuse with the I29 consumer. | Parity or better. |
| I29 | native | Fused elementwise: load 6 streams, one FMA chain, store; or fuse into gradient kernels. | Roofline; faster than C++ if fused. |
| I30 | native | Dense masked FMA with a uint8 mask, or int32 index list scatter for tiny R sets. | Roofline-bound; mask 1 B/pt versus 4 B/pt. |
| I31 | native | 2D-tile face fill with computed strides, or where(dir_mask, value, u). | Negligible. |
| I32 | workaround | Triton has no graph object. Use torch.cuda.CUDAGraph around the launch sequence, or torch.compile(mode="reduce-overhead") with custom ops. No graph capture in the community Metal backends (unverified in this session). | CUDA graphs: parity. Without: 10-30 us per launch (estimate) times ~10-60 launches, so 5-30x slower than the 32 us C++ RHS at small 2D. |
| I33 | native | Plain Python function issuing kernel launches. | Dispatch-bound when small. |
| I34 | native | Elementwise kernel with tl.exp/tl.cos/tl.sqrt over coordinates decoded from tl.arange, or torch broadcast ops; SymPy-printed source via codegen. | Faster than C++ host virtual calls; fusable into the source add. |
| I35 | unsupported | No per-point host callback inside a kernel; replace by array-expression MMS (SymPy to Triton or torch). | N/A. |
| I36 | workaround | Pass t as a runtime scalar and evaluate the source in-kernel (I34); no host sync. | Removes the sync; faster than C++. |
| I37 | native | One fused kernel per stage over the packed field with pointer args and scalar constants. | Roofline; one launch instead of four buffers. |
| I38 | native | torch copy_ or Python pointer swapping (no baked pointers unless CUDA graphs). | Free. |
| I39 | native | Evaluate boundary values only on face/object index sets in a small kernel, time as scalar. | Algorithmic win over C++. |
| I40 | native | Hoist to setup as a Python attribute. | Free. |
| I41 | native | Masked block tl.max/tl.min then atomic_max/atomic_min (fp64 atomics unverified) or two-pass torch reduction; read back every K steps. | Parity; per-step sync dominates on GPU. |
| I42 | native | Fused error kernel gated on dump-due; dump via tensor.cpu().numpy().tofile. | Equal or better. |
| I43 | native | Python loop around launches. | Negligible. |
| I44 | native | Host scipy.sparse from the same tables; numpy.linalg.eigvals. | Diagnostic. |
| I45 | native | NumPy one-liners. | Test only. |
| I46 | native | Triton kernels are the expression fuser; write or generate fused jit functions. | Roofline. |
| I47 | workaround | Pointwise physics kernel (sqrt, division) over the component axis, flux derivatives batched with a program axis, dt by max reduction. | Fusion could be a large win on CUDA; unproven. |
| I48 | native | NumPy/scipy unchanged. | Diagnostic. |
| I49 | native | Keep SymPy; retarget printer.py to emit Triton jit functions or NumPy via lambdify. | Setup. |
| I50 | native | Host scipy fp64 with the operator exported as a LinearOperator. | Diagnostic. |
| I51 | native | In-process Python solver replaces the subprocess loop. | Big win, but JIT cold start (seconds per specialization) may offset it for many small solves. |

## Code sketches

Sketches are illustrative and untested. One precedence bug in the survey draft was fixed: the interior mask comparison needs parentheses, because `a & b != 0` parses as `(a & b) != 0`. The mask load is also guarded by the bounds mask.

Per-line block matvec with strides and per-line dense closures (I16, I17, I18, I19):

```python
@triton.jit
def axis_stencil(x, out, mask_int, tap_ptr, S, F, sS, sF, sA,
                 BS: tl.constexpr, BF: tl.constexpr, W: tl.constexpr):
    # tile over (slow s, fast f); sA is the stencil-axis stride; taps unrolled
    s = tl.program_id(0) * BS + tl.arange(0, BS)[:, None]
    f = tl.program_id(1) * BF + tl.arange(0, BF)[None, :]
    base = s * sS + f * sF
    inb = (s < S) & (f < F)
    m = inb & (tl.load(mask_int + base, mask=inb, other=0) != 0)  # interior mask
    acc = tl.zeros((BS, BF), tl.float64)
    for j in tl.static_range(W):
        acc += tl.load(tap_ptr + j) * tl.load(x + base + (j - W // 2) * sA,
                                              mask=m, other=0.0)
    tl.store(out + base, acc, mask=m)

@triton.jit
def closure_rows(x, out, row_out, idx, C, nrows,
                 TMAX: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = r < nrows
    t = tl.arange(0, TMAX)[None, :]
    ii = tl.load(idx + r[:, None] * TMAX + t, mask=rm[:, None], other=0)
    cc = tl.load(C + r[:, None] * TMAX + t, mask=rm[:, None], other=0.0)
    v = tl.sum(cc * tl.load(x + ii), axis=1)
    tl.store(out + tl.load(row_out + r, mask=rm), v, mask=rm)  # overwrites uniform result
# launch order: axis_stencil(dir) then closure_rows (same stream)
```

O/B CSR cut-cell coupling to R points (I21, I23, I24):

```python
@triton.jit
def coo_scatter(srcR, du, row, col, w, nnz, BN: tl.constexpr):
    # B: du.D[row] += w * u.R[col]  (compact COO, ~R*|R| entries)
    e = tl.program_id(0) * BN + tl.arange(0, BN)
    m = e < nnz
    v = tl.load(w + e, mask=m) * tl.load(srcR + tl.load(col + e, mask=m), mask=m)
    tl.atomic_add(du + tl.load(row + e, mask=m), v, mask=m)  # atomic only for dup safety

@triton.jit
def ell_DtoR(uD, uR, duR, idx, C, nR, K: tl.constexpr, BR: tl.constexpr, ND):
    # Bf and Br merged over concatenated source [D ; R]: idx >= ND selects uR
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = r < nR
    k = tl.arange(0, K)[None, :]
    ii = tl.load(idx + r[:, None] * K + k, mask=rm[:, None], other=0)
    cc = tl.load(C + r[:, None] * K + k, mask=rm[:, None], other=0.0)
    isR = ii >= ND
    xv = tl.where(isR, tl.load(uR + ii - ND, mask=isR, other=0.0),
                       tl.load(uD + ii, mask=~isR, other=0.0))
    tl.store(duR + r, tl.load(duR + r, mask=rm) + tl.sum(cc * xv, axis=1), mask=rm)
```

Gather/scatter boundary application (I30, I31):

```python
@triton.jit
def masked_src_and_bc(du, src, fluid_mask, dir_mask, k, N, BN: tl.constexpr):
    # du = k*du + src on fluid; du = 0 on Dirichlet faces/objects, as dense masks
    i = tl.program_id(0) * BN + tl.arange(0, BN)
    m = i < N
    v = tl.load(du + i, mask=m) * k
    v += tl.where(tl.load(fluid_mask + i, mask=m, other=0) != 0,
                  tl.load(src + i, mask=m), 0.0)
    v = tl.where(tl.load(dir_mask + i, mask=m, other=0) != 0, 0.0, v)
    tl.store(du + i, v, mask=m)

@triton.jit
def face_set(u, vals, off, inner, outer, stride, n, BN: tl.constexpr):
    # strided face: element(i) = off + (i//inner)*stride + i%inner
    i = tl.program_id(0) * BN + tl.arange(0, BN)
    m = i < n
    tl.store(u + off + (i // inner) * stride + i % inner,
             tl.load(vals + i, mask=m), mask=m)
```

Four-buffer scalar algebra with RK4 slot updates (I37, I38):

```python
@triton.jit
def rk4_stage(u0, rk, srhs, u1, c_stage, c_acc, N,
              BN: tl.constexpr, FIRST_STAGE: tl.constexpr):
    # field packed D|Rx|Ry|Rz in one array: one launch covers all 4 buffers
    i = tl.program_id(0) * BN + tl.arange(0, BN)
    m = i < N
    s = tl.load(srhs + i, mask=m)
    a = tl.zeros((BN,), tl.float64) if FIRST_STAGE else tl.load(rk + i, mask=m)
    tl.store(rk + i, a + c_acc * s, mask=m)
    tl.store(u1 + i, tl.load(u0 + i, mask=m) + c_stage * s, mask=m)

def step(state, t, dt):
    for c_stage, c_acc, dtc in ((dt/2, dt/6, 0), (dt/2, dt/3, dt/2),
                                (dt, dt/3, dt/2), (dt, dt/6, dt)):
        rhs(state.u1, state.srhs, t + dtc)       # ~10 fused launches
        rk4_stage[grid](state.u0, state.rk, state.srhs, state.u1,
                        c_stage, c_acc, N, BN=1024, FIRST_STAGE=(dtc == 0))
        update_boundary(state.u1, t + dtc)
    # final: u0 += rk
```

## Performance assessment

All figures here are survey estimates; none were measured or verified.

- Small 2D (2.6k to 5k points): poor. The C++ RHS is about 32 us (roughly 40 graph nodes at ~0.8 us), so an RK4 step is 0.3 to 1 ms. Triton launches are estimated at 10-30 us each from Python; with ~10-15 kernels per RHS and ~50 per step that is ~0.5-2 ms without graphs. Parity needs CUDA graph capture, which does not exist on Metal and is irrelevant on the macOS host. triton-cpu has unmeasured JIT/launch overhead and would likely lose to Kokkos serial at this size. Expect 3-30x slower than C++.
- Large 3D: good on CUDA/ROCm. Fused axis-stencil tiles are bandwidth-bound and can beat the one-team-per-line Kokkos kernel (13.7 ns/pt at 64^3 along the slow axis, per the catalog). On Apple in fp32 (~120 GB/s on M4 versus ~60 GB/s measured on host C++) at best a 1-2x gain, via unaudited community backends. triton-cpu unproven.
- Dispatch overhead: 10-30 us per launch from Python is an unverified estimate; CUDA graphs reduce it to ~2-5 us per node. Metal community backends have no documented dispatch figure and no graph capture. Cold JIT specialization was 9.48 s for one first add on triton-msl.
- Fusion: Triton fuses within a kernel only; it has no graph compiler. A mega-kernel over three axes plus closure, source and BC is feasible for tile-local stencils with a 3D halo, but gather-dot closures and R couplings still need 2-3 extra launches. Whole-RHS compilation exists only via torch.compile/Inductor on CUDA/ROCm (unverified here).

## Suitability as a DSL substrate

Good as a code-generation target for the bandwidth-bound bulk when the DSL emits Triton source text: SymPy printing and tl.static_range unrolling fit naturally. Weaknesses:

1. No whole-step compilation. The DSL would need CUDA graphs or its own scheduler that batches many small tables into one launch (concatenated closure/ELL tables with a table-id axis).
2. Weak CPU story. A second backend (Numba or C++ codegen) is needed anyway, so Triton cannot be the only emitter.
3. Metal is community-only with no fp64.
4. No ragged-segment primitive in tl.*; fragments must be flattened to masks plus tables, as in every array framework.

Conclusion: acceptable as the CUDA/ROCm emitter and an experimental Apple fp32 emitter, not a standalone substrate that meets the no-loss CPU bar.

## Risks

- No official CPU backend; triton-cpu cannot be verified for parity on the macOS host.
- Apple support unmerged and dependent on community maintainers, the Xcode Metal toolchain, and per-bump breakage. triton-msl lags (Triton 3.7.0).
- No fp64 on Apple; fp32 closure accuracy for small psi untested; double-float emulation costs several times in throughput.
- Python launch overhead versus ~0.8 us Kokkos graph nodes; no graph capture on Metal.
- Ragged, ELL and COO tables are workable but not what Triton's autotuner and layouts optimise.
- JIT cold start (seconds per specialization) hurts short runs and the Bayesian-optimization loop that spawns many small solves.
- Fast release cadence and MLIR/LLVM pin churn require version pinning.
- fp64 tl.dot and atomics on any backend were not verified.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| 3.8.0 (28 Aug) is latest; no Apple backend in mainline | Confirmed | Releases page lists 3.8.0 Latest, 3.7.1, 3.7.0, 3.6.0, no Apple backend. Year inferred from footer. |
| Apple support only via open triton-ext PR #126 (opened 2026-08-31) with MSL lowering, MPS dispatch, Xcode Metal toolchain | Confirmed | PR open, unmerged, 186 commits, latest 2026-10-04 builds against 3.8 and 3.9; 8x8 simdgroup MMA encoding; metallib via toolchain after in-process serialization failed. |
| 19-34% speedup on M4 Pro | Unverified | Not seen in the fetched PR summary; self-reported at best. A 3x3 conv regression is reported. |
| triton-msl 0.3.0 alpha, no general FP64, 9.48 s cold add, M1 not validated | Confirmed | README. Correction: it targets Triton 3.7.0 and PyTorch 2.12; the survey's "tracking 3.8/3.9" applies only to triton-ext #126. |
| triton-cpu experimental, Linux-only, not mainline | Uncertain | README says work in progress, CPU "Under development", Linux only. It also lists NVIDIA and AMD GPUs. No-benchmarks, no-op-list and lag-behind-upstream claims not checked. |
| TRITON_INTERPRET=1 is debug-only | Uncertain | Docs and thread not fetched; consistent with common knowledge. |
| No graph capture; Inductor-MPS emits MSL directly; 10-30 us launch | Uncertain | Not verified; the launch figure is the survey's own estimate. triton-msl README lists its own Inductor integration, which may conflict with the Inductor-MPS claim. |
| fp64 tl.dot on Ampere+ and fp64 atomics on CUDA | Uncertain | Not checked. |
| gfx1250 and Rubin SM107 in 3.8; Gluon in 3.x | Uncertain | Not checked against release notes. |
| triton-metal fork issues; flash-linear-attention and Liger PR links | Uncertain | Links not checked. |
| Overall score | Adjusted: 4 (unchanged) | Verifier found no contradiction of the low score; perf and fp64 claims remain unverified. |

## Sources

- https://github.com/triton-lang/triton/releases
- https://github.com/triton-lang/triton-ext/pull/126
- https://github.com/Janssena/triton-msl
- https://github.com/akashmadhu4/triton-metal
- https://github.com/triton-lang/triton-cpu
- https://dev-discuss.pytorch.org/t/no-cpu-backend-in-triton/2599
- https://github.com/fla-org/flash-linear-attention/pull/1321
- https://github.com/linkedin/Liger-Kernel/pull/1442
- https://pytorch.org/blog/triton-kernel-compilation-stages/
- https://newreleases.io/project/github/triton-lang/triton/release/v3.8.0
