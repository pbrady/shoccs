# Apple GPU, fp64 and Kokkos-Metal Facts for SHOCCS

Status as of 2026-10-12. Versions referenced: PyTorch 2.14.1 (30 Sep 2026), MLX v0.32.3 (29 Sep 2026), jax-metal 0.1.1 (8 Oct 2024), AdaptiveCpp Metal backend (experimental, macOS 26 only), Kokkos 5.x.

## Verdict box

| Question | Verdict |
|---|---|
| Fit for SHOCCS | An Apple-GPU ("mps") target is viable only as fp32 or hand-written double-float (df64). It cannot be a requirement of an fp64 Python port. |
| CPU parity | The primary bar. A fused, whole-step compiled fp64 CPU path is the realistic way to match C++ Kokkos. |
| Apple GPU | No Kokkos route exists. Python GPU stacks work in fp32 only. A GPU wins only at large 3D sizes on Pro/Max chips, or at small sizes if many RK steps are batched per command buffer. |
| fp64 | No Apple GPU (M1 to M5) has native fp64 and MSL has no usable `double`. Only emulation (df64 or software binary64) is possible, and no Python framework offers it. |
| Risk | Medium-high for any GPU path (precision study needed, experimental toolchains). Low for CPU-first with a precision-generic IR. |

## Backends and fp64 table

| Stack | Version | Apple GPU | fp64 on GPU | Status of evidence |
|---|---|---|---|---|
| Metal / MSL | n/a | Native | None. No `double`; `matrix_double*` exists only under `__HAVE_NATIVE_DOUBLE__`, never set on Apple GPUs | AdaptiveCpp docs quote confirmed; MSL spec text not re-fetched |
| PyTorch MPS | 2.14.1 | Yes (fp32, fp16, bf16) | Not supported. Creating a float64 MPS tensor raises an error | Confirmed, mostly indirect (no release note changes it) |
| MLX | 0.32.3 | Yes | CPU only. Issue #1905 (software fp64 on GPU) closed Dec 3 2025 as not going to implement. On the CPU, fp64 exp, sin, cos and erf use fp32-accuracy approximations (#3047, open) | Confirmed (#1905 and #3047, consistent with Part II and `frameworks/mlx.md`) |
| jax-metal | 0.1.1 | Experimental, stalled since Oct 2024 | np.float64 listed unsupported | Confirmed |
| Taichi | n/a | Metal backend | No f64, i64, u64, f16 on Metal; f64 on a Mac only on the CPU backend | Confirmed for f64/i64/f16; exact table glyph string unverified |
| Halide | 21.0.0 | Metal target | Float(64) cannot reach a Metal kernel: `CodeGen_Metal_Dev.cpp` raises a hard error on 64-bit float literals ("Metal does not support 64-bit floating point literals") | Confirmed from source, consistent with Part II (`frameworks/halide.md`). Correction: the type printer still maps Float(64) to `double`, so the hard error is on fp64 literals, not on every Float(64) |
| Kokkos | 5.x | No Metal or Apple-GPU backend | n/a | Confirmed from requirements page (SYCL only via IntelLLVM 2024.2.1); absence of third-party backends not searched |
| AdaptiveCpp (SYCL to Metal) | develop | Experimental, macOS 26 only | No `double` upstream; soft-double planned | Confirmed from install-metal page; PR numbers unverified |
| OpenCL on macOS | n/a | Deprecated | No cl_khr_fp64 on Apple GPUs | Survey claim, not re-verified |
| OpenMP offload to Apple GPU | n/a | Does not exist | n/a | Survey claim (LLVM list thread), not re-verified |

### fp64 emulation options

| Option | Precision and cost | Availability |
|---|---|---|
| df64 (float32 pair, Dekker/Knuth) | About 48-bit mantissa, about 1e-14 relative (fp64 is 1.1e-16). fp32 exponent range, denormals flushed, not IEEE. About 10-20 flops per add, 3-8 per FMA-based multiply. 8 B per value, same bandwidth as fp64 | No framework offers a dtype; hand-written Metal kernels only (for example `mx.fast.metal_kernel`, PyTorch MPS custom shaders, metal-cpp) |
| Correctly rounded software binary64 | warpx-metal reports about 13x fp32 cost (unverified). metal-float64 estimated 1/32 to 1/64 fp32 throughput but was archived Aug 2024 at the planning stage | Third-party patched AdaptiveCpp only; not upstream |
| Mixed precision (fp32 GPU interior, fp64 CPU closures and BCs) | Doubles traffic on shared DRAM and adds sync points | MLX maintainers recommend a CPU/GPU split; works with any framework |

Inference (unverified): with 2 to 9 taps at about 20 flops per tap, the axis stencil and gather kernels sit near 4-11 flop/B, below the Apple GPU fp32 flop:byte ratio (about 30+ for M4-class), so a df64 stencil may stay bandwidth-bound.

## Idiom mapping table

The idiom ids and labels follow the Part I catalog (`01-codebase-idioms.md` Section 2). "Support" is for an Apple-GPU (MPS or MLX Metal) target, in fp32 unless noted. CPU fp64 support is covered in the per-framework files.

| Idiom ids (Part I names) | Support (Apple GPU) | How | Expected perf |
|---|---|---|---|
| I01 (registry slot/buffer table allocation: D, Rx, Ry, Rz per field) | native | Dict of 4 arrays or one packed 1-D buffer with static offsets | Parity; packing cuts launches about 4x |
| I02, I03, I05, I06, I07, I08, I09, I11, I12, I14 (ray casting to R records; line fragmentation and dirichlet_line; psi-independent stencil constants; psi-dependent closure evaluation `nbs`; cross-direction interpolation; domain_discretization; cut_discretization; block flattening to device arrays; config and system setup; R ordering contract) | workaround | Host NumPy/SymPy/Numba fp64 at setup, then cast and upload. Evaluate psi-dependent coefficients in fp64 on the host before casting, since small-psi cancellation in fp32 is risky | Setup only |
| I04 (index-set construction), I40 (per-call rebuild of object-BC gather lists) | native | Bool masks and int index arrays built on host and uploaded once; the I40 per-call rebuild is hoisted to setup | Setup only; masks fuse at runtime |
| I10 (COO-to-CSR build), I20 (generic CSR accumulate kernel), I21 (B: R(dir) -> D) | workaround | No usable sparse type on MPS (CSR PR #196205 open, matvec not covered; COO historically NotImplementedError). I10 becomes a host build of compact ELL or COO tables at setup; I20 and I21 become gather plus scatter-add over non-empty rows | Better than the C++ full-height sweep at large N; launch-bound at small N |
| I13 (flat C-order D with per-direction strides) | native | `reshape(nx,ny,nz)`; shifted slices are strided views | Parity |
| I15 (span bridge and system-owned work buffers) | native | Intermediates inside a compiled function; functional outputs | Parity or better when fused |
| I16 (ragged line block matvec O) | workaround | Fused shifted-slice stencil, mask select, ELL closure gather-dot, scatter. Alternative: one custom Metal kernel per direction | Large 3D: bandwidth-bound; closure adds 1 to 3 kernels per direction |
| I17 (interior circulant axis stencil) | native | Slice arithmetic fused by `mx.compile`; Inductor-Metal is fragile for indexing. conv1d is a poor fit for 3 to 9 taps | Near bandwidth in fp32 |
| I18 (wall closures) | native | Small einsum or dense matmul placed via slice update | Negligible volume; launch-bound when small |
| I19 (object closures), I23 (Bf_r: D -> R(r)), I24 (Br_r: R(r) -> R(r)) | workaround | ELL gather-dot then `index_add_`/`.at[].add`; index_add_, scatter_add_, gather exist on MPS (native Metal since 2.13, claim unverified) | Small row count; 2 to 3 unfused kernels per table dominate launches at small 2D sizes |
| I22 (N: Neumann wall data to the first X rows), I31 (Dirichlet face and object fill) | native | Slab `where` or slice update | Negligible |
| I25 to I30 (zero-fill before accumulation; whole-scalar scale; accumulation policy and write ordering; gradient fan-out; variable-coefficient dot product; masked add/assign at gather index sets) | native | Functional sums with masked `where`, fused elementwise | Parity or better |
| I32 (Kokkos Graph capture, instantiate, submit) | workaround | `mx.compile` over the RK step plus one eval per step; MPS has no verified graph replay (`torch.accelerator.Graph` MPS support unverified) | 60 to 120 kernels per step; at small 2D sizes the bar is the measured 0.90 ms shipped heat.lua step and the estimated 0.15 ms fixed-up step (`05-measurements.md`) |
| I33 (eager apply path) | native | Plain Python function calls; lazy until eval | Eager adds a few us to tens of us per op |
| I34 (closed-form MMS evaluation at all D and R locations) | native with caveat | Array math on broadcast coordinates. No fp64 or df64 transcendentals on GPU; MLX CPU fp64 exp/cos use fp32-accuracy approximations (#3047, confirmed, open). Evaluate boundary or source terms in NumPy fp64 for convergence studies | Fast on Metal, accuracy limited |
| I35, I36, I39 (Lua per-point MMS callback; per-stage host source fill before graph submit; update_boundary per stage) | workaround | SymPy-generated array expressions with `t` as traced scalar | Removes host sync |
| I37, I38, I46 (RK4 and forward-Euler slot linear combinations; deep_copy_slot for pointer stability; expression templates with alias check) | native | Functional `u1 = u0 + c*k` | Fully fused |
| I41 (stats: masked reductions with per-step host readback) | native | Device reduction, read back every K steps. Reductions are not fused and each readback is a sync (about 100 us class) | Never sync per stage |
| I42, I43 (error field, binary dump and XDMF; outer time loop and dispatch) | native | Host loop around compiled step; unified memory but fp64 to fp32 forces copies on upload | Host I/O dominated |
| I44, I48, I49 (eigenvalue diagnostic; Python dense and sparse operator assembly; Python symbolic coefficient layer and codegen) | workaround | Stay on host fp64 | n/a |
| I45 (reference component matvecs) | native | NumPy one-liners | n/a |
| I47 (Euler scaffold) | workaround | Leading component axis; pointwise physics fused | Fast in fp32; accuracy question sharpens |
| I50 (Python stability-analysis cascade, sparse non-Hermitian eigensolver) | unsupported | Stays scipy/ARPACK on host | n/a |
| I51 (cpp_bridge subprocess and BO driver) | native | In-process Python solver replaces the subprocess bridge; DLPack sharing with torch unverified | Removes process-per-point cost |

## Code sketches

Precision-generic IR with a df64 option (design sketch, not tested):

```python
# dtype is a per-buffer or per-kernel parameter; GPU backend lowers df64 to Metal source
@kernel(dtype=df64)
def axis_stencil(u: Field, du: Field, w: Taps, interior: Mask):
    du[i] = where(interior[i], sum(w[k] * u[i + k] for k in taps), du[i])
```

Dense replacement for the CSR cut-cell closure (works on MPS in fp32 today, per survey; not run here):

```python
import torch
src = torch.cat([uD.flatten(), R])                  # concatenated D|R source
vals = (C * src[idx]).sum(-1)                       # ELL gather-dot, C:(rows,w), idx:(rows,w)
du.index_add_(0, rows, vals)                        # rows are unique
```

Batching steps per command buffer (MLX):

```python
step = mx.compile(rk4_step)          # whole RK4 step, one graph
for n in range(nsteps):
    u = step(u, t)
    if n % K == 0:
        mx.eval(u, err)              # sync only every K steps
```

Double-float add (Knuth two-sum, Thall 2006), the building block of a df64 layer:

```metal
float2 df_add(float2 a, float2 b) {
    float s = a.x + b.x;
    float v = s - a.x;
    float e = (a.x - (s - v)) + (b.x - v);
    e += a.y + b.y;
    float hi = s + e;
    return float2(hi, e - (hi - s));
}
```

## Performance assessment

Host GPU measurements (torch MPS fp32, MLX GPU fp32, with numpy, torch and MLX CPU fp64 as reference) are in `05-measurements.md` Section D. They ran natively on the macOS host (Apple M4, macOS 26.2, torch 2.11.0, MLX 0.32.3), because the C++ measurements ran in a Linux devcontainer that cannot reach the Apple GPU. Measured on a synthetic SHOCCS-shaped RK4 step: GPU op plus synchronisation 167 us (MLX) to 181 us (torch MPS compiled); whole-step-compiled fp32 GPU step 0.35 to 0.5 ms from 51x51 to 201x201, against 0.085 ms for the best fused CPU fp64 step from Python at 51x51; 1.0 ms at 65^3 and 43.6 ms at 257^3, 4 to 6x faster than the best CPU fp64 path at 129^3 and 257^3; fp32 error 1e-7 to 2.4e-7 relative after 10 steps on synthetic closures. Every other Apple GPU figure below is a literature or forum figure.

Dispatch overhead (medium confidence at best):

- Measured C++ floor (fresh container runs, `05-measurements.md`, 1 thread): 0.55 us per tiny host kernel, 22.7 us for the whole heat.lua RHS at 51x51 with its sphere, and 0.90 ms for the shipped heat.lua RK4 step (estimated 0.15 ms after the four fixed-up C++ changes). The March 2026 baseline gave about 0.7 to 0.8 us per kernel and about 32 us for a 16^3 RHS.
- Eager PyTorch CPU fp64 axpy at n = 5041 took 1.1 us per op on this host (local measurement, torch 2.11 CPU), close to the Kokkos floor.
- A GPU op plus synchronisation on the owner's M4 costs 167 us through MLX (compiled), 175 us through eager torch MPS and 181 us through torch.compile on MPS (`05-measurements.md` Section D, median of 200 calls of a 1024-element op plus sync). That is seven to eight times a whole C++ RHS at 51x51 (22.7 us) and about twice the best fused CPU fp64 step from Python (0.085 ms). An earlier unverified forum figure (single M2 Max report, no Apple reply) put an empty-kernel round trip at about 120 us. Isolated small dispatches also mostly measure GPU clock ramp-up.
- Consequence: a GPU path wins at 2D sizes (2.6k to 5k points, about 40 kernels per RHS, about 280 launches plus several reductions per RK4 step) only if many steps are encoded into one or few command buffers with no readback. Even then 85 to 160 KB of state gives no bandwidth advantage, so a fused compiled CPU step (5 to 10 kernels) is the realistic target.
- warpx-metal reportedly shows the same pattern (GPU trails 12-thread CPU on small 2D grids, leads only near 128^3 in 3D). Unverified.

Small 2D versus large 3D:

| Regime | Size | Assessment |
|---|---|---|
| A | 2.6k to 5k points, L2-resident | CPU wins, now measured: compiled fp32 GPU step 0.35 to 0.5 ms against 0.085 to 0.15 ms for a fused CPU fp64 step from Python and 91 us for four C++ RHS submits (`05-measurements.md` Section D). The 167 to 181 us sync floor exceeds a C++ RHS unless steps or independent cases are batched per dispatch |
| B | 32^3 to 64^3, 2 to 16 MB per buffer | Fused CPU code has large headroom (current one-team-per-line kernel at 4 to 14 ns/pt in the March baseline and 6.6 to 8.9 ns/pt in the fresh runs, versus 0.37 to 0.44 ns/elem streaming). GPU edge modest and eaten by dispatch |
| C | 256^3, 134 MB per buffer | Base chip: little GPU headroom (CPU already near rated bandwidth). Pro/Max: possible 2 to 4x window, only in fp32 or df64 |

Bandwidth measurement (local, unverified, chip model unknown): single-thread fp64 axpy at 16.8M elements reached about 96 GB/s (4.1 ms) with no gain from 4 or 10 threads. For comparison, the C++ expression kernel in the Linux devcontainer reached 67 GiB/s on 1 thread and 97 GiB/s on 8 threads at 1M elements (`05-measurements.md` Section B). The verifier flagged this as unusually high and worth rechecking for hidden parallelism or cache effects. Rated bandwidths: M4 120, M4 Pro 273, M4 Max 546 GB/s; the share of GPU-rated bandwidth reachable from CPU clusters on Pro/Max is not measured here.

Fusion: PyTorch Inductor on MPS has a Metal codegen. It works for elementwise ops and is fragile for indexing, reductions, dynamic shapes, and axes larger than the threadgroup (#152155). It has no fp64. Measured instance (`05-measurements.md` Section D, torch 2.11.0): compiling the whole synthetic RK4 step on MPS emitted invalid Metal for an in-place face-slice fill (InductorError: SyntaxError: failed to compile) and needed a static face-mask multiply instead; with that fallback the compiled MPS step ran at 0.50 ms at 51x51 and 43.6 ms at 257^3. MLX `mx.compile` fuses elementwise ops but not gathers, scatters or reductions. A custom per-direction Metal kernel (one threadgroup per line) is the robust but hand-written path.

Unified memory: CPU and GPU share DRAM and bandwidth, so readbacks cost synchronization rather than transfer, and a CPU-plus-GPU hybrid shares one bandwidth budget. MLX numpy zero-copy needs page alignment and no dtype change, so fp64 data copies. PyTorch CPU to MPS copies still use blits and zero-copy `from_numpy(...).to('mps')` is undocumented (PR #175818 claim unverified).

## Suitability as a DSL substrate

- Apple GPU is a secondary backend, not the substrate. The IR must carry a dtype parameter (f64, f32, df64) per buffer or kernel, keep geometry-derived tables (ELL, masks) as host-built setup data, and emit Metal source for the GPU target.
- Use dense index ops (gather, `index_add_`) for the unstructured cut-cell part. Do not depend on torch.sparse on MPS.
- Batch whole RK steps per command buffer; keep reductions on device and read back every K steps; never sync per stage.
- Transcendentals in MMS and sources need df64 implementations or host fp64 evaluation.
- CPU fp64 paths that qualify as the primary bar: JAX CPU (x64), Taichi CPU (f64), Numba, generated C++ or ISPC. PyTorch MPS and MLX GPU do not qualify for fp64.

## Risks

- Precision: an fp32 MPS mode needs a numerical study of small-psi high-order cut-cell closures (conditioning, stability, error floors near 1e-7, RK4 accumulation).
- Toolchain: AdaptiveCpp-Metal is experimental, macOS 26 only; jax-metal is stalled; MPS Inductor is fragile.
- Kokkos: no path to Metal. Porting its SYCL backend to AdaptiveCpp is unowned work, and fp64 would still be missing.
- Performance: df64 may become compute-bound if flop counts exceed the estimates; software binary64 at about 13x is likely compute-bound.
- Evidence quality: latency and bandwidth figures are anecdotal or local single measurements.
- Hybrid GPU plus CPU designs add syncs and contend for one DRAM budget.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| No Apple GPU has native fp64; AdaptiveCpp docs say so | Confirmed | Exact sentence on install-metal page. MSL spec and per-chip ALU details not re-fetched |
| PyTorch MPS 2.14.1 lacks float64 | Confirmed (indirect) | 2.14.1 is latest, dated 30 Sep; no release note adds float64; Lightning issue and PyTorch forum say unsupported. 2.8 notes unchecked |
| MLX fp64 CPU-only; #1905 closed Dec 3 2025 | Confirmed | zcbenz: "we are not going to implement it"; awni: emulation would be slow. CPU-only claim not read from docs |
| MLX version "0.32.x (Aug 2026)" | Corrected | Latest is v0.32.3 on 29 Sep 2026; August date unverified |
| jax-metal 0.1.1, Oct 8 2024, float64 unsupported | Confirmed | PyPI page |
| Taichi Metal lacks f64, i64 | Confirmed | Type table (also u64, f16). Glyph string and "CPU only on Mac" not separately checked |
| Kokkos has no Metal backend; SYCL only IntelLLVM 2024.2.1 | Confirmed | Requirements page. No-issues-in-repo claim not searched |
| AdaptiveCpp Metal experimental, macOS 26, no double | Confirmed | install-metal page |
| AdaptiveCpp PRs #2196, #2243, #2221, #2250, #2223; atomic64 emulation | Unverified | Not fetched |
| warpx-metal about 13x fp32 cost; GPU trails CPU on small 2D | Unverified | Not fetched |
| PR #196205 open, leaves add/addmm/addmv/mm uncovered | Confirmed | PR opened Sep 7 2026, rebased Sep 30, "not covered" list matches. COO issues #141320 etc. not fetched |
| 2.13 moved scatter/gather etc. to native Metal; 2.12 shared memory (PR #175818) | Unverified | 2.13 blog and PR not fetched |
| 2.12 `torch.accelerator.Graph`, 2.13 FlexAttention on MPS | Unverified | Only the 2.14 native linalg item was seen |
| Metal round trip about 120 us (M2 Max) | Unverified; superseded | Forum thread 767853 not fetched; single anecdote. Kept only as an earlier unverified forum figure: the measured GPU op-plus-sync floor on the owner's M4 is 167 to 181 us (`05-measurements.md` Section D) |
| Single thread about 96 GB/s; Pro/Max 2 to 4x window | Unverified | Local, not reproducible by verifier; chip model unknown; should be rechecked |
| arXiv 2607.11391 double-word FMA 17 to 29 flops | Unverified | Not checked; omitted from the decision logic |
| Halide Metal fp64: `CodeGen_Metal_Dev.cpp` raises a hard error on 64-bit float literals | Confirmed, with correction (cross-check added in editing; not in the original 11-claim tally) | Source file, as in the Part II Halide verification (`frameworks/halide.md`). The type printer still maps Float(64) to `double`, so the error is on fp64 literals. Issue #8245 not fetched |
| MLX #3047: CPU fp64 exp, sin, cos, erf use fp32-accuracy approximations | Confirmed (cross-check added in editing; not in the original 11-claim tally) | Issue open (opened 23 Jan 2026, labels bug and low priority; latest maintainer status 16 Aug 2026), as in `frameworks/mlx.md` |
| OpenCL and OpenMP-offload routes | Unverified | Survey claims only |

Tally note. The original verification pass counted 11 claims for this fact sheet: 7 confirmed, 0 refuted, 4 uncertain (the earlier unverified 120 us round trip, the 96 GB/s single-thread rate, the AdaptiveCpp PR numbers and the warpx-metal 13x; Part II Section 8). The log has since grown beyond those 11 rows: the two rows marked above were added as cross-checks in editing, and other rows were split or added in later passes (the PyTorch 2.12 and 2.13 release details, the arXiv double-word figure, the OpenCL and OpenMP-offload routes). None of the added rows is in the 11-claim tally, and the tally is not re-derived from the current log.

## Sources

- https://adaptivecpp.github.io/AdaptiveCpp/install-metal/ (verified)
- https://github.com/AdaptiveCpp/AdaptiveCpp/pulls?q=metal (unverified)
- https://github.com/Lulzx/warpx-metal (unverified)
- https://github.com/philipturner/metal-float64 (unverified)
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (not re-fetched)
- https://developer.apple.com/forums/thread/714765 ; https://developer.apple.com/forums/thread/767853 (unverified)
- https://github.com/pytorch/pytorch/releases (verified)
- https://github.com/pytorch/pytorch/pull/196205 (verified)
- https://github.com/pytorch/pytorch/pull/175818 ; https://pytorch.org/blog/pytorch-2-13-release-blog/ (unverified)
- https://github.com/pytorch/pytorch/issues/141320 ; https://github.com/pytorch/pytorch/issues/152155 (unverified)
- https://github.com/Lightning-AI/pytorch-lightning/issues/21261 ; https://discuss.pytorch.org/t/float64-tensors-on-mps/224325 (verified, indirect)
- https://github.com/ml-explore/mlx/issues/1905 ; https://github.com/ml-explore/mlx/issues/3047 ; https://github.com/ml-explore/mlx/releases (verified)
- https://github.com/halide/Halide/blob/main/src/CodeGen_Metal_Dev.cpp (verified in the Part II Halide pass)
- https://pypi.org/project/jax-metal/ (verified)
- https://docs.taichi-lang.org/docs/type (verified)
- https://kokkos.org/kokkos-core-wiki/get-started/requirements.html (verified)
- https://www.macrumors.com/guide/m4-vs-m4-pro-vs-m4-max/ (rated bandwidth)
- https://arxiv.org/pdf/2607.11391 (unverified)
- Local measurement: torch 2.11 CPU fp64 axpy bandwidth on the survey host (not reproducible; MPS probe could not run in the sandbox)
