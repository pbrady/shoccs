# Apple-silicon CPU fp64 headroom (NEON, SME/SSVE, AMX, Accelerate, bandwidth, OpenMP toolchains)

| | |
|---|---|
| Verdict | Shortlist (verifier-adjusted overall 7/10, down from 8). This is hardware, not a framework: the realistic target and baseline for the Python port. |
| Fit for SHOCCS | High. A compiled, fused fp64 NEON path reaches roofline for the packed-field stencil, ELL closure and fused RK stage. |
| CPU fp64 parity | Yes (9/10). Native IEEE fp64 with 2-lane NEON FMA. Expected 3x to 20x over current Kokkos host (modelled, not measured). |
| Apple GPU | None for fp64 (2/10). No Apple GPU has fp64; the CPU is the only native-fp64 unit on the chip and shares DRAM with the GPU. |
| Risk | Low long-term (8/10). Main risks: unmeasured perf budgets, OpenMP toolchain fragility on Apple clang, M5 core-count split (super + performance cores). |

## Status as of 2026-10-12

Version checked: Kokkos 5.2.2 (2026-09-10, confirmed latest on GitHub releases; Homebrew formula is 5.2.2 and depends on libomp). Hardware: M1 to M5 families. M4 introduced SME/SSVE (two SME units, one shared by the P-cluster, one for the E-cluster). M5 base, Pro and Max exist: Apple spec pages list M5 Pro at 307 GB/s and M5 Max at 460 GB/s (32-core GPU) or 614 GB/s (40-core GPU). M5 Max has an 18-core CPU (6 super + 12 performance cores). AMX (M1 to M3) is undocumented. Gaps: no standalone fp64 SSVE FMLA benchmark found; no official Apple clang SME-intrinsics documentation found; Kokkos 5.2.2 release notes did not load.

## Backends and fp64

| Backend | Status | fp64 |
|---|---|---|
| CPU NEON | Primary target. 128-bit, 2 fp64 lanes, 4 FMA pipes. Reachable by any LLVM-based codegen (Numba, JAX/XLA CPU, Taichi, generated C, ISPC). | Native IEEE. fp64 NEON FMLA about 56 GFLOPS per P-core is UNVERIFIED: inferred as half of the published fp32 107 GFLOPS (P-core), not a reported measurement. |
| CPU SME/SSVE (M4+) | Two units; the P-cluster unit is shared by about four user-interactive threads. Non-streaming SVE instructions fault (EXC_BAD_INSTRUCTION). Not reachable from LLVM autovectorisation in the frameworks above. | fp64 FMOPA (outer product) about 503 GFLOPS (P) and 89 (E), a quarter of fp32 (about 2008 P / 357 E). These are the unit peak, shared and not additive across threads. Suits GEMM, not banded stencils. |
| AMX (M1 to M3) | Undocumented, reachable only via Accelerate. | Unverified. |
| Accelerate (BLAS, vDSP, vForce) | Closed source. Routing of fp32 GEMM to AMX on M1 to M3 is supported by literature; routing to SME on M4 and any fp64 path on M4 are UNVERIFIED. vDSP_convD does 1D FIR with per-call overhead and no fusion. | fp64 vForce (vvexp, vvcos) available. |
| Apple GPU (Metal/MPS) | Not applicable. | None. |
| OpenMP | Apple clang has no -fopenmp (not re-verified this pass; consistent with Homebrew libomp dependency). Options: Homebrew libomp (keg-only, -Xpreprocessor -fopenmp), Homebrew/Spack LLVM clang, Kokkos Threads or Serial, Numba prange, JAX/XLA Eigen pool, GCD/pthread pool in generated C. | n/a |

## Mapping of the hard runtime shapes

| Idiom | CPU NEON mapping | Support |
|---|---|---|
| Packed field vector [D\|Rx\|Ry\|Rz] | One contiguous 64-byte-aligned fp64 buffer with static offsets. Free on CPU; one parallel region covers all parts. | native |
| Masked axis stencil (P2) | Fused loop nest, unit-stride axis vectorised (float64x2_t, vfmaq_f64); x via shifted loads, y/z via row-offset loads; mask via vbslq_f64 or interior-span splitting. About 1.25 flop/B for E2: bandwidth-bound at large N, L2/SLC-bound at 64^3. SME gives no gain. | native |
| ELL gather-dot-scatter (P3) | Scalar loop per row, acc += C[r][k]*src[idx[r][k]], then out[row[r]] += acc. Tables are 0.5 to 2.5% of points and cache-resident; unique row ids so no atomics; pad to a dummy index to stay branch-free. NEON has no gather but loads are cheap. | native |
| Face-plane assignment (P4) | Strided memcpy/memset-style loops; negligible. | native |
| Masked reduction (P5) | Fused NEON vmaxq_f64 plus mask; no sync cost on CPU; read back every K steps or fuse into the last RK stage. | native |
| Whole-step compilation (P6) | One parallel region per RK stage with spin/futex barriers (about 1 to 3 us, estimate). Compiled C/LLVM gets this directly; eager NumPy/PyTorch dispatch of about 300 ops costs 0.3 to 1.2 ms per step. | workaround |
| RK slot arithmetic | Fuse u1 = u0 + c*k into the sweep store; removes about 27 full-field touches per step (3.6 GB at 256^3, model estimate). | native |
| fp64 transcendentals (MMS, sources) | libm or Accelerate vForce in fp64. Caution: MLX CPU reportedly gives fp32 accuracy on some fp64 transcendentals (unverified). | native |
| Dense closure/wall blocks | Plain NEON FMA on 4x6 to 9x12 blocks. Accelerate call overhead and SME mode switch exceed the work. | native |
| CSR B/N full-height sweeps | Replace with compact ELL tables. Algorithmic gain, not an ISA gain. | workaround |
| Multicore threading | Partition by z-slabs or y-pencils. No affinity API on macOS; use QoS user-interactive and size threads to the P-core count (4 on base M4). For M5 Pro/Max, the super + performance split needs re-derivation (unverified). | workaround |

## Code sketch

```c
// Fused 3-axis E2 Laplacian + scale + RK stage update, fp64 NEON, one z-slab per task.
// u: current stage input, u0: stage base, ou: new stage input, k: accumulated rhs. mask: 0/1 doubles
// (or byte mask) for fluid interior. Rows: x is unit stride.
#include <arm_neon.h>
static inline void lap_row(const double* restrict u, const double* restrict u0,
        double* restrict ou, const double* restrict mask,
        long nx, long sy, long sz, double cx, double cy, double cz,
        double kappa, double dtc) {
  const float64x2_t vcx = vdupq_n_f64(cx), vcy = vdupq_n_f64(cy), vcz = vdupq_n_f64(cz);
  const float64x2_t vk = vdupq_n_f64(kappa), vdt = vdupq_n_f64(dtc);
  long i = 1;
  for (; i + 2 <= nx - 1; i += 2) {
    float64x2_t c  = vld1q_f64(u + i);
    float64x2_t xm = vld1q_f64(u + i - 1), xp = vld1q_f64(u + i + 1);
    float64x2_t acc = vmulq_f64(vcx, vaddq_f64(xm, xp));        // x taps (shifted loads)
    acc = vfmaq_f64(acc, vcy, vaddq_f64(vld1q_f64(u + i - sy), vld1q_f64(u + i + sy)));
    acc = vfmaq_f64(acc, vcz, vaddq_f64(vld1q_f64(u + i - sz), vld1q_f64(u + i + sz)));
    // coefficients cx,cy,cz fold the -2c term: acc += (-2*(cx+cy+cz)) * c
    acc = vfmaq_f64(acc, vdupq_n_f64(-2.0*(cx+cy+cz)), c);
    float64x2_t m   = vld1q_f64(mask + i);                       // 1.0 fluid, 0.0 solid/Dirichlet
    float64x2_t rhs = vmulq_f64(vk, vmulq_f64(acc, m));
    vst1q_f64(ou + i, vfmaq_f64(vld1q_f64(u0 + i), vdt, rhs));   // fused u0 + dt*c*rhs
  }
}
// Driver: parallel over (k) slabs with 4..10 pthreads/GCD threads at QoS user-interactive;
// ELL closure rows are applied afterwards for the same slab (L2-hot): out[row[r]] += dot(C[r], u[idx[r]]).
```

## Performance assessment

All numbers below are model estimates from rated bandwidth and inferred flop rates. None were measured. Both the 56 GFLOPS per-core rate and CPU-reachable bandwidth are unverified inputs, and the 1646 us C++ 64^3 baseline is inconsistent by about 2x.

Rated DRAM bandwidth (Apple/press): M1 68, M2 and M3 100, M4 120, M4 Pro 273, M4 Max 410 or 546, M5 Pro 307, M5 Max 460 or 614 GB/s. The M4 and earlier figures were not re-verified in this pass; M5 figures are confirmed. CPU-reachable bandwidth is lower on Max-class parts: AnandTech found about 240 of 408 GB/s on M1 Max (not re-verified). Treat 250 to 400 GB/s for M4 Max CPU as low-confidence inference.

Flop model: E2 Laplacian + scale + stage update is about 20 flop/point, 16 to 24 B/point, intensity 0.8 to 1.25 flop/B.

| Case | Estimate (model) |
|---|---|
| 64^3 (2.1 MB per buffer, fused working set about 6 MB, fits in M4 P-cluster L2) | Fused RHS 30 to 60 us on base M4, 20 to 40 us on Max. Bar is 1.7 ms, so headroom is roughly 30x to 60x. |
| 256^3 base M4 | Memory-bound, about 4 to 5 ms per RHS fused; unfused three-pass 10 to 14 ms. Below the survey's own 16 to 31 ms floor for the 14-double/point C++ model. |
| 256^3 M4 Max | About 1.5 to 2.5 ms per RHS (depends on the low-confidence bandwidth figure). |
| E4 to E8 (9 taps, about 60 flop/pt) | Compute-balanced, about 4 to 6 ms on base chip at 256^3. |
| Small 2D (2.6k to 5k points) | About 5 to 10 us per RHS single-threaded; threading barriers dominate, use 1 to 2 threads or none below about 100k points. |

SME/SSVE/AMX/Accelerate give no benefit: a banded stencil as a Toeplitz matmul does 8x to 32x redundant flops on a shared unit with streaming-mode entry cost; Accelerate calls add 0.1 to 1 us overhead and an extra pass per axis. Net: CPU parity is easily achievable and a fused fp64 path should beat Kokkos host by roughly 3x to 20x, limited by bandwidth at 256^3. A rewritten Kokkos kernel could reach similar headroom; the current C++ is slow because of one team per line and full-height CSR sweeps.

## What DSL designs should copy or avoid

Copy:

1. A precision-generic IR whose CPU backend emits fused loop nests with NEON.
2. Whole-step fusion: fold RK slot arithmetic, scale, source and BC into the single stencil sweep.
3. A persistent thread pool with spin barriers sized to the P-core count (QoS user-interactive), instead of OpenMP.
4. z-slab tiling so row planes stay in L1/L2 across the three axes.
5. Interior-span splitting so masks are only evaluated near fragment ends.
6. Table-driven ELL closure applied while the slab is cache-hot.

Avoid:

1. Relying on SME/SSVE or AMX (shared unit, streaming-mode switching, no documented intrinsics path, generation churn).
2. Per-axis Accelerate/vDSP calls (extra pass, call overhead).
3. Per-op eager dispatch.
4. Python-visible OpenMP dependence (Apple clang lacks it; Homebrew libomp is keg-only and version-fragile). Ship your own pool or use Kokkos Threads.
5. Treating rated DRAM bandwidth as CPU-reachable on Pro/Max.
6. Hard-coding P-core counts: M5 Max has 6 super + 12 performance cores.

## Deciding fact

fp64 stencil sweeps at 0.8 to 1.25 flop/B are bandwidth-bound (256^3) or L2-bound (64^3) on NEON alone, so SME and AMX offer nothing; the entire win over Kokkos comes from fusion, layout and launch elimination, which any LLVM-based compiled path (including a DSL emitting C) delivers. The base-M4 figure of about 224 GFLOPS on 4 P-cores versus about 100 GB/s rests on the unverified 56 GFLOPS per core; the conclusion holds even at half that rate.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| M4 SME fp64 FMOPA about 503 GFLOPS (P), 89 (E), a quarter of fp32 | Confirmed | Jena page. Figure is the per-unit peak (shared), not cluster-additive; the original wording "per P-cluster unit" was imprecise. |
| fp64 NEON FMLA about 56 GFLOPS per P-core | Uncertain | Jena page gives only fp32 NEON FMLA 107 (P) / 46 (E); 56 is inferred as half. Not shown in fetched sources. |
| Two SME units, P-unit shared by about 4 threads, non-streaming SVE faults with EXC_BAD_INSTRUCTION | Confirmed | Jena page. |
| 512-bit streaming vector length | Uncertain | Not in the fetched extract. |
| Accelerate routes fp32 GEMM to AMX on M1 to M3 | Confirmed (partial) | arXiv 2606.25426 (M1 AMX paper). |
| Accelerate routes fp32 GEMM to SME on M4; fp64 behaviour | Refuted as cited, dropped | Neither cited abstract says it; 2512.21473 only reports MpGEMM 1.23x faster than Accelerate on M4 Pro. Treated as unverified. |
| M5 Pro 307 GB/s; M5 Max 460 / 614 GB/s | Confirmed | Apple spec pages 126318, 126319. M5 Max CPU is 6 super + 12 performance cores. |
| M4 120, M4 Pro 273, M4 Max up to 546 GB/s; M1 Max CPU reach about 240 of 408 GB/s | Uncertain | Not re-fetched; no contradicting evidence. |
| Kokkos 5.2.2 (2026-09-10) latest; Homebrew formula depends on libomp | Confirmed | GitHub releases and Homebrew formula. |
| Apple clang has no -fopenmp; libomp keg-only workaround | Uncertain | Not re-verified against current Xcode docs; consistent with Homebrew libomp dependency. |
| Perf budgets (30 to 60 us at 64^3; 4 to 5 ms at 256^3; 3x to 20x over Kokkos) | Uncertain | Derived, unmeasured; depend on unverified 56 GFLOPS and CPU-reachable bandwidth. |
| 4P cluster about 220 GFLOPS; "10+ P cores" on M4 Max; threads = P-core count | Uncertain | Rests on unverified 56 GFLOPS; M5 core split differs. |

Counts (verifier's tally over its eight checked claims, with the Accelerate-to-SME item counted as uncertain): confirmed 4, refuted 0, uncertain 4. This file additionally marks one citation as refuted (Accelerate-to-SME attribution).

## Sources

- https://arxiv.org/abs/2409.18779 (Hello SME)
- https://arxiv.org/abs/2512.21473 (Demystifying SME)
- https://scalable.uni-jena.de/opt/sme/micro.html
- https://arxiv.org/abs/2606.25426 (M1 AMX, Accelerate fp32 GEMM dispatch)
- https://at-web1.www.anandtech.com/show/17024/apple-m1-max-performance-review/2 (not re-verified)
- https://mjtsai.com/blog/2024/10/30/apple-m4-pro-and-m4-max/ (not re-verified)
- https://support.apple.com/126318
- https://support.apple.com/126319
- https://github.com/kokkos/kokkos/releases
- https://formulae.brew.sh/formula/kokkos
- https://github.com/thomas-young-2013/mindware/blob/master/docs/en/installation/openmp_macos.md
- docs/python-dsl-survey/01-codebase-idioms.md
- docs/python-dsl-survey/00-apple-gpu-facts.md
