# PyFR (pyfr.backends: PyFR-Mako kernels, views, GiMMiK, per-backend graphs)

| | |
|---|---|
| Verdict | reference_only: copy the architecture, do not build on it |
| Fit for SHOCCS | Partial. Gather/scatter views match the ELL closure idiom; no structured-grid neighbour stencil primitive |
| CPU fp64 parity | Plausible on OpenMP (single C-call graph replay); estimated, not measured |
| Apple GPU | Metal backend is fp32 only; fp64 only on CPU. Per-kernel dispatch re-encoded from Python |
| Risk | Medium to high: backend layer is an internal API with no stability contract |

Overall score (verifier-adjusted): 4 out of 10.

## Status as of 2026-10-12

- Latest release: PyFR 3.1 (PyPI, 13 Mar 2026), bug fixes only. Preceded by 3.0 (5 Feb 2026: new file format, async I/O, native HIP graphs), 2.1 (1 Nov 2024: Tiny Tensor Compiler support) and 2.0.0 (3 Mar 2024: Metal backend added).
- The develop branch is active, with commits through 8 Oct 2026. Commit 73471dd (21 Sep 2026) lets the Metal backend pack scalar arguments into a struct.
- Licence BSD-3. Requires Python 3.11 or later. OpenMP needs GCC 12 or later (or another OpenMP 5.1 compiler) plus libxsmm; CUDA needs 11.4 or later; Metal needs pyobjc-framework-Metal 9.0 or later.
- Unverified: the open issue and PR counts (reported as 4 and 5, volatile), and the reference paper details (v2.0.3 paper, Comput. Phys. Commun. 311, 109567, 2025, with 1024 MI250X and 2048 GH200 scaling runs).

## Backends and fp64

| Backend | Kernel lowering | Graph replay | fp64 |
|---|---|---|---|
| OpenMP (CPU) | C generated from PyFR-Mako, compiled at run time; libxsmm or GiMMiK C for constant-matrix multiplies | Whole run list executed by one call, `backend.krunner(len(runlist), runlist)` (confirmed) | Yes |
| CUDA | NVRTC | Real CUDA graph, launched once per run (unverified in detail) | Yes |
| HIP | HIP | Native graphs since v3.0 (confirmed in release notes) | Yes |
| OpenCL | OpenCL C | Not assessed | Device permitting |
| Metal | MSL via PyObjC | Per-segment command buffers, `k.run(cbuf)` per kernel from Python, no indirect command buffers (confirmed) | No: `has_double = False`, raises `ValueError('Device does not support double precision')` (confirmed) |

On Apple silicon this means fp64 on the CPU only and fp32 on the GPU only, with no double-float emulation. The assessment's further Metal details (a new serial compute encoder per kernel, no concurrent dispatch, MPS used for matmuls) were not visible in the files the verifier fetched and are unverified. The Metal docs confirm GiMMiK is used with `gimmik-max-nnz` defaulting to 2048. The claim of testing on macOS M1 Max with Homebrew GCC is unverified.

## Mapping of the hard runtime shapes

| Idiom | PyFR form | Support |
|---|---|---|
| Packed field vector [D, Rx, Ry, Rz] | One backend matrix of shape (1, total) or several matrices in one named extent; views can address any entry. AoSoA layout and 2D matrix expectation make offsets awkward | workaround |
| Masked constant-coefficient axis stencil | No neighbour-offset access in a pointwise kernel. Either a view holding a per-point index table (12 to 36 bytes per point extra), or a constant banded matrix via GiMMiK or libxsmm. Matmul fits only the axis that is the row dimension; other axes need transposes or gathers. Mask becomes a multiply or select | workaround |
| Fixed-width ELL gather-dot-scatter (closure, B, Bf, Br) | PyFR's core idiom: view with vshape (K,) in, out view on unique rows, no atomics. Padding by hand | native |
| Face-plane assign | Pointwise kernel with an out view built on the host, or a masked kernel over the whole field | workaround |
| Masked reduction with readback | `reduction` kernel with sum or max; mask as multiplier inside the expression; host-synchronising readback; no argmax | workaround |
| Whole-step compilation / graph replay | `backend.graph()`, add with deps, commit. Replay mechanism differs per backend. No cross-node fusion, so an RK step stays N kernels | workaround |
| RK slot linear combinations and copies | Generated `axnpby` and copy kernels, as PyFR's own steppers use | native |
| Device-side MMS and boundary values with time | Config expressions in pointwise kernels with external scalars such as `t` and physical coordinates | native |
| Fused elementwise maps with masks | Pointwise kernels with Mako-time loops and macros; fusion is by hand-written template | native |
| Setup-only geometry and stencil assembly | Out of scope; stays in NumPy or SymPy | not applicable |

The view mapping scheme (per-entry triples flattened to one constant mapping array and dereferenced as `v[vix[x]]`) was stated with high confidence by the assessment but is listed by the verifier as not independently checked this session. The statement that external packages can supply templates (`DottedTemplateLookup`, public `get_backend`) is likewise unverified, and the backend layer is not documented as a stable third-party API.

## Code sketch

Illustrative only. The `view(K)` argument syntax and kernel arguments are unverified and invented for the sketch.

```python
# shoccs/kernels/elldot.mako -- ELL gather-dot-scatter in PyFR-Mako (sketch; view(K) arg syntax unverified)
# <%inherit file='base'/>
# <%pyfr:kernel name='elldot' ndim='1'
#     src='in view(${K}) fpdtype_t' c='in fpdtype_t[${K}]' out='inout view fpdtype_t'>
#   fpdtype_t acc = 0;
# % for k in range(K):
#   acc += c[${k}]*src[${k}];
# % endfor
#   out ${'+=' if accum else '='} acc;
# </%pyfr:kernel>
import numpy as np
from pyfr.inifile import Inifile
from pyfr.backends import get_backend

cfg = Inifile('[backend]\nprecision = double\n')         # 'single' forced on metal
be = get_backend('openmp', cfg)                          # or 'cuda', 'hip', 'metal'
F  = be.matrix((1, nD + nRx + nRy), tags={'align'})      # packed [D|Rx|Ry]
DU = be.matrix((1, nD + nRx + nRy))
C  = be.const_matrix(coef.T)                             # (K, nrows) closure coeffs
# idx: (nrows, K) column ids into packed vector; rows: unique output ids
src = be.view(np.full(idx.size, F), np.zeros(idx.size, int), idx.ravel(),
              vshape=(K,))
dst = be.view(np.full(len(rows), DU), np.zeros(len(rows), int), rows)
be.pointwise.register('shoccs.kernels.elldot')           # external dotted pkg OK
k_ell = be.kernel('elldot', tplargs={'K': K, 'accum': True},
                  dims=[len(rows)], src=src, c=C, out=dst)
k_ax  = be.kernel('mul', Ax_banded, F_axis_panel, out=DU_axis_panel)  # GiMMiK/libxsmm
be.commit()
g = be.graph(); g.add(k_ax); g.add(k_ell, deps=[k_ax]); g.commit()
be.run_graph(g); be.wait()
```

## Performance assessment

All figures are estimates; nothing was measured locally.

- Small 2D (51 by 51, launch-bound; bar 30 to 40 us per RHS, 0.3 to 1 ms per RK4 step). OpenMP graph replay is a single C call, so Python drops out. Each kernel still opens its own parallel region and nothing fuses across nodes. A 10 to 20 kernel RHS is estimated at 20 to 60 us, near parity with C++. CUDA and HIP replay native graphs at a few us per graph plus roughly 1 to 2 us per node.
- Metal is the weak case. Re-encoding every kernel from Python through PyObjC is estimated at 5 to 15 us per kernel, plus command-buffer commit and wait of about 100 us or more per synchronised round trip. A 15-kernel RHS is estimated at 100 to 300 us, missing the bar by 3 to 10 times. The Sep 2026 scalar-packing commit suggests dispatch cost is only now being trimmed.
- Large 3D (64^3 bar 1.7 ms per RHS; 256^3 bar about 30 to 110 ms). Generated kernels are bandwidth-efficient and the ELL part is negligible. The axis stencil dominates: as a banded matmul it runs near bandwidth for one axis, while the other two need transposed copies (two extra passes of 134 MB at 256^3) or gather views adding 0.75 to 2.25 times bytes per point. Estimate 1 to 2 times the ideal 16 to 31 ms floor, on OpenMP and CUDA in fp64. Metal at fp32 halves bytes but is not fp64.

## What the DSL designs should copy or avoid

Copy:

1. One backend-neutral pointwise kernel template language lowered by small per-backend generators (C/OpenMP, CUDA, HIP, MSL). The IR is one thread per point plus indexed views.
2. Views as a first-class argument kind: host-built index triples flattened into a mapping array, giving gather in and scatter out. This is the right abstraction for SHOCCS closure, B, Bf and Br ELL tables.
3. Graph objects with explicit dependencies, committed once and replayed by the cheapest native mechanism per backend, including a C run-list executor for OpenMP.
4. Constant-matrix deduplication and autotuned multi-provider matmul selection (GiMMiK, libxsmm, cuBLAS).
5. External scalars (time, coordinates) in pointwise kernels, moving MMS and boundary evaluation onto the device.
6. An explicit fp32-only flag per backend that fails loudly.

Avoid:

1. Re-encoding Metal kernels from Python on every run. Record Metal work once (indirect command buffer or a native-side encoder loop) and keep at most one wait per step.
2. Omitting a neighbour-offset stencil primitive. SHOCCS needs `x[r + j*stride_d]` with compile-time strides as a native argument form.
3. Non-fusion across kernels. SHOCCS's launch-bound regime wants the RHS fused into a few kernels.
4. Hard-coding AoSoA and 2D matrices, which fight the (nx, ny, nz) plus R packed layout.
5. An internal backend API with no stability contract for others to build on.

## Deciding fact

pyfr.backends offers only pointwise kernels with gather/scatter views and constant-matrix multiplies. It has no structured-grid neighbour-offset stencil and no supported third-party API, and its Metal backend rejects fp64 and re-encodes every kernel through PyObjC on every graph run. It is the best architecture to copy, not a substrate SHOCCS can sit on.

## Verification log

| Claim | Verdict | Evidence |
|---|---|---|
| v3.1 (13 Mar 2026) latest, after v3.0 (5 Feb 2026); Metal first shipped in v2.0.0 (3 Mar 2024) | confirmed | PyPI history lists 3.1, 3.0, 2.1 (1 Nov 2024), 2.0.0; release notes place Metal in v2.0.0 |
| Metal backend fp32 only, ValueError on double | confirmed | `metal/base.py`: `has_double = False` and the ValueError |
| Python 3.11+, GCC 12+/OpenMP 5.1, CUDA 11.4+, pyobjc-framework-Metal 9.0+, BSD licence | confirmed | Installation docs and PyPI |
| OpenMP graph runs whole run list in one krunner call | confirmed | `openmp/types.py` |
| Metal graph: command buffer per MPI-separated segment, `k.run(cbuf)` per kernel from Python, waitUntilCompleted only for MPI ordering, no indirect command buffers | confirmed (partly) | `metal/types.py`. Per-kernel serial encoder and no concurrent dispatch not seen in fetched files: unverified |
| Develop active through Oct 2026; commit 73471dd on 21 Sep 2026 packs scalar args | confirmed | Commit list, with commits dated 5, 6, 8 Oct 2026 |
| Metal `gimmik-max-nnz` default 2048 | confirmed | Metal backend docs. MPS use for matmuls not in docs: unverified |
| 4 open issues, 5 open PRs; CPC paper and GPU counts; GiMMiK language list; CUDA graph event-record nodes; view mapping formula; DottedTemplateLookup and get_backend as usable API | uncertain | Not fetched this session |
| Tested on macOS M1 Max with Homebrew GCC; `ndim` values and `view(K)` syntax in sketch | uncertain | No source checked; sketch flags the syntax itself |
| Performance figures (5 to 15 us per Metal kernel, 100 us per sync, 20 to 60 us OpenMP RHS, large 3D estimates) | unverified estimates | Assessment states they are not measured |

No claim was refuted. Counts: 7 confirmed (one partly), 0 refuted, 2 uncertain groups, matching the seven confirmed rows of the log above.

## Sources

- https://pypi.org/project/pyfr/
- https://github.com/PyFR/PyFR/releases
- https://github.com/PyFR/PyFR/commits/develop
- https://pyfr.readthedocs.io/en/latest/installation.html
- https://pyfr.readthedocs.io/en/latest/backends/backend-metal.html
- https://raw.githubusercontent.com/PyFR/PyFR/develop/pyfr/backends/metal/base.py
- https://raw.githubusercontent.com/PyFR/PyFR/develop/pyfr/backends/metal/types.py
- https://raw.githubusercontent.com/PyFR/PyFR/develop/pyfr/backends/openmp/types.py
- https://github.com/PyFR/GiMMiK (unverified this session)
- https://pyfr.readthedocs.io/en/latest/developer_guide.html (unverified this session)
