# Part I: What a Python re-expression of SHOCCS has to handle

This part of the report describes the C++ solver as it exists in this worktree (branch `worktree-python-dsl-survey`, base commit 4895b5a). It lists every computational idiom a replacement would have to express, says which of them run in the time loop, measures (where possible) how fast the C++ does them, and isolates exactly where the cut-cell method stops being a structured-grid problem. Parts II and III (the framework survey and the DSL design) build on the identifiers defined here: idioms I01 to I51 and primitives P0 to P6.

## 0. Sources, conventions and confidence

- The content comes from six per-subsystem code inventories (fields, matrices, operators, mesh, systems, stencils, io/mms/config, and the Python `scripts/stencil_gen` side), a consolidated catalog built from them, and a benchmark pass. Line references are `path:line` relative to the repository root and were taken from those inventories. Several excerpts were re-read from source for this document (`src/matrices/block.hpp:117-178`, `src/matrices/csr.cpp:37-51`, `src/temporal/rk4.cpp:16-47`, `src/temporal/slot_ops.hpp:11-40`, `src/mesh/mesh.cpp:29-87`, `src/mesh/object_geometry.cpp:76-95`, `src/operators/derivative.cpp:455-474`).
- **Counts.** Point counts D are exact. Counts of intersection records (R), line fragments and fluid points for the shipped configurations come from a Python replica of the C++ ray-casting and line-fragmentation algorithm, not from running the binary. Earlier analytic estimates in the inventories (for example Rx about 16 to 18 for heat.lua) agree with the replica. Sizes for a 256^3 grid are extrapolations.
- **Timings.** Fresh measurements were made for this study on 2026-10-12 inside the project devcontainer (Docker Desktop on the owner's Mac, 8 vCPU, Kokkos OpenMP, Release). They include the shipped cut-cell configurations, end-to-end RK4 steps and a per-step decomposition from the new benchmark `benchmarks/bench_rhs_cutcell.cpp` (Part IV, `05-measurements.md`). Section 6 is built on them. The March 2026 Google Benchmark baseline, which has no cut-cell work, is kept only as a historical cross-check. There was no native macOS build of the C++ until 2026-10-13; since then Sections 5.2, 6.4, 6.5 and 11 quote the native figures of `05-measurements.md` Section E with the container figures beside them (6.2 keeps the container tables).
- **Framework claims.** Part I mostly talks about the C++ code. Where it names a Python framework feature (for example a JAX or PyTorch API) as a plausible mapping, that mapping is a pointer to Part II, not a verified claim, and is marked "(unverified here)" unless a source is given. Apple GPU and fp64 facts are cited to `docs/python-dsl-survey/00-apple-gpu-facts.md`, which carries its own sources.
- Two statements in `CLAUDE.md` are stale relative to the code: `scalar_wave` is wired through `simulation_cycle` (`src/systems/system.cpp:103-132` maps "scalar wave", and `scalar_wave.lua` and the Brady-Livescu configs use it), and `src/fields/lazy_views.hpp` does not contain `zip_transform` or `bind_back`.

## 1. Data model

### 1.1 Four index spaces per scalar field

A scalar field is not one array. It is four 1-D buffers with independent lengths.

| Space | Length | Ordering | Meaning |
|---|---|---|---|
| D | nx*ny*nz | C order, z fastest: `ic(i,j,k) = i*ny*nz + j*nz + k` (`src/mesh/mesh.hpp:64-68`, `src/index_extents.hpp:15-19`) | Every grid point, including solid points. Fluid versus solid is not a property of the buffer. |
| Rx | number of ray/object intersections along x | (j, k, i): outer slow, inner fast, then increasing distance along the ray (`src/mesh/object_geometry.cpp:51-114`) | One value per intersection point on an x-directed grid line |
| Ry | intersections along y | (i, k, j) | as above |
| Rz | intersections along z | (i, j, k) | as above; empty in 2D |

Per-direction strides in D are x = ny*nz, y = nz, z = 1 (`src/indexing.hpp:51-57`). The direction-local (slow, fast) ordering is dir0 (y, z), dir1 (x, z), dir2 (x, y) (`src/indexing.hpp:18-37`). 2D is simply nz = 1, so y-lines are contiguous and x-lines have stride ny.

Each R buffer position i corresponds to record i of `m.R(dir)`, a `std::vector<mesh_object_info>` (`src/mesh/mesh_types.hpp:31-38`):

```cpp
struct mesh_object_info {
  real psi; real3 position; real3 normal; // outward
  bool ray_outside; int3 solid_coord; int shape_id; };
```

This is an array of structs of about 80 bytes per record. `psi` is the distance from the adjacent fluid grid point to the intersection in units of h, snapped at 1e-12 and clamped to [1e-12, 1-1e-12] (`src/mesh/object_geometry.cpp:87-95`). `solid_coord` is the first solid grid point beyond the intersection. It is a D coordinate and is not unique across records.

The R ordering is load-bearing. Field R buffers are positionally aligned with the records (`src/fields/field_registry.hpp:89-91`, `src/mesh/mesh.hpp:96`). `object_boundary::object_coordinate` stores the record position and is used as the column index of the B coupling matrix (`src/mesh/mesh.cpp:64-73`, `src/operators/derivative.cpp:106,200-206`). The line-fragmentation merge-join consumes the records in this order (`src/mesh/mesh.cpp:58`). No explicit permutation array exists anywhere.

**Lines.** `m.lines(dir)` is a `vector<line{integer stride; boundary start; boundary end}>`, where `boundary = {int3 mesh_coordinate; optional<object_boundary{object_coordinate, objectID, psi}>}` (`src/mesh/mesh_types.hpp:8-23`). There is one line per fluid fragment of a grid pencil. An object endpoint carries the solid coordinate next to the intersection. The count per direction is the number of pencils, minus pencils lying in a Dirichlet wall of another axis (`src/mesh/mesh.cpp:154-168`), plus the extra fragments, which equal |R(dir)|/2 for closed bodies strictly inside the domain (`src/mesh/mesh.t.cpp:86`).

### 1.2 Field storage

- `sim_registry = field_registry<8 slots, 8 scalars, 4 vectors>` (`src/fields/field_registry.hpp:246`) is a `std::array<Kokkos::View<real*>, 640>` addressed by `slot*80 + buffer_id`. Scalar i occupies ids 4i..4i+3 = {D, Rx, Ry, Rz}; vector i occupies 32+12i onward (`src/fields/handle.hpp:43-46,75-81,108-114`). `real = double` (`src/shoccs_config.hpp:8`). The memory space is `DefaultHostExecutionSpace::memory_space` (`src/kokkos_types.hpp:10-11`).
- Compute code never sees Views. It sees `scalar_span`/`scalar_view`, which are four `std::span` (raw pointer plus length) (`src/fields/scalar.hpp:13-67`).
- `simulation_cycle` allocates four slots: 0 = u0, 1 = u1 (stage and output state), 2 = RK accumulator, 3 = stage RHS (`src/simulation/simulation_cycle.cpp:44-56`). The RHS graph is bound to raw pointers of slots 1 and 3 (`src/simulation/simulation_cycle.cpp:72-75`), so slots are copied with `deep_copy_slot` rather than swapped (`src/simulation/simulation_cycle.cpp:121-125`).
- Systems also own host `std::vector<real>` work buffers in the same four-buffer shape: heat `neumann_*`, `src_*`, `error_*` (`src/systems/heat.hpp:29-31`); scalar_wave 12 coefficient buffers `gG_{x,y,z}{D,Rx,Ry,Rz}`, 12 gradient scratch buffers `du_*` and `error_*` (`src/systems/scalar_wave.hpp:29-37`). Graph nodes capture raw pointers into these vectors.

### 1.3 Index sets

All index sets are defined in `src/fields/selection_desc.hpp`.

| Set | Representation | Equivalent array form |
|---|---|---|
| x face i | `contiguous{offset=i*ny*nz, count=ny*nz}` | `D3[i,:,:]` |
| y face j | `strided{j*nz, inner=nz, outer=nx, stride=ny*nz}` | `D3[:,j,:]` |
| z face k | `strided{k, 1, nx*ny, nz}` | `D3[:,:,k]` |
| fluid points | `gather_selection{Kokkos::View<const int*>}` built from merged [first,last) slices of the highest active direction's fragments (`src/mesh/mesh.cpp:89-111,135-137`); ascending int32 | a D-sized boolean mask |
| Dirichlet / non-Dirichlet object points | int32 positions in R(dir) where `obj_bcs[shape_id]` is (not) Dirichlet (`src/mesh/mesh.hpp:97-114`, `selection_desc.hpp:140-158`); rebuilt on every call | an R-sized boolean mask |
| heat Dirichlet faces | all face indices flattened into one gather View (`src/systems/heat.cpp:192-205`) | union of face slices |

`element(i) = offset + (i/inner)*stride + i%inner` (`selection_desc.hpp:22-45,77-103`). All indices are int32 with overflow checked by assert only, which caps D at 2^31 points.

### 1.4 The per-direction operator: nine matrices

One `derivative` along axis `dir` (`src/operators/derivative.cpp:455-474`) holds nine matrices:

| Matrix | Maps | Storage | Content |
|---|---|---|---|
| O | D -> D | `matrix::block` (ragged list of line blocks, flattened) | interior stencil and all closures, one block per line fragment |
| B | R(dir) -> D | host CSR, rows = \|D\| | the intersection-value column of each object closure |
| N | D -> D | host CSR, rows = \|D\| | Neumann wall data |
| Bf_r, r = 0,1,2 | D -> R(r) | host CSR, rows = \|R(r)\| | derivative at the intersection points of ray direction r, fluid part |
| Br_r, r = 0,1,2 | R(r) -> R(r) | host CSR | same, intersection-point part |

**O layout.** `block` (`src/matrices/block.hpp:20-217`) flattens a `vector<inner_block>` into `meta_d[n_lines]` of `inner_block_meta`, 13 int32 per line: `row_offset, col_offset, stride, left_rows, left_cols, left_coeff_offset, interior_rows, interior_coeff_offset, stencil_width, right_rows, right_cols, right_coeff_offset, right_col_offset` (`src/matrices/inner_block_meta.hpp:9-23`), and `coeffs_d`, which stores per line `[left R_L x T_L row-major | interior w = 2P+1 | right R_R x T_R]` (`block.hpp:30-94`). Row r of a line writes `b[row_offset + r*stride]`; left rows read `x[col_offset + j*stride]`; interior rows read `x[out + (j - w/2)*stride]`; right rows read `x[right_col_offset + j*stride]` with `right_col_offset = col_offset + stride*(columns - right.columns)` (`src/matrices/inner_block.cpp:17-40`). Interior coefficients are identical for every line of a direction (`derivative.cpp:436,465-467`) and wall closures are identical for all wall-terminated ends, but `coeffs_d` stores a full copy per line.

**Per-line shape rules** (`derivative.cpp:259-303` `submatrix_size`, `328-438`):

- Wall end, Dirichlet: (R-1) x T. The wall row is dropped and the wall column kept, so `u.D` at the wall carries the boundary value (flag `ldd`/`rdd`).
- Wall end, Floating: R x T.
- Wall end, Neumann: R x T plus X entries in N.
- Object end: `nbs(h, bc, psi, right)` gives R x T. The boundary row is dropped if the object BC is not Dirichlet (that row moves to Bf/Br). Column 0, the intersection point, is stripped into `B[row, object_coordinate]`; the remaining T-1 columns are dense, with offsets shifted past the solid point.
- `n_interior = rows - left_rows - right_rows`, unguarded for values <= 0.

**Cut-point rows.** For r == dir, Bf_r/Br_r hold the boundary row of the psi closure: T-1 entries in Bf, the diagonal in Br (`derivative.cpp:188-208`). For r != dir, they hold a t-point derivative line in `dir` whose off-grid points are each interpolated along r (`derivative.cpp:209-253,22-70,92-135`; `src/stencils/stencil.hpp:213-268`; `src/mesh/mesh.cpp:170-207`), giving up to t_deriv x t_interp entries per row (E2_2: 4x3 = 12; polyE2_1: 4x4 = 16). Duplicate (row, col) entries are not merged.

CSR storage is host `std::vector<real> w`, `std::vector<int64> v` (columns), `std::vector<int64> u` (row pointers) (`src/matrices/csr.hpp:19-21`). The kernel captures raw host pointers (`csr.cpp:37-51`, `csr.hpp:47-61`).

`laplacian` is three second-derivative `derivative`s accumulated (`plus_eq`) into one four-buffer output after a four-buffer zero fill (`src/operators/laplacian.hpp:45-118`). `gradient` is three first-derivative `derivative`s into three independent outputs (`eq`) (`src/operators/gradient.hpp:34-122`). Each operator builds its own three derivatives.

### 1.5 Stencil dimensions

| Scheme | P (half-width) | R (closure rows) | T (closure cols) | X (Neumann extras) | interp {p, t} | Used by |
|---|---|---|---|---|---|---|
| E2_1 | 1 | 4 | 5 | 0 | 2, 3 | cut-cell first derivative |
| E2_2 | 1 | 2 | 4 | 2 | 2, 3 | heat.lua (Laplacian) |
| E4_1 | 2 | 5 | 7 | 0 | none | cut-cell first derivative |
| E4_2 | 2 | 3 | 5 | 3 | 5, 5 | |
| E4u_1 | 2 | 3 | 5 | 0 | none | Brady-Livescu configs |
| E6u_1 | 3 | 5 | 8 | 0 | none | |
| E8u_1 | 4 | 7 | 11 | 0 | none | |
| polyE2_1 | 1 | 3 | 4 | 0 | 2, 4 | scalar_wave.lua, eigenvalues.lua |

Dirichlet returns R-1 rows. First-derivative schemes return `{}` for Neumann. E4_1, E4u_1, E6u_1 and E8u_1 return an empty `interp_info`, so they cannot build the r != dir rows needed for floating objects in 2D and 3D (open question in Section 11).

### 1.6 Sizes

| Config | Grid, scheme, BCs | D | Rx, Ry, Rz | lines/dir | fluid | Notes |
|---|---|---|---|---|---|---|
| heat.lua | 51x51, E2_2, 4 Dirichlet walls, 1 floating circle r = sqrt(3)/10, h = 0.04 | 2601 | 16, 16, 0 | 59 (51+8) | 2545 (56 solid) | faces 4x51; B nnz about 16; coeffs_d about 0.6k doubles/dir; fragment length 19..51 |
| scalar_wave.lua | 71x71, polyE2_1, xmin/ymin Dirichlet, 2 quarter circles (1 Dirichlet, 1 floating) | 5041 | 22, 22, 0 | 71 | 4833 (208 solid) | Dirichlet object list 11/dir, non-Dirichlet 11/dir; fragment length 61..71 |
| brady_livescu_4_3_n61 / long | 61x61 / 31x31, E4u_1, no shapes | 3721 / 961 | 0 | N | N^2 (one slice) | B, Bf, Br empty |
| scalar_wave_1d.lua | 71, one yz_rect | 71 | 1, 0, 0 | 1 | 70 | |
| eigenvalues.lua | 21 (1D), two yz_rects | 21 | 2, 0, 0 | 1 | 19 | dense 21-23 square |
| bench_rhs | 16^3 / 32^3 / 64^3 heat, no shapes | 4096 / 32768 / 262144 | 0 | N^2 | all | |
| 64^3 sphere (replica) | [0,1]^3, centred sphere r = 0.25 | 262144 | 1576 per dir | 4884 | 245,776 | shortest fragment 17 |
| 256^3 sphere (extrapolated) | same geometry | 16.78M | about 25.5k per dir (0.46% of D in total) | about 78.3k (65,536 + 12.8k) | about 15.7M (6.5% solid) | see below |

Memory consequences:

- heat.lua: 4 slots x 2633 doubles is about 85 KB. The whole state is L2-resident, so every kernel is limited by launch and synchronisation overhead.
- 256^3: 134 MB per D buffer and about 540 MB for four slots. The fluid gather list is 63 MB of int32 versus 16.8 MB for a boolean mask. The R records are about 2 MB per direction. `meta_d` is 4 MB per direction and `coeffs_d` 22 MB per direction for E4u_1 (about 85 MB per derivative for E8u_1) without deduplication. The B and N row-pointer arrays are 134 MB each per derivative. Closure rows are about 400k per direction (2.4% of D). Bf nnz reaches about 1e5 to 1e6 for r != dir.

## 2. Idiom catalog

Pattern names: `axis_stencil_conv` (constant stencil along one axis), `dense_boundary_block` (small dense closure), `csr_matvec`, `gather`, `scatter_add`, `masked_fill`, `elementwise`, `axpby`, `reduction`, `point_eval_host` (host callback per point), `graph_launch`, `control_flow`, `host_setup_assembly`, `io`, `other`. Phase is when the idiom runs. "Hot" means it runs inside the time loop. Difficulty is the expected difficulty of expressing the idiom efficiently in an array framework: trivial, easy, workaround (needs restructuring), hard, blocker.

| Id | Name | Pattern | Phase | Hot | Difficulty | Example |
|---|---|---|---|---|---|---|
| I01 | Registry slot/buffer table allocation | host_setup_assembly | setup | no | trivial | src/fields/field_registry.hpp:72-131 |
| I02 | Ray casting to R records; solid-point lists | host_setup_assembly | setup | no | easy | src/mesh/object_geometry.cpp:32-116 |
| I03 | Line fragmentation (init_line) and dirichlet_line | host_setup_assembly | setup | no | easy | src/mesh/mesh.cpp:29-87 |
| I04 | Index-set construction (fluid, object BC, faces) | host_setup_assembly | setup | no | trivial | src/mesh/mesh.cpp:89-111 |
| I05 | psi-independent stencil constants and dispatch | host_setup_assembly | setup | no | trivial | src/stencils/stencil.hpp:75-160 |
| I06 | psi-dependent closure evaluation `nbs` | elementwise | setup | no | easy | src/stencils/E2_1.cpp:123-1307 |
| I07 | Cross-direction interpolation (interp, interp_line) | control_flow | setup | no | hard | src/stencils/stencil.hpp:196-254 |
| I08 | domain_discretization (line closures, B/N COO) | host_setup_assembly | setup | no | easy | src/operators/derivative.cpp:305-449 |
| I09 | cut_discretization (Bf/Br rows) | host_setup_assembly | setup | no | workaround | src/operators/derivative.cpp:159-257 |
| I10 | COO to CSR build | host_setup_assembly | setup | no | trivial | src/matrices/csr.cpp:10-35 |
| I11 | Block flattening to meta/coeffs | host_setup_assembly | setup | no | trivial | src/matrices/block.hpp:30-94 |
| I12 | Config and system setup (Lua) | host_setup_assembly | setup | no | trivial | src/app/shoccs.cpp:17-55 |
| I13 | Flat C-order D with per-direction strides | other | per_rhs | yes | trivial | src/index_extents.hpp:15-19 |
| I14 | R ordering contract | other | setup | no | easy | src/mesh/object_geometry.cpp:51-114 |
| I15 | Span bridge and system-owned work buffers | other | per_rhs | yes | easy | src/fields/scalar.hpp:13-67 |
| I16 | Ragged line block matvec O | other | per_rhs | yes | workaround | src/matrices/block.hpp:117-178 |
| I17 | Interior circulant axis stencil | axis_stencil_conv | per_rhs | yes | trivial | src/matrices/block.hpp:150-160 |
| I18 | Wall closures | dense_boundary_block | per_rhs | yes | easy | src/operators/derivative.cpp:367-381 |
| I19 | Object closures (psi-dependent) | dense_boundary_block | per_rhs | yes | workaround | src/operators/derivative.cpp:336-366 |
| I20 | Generic CSR accumulate kernel | csr_matvec | per_rhs | yes | easy | src/matrices/csr.cpp:37-51 |
| I21 | B: R(dir) -> D | csr_matvec | per_rhs | yes | easy | src/operators/derivative.cpp:441 |
| I22 | N: Neumann wall data -> D | csr_matvec | per_rhs | yes | trivial | src/operators/derivative.cpp:442 |
| I23 | Bf_r: D -> R(r) | gather | per_rhs | yes | easy | src/operators/derivative.cpp:188-253 |
| I24 | Br_r: R(r) -> R(r) | gather | per_rhs | yes | easy | src/operators/derivative.cpp:484-486 |
| I25 | Zero-fill before accumulation | masked_fill | per_rhs | yes | trivial | src/operators/laplacian.hpp:62-75 |
| I26 | Whole-scalar scale | elementwise | per_rhs | yes | trivial | src/systems/heat.cpp:221-232 |
| I27 | Accumulation policy and write ordering | axpby | per_rhs | yes | easy | src/operators/laplacian.hpp:77-80 |
| I28 | Gradient fan-out | elementwise | per_rhs | yes | trivial | src/operators/gradient.cpp:34-46 |
| I29 | Variable-coefficient dot product | elementwise | per_rhs | yes | trivial | src/systems/scalar_wave.cpp:221-239 |
| I30 | Masked add/assign at gather sets | scatter_add | per_rhs | yes | easy | src/systems/heat.cpp:236-260 |
| I31 | Dirichlet face/object fill | masked_fill | per_rhs | yes | trivial | src/fields/selection_desc.hpp:169-186 |
| I32 | Kokkos Graph with baked pointers, fences | graph_launch | per_rhs | yes | workaround | src/systems/heat.cpp:163-294 |
| I33 | Eager apply path | control_flow | per_rhs | yes | trivial | src/operators/derivative.cpp:475-519 |
| I34 | MMS evaluation at all D and R points | point_eval_host | per_rhs | yes | workaround | src/systems/detail/scalar_system_utils.hpp:16-73 |
| I35 | Lua per-point MMS callback | point_eval_host | per_rhs | yes | blocker | src/mms/lua_mms.hpp:13 |
| I36 | Per-stage host source fill before submit | control_flow | per_rhs | yes | hard | src/systems/system.cpp:57-69 |
| I37 | RK4 / Euler slot linear combinations | axpby | per_step | yes | trivial | src/temporal/slot_ops.hpp:11-59 |
| I38 | deep_copy_slot for pointer stability | elementwise | per_step | yes | trivial | src/simulation/simulation_cycle.cpp:121-125 |
| I39 | update_boundary per stage | host_setup_assembly | per_step | yes | easy | src/systems/heat.cpp:296-354 |
| I40 | Per-call rebuild of object-BC lists | host_setup_assembly | per_step | yes | trivial | src/mesh/mesh.hpp:97-115 |
| I41 | Stats reductions with host readback | reduction | per_step | yes | easy | src/systems/detail/scalar_system_utils.hpp:77-182 |
| I42 | Error field, binary dump, XDMF | io | per_output | yes | easy | src/systems/detail/scalar_system_utils.hpp:224-302 |
| I43 | Outer time loop and dispatch | control_flow | per_step | no | easy | src/simulation/simulation_cycle.cpp:31-143 |
| I44 | Eigenvalue diagnostic (dense + geev) | host_setup_assembly | diagnostic | no | trivial | src/operators/eigenvalue_visitor.cpp:10-52 |
| I45 | Reference component matvecs | other | diagnostic | no | trivial | src/matrices/dense.cpp:9-37 |
| I46 | Expression templates with alias check | elementwise | per_rhs | no | trivial | src/fields/expr.hpp:20-150 |
| I47 | Euler scaffold (inviscid_vortex) | other | per_rhs | yes (future) | workaround | src/systems/inviscid_vortex.cpp:13-116 |
| I48 | Python dense/sparse operator assembly | dense_boundary_block | diagnostic | no | easy | scripts/stencil_gen/stencil_gen/phs.py:622-697 |
| I49 | Python symbolic coefficient layer and codegen | host_setup_assembly | setup | no | trivial | scripts/stencil_gen/stencil_gen/codegen.py:136-233 |
| I50 | Python stability-analysis cascade | reduction | diagnostic | no | hard | scripts/stencil_gen/stencil_gen/non_normality.py:72-148 |
| I51 | cpp_bridge subprocess and BO driver | io | diagnostic | no | trivial | scripts/stencil_gen/stencil_gen/cpp_bridge.py:52-205 |

Summary by phase: 14 setup idioms (I01 to I12, I14, I49); 29 hot idioms (I13 and I15 to I42); 7 cold or diagnostic idioms (I43 to I46, I48, I50, I51); and the not-yet-implemented Euler system (I47). The one blocker (I35) and the hard items (I36 at runtime, I07 and I50 off the hot path) concern host callbacks, setup geometry or host-only analysis, not the arithmetic of the RHS.

## 3. Idiom details

### Setup: geometry, stencils and operator assembly

#### I01 Registry slot/buffer table allocation

A fixed 2-D table of 1-D Views indexed by (slot, buffer id). Allocation is sequential per slot (asserted). `field_ref` is a 12-byte `{slot, n_scalars, n_vectors}` token and lengths come from View extents at launch time (`src/fields/field_registry.hpp:46-55,163-168`).

```cpp
buffers_[base + sh.D().id]  = Kokkos::View<real*>(label("D"),  d_sz);
buffers_[base + sh.Rx().id] = Kokkos::View<real*>(label("Rx"), rx_sz);
buffers_[base + sh.Ry().id] = Kokkos::View<real*>(label("Ry"), ry_sz);
buffers_[base + sh.Rz().id] = Kokkos::View<real*>(label("Rz"), rz_sz);
...
using sim_registry = field_registry<8, 8, 4>;
```

Shipped runs have 16 non-empty Views (D 21 to 5041 doubles, R 0 to 22). Only two semantics matter for a port: stable buffer identity where graphs capture pointers, and four unrelated lengths per field. Packing D|Rx|Ry|Rz into one array per field with static offsets would turn four launches per elementwise op into one.

#### I02 Ray casting to R records; solid-point lists

For each active direction and each (slow, fast) pencil, the code repeatedly takes the closest hit over all shapes (type-erased sphere or yz_rect, `src/mesh/shapes.hpp:35-120`) in (t_min, t_max), then advances `t_min = nextafter(t)`. Each hit becomes a `mesh_object_info` pushed in (slow, fast, t) order (`src/mesh/object_geometry.cpp:32-116`). The snapping logic, re-read from source (`object_geometry.cpp:76-95`):

```cpp
real t_over_h = hit->t / iline.h;
int i_cell = static_cast<int>(std::round(t_over_h));
constexpr real snap_tol = 1e-12;
if (std::abs(t_over_h - i_cell) > snap_tol)
    i_cell = static_cast<int>(t_over_h);
coord[I] = i_cell + hit->ray_outside;
int off = 1 - 2 * hit->ray_outside;
real fluid_pos = min + h * (coord[I] + off);
real psi = off * (fluid_pos - hit->position[I]) / h;
psi = std::clamp(psi, snap_tol, 1.0 - snap_tol);
```

It also builds per-shape duplicate R vectors (`object_geometry.cpp:105-110`) and solid-point lists Sx/Sy/Sz (`object_geometry.cpp:124-211`), neither of which has a production consumer. Sizes: heat.lua casts 102 rays and finds 32 hits; 256^3 casts 3 x 65,536 rays per shape and finds about 25.5k hits per direction. The computation vectorises (quadratic roots for all lines and shapes, filter, lexsort), but the round-versus-truncate snap, the clamp and the `nextafter` tie-breaking must be reproduced bit-for-bit or `solid_coord` and psi shift. Fully solid pencils are unsupported (asserts at `object_geometry.cpp:186`, `mesh.cpp:21-22`).

#### I03 Line fragmentation and dirichlet_line

For each pencil, `init_line` consumes the R records on the same plane in order. A `ray_outside` hit closes the open fragment [left, object]; a non-`ray_outside` hit opens a new left end; at the end of the pencil the open fragment is closed at the wall. Re-read from `src/mesh/mesh.cpp:58-85` (abbreviated):

```cpp
while (first != last && same_plane<S, F>(left, first->solid_coord)) {
    if (first->ray_outside) {
        v.emplace_back(index::stride<I>(extents), *left_boundary,
            boundary{.mesh_coordinate = first->solid_coord,
                     .object = object_boundary{first - std::ranges::begin(r),
                                               first->shape_id, first->psi}});
        left_boundary.reset();
    } else {
        left_boundary = boundary{.mesh_coordinate = first->solid_coord, .object = ...};
    }
    ++first;
}
if (left_boundary)
    v.emplace_back(index::stride<I>(extents), *left_boundary,
                   boundary{.mesh_coordinate = right, .object = std::nullopt});
```

`dirichlet_line` (`mesh.cpp:154-168`) flags fragments whose start lies on a Dirichlet wall of another axis; those lines get no operator rows (`derivative.cpp:328-329`). Counts: heat 59 per direction, scalar_wave 71, 64^3 sphere 4884, 256^3 about 78.3k; fragment lengths 17 to n. The resulting ragged (offset, stride, length) segment list is the first carrier of unstructuredness. Array frameworks have no ragged strided-segment primitive (unverified here, see Part II), so a runtime formulation must flatten it into a uniform axis stencil plus a closure table plus masks.

#### I04 Index-set construction

The fluid set merges contiguous [first, last) slices of the highest-direction fragments, excluding object endpoints, and flattens them to a sorted int32 View (`src/mesh/mesh.cpp:89-111`, `selection_desc.hpp:110-129`). Object lists come from a two-pass host predicate scan over R(dir) on `obj_bcs[shape_id]`, then a View, mirror and `deep_copy` (`selection_desc.hpp:140-158`):

```cpp
template <typename Pred>
gather_selection make_gather_from_predicate(std::span<const mesh_object_info> infos, Pred pred) {
  int total = 0; for (...) if (pred(infos[i])) ++total;
  Kokkos::View<int*, memory_space> indices("gather_pred_indices", total);
  auto h = Kokkos::create_mirror_view(indices);
  int pos = 0; for (...) if (pred(infos[i])) h(pos++) = i;
  Kokkos::deep_copy(indices, h); return gather_selection{indices}; }
```

Face descriptors are pure arithmetic from extents; `heat::build_rhs_graph` concatenates all Dirichlet faces into one gather View (`src/systems/heat.cpp:185-211`). All sets are time-invariant functions of geometry and BC type. With 2 to 6.5% solid points, a dense boolean mask is smaller than the gather list at 256^3 (16.8 MB versus 63 MB). See I40 for the fact that the C++ rebuilds the object lists on every use.

#### I05 psi-independent stencil constants and dispatch

`query`/`query_max`/`query_interp` return `{p, r, t, nextra}` (`src/stencils/stencil.hpp:21-31,49-69`). `interior(h, c)` writes 2P+1 rational constants divided by h or h^2 (for example E2_1 `{-1/(2h), 0, 1/(2h)}`, `src/stencils/E2_1.cpp:97-104`). E4u/E6u/E8u closures are alpha-dependent R x T constants that ignore psi. The RBF E4u variants solve one 10x10 augmented `[Phi P^T; P 0]` system at construction, with a Taylor branch for small arguments (`src/stencils/tension_E4u_1.cpp:63-170`). `from_lua` reads alpha arrays through `copy_zero_padded`, which silently truncates (`src/stencils/stencil.cpp:10-107`). Every method is a virtual call on a heap `any_stencil` returning spans into caller scratch (`stencil.hpp:75-160`). In a port these become constants of the axis-stencil primitive (P2). Alpha optimisation already lives in Python (`scripts/stencil_gen` Bayesian optimisation).

#### I06 psi-dependent closure evaluation `nbs(h, bc, psi, right)`

`nbs` evaluates an R x T block as closed-form rational functions of the scalar psi and the alpha parameters. These are SymPy CSE outputs: E2_1 has about 1170 lines per BC with `pow(x,-1)` (`src/stencils/E2_1.cpp:123-1307`); E4_1 has denominators with poles at psi = 0 and 1 and a hand-added clamp to [1e-4, 1-1e-4]; polyE2_1 uses `fa[6]`, `da[3]`. Post-processing divides by h (or h^2); for right ends odd derivatives are negated and the whole flat block is reversed (a 180-degree rotation); Neumann fills X extra coefficients, also flipped and negated. The emitted pattern (from the codegen inventory):

```cpp
for (auto&& v : c) v /= h;
if (right) { for (auto&& v : c) v *= -1; std::ranges::reverse(c); }
```

Callers re-slice rows: `derivative.cpp:199-203` takes the last row and reverses it again for `ray_outside`. Blocks have 8 (E2_2), 12 (polyE2_1), 20 (E2_1) or 35 (E4_1) entries, one block per R point per axis (32 for heat.lua, about 76k at 256^3) plus psi = 1 wall blocks. Because every entry is a scalar function of psi, a batched evaluation over a psi array is one elementwise pass. The E4_1 guards exist only in the hand-edited `src/stencils/E4_1.cpp`, not in `scripts/stencil_gen/output/`.

#### I07 Cross-direction interpolation

For an R(r) point and derivative direction dir != r, each of the t points on the derivative line lies off-grid by y = +/-psi or +/-(psi-1) and is interpolated along r. `stencil::interp` compares the closest cell with the left and right bounds of the fluid line (from `mesh.interp_line`) and chooses an interior interpolant (2 weights, branch on sign(y)) or a left/right wall interpolant (3 or 4 weights, rational in y and psi), adjusting y when the closest point is solid (`src/stencils/stencil.hpp:196-254`). The assembly call site and one generated interpolant body:

```cpp
// derivative.cpp:239-246
auto&& [interp_v, left, right] = st.interp(r, cp, y, left_bounds, right_bounds, interp_c);
builder.add_interp_row(shape_row, v, interp_v, left, right, r_stride, m, msg);
// polyE2_1.cpp interp_wall, left, case 0
c[0] = t6*(ia[1] + psi + t7 + t8);
c[1] = (1.0 / 2)*psi + t10 + t11;
```

`mesh.interp_line` (`src/mesh/mesh.cpp:170-207`) finds the enclosing fragment by a linear scan of R(r) that classifies sides by the sign of the normal, not by `ray_outside`. The output footprint has data-dependent length and may end on another R point (which turns a D column into an R column). Calls number |R(r)| x t per (r, dir) pair, over 6 pairs with r != dir. At 256^3 that is an estimated 1e9 comparisons per pair. Derivation (estimate, order of magnitude only): about 25.5k R(r) records x t = 5 line points gives about 1.3e5 calls per pair, and each linear scan covers up to about 12k records. This is the main carrier of unstructuredness, but it runs only at setup. The code assumes convex bodies and is marked "no good for planar objects" (`derivative.cpp:46-47,89-90`). A port would keep it as host code (NumPy, Numba or C++) with a per-pencil fragment lookup (searchsorted) and emit padded (index, coefficient) tables.

#### I08 domain_discretization

A host loop over `m.lines(dir)` skips Dirichlet lines, then for each end branches on wall versus object and on Dirichlet/Floating/Neumann, calls `st.query`/`st.nbs`, slices rows and columns, computes `inner_block` component offsets and pushes an `inner_block` (`src/operators/derivative.cpp:305-449`). The offset arithmetic (`src/matrices/inner_block.cpp:17-40`):

```cpp
left_boundary.row_offset(row_offset).col_offset(col_offset).stride(stride);
interior.row_offset(row_offset + stride * left_boundary.rows()).stride(stride);
right_boundary.row_offset(row_offset + stride * (left_boundary.rows() + interior.rows()))
  .col_offset(col_offset + stride * (columns - right_boundary.columns())).stride(stride);
```

heat.lua runs about 59 lines per direction; 256^3 about 78k lines per direction, times 3 directions, times the number of operators (gradient and laplacian each build their own three derivatives). The loop vectorises in NumPy by grouping lines by (left kind, right kind, BC). It is the natural precompute stage of a DSL. `n_interior <= 0` for very short fragments is not guarded.

#### I09 cut_discretization

For each R(r) point whose shape BC is not Dirichlet: if r == dir, take the boundary row of `nbs` (reversed if `ray_outside`), put the diagonal into Br and T-1 entries into Bf at `solid_ic +/- i*stride` (`derivative.cpp:188-208`); if r != dir, choose the derivative line (a one-sided `nbs` row with psi = 1, or the interior stencil) and compose derivative coefficients with interpolation coefficients through I07, with columns in D (Bf) or R(r) (Br) (`derivative.cpp:209-253`). The function exits early if there are no shapes or all object BCs are Dirichlet. When logging is on it formats a log string per row. Rows number |R(r)| (16 to 22 shipped, about 25.5k at 256^3), with up to 12 to 16 entries per row for r != dir. Embarrassingly parallel over intersections once I06 and I07 are batched.

#### I10 COO to CSR build

`csr::builder` appends (row, col, v) triples, sorts them lexicographically and counts per row (`src/matrices/csr.cpp:10-35`). Duplicate (row, col) pairs are kept as separate nonzeros. B and N are built with `nrows = m.size() = |D|` (`derivative.cpp:441-442`).

```cpp
std::ranges::sort(p);
for (integer i = 0; i < nrows; i++) { u[i+1] = u[i];
  while (first != last && first->row == i) { ++u[i+1]; ++first; } }
```

nnz is 20 to 300 shipped and up to about 1e6 at 256^3, while the B/N row pointer has 16.8M+1 int64 entries at 256^3. The full-height row pointer is the most wasteful artifact in the runtime (I21, I22). A port would keep compact COO or ELL tables.

#### I11 Block flattening to device arrays

Two host passes over `vector<inner_block>`: an exclusive scan of per-line coefficient sizes into `inner_block_meta`, then `std::copy` of each component into one host vector and `deep_copy` of both to device Views (`src/matrices/block.hpp:30-94`).

```cpp
m.left_coeff_offset = total_coeffs; total_coeffs += L.size();
m.interior_coeff_offset = total_coeffs; m.stencil_width = C.size(); total_coeffs += C.size();
m.right_coeff_offset = total_coeffs; m.right_col_offset = R.col_offset(); total_coeffs += R.size();
```

Each `dense` and `circulant` instance had already allocated its own device View (`src/matrices/dense.hpp:22-52`, `src/matrices/circulant.cpp:11-32`), three per line: about 350 tiny allocations for heat.lua and an estimated 700k allocations plus `deep_copy`s at 256^3. Derivation (estimate): 3 Views per line x about 78.3k lines per direction x 3 directions is about 705k for one three-derivative operator such as the laplacian. A debug-only O(n_lines^2) interval-disjointness assert (`block.hpp:230-246`) is wrong for interleaved strided lines: by inspection it would fire for any 2D or 3D mesh in a Debug build. CI uses RelWithDebInfo, so it never runs. A DSL should emit (line start, length, stride, left closure id, right closure id) plus a deduplicated closure table.

#### I12 Config and system setup

One-time work: `cartesian`, `object_geometry`, BCs, `field_io` and MMS from Lua (sol2 with base and math libraries only, `src/app/shoccs.cpp:35`). `bcs::Grid` is 3 axes x {left, right}, Floating by default, with Dirichlet/Neumann downgraded to Floating on an extent-1 axis (`src/operators/boundaries.cpp:10-90`). scalar_wave evaluates `gG = -(x-c)/|x-c|` at all D and R points and zeroes Dirichlet sets (`src/systems/scalar_wave.cpp:44-102`); heat allocates 12 member vectors (`src/systems/heat.cpp:22-51`); the initial condition is `u0 = where(fluid, sol, 0)`, `u.R = sol.R` (`src/systems/detail/scalar_system_utils.hpp:186-219`); `system::build_rhs_graph` handles `scalar_handle{0}` only (`src/systems/system.cpp:41-55`); the mesh constructor logs every R record to `geometry.csv` (`src/mesh/mesh.cpp:139-151`). This is the surface a Python front end would replace. Several inputs are unvalidated: variance = 0 gives `pow(0,-2) = inf`, zero Gaussians give a null MMS, and `write_every_step <= 0` fires every step.

#### I14 R ordering contract

`Rx[i]` holds the value at `R(0)[i]`; R(dir) is ordered (slow, fast, t); B/Bf/Br use that position as row or column; `object_coordinate` stores it. Only Rz matches flat-D order.

```cpp
// Indices into R(dir); R(dir) buffer layout matches the data buffer by construction.
gather_selection dirichlet_object_desc(int dir, const bcs::Object& o) const {
  return make_gather_from_predicate(R(dir), [&o](const mesh_object_info& info){
    return o[info.shape_id] == bcs::Dirichlet; }); }
```

A port must reproduce the lexsort exactly or re-index every table consistently. This is a silent-corruption hazard, not a performance one. A single global R space with explicit permutations is an alternative design.

### Runtime: data access

#### I13 Flat C-order D with per-direction strides

Every kernel addresses D as a flat index with z fastest. Line operators walk `idx = row_offset + stride*t` with stride in {ny*nz, nz, 1}. Point evaluation decodes `i = idx/(ny*nz)`, `j = (idx/nz)%ny`, `k = idx%nz` (`src/systems/detail/scalar_system_utils.hpp:27-33`). Coordinates are 1-D `linear_distribute` arrays (`src/mesh/cartesian.cpp:41-43`).

```cpp
constexpr integer operator()(int3 ijk) const {
  auto&& [nx, ny, nz] = extents;
  return (integer)ijk[0]*ny*nz + (integer)ijk[1]*nz + ijk[2];
}
```

A reshape to (nx, ny, nz) needs no padding or copy. x-lines are the strided axis, so a framework that prefers contiguous last-axis operations may need a per-direction transpose (cost unverified here).

#### I15 Span bridge and system-owned work buffers

Operators and systems compute on four (pointer, length) pairs extracted from Views. `scalar_span::operator=(Fn)` applies an operator closure in place, which is how `u_rhs = lap(u, nu)` works.

```cpp
struct scalar_span {
    std::span<real> D{}, Rx{}, Ry{}, Rz{};
    template <std::invocable<scalar_span&> Fn>
    scalar_span& operator=(Fn&& fn) { FWD(fn)(*this); return *this; }
};
```

heat holds 12 and scalar_wave 28 extra `std::vector` buffers in this shape, passed into kernels and graph nodes (`src/systems/heat.hpp:29-31`, `src/systems/scalar_wave.hpp:29-37`). Spans carry no memory-space information, so on any device backend every work buffer must become a device array.

### Runtime: the derivative operator

#### I16 Ragged line block matvec O

One `TeamPolicy` kernel per direction: league = n_lines (one team per fragment), `TeamThreadRange` over the line's rows, a three-way branch on row region, a `ThreadVectorRange` reduction (vector length 8) over the row width, and `Kokkos::single` writing `op(b[out], dot)` with `op` either `eq_t` or `plus_eq_t`. Re-read from `src/matrices/block.hpp:117-178` and abbreviated:

```cpp
const auto m = meta(team.league_rank());
const int total_rows = m.left_rows + m.interior_rows + m.right_rows;
Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_rows), [&](int local_row) {
    int out_idx; real dot = 0;
    if (local_row < m.left_rows) {
        int r = local_row; out_idx = m.row_offset + r * m.stride;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, m.left_cols),
            [&](int j, real& s) { s += coeffs(m.left_coeff_offset + r * m.left_cols + j)
                                       * x_ptr[m.col_offset + j * m.stride]; }, dot);
    } else if (local_row < m.left_rows + m.interior_rows) {
        int r = local_row - m.left_rows;
        out_idx = m.row_offset + (m.left_rows + r) * m.stride;
        const int half_w = m.stencil_width / 2;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, m.stencil_width),
            [&](int j, real& s) { s += coeffs(m.interior_coeff_offset + j)
                                       * x_ptr[out_idx + (j - half_w) * m.stride]; }, dot);
    } else { /* right block, reads x_ptr[m.right_col_offset + j * m.stride] */ }
    Kokkos::single(Kokkos::PerThread(team), [&]() { op(b_ptr[out_idx], dot); });
});
```

Outputs not covered by any row (solid points, Dirichlet rows, skipped Dirichlet lines) are never written. Sizes: heat.lua about 59 teams x 19 to 51 rows with w = 3; 256^3 about 78k teams x up to 256 rows, about 15.7M rows per direction (one per fluid point, less Dirichlet wall rows), w = 3 to 9. On the host backend the team size degenerates to 1 and parallelism equals the line count (59 for heat.lua). The March baseline gives 13.7 ns/point for a slow-axis block matvec at 64^3, with 28x the time for 8x the points from 32^3 to 64^3. The fresh 8-thread run gives 8.9 ns/point at 64^3 and 21x the 32^3 time (Section 6, Part IV Section B). Both point to poor locality of one team per strided line. A coalesced axis-stencil formulation (threads across the fast index) decomposes the kernel exactly into I17 + I18/I19 + a mask. The `eq_t` semantics (unwritten points keep stale values) must then be reproduced explicitly with `where(mask, new, old)` or a zero-initialised output.

#### I17 Interior circulant axis stencil

`b[r] (=|+=) sum_{j<w} c[j]*x[r + (j-w/2)*stride]` for every interior row of every fragment; c has 2P+1 taps scaled by h and is identical for all lines of a direction (`block.hpp:150-160`, `derivative.cpp:436,465-467`). The standalone version (`src/matrices/circulant.cpp:59-66`):

```cpp
parallel_for(RangePolicy(0, nr), [=](int i){
  real dot = 0.0;
  for (integer j = 0; j < vs; j++) dot += vp[j] * xp[(i + j) * st];
  op(bp[i * st], dot);
});
```

It covers about 91% of D rows and about 97% of fluid rows: about 2.4k per direction for heat.lua, and about 15.3M per direction at 256^3. The 256^3 count is about 15.7M fluid points (D = 16.78M, about 6.5% solid) minus about 400k closure rows per direction (2.4% of D). With 2w flops per 16 to 24 bytes it is memory-bound (0.25 to 1.1 flop/byte). Expressed as shifted slices along one axis of (nx, ny, nz), it must be restricted to rows at least R_L / R_R from fragment ends, or overwritten afterwards by I18/I19.

#### I18 Wall closures

For wall-terminated line ends, `st.nbs` with psi = 1 gives an R x T block identical on every line ending at that wall (`derivative.cpp:367-381,414-428`). Dirichlet drops the wall row but keeps the wall column. Lines lying in another axis's Dirichlet wall are skipped entirely. Block sizes range from 1x4 (E2_2 Dirichlet) to 7x11 (E8u), applied on 2R planes per axis (heat.lua about 2 x 51 rows per direction; 256^3 about 2 x 3 x 65k rows). Equivalent to a small fixed matmul of the first or last T planes into the first or last R planes; the skipped Dirichlet-line planes become a static mask.

#### I19 Object closures

Per object endpoint, the psi-dependent `nbs` block minus column 0 (moved to B) and minus the boundary row for non-Dirichlet objects (moved to Bf/Br), with offsets shifted past the solid point (`derivative.cpp:336-366,385-413`). Each row is a unique small dense dot over a strided window of D:

```cpp
// block.hpp:141-149
int r = local_row;
out_idx = m.row_offset + r * m.stride;
parallel_reduce(ThreadVectorRange(team, m.left_cols), [&](int j, real& s){
  s += coeffs(m.left_coeff_offset + r*m.left_cols + j) * x_ptr[m.col_offset + j*m.stride];
}, dot);
```

Counts equal |R(dir)| endpoints: heat.lua 16 per direction of 1x3 rows; scalar_wave 22 of 2x3; 256^3 about 25.5k of up to 7x10. With wall closures this is about 2.4% of D rows at 256^3. Not a convolution. The portable form is an ELL table (`out_idx[n]`, `idx[n, Tmax]`, `C[n, Tmax]`) applied as gather, dot, scatter that overwrites the uniform-stencil result.

#### I20 Generic CSR accumulate kernel

`b[row] += sum w[i]*x[v[i]]`, always accumulating, in a plain `[=]` lambda (not `KOKKOS_LAMBDA`), over host vectors. Re-read from `src/matrices/csr.cpp:37-51`:

```cpp
void csr::operator()(std::span<const real> x, std::span<real> b) const
{
    const auto nr = rows();
    const auto* w_ptr = w.data(); const auto* v_ptr = v.data(); const auto* u_ptr = u.data();
    const auto* x_ptr = x.data(); auto* b_ptr = b.data();
    Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, nr),
        [=](integer row) {
            for (integer i = u_ptr[row]; i < u_ptr[row + 1]; i++)
                b_ptr[row] += w_ptr[i] * x_ptr[v_ptr[i]];
        });
}
```

The row and column spaces are recorded only in a uint8 flag (`src/matrices/common.hpp:23-41`). It is used for B, N, Bf_x/y/z and Br_x/y/z: eight launches per derivative. Arithmetic intensity is at most 0.083 flop/byte plus 16 bytes per row of int64 row-pointer traffic. This kernel is the largest Kokkos-device blocker in the matrices subsystem: host pointers inside a kernel, and a lambda that would not compile for CUDA or HIP. On Apple GPUs, sparse tensor types are not a dependable path (`00-apple-gpu-facts.md`, idiom table row I10/I20/I21, and `frameworks/pytorch.md`), so the portable form is plain gather plus scatter-add over non-empty rows only. Duplicate (row, col) entries are harmless for scatter-add.

#### I21 B: object-boundary column R(dir) -> D

`du.D[row] += w * u.R{dir}[object_coordinate]` for the fluid rows beside each intersection (`derivative.cpp:362-365,407-412`). Rows = |D| (`derivative.cpp:441`), so every call sweeps the full row-pointer array for about R x |R(dir)| nonzeros. Chained after O in the graph (`derivative.hpp:113`). heat.lua: 2601 row iterations for about 16 nonzeros. 256^3: 16.8M rows (134 MB of int64 row pointers) for about 1e5 nonzeros, three times per laplacian. A compact COO table removes a full-D-length pass per call, which is a straightforward improvement over the current code.

#### I22 N: Neumann wall data -> first X rows

`du.D[wall_row + k*stride] += extra[k]*nu.D[wall_ic]` for k < X (`derivative.cpp:375-379,422-426`). `nu` is a full D-sized host `std::vector` that is non-zero only on Neumann faces, filled in `update_boundary` (`heat.cpp:325-353`). heat always calls the Neumann overload (`heat.cpp:128,218`), so an empty full-height sweep runs for each derivative of every RHS even with all-Dirichlet walls. Object Neumann is never assembled and `nu.R*` is never read. Equivalent to a broadcast update of X slabs, `du[k-th plane] += extra[k]*g_face`, and a no-op when there is no Neumann BC.

#### I23 Bf_r: D -> R(r)

`du.R(r)[q] += sum_k w[q,k]*u.D[idx[q,k]]`. For r == dir, k runs over T-1 points along the ray from `solid_ic` (reversed if `ray_outside`); for r != dir, up to t_deriv x t_interp columns (12 to 16) spanning two axes (`derivative.cpp:76-87,188-253,480-482`). heat.lua has about 16 rows of 3 to 12 entries per (r, dir); 256^3 about 25.5k rows of up to 25 entries, three per derivative and nine per laplacian. A fixed-width gather-dot with precomputed index and weight tables. At shipped sizes these are tiny launch-bound kernels that should fuse with I24 over a concatenated [D; R(r)] source.

#### I24 Br_r: R(r) -> R(r)

`du.R(r) += Br*u.R(r)`: the diagonal boundary coefficient plus off-diagonal entries when an interpolation line ends on another intersection (`derivative.cpp:79,84-87,107,121,484-486`). Chained after Bf_r because both write `du.R(r)`. Merging it into the Bf table removes the dependency.

### Runtime: composition, fills and pointwise work

#### I25 Zero-fill before accumulation

Four `RangePolicy` launches plus a fence (eager) or four graph nodes (12 for gradient), because B, N, Bf and Br always accumulate and O never writes solid or Dirichlet rows (`src/fields/scalar.hpp:21-45`, `laplacian.hpp:62-75`, `gradient.hpp:68-114`).

```cpp
Kokkos::parallel_for(RangePolicy<execution_space>(0, D.size()),
    KOKKOS_LAMBDA(int i) { d_ptr[i] = v; });
// ... Rx, Ry, Rz likewise
Kokkos::fence();
```

A compiler that builds the output as one expression removes this pass.

#### I26 Whole-scalar scale

`du *= diffusivity` as four graph nodes `heat_scale_{D,Rx,Ry,Rz}` (`heat.cpp:221-232`) or `times_assign_scalar` over four buffers (`src/fields/expr.hpp:156-169`). Should fold into the operator output expression.

```cpp
auto s_d = lap_done.then_parallel_for("heat_scale_D", rp_t(0, n_d),
    KOKKOS_LAMBDA(int i) { d_ptr[i] *= k; });
```

#### I27 Accumulation policy and write ordering

Within a direction, O writes `du.D` with `eq` (gradient) or `plus_eq` (laplacian), then B and N accumulate after O; Bf_r then Br_r accumulate into `du.R(r)`. The laplacian zeroes `du` then chains dx, dy, dz strictly sequentially, since all three write the same buffers (`src/types.hpp:69-83`, `laplacian.cpp:36-59`):

```cpp
// derivative.cpp:546-549
auto o = O.graph_node(root, u_D, du_D, op);
B.graph_node(o, b_src, du_D);
// laplacian.hpp:78-80
auto d0 = dx.add_graph_nodes(zeroed, u, du, plus_eq);
auto d1 = dy.add_graph_nodes(d0, u, du, plus_eq);
return dz.add_graph_nodes(d1, u, du, plus_eq);
```

A laplacian has 4 zero nodes + 3 x (1 block + 7 or 8 CSR) = 28 to 31 nodes, with a critical path of about 10 dependent kernels. dz nodes are present even in 2D, with zero work (the `derivative` constructor returns early for extent < 2, `derivative.cpp:460`). In functional form, `du = Ox(u) + Oy(u) + Oz(u) + B... + N...` has no ordering constraints. The one semantic hazard is `eq_t` on partially covered outputs (gradient). Across directions the outputs overlap, so a single-kernel laplacian has to sum three directional contributions per point.

#### I28 Gradient fan-out

`du_x`, `du_y`, `du_z` are separate four-buffer outputs using `eq`, built as independent graph chains joined by `when_all`: 36 nodes including 12 zero fills (`gradient.cpp:34-46`, `gradient.hpp:116-121`). Fusing with the consumer (I29) avoids materialising three scratch fields, which would be 400 MB at 256^3.

#### I29 Variable-coefficient dot product

scalar_wave computes `u_rhs = gGx*dux + gGy*duy + gGz*duz` per buffer: six input streams and one output. The eager path fences after each buffer (four fences per RHS); the graph path has four nodes after `grad_done` (`scalar_wave.cpp:221-239,241-327`).

```cpp
Kokkos::parallel_for(RangePolicy<execution_space>(0, out.size()), [=](int i) {
    o[i] = gxp[i]*dxp[i] + gyp[i]*dyp[i] + gzp[i]*dzp[i]; });
Kokkos::fence();
```

This "linear operator, then pointwise product" shape is the same one the Euler equations need.

#### I30 Masked add/assign at gather index sets

`parallel_for` over `desc.count()`; `idx = desc.element(i)`; `dst[idx] (=|+=) expr(idx)`, with source and destination in the same index space and unique indices, so no atomics (`selection_desc.hpp:169-197`). Uses: heat source add at fluid D points and non-Dirichlet R points (four graph nodes per RHS), the initial-condition copy at fluid points, and the error field.

```cpp
template <typename Desc, typename Expr>
void plus_assign_selected(real* dst, Desc desc, Expr expr) {
  Kokkos::parallel_for(RangePolicy<execution_space>(0, desc.count()),
    KOKKOS_LAMBDA(int i) { int idx = desc.element(i); dst[idx] += expr(idx); });
}
```

heat.lua: 2545 + 16 + 16 indices per RHS. 256^3: 15.7M fluid indices, about 63 MB of extra index traffic (about 30% over a dense masked add). Because the fluid indices are sorted and unique, they are equivalent to a boolean mask, and a dense masked update does 2 to 6.5% redundant work.

#### I31 Dirichlet face and object fill

In the RHS, `du[face or Dirichlet object] = 0` (graph: one flattened gather node for D faces plus per-R nodes). In `update_boundary`, `u[face] = sol[face]` with one kernel per face, and `u.R[Dirichlet object] = sol.R[...]` (`heat.cpp:152-160,262-284,310-322`; `scalar_wave.cpp:342-354`).

```cpp
struct strided_selection { int offset_, inner_count_, outer_count_, outer_stride_;
  int element(int i) const {
    return offset_ + (i / inner_count_) * outer_stride_ + (i % inner_count_); } };
// x: {i*ny*nz, ny*nz}  y: {j*nz, nz, nx, ny*nz}  z: {k, 1, nx*ny, nz}
```

heat.lua: 204 face points, no R; scalar_wave.lua: 142 face points + 11 Rx + 11 Ry; 256^3: up to 393k face points. In the March baseline, contiguous and strided fills cost a negligible 0.1 to 0.17 ns/element and gather fills 0.5 to 8 ns/element. The fresh runs show the same pattern: gather is 4 to 25x the contiguous cost (Part IV Section B).

### Runtime: launch structure

#### I32 Kokkos Graph capture, instantiate, submit

Operators and systems capture raw pointers (u, du, nu, src, gather Views, CSR vectors) once into `then_parallel_for` nodes. The heat graph is the laplacian (4 zero + 3 x 9) + 4 scale + 4 source + 1 to 4 Dirichlet fills, about 40 nodes; the scalar_wave graph is 12 zero + 3 x 8 + 4 dot = 40 nodes. Each submit is followed by `Kokkos::fence` (`heat.cpp:290-294`).

```cpp
rhs_graph_ = create_graph<execution_space>([&](auto root) {
  auto lap_done = lap.add_graph_nodes(root, u, nu, du);
  auto s_d = lap_done.then_parallel_for("heat_scale_D", rp_t(0, n_d),
      KOKKOS_LAMBDA(int i) { d_ptr[i] *= k; });
  ...
  auto src_d_node = s_d.then_parallel_for("heat_src_D", rp_t(0, fluid.count()),
      KOKKOS_LAMBDA(int i) { int idx = fluid.element(i); d_ptr[idx] += src_d_ptr[idx]; });
  ...
  src_d_node.then_parallel_for("heat_fill_dir_D", rp_t(0, dir_d.count()),
      KOKKOS_LAMBDA(int i) { d_ptr[dir_d.element(i)] = 0; });
});
rhs_graph_->instantiate();
```

Buffers must never be reallocated or swapped after the build. Four submits per RK4 step give 160 node launches per step. On the host backend, graph nodes execute as ordinary `parallel_for` calls, so the graph saves no launch overhead today (systems inventory, based on reading the code path; not measured). Further fences sit in slot ops, `eval_at_locations`, scalar fill, stats and the scalar_wave dot. A whole-RHS or whole-step compiled function replaces this. The pointer-stability constraint disappears in functional frameworks with buffer donation and reappears in static-buffer graph replay (unverified here; see Part II). PyTorch has no graph capture or replay API for MPS (strong inference reported in `frameworks/pytorch.md`).

#### I33 Eager apply path

`derivative::operator()` launches 8 or 9 kernels then fences (`derivative.cpp:475-519`). `laplacian::operator()` and `gradient::operator()` return a heap-allocated `std::function` invoked by `scalar_span::operator=` (`laplacian.cpp:37-59`, `gradient.cpp:34-46`). A 2D heat eager RHS is 4 zero kernels + fence + 2 x (9 kernels + fence) = 22 launches and 3 fences; a 3D gradient is 36 launches and 6 fences. Used by tests and by the non-graph system path. In Python this is just a function; its fences correspond to the sync points a compiled path removes.

### Runtime: manufactured solutions and host callbacks

#### I34 Closed-form MMS evaluation at all D and R locations

`func(real3)` is evaluated at `(x[i], y[j], z[k])` for every D index and at `R(dir)[i].position` for every intersection, through `parallel_for` with a non-`KOKKOS_LAMBDA` `[=, &func]` lambda (host only), followed by a fence (`src/systems/detail/scalar_system_utils.hpp:16-73`):

```cpp
Kokkos::parallel_for(RangePolicy<execution_space>(0, nx*ny*nz), [=, &func](int idx) {
    int i = idx/(ny*nz); int j = (idx/nz)%ny; int k = idx%nz;
    d[idx] = func(real3{xv[i], yv[j], zv[k]}); });
Kokkos::parallel_for(RangePolicy<execution_space>(0, (int)m.Rx().size()),
    [=, &func](int i){ rx_out[i] = func(rx_data[i].position); });
Kokkos::fence();
```

`func` is a virtual `any_sol` call into Gaussian sums: per Gaussian one `exp`, one `cos` and several `pow(v,-2)` for the value (`src/mms/gauss2d.cpp:12-22`); the 3D Laplacian is hand-expanded with about 20 `pow` calls per Gaussian (`src/mms/gauss3d.cpp:82-125`). heat performs about 14 evaluations of (D+R) per step: 8 in `fill_source` (ddt and Laplacian at 4 stages), 4 in `update_boundary` (whole field, of which only faces and objects are used), 1 in stats and 1 in write. That is about 37k virtual calls per step for heat.lua and about 235M at 256^3, of which the 4 x 16.8M in `update_boundary` are wasted. scalar_wave performs 6 per step. Expressed as array math over broadcast coordinate grids and an (nR, 3) position array, this becomes one fused elementwise kernel, and the Gaussian time factor `cos(wt)` is separable. This is the one place where a Python port could plausibly beat the C++ at shipped sizes (inference, not measured). Restricting `update_boundary` to boundary sets helps in either language. Measured cost (heat.lua at 51x51, 1 thread; Part IV Section E natively, Section C in the container): `fill_source` takes 59.8 us per stage (28 to 37% of the 0.65 to 0.86 ms native step; container 132 us, 59% of 0.90 ms), `update_boundary` 24.4 us per stage (11 to 15%; container 34 us, 15%) and stats 26.9 us per step (3 to 4%; container 32 us, 4%).

#### I35 Lua per-point MMS callback

`lua_mms` sets `thread_safe = false` (`src/mms/lua_mms.hpp:13`), which forces the serial branch of `eval_at_locations` (`scalar_system_utils.hpp:60-69`). Each query marshals the location as a Lua table and calls a Lua function through a sol2 `std::function`; `grad` returns three values (`src/mms/mms.cpp:34-82`, `mms.t.cpp:164-168`). It runs wherever I34 does when `manufactured_solution.type = 'lua'`: about 14 x 2633 Lua calls per step at heat.lua scale. The cost per call was not measured; the io inventory estimates microseconds, which would make each evaluation take milliseconds. This is inherently a host per-point callback and the only blocker in the catalog. A Python front end would replace it with NumPy- or SymPy-generated vectorised callables rather than port it.

#### I36 Per-stage host source fill before graph submit

`system::submit_rhs_graph` calls `fill_source(time)` on the host (two MMS evaluations per point into member vectors) and then submits the graph whose nodes read those vectors (`src/systems/system.cpp:57-69`, `heat.cpp:111-117,290-294`):

```cpp
if constexpr (requires { s.submit_rhs_graph(); }) {
    if constexpr (requires { s.fill_source(time); }) s.fill_source(time);
    s.submit_rhs_graph();
} else { s.rhs(creg, input, reg, output, time); }
```

Each RK stage therefore has a host-visible boundary: a fence after the evaluation and another after the submit. It is hard only because of the host callback. Once I34 is array math, the source folds into the compiled RHS with t as a traced scalar.

### Runtime: time integration and boundary updates

#### I37 RK4 and forward-Euler slot linear combinations

Re-read from `src/temporal/rk4.cpp:21-46` (profiling regions removed):

```cpp
slot_zero(reg, rk_rhs_ref);
slot_zero(reg, system_rhs_ref);
reg.deep_copy_slot(output.slot, u0.slot);
for (int i = 0; i < 4; ++i) {
    if (i > 0) {
        slot_assign_lc(reg, output, u0, dt * rki[i], system_rhs_ref);
        sys.update_boundary(reg, output, time + dt * rki[i]);
    }
    sys.submit_rhs_graph(reg, output, reg, system_rhs_ref, time + dt * rki[i]);
    slot_accumulate(reg, rk_rhs_ref, dt * rkf[i], system_rhs_ref);
}
slot_assign_lc(reg, output, u0, 1.0, rk_rhs_ref);
sys.update_boundary(reg, output, time + dt);
```

Each slot op loops over n_scalars x 4 buffers with one `parallel_for` per buffer (including the empty Rz) and a trailing fence, and asserts `n_vectors == 0` (`src/temporal/slot_ops.hpp:11-59`):

```cpp
for (auto bh : sh.all()) {
    int n = reg.size(dst, bh);
    real* d = reg.data(dst, bh); const real* s0 = reg.data(src, bh);
    const real* r = reg.data(rhs, bh);
    Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i) { d[i] = s0[i] + coeff * r[i]; });
}
Kokkos::fence();
```

Solid D points are integrated too. The inventories count this differently: the systems inventory gives 9 slot passes (36 `parallel_for` + 8 `deep_copy` zeroes) per RK4 step; the fields inventory counts 13 passes (about 52 launches) by including the two slot copies and the zeroes as passes. Either way, at shipped sizes each launch moves 2633 doubles and is bound by the launch floor: 0.55 us per tiny kernel at 1 thread in the fresh runs (Part IV Section B), 0.7 us in the March baseline. At 256^3 the slot ops alone touch about 27 full-field arrays per step, about 3.6 GB. Forward Euler (`src/temporal/euler.cpp:11-27`) is the one-stage analogue and is unused by the shipped configs.

#### I38 deep_copy_slot for pointer stability

All non-empty buffers of a slot are copied (`src/fields/field_registry.hpp:172-207`). `swap_slots` exists but is avoided:

```cpp
// Uses deep_copy (not swap_slots) to preserve stable data pointers
// required by the pre-built RHS graph.
reg.deep_copy_slot(u0_ref.slot, u1_ref.slot);
```

Two full copies per step (u1 <- u0 at RK start, u0 <- u1 at step end; `rk4.cpp:25`, `simulation_cycle.cpp:121-125`).

#### I39 update_boundary per stage

heat allocates four zeroed `std::vector`s, evaluates the exact solution everywhere (I34), assigns Dirichlet faces with one kernel per face, and assigns object-Dirichlet R entries through descriptors rebuilt on the spot (I40). For each Neumann direction it allocates four more vectors, evaluates `gradient[dir]` everywhere and copies the face values into `neumann_d` (`src/systems/heat.cpp:296-354`):

```cpp
std::vector<real> sol_d(m.size()); ... // 4 allocs per call
eval_at_locations(m, [&](const real3& loc){ return m_sol(time, loc); }, sol, m_sol.is_thread_safe());
for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
    assign_selected(u_D, desc, handle_expr{sol.D.data()}); });
for (int dir = 0; dir < 3; ++dir) {
    auto gd = m.dirichlet_object_desc(dir, object_bcs); // host loop + View alloc + deep_copy
    assign_selected(reg.data(ref, R[dir]), gd, handle_expr{sol_R[dir]}); }
```

Called 4 times per RK4 step. Per heat.lua step: 16 vector allocations, 12 View allocations and `deep_copy`s, 44 `parallel_for`, 4 fences. At 256^3: 4 x 134 MB of host allocations per step. Everything except the time-dependent boundary values is invariant, so a port evaluates only on boundary points. Together with I36 this is the dominant per-step host cost and the source of forced sync points if data lived on a GPU.

#### I40 Per-call rebuild of object-BC gather lists

`m.dirichlet_object_desc` and `non_dirichlet_object_desc` return a new `gather_selection` on every call (`src/mesh/mesh.hpp:97-115`): 3 per `update_boundary` (4 per step), 3 per stats, 6 per write and 6 per eager `heat::rhs`. Only `heat::build_rhs_graph` hoists them. This means about 18 to 30 small View allocations and host-to-memory-space copies per RK4 step. The lists are 16 to 22 entries shipped and about 25.5k at 256^3. It is an incidental inefficiency, not an inherent need, and a port hoists it to setup (I04).

### Runtime: diagnostics, output and loop control

#### I41 Stats: masked reductions with per-step host readback

The exact solution is re-evaluated (I34), then over the fluid gather: `MinMax(u)` and `MaxLoc(|u-sol|)`; for each R direction with a non-empty non-Dirichlet list (rebuilt, I40): MinMax and MaxLoc (`src/systems/detail/scalar_system_utils.hpp:77-182`):

```cpp
Kokkos::parallel_reduce(RangePolicy<execution_space>(0, fd.count()),
  KOKKOS_LAMBDA(int k, Kokkos::ValLocScalar<real,int>& update) {
    int i = fd.element(k); real e = Kokkos::abs(u_D[i] - sol_D[i]);
    if (e > update.val) { update.val = e; update.loc = i; } },
  Kokkos::MaxLoc<real,int>(maxloc_result));
```

That is 2 + 2 x (non-empty R lists) blocking reductions into host scalars per step (6 for heat.lua). The 11 returned values gate `while (controller && sys.valid(stats))` and feed the CSV log every step (`simulation_cycle.cpp:78,101-104`). This is a mandatory device-to-host synchronisation per step unless validity is checked every K steps and logs are buffered.

#### I42 Error field, binary dump and XDMF

Every accepted step: allocate four solution vectors, evaluate the exact solution everywhere (I34), zero four error buffers, compute `err = |u-sol|` at fluid D and non-Dirichlet R points, fence, zero Dirichlet faces and objects (six descriptors rebuilt), then call `field_io::write`, which returns early unless the dump interval fires (`scalar_system_utils.hpp:224-302`, `src/io/field_io.cpp:32-66`).

```cpp
Kokkos::parallel_for(RangePolicy<execution_space>(0, fd.count()),
    KOKKOS_LAMBDA(int k) { int i = fd.element(k);
        err_d_ptr[i] = Kokkos::abs(u_D[i] - sol_D[i]); });
...
return io.write(io_names, io_scalars, c, dt, m.R()); // early-returns if !dump_interval
```

On a dump: per variable one headerless float64 file holding D then Rx, Ry, Rz (solid points included) (`src/io/field_data.cpp:39-63`); the pugixml DOM of `view.xmf` is reloaded, appended and saved with `Seek` offsets F at 0, RX at 8D, RY at 8(D+Rx), RZ at 8(D+Rx+Ry) (`src/io/xdmf.cpp:60-173`); dump 0 also writes R coordinates, and in 2D writes (z, y, x) into an XYZ geometry (`field_data.cpp:16-17`, open question). heat.lua runs about 18 kernels per step even when nothing is written and produces about 31 dumps of 2 x 8 x 2633 bytes. scalar_wave.lua dumps every step and reparses an XML file that grows with the step count. Gating the error computation on the dump interval is an easy fix in either language.

#### I43 Outer time loop and dispatch

```cpp
while (controller && sys.valid(stats)) {
  const std::optional<real> dt = sys.timestep_size(reg, u0_ref, controller);
  integrate(sys, reg, u0_ref, u1_ref, rk_ref, srhs_ref, controller, *dt);
  controller.advance(*dt);
  stats = sys.stats(reg, u0_ref, u1_ref, controller);
  sys.write(io, reg, u1_ref, controller, *dt);
  sys.log(stats, controller);
  reg.deep_copy_slot(u0_ref.slot, u1_ref.slot);
}
```

(`src/simulation/simulation_cycle.cpp:31-143`). dt is a constant CFL formula: heat `p_cfl*hmin^2/(4k)`, scalar_wave `h_cfl*hmin` with speed 1 assumed (`heat.cpp:356-361`, `scalar_wave.cpp:357-362`). Systems are a `std::variant<empty, scalar_wave, inviscid_vortex, heat, hyperbolic_eigenvalues>` with graph gating through `if constexpr (requires {...})`; the integrator is `variant<empty, rk4, euler>` (`src/systems/system.hpp:22-65`). Output intervals OR step and time criteria when both are set (`src/io/interval.hpp:58-99`, `field_io.cpp:100-101`). Step counts: heat.lua about 50, scalar_wave.lua 20, brady n61 about 530, brady long about 2650. Only the per-step readback (I41) constrains a host Python loop around a compiled step. Euler would need dt from a device reduction.

#### I44 Eigenvalue diagnostic

Double-dispatch visitors walk every dense/circulant/CSR piece of `dx` (1D only, stride 1 asserted). Pass 1 (`src/matrices/unit_stride_visitor.cpp:11-197`) assigns first-come dense indices in a concatenated [D | Rx | Ry | Rz] space, skipping `ldd`/`rdd` and Dirichlet R rows and columns. Pass 2 (`src/matrices/coefficient_visitor.cpp:10-61`) overwrites coefficients into a dense n x n host matrix. Then lapackpp `geev` (`src/operators/eigenvalue_visitor.cpp:10-52`):

```cpp
auto v = eigenvalue_visitor{m.extents(), p, std::vector<bool>{}, std::vector<bool>{}};
grad.visit(v);
return system_stats{.stats = {-m.h(0) * std::ranges::min(v.eigenvalues_real())}};
```

eigenvalues.lua has 21 to 23 rows. The design requirement this sets for a DSL: operators must be assemblable as explicit matrices, not only applicable. The overwrite semantics would drop duplicate CSR entries if the path were used in 2D.

#### I45 Reference component matvecs

Oracle implementations used only in tests: `dense` is a serial host double loop over device-View data (`src/matrices/dense.cpp:9-37`), `circulant` a `RangePolicy` over rows (`circulant.cpp:34-68`), `inner_block` calls the three in sequence (`inner_block.cpp:42-48`). Test sizes 15 to 45 points.

#### I46 Expression templates with alias check

Trivially copyable expression nodes (`handle_expr`, `scalar_literal_expr`, `binary_expr`, `unary_expr`) evaluated by `operator()(int i)` inside a `parallel_for`; `assign()` walks the tree for aliasing and falls back to a temporary View plus `deep_copy` (`src/fields/expr.hpp:20-150`):

```cpp
template <typename Expr>
void assign(real* dst, int n, Expr expr) {
  if (contains_ptr(expr, dst)) {
    Kokkos::View<real*, memory_space> tmp("expr_tmp", n);
    ... parallel_for tmp[i] = expr(i); deep_copy(dst_um, tmp);
  } else
    parallel_for(RangePolicy(0,n), KOKKOS_LAMBDA(int i){ dst[i] = expr(i); });
}
```

Production uses only leaf nodes through the `_selected` helpers; composite trees appear only in tests and `benchmarks/bench_expr.cpp:39-60`. No offset (stencil) access is possible. The C++ operator DSL was deliberately deleted (`docs/reference/fields.md`), so there is nothing to port here.

#### I47 Euler scaffold

`inviscid_vortex` contains only 2D isentropic-vortex exact-solution functors (rho, rhoU, rhoV, P, rhoE, gamma = 1.4) (`src/systems/inviscid_vortex.cpp:13-116`); `size()` returns `{}`, `rhs` is empty and `valid()` is false (`inviscid_vortex.cpp:118-154`).

```cpp
enum class vars { rho, rhoU, rhoV, rhoE, P };
system_size inviscid_vortex::size() const { return {}; }
void inviscid_vortex::rhs(...) {}
```

There is no multi-component integrator support (slot ops assert `n_vectors == 0`; `build_rhs_graph`, stats and extract helpers hard-code `scalar_handle{0}`). Section 10 lists what a working Euler system would add.

### Python side (scripts/stencil_gen)

#### I48 Python dense and sparse operator assembly

An r x t closure block is written top-left, a 2p+1 band in the middle and `(-1)^nu` times the reflected block bottom-right into `np.zeros((n, n))` with Python loops, reimplemented in at least five places (`scripts/stencil_gen/stencil_gen/phs.py:622-697`, `brady2d_stability.py:438-489`, `sweeps/alpha_extraction.py:138-159`, and others):

```python
D = np.zeros((n, n))
for i in range(r):
    w = uniform_boundary_weights_rbf(i, t, nu, q, epsilon, kernel=kernel)
    for j in range(t): D[i, j] = w[j]
for i in range(r, n - r):
    for k_idx, j in enumerate(range(i - p, i + p + 1)):
        D[i, j] = interior_w[k_idx]
sign = (-1.0) ** nu
for i in range(r):
    row = n - 1 - i
    for j in range(t): D[row, n - 1 - j] = sign * w[j]
```

Dirichlet is modelled by deleting row and column 0. The 2D operator is `L = -(diag(cx) kron(I, D/h) + diag(cy) kron(D/h, I))` in Fortran order with `np.ix_` DOF masks (`brady2d_stability.py:834-914`). Sizes: 1D n = 20 to 160; 2D (N-1)^2 = 400, 900, 3600 DOFs. These cover only uniform, unfragmented lines: no R DOFs, no O/B split, no fragments, no cross-direction interpolation rows, no 3D. The only cut-cell line matrix is a 1D test helper with a Dirichlet cut wall on the left (`tests/test_group_velocity.py:994-1063`).

#### I49 Python symbolic coefficient layer and codegen

`CutCellResult` and `InterpResult` hold exact `sympy.Matrix` blocks in (psi, y, alpha/fa/da/ia) whose dimensions reproduce the C++ {P, R, T, X} exactly (`scripts/stencil_gen/stencil_gen/temo.py:80-96,3041`; `interp.py:73-340`). Numeric evaluation is `float(F[i,j].xreplace(subs))` per entry (`group_velocity.py:628-687`); `lambdify` is not used anywhere. Codegen applies `sympy.cse` and prints the C++ struct, including the right-side flip and h scaling (`codegen.py:15-40,136-233,354-428`; `printer.py:14-96`):

```python
replacements, reduced = cse(exprs, symbols=numbered_symbols('t', start=start))
```

Also: the RBF numeric solve (`phs.py:258-462`), symbolic conservation weights (`conservation.py:38-289`), and tools that regex-parse committed C++ bodies and `exec` them to cross-check (`tools/eval_poly.py`, `tools/eval_e2_1.py`). Several shipped stencils (E4_2, the spline families, the E4_1 guards) are hand-written and cannot be regenerated. E4 derivation takes about 30 s with `SYMPY_CACHE_SIZE=50000`. This is the strongest reusable asset for a DSL (Section 9).

#### I50 Python stability-analysis cascade

Rightmost eigenvalue via `scipy.sparse.linalg.eigs(which='LR')` with a shift-invert retry and a dense fallback for n <= 900; `sigma_min(sI - L)` over a complex s-grid by dense SVD for n <= 1200; GKS Kreiss determinant sweeps; group velocity tensors of shape (Ny, Nx, N_xi); RK4 CFL from the spectral radius (`non_normality.py:72-148,274-345`; `phs.py:781-846`; `gks_kreiss.py:90-340`; `group_velocity.py:19-160,1100-1166`; `sweeps/comparison.py:40-51`).

```python
evals = eigs(L, k=k_use, which="LR", return_eigenvectors=False, rng=rng_seed)
...
M = s * np.eye(n) - L_dense
sv = np.linalg.svd(M, compute_uv=False)
```

A code comment reports dense SVD at n = 961 at about 0.2 s per s-point (`non_normality.py:292-295`). The cascade stays fp64 on the host; `00-apple-gpu-facts.md` lists the sparse non-Hermitian eigensolver as unsupported on Apple GPU stacks. It would benefit directly from a DSL that can export the real cut-cell operator as a matrix, which would close the current gap: no code computes 2D eigen-stability with embedded boundaries.

#### I51 cpp_bridge and Bayesian optimisation driver

`cpp_bridge` renders a Lua template (`--{{N}}--`, `--{{T_FINAL}}--`, `--{{SCHEME_TABLE}}--`), runs the `shoccs` binary per parameter point in a temporary directory, and parses `logs/system.csv` (stable if Linf < 10) (`scripts/stencil_gen/stencil_gen/cpp_bridge.py:52-205`):

```python
src = src.replace("--{{N}}--", str(int(N)))
src = src.replace("--{{SCHEME_TABLE}}--", _scheme_table_for(scheme_type, params))
completed = subprocess.run([str(binary), str(lua_path)], cwd=tmp, capture_output=True, text=True, timeout=timeout)
```

The Bayesian optimisation driver uses float64 CPU torch tensors for GP surrogates (`bo.py:42-55,432-433`). torch 2.11.0 is a locked dependency (`scripts/stencil_gen/pyproject.toml:9-22`, `uv.lock`). There are no pybind or ctypes bindings and no in-process operator access. An in-process Python solver would remove this boundary.

## 4. Minimal primitive set

Every hot idiom reduces to seven primitives. Setup idioms need no runtime primitive: they produce tables.

| Primitive | Definition | Covers | Performance character |
|---|---|---|---|
| P0 field container | scalar = {D as (nx,ny,nz) \| Rx \| Ry \| Rz} with static offsets; optionally one packed array per field | I01, I13, I14, I15 | Determines launch count: 4 (or 12) launches per op today versus 1 when packed. No bandwidth cost of its own. |
| P1 fused elementwise map | `out[i] = f(a[i], b[i], ..., mask[i], x[i], y[j], z[k], t)`, arbitrary pointwise trees fused into one pass; masks replace fluid and Dirichlet gather lists on D; also realises zero-fill + accumulate as one expression | I22, I25, I26, I27, I29, I30, I31, I34, I36, I37, I38, I39, I42, I46, I47 | Bandwidth-bound at large N: 0.44 ns/element at 1M doubles on 1 thread (67 GiB/s effective at 24 B/element) and 0.37 ns/element on 8 threads (97 GiB/s); the March baseline gave 0.39 ns/element, about 61 GB/s. Launch-bound at shipped sizes (0.55 us floor at 1 thread). |
| P2 constant-coefficient axis stencil with interior mask | `out[r] (=\|+=) sum_j c_d[j]*x[r + (j-w/2)*stride_d]`, c_d of width 3 to 9 shared by all lines of a direction, applied over the whole grid and masked or overwritten near fragment ends, at solid points and at Dirichlet rows; threads across the fast index | I16, I17, I28 | 0.25 to 1.1 flop/byte, memory-bound. At 256^3 one read and one write of D per axis (134 MB each), about 2 to 3 ms per axis at 100 GB/s. The current kernel does 4 to 14 ns/point per axis at 64^3 in the March baseline (6.6 to 8.9 ns/point in the fresh 8-thread single-axis runs), so there is headroom over C++ even on CPU. |
| P3 fixed-width ELL gather-dot with scatter | `out[row_ids[r]] (=\|+=) sum_k C[r,k]*src[idx[r,k]]`, padding to a dummy index with zero coefficient; one table per (direction, role): closure rows, B, N, Bf_r, Br_r, optionally merged over a concatenated [D; R(r)] source; row ids unique within a table, so no atomics | I18, I19, I20, I21, I22, I23, I24, I30 | Latency-bound at shipped sizes (16 to 200 rows per table, 6 to 9 tables per derivative), so it must fuse or batch. At 256^3 about 400k closure rows per direction x T = 5 to 11, and about 25.5k R rows x up to 25 entries: a few ms per RHS at the baseline gather rate of 0.5 to 8 ns/element, small next to three 134 MB D passes. Replaces the full-height B/N sweeps. |
| P4 strided slice assign/read on the (nx,ny,nz) view | `u3[0,:,:]`, `u3[:,j,:]`, `u3[:,:,k]`; may lower to P1 with a static face mask in functional frameworks | I22, I31, I39 | Negligible: 0.1 to 0.17 ns/element in the baseline; 51 to 71 elements per face shipped, 65,536 at 256^3. |
| P5 masked reduction with controlled readback | `where(mask, f(u), -inf).max()` / argmax, fused into one pass per buffer; results stay on device and are read back every K steps or asynchronously | I41 | One extra read of D per step (about 1 to 2 ms at 256^3). The real cost on a discrete or asynchronous device is the per-step synchronisation. |
| P6 whole-RHS or whole-step compilation / graph replay | traces a pure function `step(state, t) -> state` covering P1 to P5, fusing about 40 RHS nodes and about 44 slot/BC launches per stage into O(1 to 10) kernels; t enters as a traced scalar; state is donated (functional) or static (graph replay) | I27, I32, I33, I36, I43 | Decides the small-size regime: the C++ graph submit is 21.0 us per RHS for heat.lua at 51x51 and 25.2 us for scalar_wave.lua at 71x71 (1 thread, native, Part IV Section E; 22.7 and 30.7 us in the container, Section C). Compile cost is per (shape, geometry), amortised over 20 to 2650 steps. |

Nothing in the runtime needs a general sparse matrix type, atomics, team scratch memory or a library solver. No KokkosKernels, atomics or scratch are used in `src/matrices/` today.

## 5. The hot loop

### 5.1 One heat RHS (graph path)

Sources: `src/systems/heat.cpp:163-294`, `src/operators/laplacian.hpp:85-118`, `src/operators/derivative.hpp:121-147`. Sizes are given for heat.lua (|D| = 2601, |Rx| = |Ry| = 16, |Rz| = 0, fluid 2545, Dirichlet faces 204) and, in brackets, for the hypothetical 256^3 sphere (|D| = 16.8M, |R| about 25.5k per axis, fluid 15.7M, faces 6 x 65k, lines about 78k per direction).

| Step | Kernels | Work | Sync |
|---|---|---|---|
| H0 | host `fill_source(t)`: `eval_at_locations` twice (ddt and Laplacian of the MMS) into `src_*` vectors | 2 x (2601 + 32) virtual calls [2 x 16.9M] | 2 fences; host compute before submit |
| H1 | 4 nodes `lap_zero_{D,Rx,Ry,Rz}` | 2633 [16.9M] | `when_all` |
| H2 | dx, 9 nodes: Bfx -> Brx (16 rows); Bfy -> Bry (16); Bfz -> Brz (0); O_x block `plus_eq` (59 teams x 19 to 51 rows, w = 3, stride 51) -> B_x (2601-row sweep, about 16 nnz) -> N_x (2601-row sweep, 0 nnz) | [78k teams x up to 256 rows; B and N sweeps of 16.8M rows each] | |
| H3 | dy, 9 nodes, same shape, chained after dx (stride 1 lines) | | |
| H4 | dz, 9 nodes with 0 teams and 0 rows in 2D (still launched) | [third full pass] | |
| H5 | 4 nodes `heat_scale_*`: `du *= k` | 2633 | |
| H6 | 4 nodes `heat_src_*`: `du[idx] += src[idx]` over fluid and non-Dirichlet R lists | 2545 + 16 + 16 | |
| H7 | 1 to 4 nodes `heat_fill_dir_*`: `du[Dirichlet] = 0` | 204 face points; object lists empty for heat.lua | |
| H8 | graph submit + `Kokkos::fence` (`heat.cpp:290-294`) | | fence |

Total: about 40 graph nodes, 2 host evaluations and 3 fences per RHS. The 2D critical path is zero -> (O_x -> B_x -> N_x) -> (O_y -> B_y -> N_y) -> (dz nodes) -> scale -> source -> fill, about 13 dependent kernels.

scalar_wave RHS instead: 12 zero nodes; dx, dy, dz in parallel (8 nodes each, `eq`, no N) into scratch buffers `dux`/`duy`/`duz`; 4 dot nodes. 40 nodes, no `fill_source`.

### 5.2 One RK4 step

Sources: `src/temporal/rk4.cpp:16-47` inside `src/simulation/simulation_cycle.cpp:78-126`, for heat.lua.

| Step | Work | Sync |
|---|---|---|
| S1 | `slot_zero(rk)` and `slot_zero(srhs)`: 4 `deep_copy(0)` each | 2 fences |
| S2 | `deep_copy_slot(u1 <- u0)`: 3 `deep_copy` (D, Rx, Ry) | |
| S3 | stage 0: RHS(u1, t) = H0..H8; `slot_accumulate rk += dt/6*srhs` (4 `parallel_for`) | fence |
| S4 | stage 1: `slot_assign_lc u1 = u0 + dt/2*srhs` (4 + fence); `update_boundary(u1, t+dt/2)`: 4 vector allocations, host MMS at all D and R + fence, 4 face kernels, 3 object-descriptor rebuilds + 3 assign kernels (0 entries); RHS; accumulate dt/3 | 3+ fences |
| S5 | stage 2: as S4 | |
| S6 | stage 3: `u1 = u0 + dt*srhs`; `update_boundary(t+dt)`; RHS; accumulate dt/6 | |
| S7 | final `u1 = u0 + rk` (4 `parallel_for` + fence); `update_boundary(u1, t+dt)` (fourth call) | |
| S8 | `controller.advance(dt)` | |
| S9 | stats(u1): host MMS at all D and R (4 allocations), 3 descriptor rebuilds, 6 blocking reductions into 11 host doubles | device-to-host readback that gates the loop |
| S10 | write(u1): host MMS (4 allocations), 4 error zero kernels, 3 error kernels, fence, 6 descriptor rebuilds + Dirichlet zero kernels; on a dump (every 0.01 time units): 2 binary files of 8 x 2633 bytes and an XDMF reparse/append/save | fence + host I/O |
| S11 | 2 spdlog lines; `deep_copy_slot(u0 <- u1)` | |

Per-step totals for heat.lua: 4 graph submits x about 40 nodes = 160 node launches; about 92 integrator `parallel_for` (16 `fill_source` evaluations, 16 accumulate, 16 assign, 44 `update_boundary`); about 32 `deep_copy` calls (8 slot zeroes, 6 slot copies, about 18 index-View uploads); about 26 explicit fences; 6 blocking reductions; about 14 full-field host MMS evaluations (about 37k virtual calls); 24 `std::vector` allocations; about 280 launches in total. Beyond the operators, memory traffic is about 27 full-field array touches per step (3.6 GB per step at 256^3).

**Measured cost of one step** (Part IV Sections A, C and E, heat.lua at 51x51, 1 thread, io off). The native figures from the macOS build of 2026-10-13 (Part IV E) are the ones quoted; the container figures of 2026-10-12 (Part IV A and C) stand beside them. These measurements replace the earlier estimate of the host MMS cost (about 37k virtual calls, plausibly 100 to 300 us).

| Component | Per call, native | Calls per step | Per step, native | Share of the native 0.65 to 0.86 ms step | Per call, container | Share of the 0.90 ms container step |
|---|---|---|---|---|---|---|
| `fill_source` (host MMS, H0) | 59.8 us | 4 | 239 us | 28 to 37% | 132 us | 59% |
| `update_boundary` (host MMS at every point plus face and object assigns, S4 to S7) | 24.4 us | 4 | 98 us | 11 to 15% | 33.9 us | 15% |
| RHS graph submit (H1 to H8) | 21.0 us | 4 | 84 us | 10 to 13% | 22.7 us | 10% |
| stats (S9) | 26.9 us | 1 | 27 us | 3 to 4% | 31.9 us | 4% |
| remainder: slot ops and copies, timestep logic, logging, write with io off (S1, S2, S10, S11) | | | 0.2 to 0.4 ms, noisy | about 30 to 50% | about 114 us | about 13% |

The sum of the measured native components is 448 us against a native one-thread step whose median is 0.856 ms, minimum 0.648 ms and p90 1.35 ms: a lone busy thread is not pinned to a performance core, so the one-thread step is noisy, while the stage micro-benchmarks average over many iterations and are not (container: 786 us against a 0.90 ms step). The host-MMS components alone (`fill_source`, `update_boundary` and stats) are 42 to 56% of the native shipped step (container: 78%), before counting the error-field evaluation in write. The discretization itself is about a tenth in both builds. At 65^3 the same split gives `fill_source` 49% of the 19.96 ms native step on four threads with libomp defaults (4 x 2429 us; container: 72% of a 37.9 ms step on 8 threads). With four spin-waiting threads the native heat.lua step is 0.266 ms, below the one-thread sum of stage work; under spinning the stage micro-benchmarks overstate the in-step cost (Part IV E), so that step is not decomposed here.

### 5.3 Host synchronisation points

On the host backend every kernel completes before returning, so fences cost little today. On any asynchronous device backend, these become real synchronisations:

1. Before each RHS submit: host source evaluation (I36), 4 per step.
2. Inside each `update_boundary`: host MMS evaluation, host allocations, descriptor uploads (I39, I40), 4 per step.
3. After every slot op: a fence (I37), about 9 per step.
4. After each graph submit: a fence (I32), 4 per step.
5. Stats: blocking reductions whose values gate the loop (I41), once per step.
6. Write: host MMS and error field every step, host I/O on dumps (I42).

Only item 5 is semantically required per step by the current loop, and even it could be relaxed to every K steps. Items 1, 2 and 6 disappear once MMS and boundary values are device-side array expressions. Items 3 and 4 disappear under whole-step compilation.

## 6. Measured performance bar

### 6.1 Machine and provenance

The bar is built on fresh measurements made on 2026-10-12 (Part IV, `05-measurements.md`). The solver and benchmarks were built from this worktree inside the project devcontainer image (shoccs-devcontainer:latest of 2026-07-30: Debian trixie, gcc 14, Kokkos 5.1.1 with Serial and OpenMP, Release). They ran under Docker Desktop on the owner's Apple-silicon Mac with 8 vCPU and 12 GB. The default host execution space is Kokkos::OpenMP. "1 thread" means OMP_NUM_THREADS=1. "8 threads" means OMP_NUM_THREADS=8 with OMP_PROC_BIND=spread, except in the original micro-benchmarks, which ran with OMP_PROC_BIND unset. The measurements were made in the container because at the time there was no native macOS build: the C++ dependency stack existed only in the container's Spack view. A native build has existed since 2026-10-13 (`docs/reference/macos-host-build.md`), and `05-measurements.md` Section E repeats these measurements on it; Sections 5.2 and 6.4 quote the native figures with the container ones beside them. The earlier benchmark pass identified the host as an Apple M4 (10 cores, 4P + 6E, 16 GB, Darwin 25.2). A Linux container cannot reach the Apple GPU, so the Apple GPU measurements (torch MPS fp32, MLX fp32, with numpy, torch and MLX CPU fp64 as their reference) ran natively on the host; they are in `05-measurements.md` Section D and summarised in Section 6.5 below.

The new benchmark `benchmarks/bench_rhs_cutcell.cpp`, added by this study, embeds the shipped heat.lua and scalar_wave.lua geometries in 2D and 3D. It times the RHS graph submit and the host-side work of the time loop (`fill_source`, `update_boundary`, stats) separately, so one RK4 step can be decomposed. End-to-end step times come from the Wall_ms column of `logs/system.csv` with io off, as the median over all steps after step 0. Raw JSON and run scripts are in `docs/python-dsl-survey/experiments/container/`.

Historical cross-check: the earlier version of this section used `benchmarks/baselines/baseline-20260320-024030.json`. That baseline was run on 2026-03-20 in a Linux container on an Apple-silicon host with Google Benchmark 1.9.5, `num_cpus = 4` and a release build, with unknown thread count. It contains no cut-cell work. It is kept below (6.2, last table) only for comparison.

### 6.2 Measurements

**Discretization: RHS graph submit on the shipped cut-cell geometries** (Part IV Section C). R is the total number of Rx + Ry + Rz points. The R counts match the replica counts in Section 1.6 (16 + 16 for heat.lua, 22 + 22 for scalar_wave.lua).

| Case | D points | R | 1 thread | 8 threads | ns/point (faster run) |
|---|---|---|---|---|---|
| heat.lua 51x51 with its sphere | 2601 | 32 | 22.7 us | 87.8 us | 8.7 (1t) |
| heat 71x71 | 5041 | 50 | 34.5 us | 92.0 us | 6.8 (1t) |
| scalar_wave.lua 71x71 with its two spheres | 5041 | 44 | 30.7 us | 86.1 us | 6.1 (1t) |
| heat 201x201 | 40401 | 138 | 198 us | 152 us | 3.8 (8t) |
| scalar_wave 201x201 | 40401 | 128 | 177 us | 123 us | 3.1 (8t) |
| heat 401x401 | 160801 | 276 | 772 us | 422 us | 2.6 (8t) |
| heat 65^3 with sphere | 274625 | 586 | 1867 us | 798 us | 2.9 (8t) |

**Shipped RK4 step** (Part IV Section A). io is off. Wall_ms covers integrate, stats and write; write still evaluates the MMS and the error field with io off.

| Case | D points | Steps | 1 thread | 8 threads |
|---|---|---|---|---|
| heat.lua 51x51 as shipped | 2601 | 50 | 0.90 ms | 0.94 ms |
| scalar_wave.lua 71x71 as shipped (max_step 60) | 5041 | 60 | 0.43 ms | 0.68 ms |
| heat 201x201 | 40401 | 80 | 12.2 ms | 4.6 ms |
| heat 401x401 | 160801 | 64 | 48.0 ms | 13.7 ms |
| heat 65^3 with sphere | 274625 | 11 | 148.9 ms | 37.9 ms |

**Step decomposition** (Part IV Section C). Times are per call. Each RK4 step makes four `fill_source`, four `update_boundary` and four RHS calls, plus one stats call.

| Component | 51x51, 1 thread | Share of 0.90 ms step | 65^3, 8 threads | Share of 37.9 ms step |
|---|---|---|---|---|
| `fill_source` (host MMS: ddt minus k times the Laplacian at every D and R point) | 132 us per stage | 59% | 6.84 ms per stage | 72% |
| `update_boundary` (host MMS at every point plus face and object assigns) | 34 us per stage | 15% | 1.12 ms per stage | 12% |
| RHS graph submit | 22.7 us per stage | 10% | 0.80 ms per stage | 8% |
| stats (host MMS at every point plus masked reductions) | 32 us per step | 4% | 1.08 ms per step | 3% |
| Sum of the above against the measured step | 786 us against 900 us | | 36.1 ms against 37.9 ms | |

The remainder (114 us at 51x51) is slot ops and copies, timestep logic, logging and the write call. Host-side manufactured-solution evaluation dominates the shipped step at every size measured.

**Fresh micro-benchmarks** (Part IV Section B).

| Benchmark | Case | 1 thread | 8 threads | Note |
|---|---|---|---|---|
| BM_assign_fma (`dst = a + alpha*b`) | 1024 elements | 0.551 us | 1.94 us | launch floor 0.55 us per tiny kernel at 1 thread |
| BM_assign_fma | 1048576 elements | 463 us | 385 us | 0.44 ns/element and 67 GiB/s at 1 thread; 0.37 ns/element and 97 GiB/s at 8 threads |
| BM_heat_rhs (no shapes, Dirichlet xmin/xmax only) | 16^3 / 32^3 / 64^3 | 40 us / 418 us / 4.51 ms | 84 us / 169 us / 1.16 ms | 64^3: 17.2 ns/point at 1 thread, 4.4 at 8 threads |
| BM_block_matvec | 16^3 / 32^3 / 64^3 | not run | 4 us / 109 us / 2.32 ms | 64^3: 8.9 ns/point; 21x the 32^3 time for 8x the points |
| BM_derivative_apply E2 / E4, one axis | 64^3 | not run | 1.74 ms / 1.81 ms | 6.6 / 6.9 ns/point |
| BM_assign_selected gather | 1024 / 131072 / 1048576 | 29.1 us / 2905 us / 440 us | 8.84 us / 534 us / 508 us | 4 to 25x the contiguous cost; 131072 anomalous in both runs |

**Historical cross-check: March 2026 baseline** (`benchmarks/baselines/baseline-20260320-024030.json`, thread count unknown, no cut cells). ns/point values are time divided by point count.

| Benchmark | Case | Size | Time (us) | ns/pt or ns/elem | Note |
|---|---|---|---|---|---|
| BM_heat_rhs | full heat RHS, Kokkos Graph submit, E2, 3D, Dirichlet xmin/xmax, no shapes | 16^3 | 32 | 7.8 | 22678 iterations |
| BM_heat_rhs | same | 32^3 | 157 | 4.8 | |
| BM_heat_rhs | same | 64^3 | 1646 | 6.3 | about 2 MB per D buffer |
| BM_block_matvec | N^2 line blocks along one axis, width 5 | 16^3 | 3 | 0.8 | |
| BM_block_matvec | same | 32^3 | 129 | 3.9 | |
| BM_block_matvec | same | 64^3 | 3585 | 13.7 | 28x time for 8x points; slower than the whole RHS |
| BM_derivative_apply | E2, one axis | 16^3 / 32^3 / 64^3 | 8 / 150 / 1802 | 2.0 / 4.6 / 6.9 | one axis exceeds the 3-axis RHS at 64^3 |
| BM_derivative_apply | E4, one axis | 16^3 / 32^3 / 64^3 | 8 / 137 / 3493 | 2.0 / 4.2 / 13.3 | |
| BM_assign_fma | `dst = a + alpha*b` | 1024 / 16384 / 131072 / 1048576 | 0.731 / 2.547 / 64.751 / 409.503 | 0.71 / 0.16 / 0.49 / 0.39 | launch floor about 0.7 us; 1M elements about 61 GB/s effective |
| BM_plus_assign_mul | `dst += a*b` | 1024 / 16384 / 131072 / 1048576 | 1.897 / 2.278 / 63.033 / 259.048 | 1.85 / 0.14 / 0.48 / 0.25 | 1024 is an outlier |
| BM_assign_selected_contiguous | face copy | 1024 / 16384 / 131072 / 1048576 | 0.759 / 1.701 / 11.514 / 137.145 | 0.74 / 0.10 / 0.09 / 0.13 | |
| BM_assign_selected_strided | strided face | same | 0.758 / 2.864 / 22.403 / 174.901 | 0.74 / 0.17 / 0.17 / 0.17 | |
| BM_assign_selected_gather | index-list gather | same | 9.372 / 40.661 / 1013.622 / 530.356 | 9.2 / 2.5 / 7.7 / 0.51 | 1M faster than 131k: anomalous |
| BM_circulant_apply | single 1D circulant | 64 / 1024 | 0.64 / 1.036 | about 1 ns/pt at 1024 | dispatch-dominated |

The baseline 16^3 and 64^3 RHS times and the 1M axpby time each fall between the fresh 1-thread and 8-thread figures. That fits a multi-threaded baseline on 4 CPUs.

### 6.3 Caveats

- **Container, not native.** Docker Desktop on macOS runs Linux in a virtual machine, so memory bandwidth and scheduling may differ from native. The tables in 6.2 are container figures; until 2026-10-13 there was no native macOS build, and Apple clang ships without OpenMP, so the native build needed its own toolchain decision (Spack llvm-openmp, `docs/reference/macos-host-build.md`). Its figures are in `05-measurements.md` Section E and are the ones quoted in 5.2 and 6.4: the one-thread RHS submit is within 10% of the container, host MMS work is 2 to 3.5x cheaper, and four threads at libomp defaults are slower than one on every case below 201x201; spin-waiting recovers the end-to-end step (0.266 against 0.856 ms at heat51) but not the RHS submit alone (54.1 against 21.0 us).
- **Thread count changes the bar.** With 8 threads, runs below about 40k points are slower than with one thread: the fork and join per Kokkos node costs more than the work. At 51x51 the RHS takes 87.8 us on 8 threads against 22.7 us on 1. Every figure in 6.4 names its thread count, and the small-size bars use the 1-thread figures. The original micro-benchmarks ran their 8 threads unpinned and the cut-cell benchmark pinned them (spread), so the two 8-thread columns are not strictly comparable.
- **The fixed-up step bar is an estimate.** Part IV derives it by modelling only change (3) of the fixed-up list (6.4), with assumed residual MMS costs. It also leaves out the 114 us remainder of the 51x51 step. Even if the whole remainder stayed, the 51x51 estimate would rise to at most about 0.26 ms.
- **Two 64^3-class RHS fixtures disagree.** On 1 thread the shape-free `bench_rhs` at 64^3 (17.2 ns/point; Dirichlet on xmin/xmax only, floating elsewhere) is 2.5x slower per point than the cut-cell fixture at 65^3 (6.8 ns/point; Dirichlet on every face plus the sphere). On 8 threads the gap is 4.4 against 2.9 ns/point. The cause was not investigated. The cut-cell figure is the one used below, because it is the configuration a replacement must match.
- **Single-axis benchmarks are still inconsistent.** In the fresh 8-thread run, a single-axis E2 derivative at 64^3 (1.74 ms) still costs more than the whole three-axis shape-free RHS (1.16 ms), as in the baseline. `bench_block` and `bench_derivative` use their own fixtures, so they are not RHS components. The regime-C extrapolation therefore scales whole-RHS rates, not the 13.7 ns/point single-axis figure.
- **Gather at 131072 elements is anomalous** in both fresh runs and in the baseline (slower than at 1M).
- **The streaming rates may be optimistic for 256^3.** The 1M-element axpby moves about 25 MB, which may be partly cache-resident. A 256^3 buffer is 134 MB.

### 6.4 Derived requirements for a replacement

There are three bars, and a replacement should be judged against all three.

- **Discretization bar:** the RHS graph submit time per size (6.2, first table). It is what a replacement's kernels must match.
- **Shipped step bar:** today's measured RK4 step (6.2, second table). Host manufactured-solution evaluation dominates it.
- **Fixed-up step bar (estimate):** the step after a frozen list of exactly four changes to the C++: (1) hoist the object-BC gather lists out of the per-call path; (2) compact B and N tables with no full-height row pointers; (3) manufactured-solution evaluation once per step and only where needed, with time factors hoisted (fill_source and update_boundary); (4) packed [D | Rx | Ry | Rz] buffers. Restated from Part IV E, the estimate is 131 us at 51x51 (1 thread: 4 x (21.0 + 5) + 26.9 us), 6.0 ms at 65^3 (4 threads, libomp defaults: 4 x (0.961 + 0.3) + 0.955 ms) and 12.4 ms at 65^3 (1 thread: 4 x (2.065 + 0.3) + 2.96 ms), each modelling change (3) alone with a residual of 5 us or 0.3 ms per stage for fill_source and update_boundary together; the container estimates were about 0.15 ms and about 5.5 ms (8 threads). All are estimates. There is no decomposed estimate at 4 threads spin, where the fixed-up heat.lua step lies between 131 us and the measured 0.266 ms. A Python replacement should be compared against this bar, because every source of its expected advantage at small sizes is on this list.

**Small 2D (shipped configs: D = 2.6k to 5k, R = 32 to 50, state about 85 to 160 KB in L2).** These are launch- and latency-bound.

- The launch floor is 0.28 us per tiny kernel at 1 thread on the native build (0.55 us in the container, 0.7 us in the March baseline). The 51x51 heat RHS takes 21.0 us natively (22.7 us in the container) for about 40 graph nodes, about 0.5 us per node: pure dispatch.
- Requirement S1 (discretization bar): an RHS in at most 21.0 us for heat.lua (51x51) and 25.2 us for scalar_wave.lua (71x71), both at 1 thread on the native build (container: 22.7 and 30.7 us); with four spin-waiting threads the native submits are 54.1 and 54.6 us, so one thread sets this bar.
- Requirement S2 (step bars): an RK4 step under the shipped native 0.856 ms for heat.lua and 0.278 ms for scalar_wave.lua at 1 thread (0.266 and 0.198 ms with four spin-waiting threads; container: 0.90 and 0.43 ms), and under the estimated fixed-up 131 us for heat.lua (container estimate 0.15 ms). These replace the earlier unmeasured estimate of 0.3 to 1 ms. The host runs of `05-measurements.md` Section D show S2 is reachable from Python in fp64 on the CPU, on a synthetic step without host MMS or update_boundary: 0.085 ms (compiled MLX CPU), 0.103 ms (eager numpy) and 0.149 ms (torch eager) at 51x51, all under the shipped bar and at or near the fixed-up estimate, and 1.0x to 1.8x the four native one-thread RHS submits of S1 (84 us; 0.9x to 1.6x of the container's 91 us). The fair S2 bar for such a step is therefore four RHS submits plus stats, 111 us at 51x51 on the native build (4 x 21.0 + 26.9 us; container 123 us = 4 x 22.7 + 31.9 us), not the shipped step.
- Consequence: a port must compile the whole step (P6) into O(5 to 10) fused kernels plus at most one readback. Eager per-op dispatch at 5 to 20 us per op is an estimate from the inventories (unverified). Local measurements in Part II put eager CPU dispatch at about 0.6 to 4 us per op for PyTorch and JAX, which would still give 0.2 to 1.2 ms per step for 300 ops. The host runs of `05-measurements.md` Section D measure the floors on the owner's M4: 0.46 us per numpy op, 0.75 us per eager torch op, 6.4 us per torch.compile call, 8.7 us per compiled MLX CPU call, and 167 to 181 us for any GPU op plus synchronisation (torch MPS or MLX); the last replaces the earlier unverified forum figure of about 120 us for a Metal round trip in `00-apple-gpu-facts.md`. The same runs show what fusion buys: the synthetic step takes 0.085 to 0.15 ms on the CPU in fp64 with about 20 ops per RHS, and eager MPS, at about a hundred kernels per step, takes 1.5 to 2.2 ms. At 5 to 20 us per op, about 300 ops would take 1.5 to 6 ms per step: 1.8 to 7x the native one-thread shipped step and 11 to 46x the native fixed-up estimate (container: 1.7 to 7x and 10 to 40x).
- Where Python can win at this size: vectorised MMS (one fused kernel instead of thousands of virtual calls per evaluation), boundary-only evaluation in `update_boundary`, hoisted object lists, and an error field gated on the dump interval. `fill_source` alone is 28 to 37% of the native one-thread shipped heat.lua step (59% in the container). These are algorithmic changes that also speed up the C++, which is why the fixed-up bar exists (Section 11, items 5 and 15).
- A serial path is part of the bar: below 201x201 the native RHS submit is fastest on one thread (21.0 us against 54.1 us with four spin-waiting threads and 309 us with libomp defaults at 51x51), even though the whole shipped step is faster with four spin-waiting threads (0.266 against 0.856 ms); in the container the C++ was fastest on one thread below about 40k points.

**Mid sizes (2D 201x201 to 401x401, 3D 32^3 to 65^3; 0.3 to 2.2 MB per buffer).** Requirement M1 (discretization bar), native: an RHS in at most 217 us at 201x201 (1 thread; 228 us with four spin-waiting threads), 575 us at 401x401 (4 threads, libomp defaults; 827 us at 1 thread) and 961 us at 65^3 with the sphere (4 threads, libomp defaults; 2065 us at 1 thread, 1810 us spinning); container: 152, 422 and 798 us at 8 threads, 198, 772 and 1867 us at 1 thread. For shape-free 3D: 0.313 ms at 32^3 (4 threads spin) and 1.36 ms at 64^3 (4 threads default) natively; container 169 us and 1.16 ms at 8 threads. Step bars: shipped native 2.17 ms (201x201), 8.18 ms (401x401) and 17.49 ms (65^3), all with four spin-waiting threads (container: 4.6, 13.7 and 37.9 ms at 8 threads); estimated fixed-up 6.0 ms at 65^3 at 4 threads and 12.4 ms at 1 thread (container about 5.5 ms). Slot ops add an estimated 0.5 to 0.8 ms per step at 64^3: about 27 full-field array touches of 2.1 MB at the measured 67 to 108 GiB/s streaming rates. The one-team-per-strided-line kernel runs at 6.6 to 8.9 ns/point per axis at 64^3 (container, 8 threads), against 0.37 to 0.44 ns/element for streaming. That leaves a coalesced axis stencil large headroom on CPU.

**Large 3D (256^3 = 16.78M points, 134 MB per buffer, bandwidth-bound; extrapolated, not measured).**

- Lower bound per RHS: at least 14 doubles per point, 1.88 GB moved. At the measured single-core streaming rate of 67 GiB/s (about 72 GB/s) that is about 26 ms (1.6 ns/point). At the 8-thread 97 GiB/s (about 104 GB/s) it is about 18 ms. At an assumed (not measured) 120 GB/s M4 unified-memory peak it is about 16 ms (0.9 ns/point).
- Linear scaling of the fresh 65^3 cut-cell RHS rate gives about 49 ms at 8 threads (2.9 ns/point) and about 114 ms at 1 thread (6.8 ns/point). The shape-free 64^3 rate gives about 74 ms at 8 threads (4.4 ns/point). At 256^3 the buffers no longer fit in cache, so linear scaling from 2 MB buffers is optimistic.
- Requirement L1: about 50 ms per RHS at 8 threads and about 115 ms at 1 thread (estimates, scaled from 65^3 with the sphere). A fused design that reads D once per axis cannot go below about 16 to 26 ms.
- Requirement L2: per RK4 step, add about 3.6 GB of slot-op traffic, about 35 to 50 ms at the measured 97 and 67 GiB/s, unless stage update, BC and accumulate are fused. For whole steps, linear scaling of the shipped 65^3 step (138 ns/point, 8 threads) gives about 2.3 s per step at 256^3. Scaling the estimated fixed-up 65^3 step (about 20 ns/point) gives about 0.34 s. Both are estimates.
- Requirement L3: memory. 540 MB for four fp64 state slots, plus operator tables. Use masks rather than gather lists (16.8 MB versus 63 MB for the fluid set), compact B/N tables rather than 134 MB row pointers, and a deduplicated closure table rather than 22 to 85 MB of per-line coefficients per derivative.
- The current C++ also sweeps 134 MB of int64 row pointers for B and N per direction (6 per laplacian) and does 4 x 16.8M host MMS evaluations per step in `update_boundary`, on top of `fill_source`, which is already 49% of the native four-thread step at 65^3 (72% in the container at 8 threads). A compact-table, boundary-only port therefore has large headroom over the shipped step at this size, and much less over the fixed-up C++.
- Cut-cell cost: R work is O(N^2) against O(N^3) for D (0.46% of D at 256^3). The fresh runs measure it only indirectly. At 65^3 the cut-cell RHS (2.9 ns/point at 8 threads) is not slower per point than the shape-free RHS (4.4 ns/point), so the 9 to 27 tiny CSR launches per RHS matter only at small sizes.

**Precision.** Every number above is fp64. No Apple GPU has native fp64, PyTorch MPS rejects float64 tensors, and MLX runs fp64 on the CPU only (`00-apple-gpu-facts.md`, backends table). An MPS port therefore runs fp32 or a hand-written double-float scheme at several times the cost. Whether fp32 is accurate and stable enough for small-psi high-order closures is an open question (Section 11). The CPU fp64 path is the primary performance bar. The measured Apple GPU timings are in `05-measurements.md` Section D and Section 6.5 below.

### 6.5 Host-side Python measurement

`05-measurements.md` Section D (source: `experiments/host_primitives_results.md`) times a synthetic SHOCCS-shaped RK4 step built from the primitives P0 to P6 of Section 4 (E2 interior, random closure coefficients, no host MMS, no update_boundary, one Linf readback) natively on the owner's M4 on seven backends. Against the bars above:

- S1 and S2 at 51x51: a fused fp64 step on the CPU from Python costs 0.085 ms (compiled MLX CPU), 0.103 ms (eager numpy) and 0.149 ms (torch eager), 1.0x to 1.8x the four native one-thread RHS submits (84 us), 0.4x to 0.7x of the four-thread-spin submits (216 us), 6 to 10x under the native one-thread shipped step (0.856 ms) and 1.8 to 3.1x under the four-thread-spin step (0.266 ms) (container: 0.9x to 1.6x of 91 us and 6 to 11x under 0.90 ms).
- M1 at 65^3: the best CPU fp64 paths cost 4.2 ms (compiled MLX CPU) and 4.5 ms (torch.compile CPU, 4 intra-op threads), 1.1x to 1.2x of the native four-thread submits (3.8 ms) and about half the one-thread submits (8.3 ms) (container: within 1.4x of 3.2 ms at 8 threads and under 7.5 ms at 1). CPU parity is plausible, not shown: synthetic coefficients, no MMS; since Part IV E the C++ bar is at least on the same machine.
- Apple GPU in fp32: held at 0.35 to 0.5 ms per step by a 167 to 181 us sync floor from 51x51 to 201x201, so it loses to the CPU at the shipped sizes; 1.0 ms at 65^3 and 43.6 ms at 257^3, 4 to 6x faster than the best CPU fp64 path at 129^3 and 257^3 (2.6 to 4.2 ns per point per step).
- Regime C in Python: 272 ms per step on the CPU in fp64 (torch.compile, 16 ns per point) and 43.6 ms on the GPU in fp32 at 257^3, against the L1 and L2 estimates above for the C++.
- fp32 error after 10 steps: 1e-7 to 2.4e-7 relative on every case, on synthetic E2 closures; it says nothing about small-psi conditioning (Section 11, item 1).

## 7. Cut-cell unstructuredness, precisely

The owner's earlier attempts were slowed by the "unstructuredness" the cut cells introduce. Reading the code, regularity breaks in exactly six places at runtime and three at setup. Everything else is a dense Cartesian stencil problem: about 80% of lines are full-length, and interior rows with a shared stencil make up about 91% of D rows and about 97% of fluid rows (at 256^3: about 15.3M interior rows per direction, from about 15.7M fluid points less about 400k closure rows).

**Runtime carriers**

1. **Fragmented lines** (`src/mesh/mesh.cpp:29-87`; `derivative.cpp:328-438`). A pencil crossing a body yields two or more fluid fragments. Each fragment is one `inner_block` with its own `row_offset = ic(start)`, `col_offset`, stride, left/right/interior row counts and coefficient offsets (13 int32 per line, `block.hpp:30-94`). Fragment endpoints are the solid coordinate next to the intersection; rows are trimmed by `submatrix_size` (`derivative.cpp:259-303`). Pencils in a Dirichlet wall of another axis are dropped (`mesh.cpp:154-168`). Counts: heat.lua 59 = 51 + 8 per direction; 256^3 sphere about 78.3k = 65,536 + 12.8k per direction (19.5% of pencils fragmented); shortest fragment 17 to 19 points in the examined cases. The output rows of one direction are disjoint arithmetic progressions, not contiguous ranges.
2. **Irregular-length R buffers** (`object_geometry.cpp:51-114`; `field_registry.hpp:89-91`). A scalar is four buffers of unrelated lengths. |R(dir)| depends on geometry (16/16/0 heat.lua; 22/22/0 scalar_wave.lua; about 25.5k per axis at 256^3). The implicit (slow, fast, t) order is the index contract for field buffers, B columns, Bf/Br rows and `object_coordinate`. Records are AoS. Every scalar op becomes 4 launches, three of which touch about 20 elements.
3. **psi-dependent dense closure blocks** (`derivative.cpp:336-366,383-413`). Each object-terminated line end has a unique (R-1 or R) x (T-1) block from closed-form rational functions of that intersection's psi. Wall closures (psi = 1) and the interior are identical across lines but stored per line anyway (11 to 163 doubles per line, up to 85 MB per derivative at 256^3, against O(|R| x R x T) unique data).
4. **CSR couplings B, N, Bf, Br** (`derivative.cpp:71-157,159-257,441-448`; `csr.hpp:16-68`). B (R(dir) -> D) and N (Neumann -> D) are full-height host CSRs with |D|+1 int64 row pointers for O(R x |R|) or zero nonzeros. Bf_r (D -> R(r)) and Br_r (R(r) -> R(r)) have |R(r)| rows with T-1 entries (r == dir) or up to 12 to 16 entries (r != dir, derivative of interpolants across two axes, whose interpolation lines may end on other R points). Duplicates are not merged. Row and column spaces are encoded only in a uint8 flag. Kernels capture host vector pointers.
5. **Gather lists instead of masks** (`selection_desc.hpp:110-158`; `mesh.cpp:89-111`; `mesh.hpp:97-115`). The fluid set is a sorted int32 list (2545 of 2601 for heat.lua; 15.7M = 63 MB at 256^3). Object sets are int32 lists into R built by predicate scans and rebuilt on every use.
6. **Partial-coverage write semantics** (`block.hpp:173-175`; `types.hpp:69-83`; `laplacian.hpp:62-80`). O never writes solid points, Dirichlet wall rows or Dirichlet lines; `eq_t` leaves them stale; all CSR terms and `du.R` always accumulate. Hence the zero fill before every operator and the serial dx -> dy -> dz chain. A vectorised port must make the output definition total (zeros plus scatter, or `where(mask, new, old)`).

**Setup-only carriers**

- `interp()` dispatch with integer, data-dependent branching and variable-length footprints (`stencil.hpp:196-254`).
- The O(|R|) `interp_line` scan per query (`mesh.cpp:170-207`).
- Bit-exact snap, clamp and `nextafter` ray casting (`object_geometry.cpp:76-95`).

**What this means.** Everything that depends on psi, BC type, wall-versus-object or interpolation is baked into coefficients at setup. Nothing geometric is recomputed per step. The runtime unstructuredness is therefore a fixed set of small index and coefficient tables (closure ELL, B, Bf, Br) plus masks, all of size O(|R|), applied around a regular O(N^3) axis stencil. The unstructured work at runtime is a gather-dot-scatter over about 0.5 to 2.5% of the points, not a general sparse matrix. The genuinely irregular control flow (carrier 1's fragment search, and I07's interpolation dispatch) runs once, on the host, and can stay in Python or C++ outside any compiled kernel. This holds only while geometry is static: moving bodies would pull I02 to I11 into the step and would require device-side `nbs` and `interp`. The generated scalar arithmetic would port to a device; the dispatch logic would not port easily.

## 8. Setup versus runtime

**Once per (geometry, BCs, scheme, alpha)** (can stay in Python/NumPy or C++, and produces device tables):

- Grid coordinates, extents, strides (I12, I13).
- Ray casting to SoA R records `psi[nR]`, `ray_outside`, `solid_ic`, `shape_id`, `position[nR,3]`, `normal[nR,3]` per direction, in (slow, fast, t) order (I02, I14).
- Fragment table per direction: `start_ic`, `len`, stride, left/right kind (wall Dirichlet/Floating/Neumann, or object with R index and psi), `dirichlet_line` mask (I03).
- Index sets: `fluid_mask[D]`, Dirichlet face slices, `obj_dirichlet_mask[nR]`, `obj_nondirichlet_mask[nR]` (I04; hoisting the per-call rebuilds of I40).
- Stencil constants: interior taps per direction, psi = 1 wall closures, uniform closures, RBF solve (I05); psi-dependent closures for all R points in one vectorised fp64 call (I06); interpolation rows via a per-pencil lookup (I07).
- Operator tables per direction and operator: deduplicated closure table and ELL closure rows; B, N, Bf_r, Br_r as compact COO/ELL; no full-height row pointers; no per-line coefficient copies (I08 to I11).
- Constant coefficient fields (scalar_wave `gG`), MMS parameters, initial condition (I12).
- Compile or trace the RHS and the step (P6); matrix export for diagnostics (I44, I50).

Setup cost today at 256^3 includes an estimated 700k tiny Kokkos allocations (I11: 3 Views per line x about 78.3k lines x 3 directions), an estimated 1e9 comparisons per (r, dir) pair in `interp_line` (I07: about 1.3e5 calls per pair, each scanning up to about 12k records), and per-line virtual stencil calls. All of it is fixable in a NumPy setup.

**Every RHS** (4 per RK4 step): zero/accumulate (I25, I27), three axis stencils (I17), closure rows (I18, I19), B/N (I21, I22), Bf/Br (I23, I24), scale (I26), masked source add (I30), Dirichlet zero (I31); for scalar_wave also gradient fan-out (I28) and the dot product (I29); source values at time t (I34, a host evaluation today, I36).

**Every stage:** slot axpby (I37); boundary values at time t on faces and Dirichlet R points (I39, a full-field host evaluation plus allocations today); object-list rebuild (I40, should be setup).

**Every step:** slot copies (I38), stats reductions and readback (I41), error field and dump-due test (I42; the error field should be gated), logging and loop control (I43), dt (constant today).

**Every output:** device-to-host copy, binary dump, XDMF append (I42).

The only runtime inputs that change are the state u and time t. psi, masks, tables and coefficients are immutable.

## 9. What the Python stencil_gen side already provides

`scripts/stencil_gen` is a stencil design and stability toolkit, not a solver.

| Asset | Status | Faithfulness to C++ | Reuse in a DSL |
|---|---|---|---|
| Exact symbolic coefficients (I49): `CutCellResult`, `InterpResult` in (psi, y, alpha/fa/da/ia) | Working for TEMO and poly families | Dimensions match {P, R, T, X} exactly (`temo.py:80-96`); T-frame column 0 is the wall/R value, matching the B-column split; psi convention matches the mesh | Direct source for batched I06/I07 evaluation |
| Numeric evaluation | `float(F[i,j].xreplace(subs))` per entry; no `lambdify` anywhere | Exact but slow (minutes at 256^3 point counts, inference) | Replace with `lambdify` or a retargeted printer |
| CSE codegen (`codegen.py`, `printer.py`) | Working; output copied into `src/stencils/` by hand | Emits exact `nbs`/`interp_wall`/`interp_interior` bodies including right-side flip and h scaling | Retarget the printer to NumPy, JAX or Metal |
| Hand-written C++ stencils | E4_2, spline families, E4_1 guards (psi clamp [1e-4, 1-1e-4], `alpha[1] >= 197/288`) | The C++ files are the source of truth; generator output lacks the guards; E4_2 symbolic cut-cell derivation is broken | Must be ingested (as `tools/eval_poly.py` does) or re-derived |
| Numeric operators (I48) | Dense 1D and 2D Kronecker operators | Faithful to one unfragmented line with stride 1; Dirichlet as row/column deletion, verified equivalent to `nbs_dirichlet` for E4u_1 | Dense twin of I16 for regression tests only |
| Missing entirely | | Four-buffer scalar, R DOFs, fragmented lines, per-end psi, O/B/Bf/Br, cross-direction interpolation rows, selection descriptors, runtime Neumann extras, 3D, time integration, RHS | Everything at runtime |
| Analysis cascade (I50) | Host fp64 scipy | Runs only on uniform-line operators | Benefits from a matrix-export backend |
| Integration (I51) | Subprocess per parameter point | Black box | Replaced by an in-process solver |
| torch / botorch | torch 2.11 locked (CPU wheel; MPS reported built in the sibling venv) | Used only for float64 GP surrogates | No new dependency if torch is chosen |

Net: Python owns coefficient derivation and analysis. None of the runtime data layout (D|Rx|Ry|Rz, fragments, meta/coeffs, CSR tables, masks) and none of the runtime kernels has a Python counterpart. A DSL would reuse the symbolic layer and the CSE printer directly, replace the subprocess bridge, and should expose a matrix-export backend for the stability cascade.

## 10. Euler requirements

The Euler scaffold (I47) contains only exact-solution functors. A working system adds the following on top of the catalog.

1. **Multi-component state.** 4 to 5 conservative fields (rho, rhoU, rhoV, [rhoW], rhoE), each in the four-buffer layout: 16 to 20 buffers per slot. The registry supports 8 scalars per slot, but slot ops assert `n_vectors == 0` and `build_rhs_graph`, stats and extract helpers hard-code `scalar_handle{0}`. Either the integrator and graph glue become multi-scalar, or fields become (ncomp, D|R) arrays.
2. **Pointwise nonlinear maps (P1).** Velocity m/rho, pressure (gamma-1)(E - |m|^2/(2 rho)), sound speed sqrt(gamma p/rho); flux products F_d = (m_d, m_d m/rho + p e_d, m_d (E+p)/rho): 5 components x 3 directions = 15 flux fields, 2 GB at 256^3 in fp64 unless fused into the derivative input.
3. **Batched linear operators.** 15 directional first derivatives summed into 5 RHS fields (10 in 2D), that is, gradient-type `derivative` applications over a leading component axis in one launch (P2/P3 with a batch dimension). The per-scalar operator API, the one-graph-per-slot-pair binding and the Bf/Br R-point rows must all accept batches. The "operator then pointwise product" shape already appears in scalar_wave (I29).
4. **R-point and wall physics.** Slip or no-penetration (m . n = 0) at R points needs the per-record normal (SoA `normal[nR,3]`) and a projection kernel on the R buffers. Characteristic or inflow/outflow face BCs need per-face pointwise eigen-decompositions (P1 on faces, P4). Dirichlet object R values and floating R unknowns must be distinguished per component.
5. **Time step from a device reduction.** dt = CFL x h / max over fluid of (|u| + c) every step (P5), with a host-visible or device-resident dt.
6. **Stability aids.** Filtering or artificial dissipation (another axis stencil, P2, possibly with per-line closures); possibly conservative quadrature weights (the SymPy conservation weights exist, I49) for monitoring discrete conservation at R points.
7. **Diagnostics.** Per-component Linf against the vortex exact solution (I41 x 5), multi-variable output (I42), matrix export of the linearised operator for I50.
8. **Precision.** Euler with shocks or strong vortices in fp32 (as on Apple GPUs) is a sharper accuracy question than the heat equation. fp64 on CPU or CUDA remains the reference.

## 11. Open questions

1. **fp32 viability.** `real = double` everywhere and Apple GPUs have no fp64 (`00-apple-gpu-facts.md`). Are fp32 cut-cell closures acceptable? Small-psi rows have large cancelling rational coefficients; E4_1 clamps psi to [1e-4, 1-1e-4]; the RBF schemes need a Taylor branch to avoid cancellation. This decides whether the MPS target is viable at all.
2. **Native benchmark.** Fresh container measurements with per-region timers now exist (Part IV, Section 6). Still missing: a native macOS build (same CMake flags; Apple clang has no OpenMP, so the toolchain must be chosen). The host-side Apple GPU and Python CPU runs now exist (`05-measurements.md` Section D, Section 6.5), on synthetic coefficients; they need repeating on the real exported tables. Exact |R|, line, fluid and nnz counts should still be read from `geometry.csv` of a real run. The benchmark's R totals (32 for heat.lua, 44 for scalar_wave.lua) already match the replica.
3. **Uncovered points under `eq_t` (gradient).** Keep stale values (current behaviour) or zero? A total definition changes bitwise results at solid and Dirichlet points.
4. **MMS surface.** Can manufactured solutions and sources be restricted to array expressions, so the Lua per-point callback (I35, the one blocker) can be dropped? Must the hand-expanded `gauss3d` Laplacian be reproduced bit-for-bit, or is SymPy regeneration or autodiff acceptable?
5. **Algorithmic changes.** May a port change clearly wasteful steps (boundary-only evaluation in `update_boundary`, hoisted object lists, compact B/N tables, gated error field, coefficient deduplication)? If so, C++ versus Python timings are not comparable unless the C++ is fixed too. The report therefore compares against the fixed-up C++, a frozen list of exactly four changes: (1) hoist the object-BC gather lists out of the per-call path; (2) compact B and N tables with no full-height row pointers; (3) manufactured-solution evaluation once per step and only where needed, with time factors hoisted (fill_source and update_boundary); (4) packed [D | Rx | Ry | Rz] buffers. Gating the error field and deduplicating coefficients are not on the list.
6. **Short fragments.** `n_interior = rows - L.rows - R.rows` can be <= 0 for thin gaps between bodies. Is that excluded by construction, or must a DSL support merged left/right closures? Fully solid pencils are asserted unsupported. Should a new design handle them, along with multiple or overlapping shapes, concave bodies and planar objects (where `add_interp_row` is marked "no good")?
7. **Scheme coverage.** E4_1, E4u_1, E6u_1 and E8u_1 return an empty `interp_info`, so floating objects in 2D and 3D appear unsupported for them. Is that intended? Object Neumann is never assembled (`nu.R` never read). Unsupported by design?
8. **Trusted reference on R points.** `t-laplacian` currently fails on R-point values for E2 with floating objects, which is exactly the Bf/Br path. A reimplementation needs a trusted reference for parity there.
9. **Debug overlap assert.** The assert in `block::builder::to_block` (`block.hpp:230-246`) appears wrong for interleaved strided lines. Latent bug or dead code?
10. **R layout.** Keep the per-direction (slow, fast, t) order and the four-buffer split, or move to one global R space or one packed D|Rx|Ry|Rz array per field with explicit permutation tables?
11. **Fluid-set consistency.** The fluid set is derived only from the highest-direction fragments. Can x, y and z fragments disagree about solid points in grid-tangent cases, so that an operator writes rows outside the fluid set?
12. **Slot binding.** The RHS graph is bound to slots 1 and 3, and RK stages reuse them through copies. Does a functional Python RHS (state in, state out) drop in, or does any caller rely on in-place mutation of specific slots?
13. **2D R geometry in output.** Does writing (z, y, x) into an XYZ geometry (`field_data.cpp:16-17`) misplace cut points in ParaView? Should a Python writer reproduce or fix it?
14. **Euler design.** Flux-form derivatives per component (15 applications) or another form; wall BC formulation at R points (slip via normals, characteristic); dissipation or filter; adaptive dt; multi-component integrator support. None of this is specified by the scaffold.
15. **Where the shipped step goes, and what the fixed-up C++ changes.** The shipped heat.lua step (0.856 ms at 51x51, 1 thread, native) is 28 to 37% `fill_source`, 11 to 15% `update_boundary`, 10 to 13% the four RHS graph submits and 3 to 4% stats (Part IV Section E; container: 0.90 ms with 59%, 15%, 10% and 4%, Section C). At 65^3 on four threads `fill_source` is 49% of a 19.96 ms step (container: 72% of 37.9 ms at 8 threads). Today's step time therefore measures host manufactured-solution evaluation through virtual calls, not the discretization. Change (3) of the fixed-up list removes most of that work. The estimate is 131 us at 51x51 (1 thread) and 6.0 ms at 65^3 (4 threads; container about 0.15 ms and about 5.5 ms), which leaves the RHS graph (21.0 us and 0.96 ms per stage) as the largest remaining cost. Changes (1), (2) and (4) then shrink the RHS and boundary passes further, by an amount not estimated. Open: should the fixed-up C++ be built and measured on a tagged branch before any Python timing is compared with it? Until it is, the fixed-up bar is an estimate, and a Python port that beats the shipped step may only be beating the MMS overhead.
