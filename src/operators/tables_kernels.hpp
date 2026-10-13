#pragma once

// Kokkos kernels over the compact tables of tables.hpp for the heat system: the
// structured RHS (axis taps on the rows flagged interior, one exception CSR for
// every other entry, the diffusivity folded into both, the source added where the
// live heat RHS adds it) and the fused RK stage; primitives B3, B5 and B11 of
// docs/python-dsl-survey/03-dsl-designs.md Section 9.1. Every vector is packed,
// V = [D | Rx | Ry | Rz] (docs/reference/pyshoccs-export.md Section 1).
//
// Row ownership: a V row is written by exactly one thread of one launch, so no
// atomics are needed. D rows without an exception row belong to the point launch,
// rows with an exception row (D or R) and R rows without one belong to the row
// launch; the one-launch form runs both index sets in a single parallel_for.

#include "kokkos_types.hpp"
#include "tables.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ccs
{

namespace role_bit
{
// Kernel-side bits of device_tables::role, which runs over all of V. On an R
// point, bit 4 (dirichlet_face) marks a Dirichlet-object point.
constexpr uint8_t exception = 1u << 5; // the row has an exception row
constexpr uint8_t source = 1u << 6;    // the source is added on this row
} // namespace role_bit

// The tables of one heat RHS on the device: k folded into taps and exception
// values, one role byte per V row, exception rows on Dirichlet rows dropped
// (heat::build_rhs_graph zeroes those rows last, whatever they hold).
struct device_tables {
    static constexpr int max_width = 11;

    int n_D{};
    int n_V{};
    int n_exc{};
    int n_exc_dropped{};
    int width{};
    int3 extents{};
    std::array<int, 3> stride{};
    std::array<real, 3 * max_width> taps{}; // k * taps[d][j] at d * width + j

    device_view<uint8_t*> role;       // [n_V]
    device_view<int32_t*> exc_rows;   // [n_exc], ascending V rows
    device_view<int32_t*> exc_rowptr; // [n_exc + 1]
    device_view<int32_t*> exc_col;    // V columns
    device_view<real*> exc_val;       // k * value

    int n_R() const { return n_V - n_D; }
};

inline device_tables make_device_tables(const compact_tables& T,
                                        const mesh& m,
                                        const bcs::Object& object_bcs,
                                        real k)
{
    const packed_layout& L = T.layout;
    if (T.width > device_tables::max_width || T.width % 2 == 0)
        throw std::invalid_argument("make_device_tables: unsupported stencil width");

    device_tables D;
    D.n_D = (int)L.n_D;
    D.n_V = (int)L.n_V;
    D.width = T.width;
    D.extents = L.extents;
    for (int d = 0; d < 3; ++d) {
        D.stride[d] = (int)L.stride(d);
        for (int j = 0; j < T.width; ++j) D.taps[d * T.width + j] = k * T.taps[d][j];
    }

    // role over V: the Section 4 bits on D, the Dirichlet and source flags on R
    std::vector<uint8_t> role(L.n_V, 0);
    for (integer i = 0; i < L.n_D; ++i) {
        uint8_t bits = T.role[i];
        if ((bits & role_bit::fluid) && !(bits & role_bit::dirichlet_face))
            bits |= role_bit::source;
        role[i] = bits;
    }
    for (int d = 0; d < 3; ++d) {
        const auto R = m.R(d);
        for (std::size_t r = 0; r < R.size(); ++r)
            role[L.off_R(d) + r] |= object_bcs[R[r].shape_id] == bcs::Dirichlet
                                        ? role_bit::dirichlet_face
                                        : role_bit::source;
    }

    // exception CSR with k folded in and the Dirichlet rows left out
    std::vector<int32_t> rows, rowptr{0}, col;
    std::vector<real> val;
    for (std::size_t r = 0; r < T.exc_rows.size(); ++r) {
        const int32_t row = T.exc_rows[r];
        if (role[row] & role_bit::dirichlet_face) {
            ++D.n_exc_dropped;
            continue;
        }
        role[row] |= role_bit::exception;
        rows.push_back(row);
        for (integer p = T.exc_rowptr[r]; p < T.exc_rowptr[r + 1]; ++p) {
            col.push_back(T.exc_col[p]);
            val.push_back(k * T.exc_val[p]);
        }
        rowptr.push_back((int32_t)col.size());
    }
    D.n_exc = (int)rows.size();

    auto upload = [](const char* label, const auto& v) {
        using T_ = std::remove_cv_t<typename std::decay_t<decltype(v)>::value_type>;
        device_view<T_*> out(label, v.size());
        auto h = Kokkos::create_mirror_view(out);
        for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
        Kokkos::deep_copy(out, h);
        return out;
    };
    D.role = upload("tables_role", role);
    D.exc_rows = upload("tables_exc_rows", rows);
    D.exc_rowptr = upload("tables_exc_rowptr", rowptr);
    D.exc_col = upload("tables_exc_col", col);
    D.exc_val = upload("tables_exc_val", val);
    return D;
}

// How the two index sets are launched.
struct launch_config {
    enum kind_t {
        two,        // RangePolicy over D, then RangePolicy over the rows
        two_md,     // MDRangePolicy<Rank<3>> over (i, j, k), then the rows
        one,        // one RangePolicy over both index sets
        pencil,     // RangePolicy over the fast-axis lines of D, each walked in
                    // runs of equal role bytes with a branch-free inner loop,
                    // then the rows
        pencil_one, // the pencils and the rows in one RangePolicy
    };
    kind_t kind = pencil;
    std::array<int, 3> tile{2, 4, 64}; // two_md only; clipped to the extents
};

namespace tables_detail
{

struct points_tag {
};
struct rows_tag {
};
struct all_tag {
};
struct md_tag {
};
struct pencil_tag {
};
struct pencil_all_tag {
};

// out[row] = r
struct assign_epilogue {
    real* out;
    KOKKOS_INLINE_FUNCTION void operator()(int i, real r) const { out[i] = r; }
};

// The RK epilogue: acc = (first ? 0 : acc) + w r; u_next = u0 + c r, or on the
// last stage the completed step u0 + acc (skipped when u_next is null).
struct rk_epilogue {
    const real* u0;
    real* acc;
    real* u_next;
    real w;
    real c;
    bool first;
    bool last;

    KOKKOS_INLINE_FUNCTION void operator()(int i, real r) const
    {
        const real a = (first ? 0.0 : acc[i]) + w * r;
        acc[i] = a;
        if (!last)
            u_next[i] = u0[i] + c * r;
        else if (u_next)
            u_next[i] = u0[i] + a;
    }
};

template <int W, typename Epilogue>
struct kernel {
    static constexpr int half = W / 2;

    int n_D;
    int n_exc;
    int ny;
    int nz;
    int pencil_len; // points per fast-axis line (nz, or ny in 2D)
    int stride[3];
    real taps[3][W];
    const uint8_t* role;
    const int32_t* exc_rows;
    const int32_t* exc_rowptr;
    const int32_t* exc_col;
    const real* exc_val;
    const real* u;
    const real* src;
    Epilogue epi;

    kernel(const device_tables& T, const real* u_, const real* src_, Epilogue e)
        : n_D{T.n_D},
          n_exc{T.n_exc},
          ny{T.extents[1]},
          nz{T.extents[2]},
          pencil_len{T.extents[2] > 1 ? T.extents[2] : T.extents[1]},
          stride{T.stride[0], T.stride[1], T.stride[2]},
          role{T.role.data()},
          exc_rows{T.exc_rows.data()},
          exc_rowptr{T.exc_rowptr.data()},
          exc_col{T.exc_col.data()},
          exc_val{T.exc_val.data()},
          u{u_},
          src{src_},
          epi{e}
    {
        for (int d = 0; d < 3; ++d)
            for (int j = 0; j < W; ++j) taps[d][j] = T.taps[d * W + j];
    }

    KOKKOS_INLINE_FUNCTION real axis(int i, uint8_t bits) const
    {
        real r = 0.0;
        for (int d = 0; d < 3; ++d) {
            if (!(bits & role_bit::interior(d))) continue;
            const int s = stride[d];
            real a = 0.0;
            for (int j = 0; j < W; ++j) a += taps[d][j] * u[i + (j - half) * s];
            r += a;
        }
        return r;
    }

    // a D row without an exception row
    KOKKOS_INLINE_FUNCTION void point(int i) const
    {
        const uint8_t bits = role[i];
        if (bits & role_bit::exception) return;
        real r = axis(i, bits);
        if (bits & role_bit::source) r += src[i];
        epi(i, r);
    }

    // k < n_exc: exception row k (taps included for a D row); then the R rows
    // without an exception row (source or 0)
    KOKKOS_INLINE_FUNCTION void row(int k) const
    {
        if (k < n_exc) {
            const int i = exc_rows[k];
            real r = 0.0;
            for (int p = exc_rowptr[k]; p < exc_rowptr[k + 1]; ++p)
                r += exc_val[p] * u[exc_col[p]];
            const uint8_t bits = role[i];
            if (i < n_D) r += axis(i, bits);
            if (bits & role_bit::source) r += src[i];
            epi(i, r);
        } else {
            const int i = n_D + (k - n_exc);
            const uint8_t bits = role[i];
            if (bits & role_bit::exception) return;
            epi(i, (bits & role_bit::source) ? src[i] : 0.0);
        }
    }

    // the points [i0, i1) of one line, all with the interior bits B and the same
    // source flag: no branch in the loop body
    template <int B>
    KOKKOS_INLINE_FUNCTION void run(int i0, int i1, bool add_src) const
    {
        for (int i = i0; i < i1; ++i) {
            real r = 0.0;
            if constexpr (B & 1) {
                real a = 0.0;
                for (int j = 0; j < W; ++j) a += taps[0][j] * u[i + (j - half) * stride[0]];
                r += a;
            }
            if constexpr (B & 2) {
                real a = 0.0;
                for (int j = 0; j < W; ++j) a += taps[1][j] * u[i + (j - half) * stride[1]];
                r += a;
            }
            if constexpr (B & 4) {
                real a = 0.0;
                for (int j = 0; j < W; ++j) a += taps[2][j] * u[i + (j - half) * stride[2]];
                r += a;
            }
            if (add_src) r += src[i];
            epi(i, r);
        }
    }

    // one fast-axis line of D, in runs of equal role bytes
    KOKKOS_INLINE_FUNCTION void pencil(int p) const
    {
        const int base = p * pencil_len;
        int k = 0;
        while (k < pencil_len) {
            const uint8_t bits = role[base + k];
            int k1 = k + 1;
            while (k1 < pencil_len && role[base + k1] == bits) ++k1;
            if (!(bits & role_bit::exception)) {
                const int i0 = base + k;
                const int i1 = base + k1;
                const bool add_src = bits & role_bit::source;
                switch (bits & role_bit::any_interior) {
                case 0:
                    run<0>(i0, i1, add_src);
                    break;
                case 1:
                    run<1>(i0, i1, add_src);
                    break;
                case 2:
                    run<2>(i0, i1, add_src);
                    break;
                case 3:
                    run<3>(i0, i1, add_src);
                    break;
                case 4:
                    run<4>(i0, i1, add_src);
                    break;
                case 5:
                    run<5>(i0, i1, add_src);
                    break;
                case 6:
                    run<6>(i0, i1, add_src);
                    break;
                default:
                    run<7>(i0, i1, add_src);
                    break;
                }
            }
            k = k1;
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()(const points_tag&, int i) const { point(i); }
    KOKKOS_INLINE_FUNCTION void operator()(const pencil_tag&, int p) const { pencil(p); }
    KOKKOS_INLINE_FUNCTION void operator()(const pencil_all_tag&, int p) const
    {
        const int n_pencils = n_D / pencil_len;
        if (p < n_pencils)
            pencil(p);
        else
            row(p - n_pencils);
    }
    KOKKOS_INLINE_FUNCTION void operator()(const rows_tag&, int k) const { row(k); }
    KOKKOS_INLINE_FUNCTION void operator()(const all_tag&, int i) const
    {
        if (i < n_D)
            point(i);
        else
            row(i - n_D);
    }
    KOKKOS_INLINE_FUNCTION void operator()(const md_tag&, int i, int j, int k) const
    {
        point((i * ny + j) * nz + k);
    }
};

template <int W, typename Epilogue>
void run(const device_tables& T, const kernel<W, Epilogue>& f, const launch_config& cfg)
{
    const int n_rows = T.n_exc + T.n_R();
    switch (cfg.kind) {
    case launch_config::one:
        Kokkos::parallel_for("tables_all",
                             Kokkos::RangePolicy<execution_space, all_tag>(0, T.n_D + n_rows),
                             f);
        break;
    case launch_config::two:
        Kokkos::parallel_for(
            "tables_points", Kokkos::RangePolicy<execution_space, points_tag>(0, T.n_D), f);
        Kokkos::parallel_for(
            "tables_rows", Kokkos::RangePolicy<execution_space, rows_tag>(0, n_rows), f);
        break;
    case launch_config::pencil:
        Kokkos::parallel_for(
            "tables_pencils",
            Kokkos::RangePolicy<execution_space, pencil_tag>(0, T.n_D / f.pencil_len),
            f);
        Kokkos::parallel_for(
            "tables_rows", Kokkos::RangePolicy<execution_space, rows_tag>(0, n_rows), f);
        break;
    case launch_config::pencil_one:
        Kokkos::parallel_for("tables_pencils_all",
                             Kokkos::RangePolicy<execution_space, pencil_all_tag>(
                                 0, T.n_D / f.pencil_len + n_rows),
                             f);
        break;
    case launch_config::two_md: {
        const int3& e = T.extents;
        using policy = Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<3>, md_tag>;
        Kokkos::parallel_for("tables_points_md",
                             policy({0, 0, 0},
                                    {e[0], e[1], e[2]},
                                    {std::min(cfg.tile[0], e[0]),
                                     std::min(cfg.tile[1], e[1]),
                                     std::min(cfg.tile[2], e[2])}),
                             f);
        Kokkos::parallel_for(
            "tables_rows", Kokkos::RangePolicy<execution_space, rows_tag>(0, n_rows), f);
        break;
    }
    }
    Kokkos::fence();
}

template <typename Epilogue>
void apply(const device_tables& T,
           const real* u,
           const real* src,
           Epilogue epi,
           const launch_config& cfg)
{
    switch (T.width) {
    case 3:
        run(T, kernel<3, Epilogue>(T, u, src, epi), cfg);
        break;
    case 5:
        run(T, kernel<5, Epilogue>(T, u, src, epi), cfg);
        break;
    case 7:
        run(T, kernel<7, Epilogue>(T, u, src, epi), cfg);
        break;
    case 9:
        run(T, kernel<9, Epilogue>(T, u, src, epi), cfg);
        break;
    case 11:
        run(T, kernel<11, Epilogue>(T, u, src, epi), cfg);
        break;
    default:
        throw std::invalid_argument("tables_kernels: unsupported stencil width");
    }
}

} // namespace tables_detail

// out = the heat RHS of u: k lap(u) on every row with entries plus src on fluid
// D rows away from Dirichlet faces and on non-Dirichlet R rows, 0 elsewhere. All
// three vectors have n_V entries; out may not alias u.
inline void structured_rhs(const device_tables& T,
                           const real* u,
                           const real* src,
                           real* out,
                           const launch_config& cfg = {})
{
    tables_detail::apply(T, u, src, tables_detail::assign_epilogue{out}, cfg);
}

struct rk_stage {
    real w;     // weight of this stage's rhs in the accumulator (dt * b_s)
    real c;     // coefficient of this stage's rhs in the next stage input (dt * a_{s+1})
    bool first; // acc = w r instead of acc += w r
    bool last;  // no next stage: u_next (when given) receives u0 + acc
};

// One RK stage fused with the RHS: r = rhs(u_stage); acc (+)= w r; u_next = u0 +
// c r, or the completed step u0 + acc on the last stage. u_next may not alias
// u_stage; the other buffers are distinct n_V vectors.
inline void fused_rk_stage(const device_tables& T,
                           const real* u_stage,
                           const real* src,
                           const real* u0,
                           real* acc,
                           real* u_next,
                           rk_stage s,
                           const launch_config& cfg = {})
{
    tables_detail::apply(
        T,
        u_stage,
        src,
        tables_detail::rk_epilogue{u0, acc, u_next, s.w, s.c, s.first, s.last},
        cfg);
}

// u[idx] = values[idx] over an index list (the Dirichlet refresh between stages).
inline void assign_at(const device_view<const int*>& idx, const real* values, real* u)
{
    Kokkos::parallel_for(
        "tables_assign_at",
        Kokkos::RangePolicy<execution_space>(0, (int)idx.extent(0)),
        KOKKOS_LAMBDA(int i) {
            const int j = idx(i);
            u[j] = values[j];
        });
    Kokkos::fence();
}

} // namespace ccs
