#include "tables.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace ccs
{

packed_layout::packed_layout(const mesh& m)
    : extents{m.extents().extents},
      n_D{m.size()},
      n_R{(integer)m.Rx().size(), (integer)m.Ry().size(), (integer)m.Rz().size()},
      offsets{0, n_D, n_D + n_R[0], n_D + n_R[0] + n_R[1]},
      n_V{n_D + n_R[0] + n_R[1] + n_R[2]}
{
}

integer packed_layout::stride(int dir) const
{
    integer s = 1;
    for (int i = dir + 1; i < 3; ++i) s *= extents[i];
    return s;
}

std::vector<int32_t> block_line_map(const mesh& m, int dir, const bcs::Grid& grid_bcs)
{
    std::vector<int32_t> out;
    if (m.extents()[dir] < 2) return out;
    const auto& lines = m.lines(dir);
    for (int32_t i = 0; i < (int32_t)lines.size(); ++i)
        if (!m.dirichlet_line(lines[i].start.mesh_coordinate, dir, grid_bcs))
            out.push_back(i);
    return out;
}

namespace
{

void append_csr(const matrix::csr& M,
                integer row_offset,
                integer col_offset,
                uint8_t part,
                std::vector<coo_entry>& out)
{
    for (integer r = 0; r < M.rows(); ++r) {
        auto cols = M.column_indices(r);
        auto vals = M.column_coefficients(r);
        for (std::size_t k = 0; k < cols.size(); ++k)
            out.push_back({row_offset + r, col_offset + cols[k], vals[k], part, -1});
    }
}

} // namespace

derivative_coo extract_coo(const derivative& d,
                           const packed_layout& L,
                           std::span<const int32_t> block_lines)
{
    derivative_coo out;
    const auto mv = d.matrices();

    const auto& meta = mv.O.metadata_view();
    const auto& coef = mv.O.coefficients_view();
    const int n_blocks = mv.O.num_lines();
    if (!block_lines.empty() && (int)block_lines.size() != n_blocks)
        throw std::invalid_argument("extract_coo: block_lines does not match the O matrix");

    for (int b = 0; b < n_blocks; ++b) {
        const auto m = meta(b);
        const int32_t line = block_lines.empty() ? -1 : block_lines[b];

        for (int r = 0; r < m.left_rows; ++r) {
            const integer row = m.row_offset + (integer)r * m.stride;
            for (int j = 0; j < m.left_cols; ++j)
                out.A.push_back({row,
                                 m.col_offset + (integer)j * m.stride,
                                 coef(m.left_coeff_offset + r * m.left_cols + j),
                                 coo_part::O_left,
                                 line});
        }

        const int half = m.stencil_width / 2;
        for (int r = 0; r < m.interior_rows; ++r) {
            const integer row = m.row_offset + (integer)(m.left_rows + r) * m.stride;
            for (int j = 0; j < m.stencil_width; ++j)
                out.A.push_back({row,
                                 row + (integer)(j - half) * m.stride,
                                 coef(m.interior_coeff_offset + j),
                                 coo_part::O_interior,
                                 line});
        }

        for (int r = 0; r < m.right_rows; ++r) {
            const integer row =
                m.row_offset + (integer)(m.left_rows + m.interior_rows + r) * m.stride;
            for (int j = 0; j < m.right_cols; ++j)
                out.A.push_back({row,
                                 m.right_col_offset + (integer)j * m.stride,
                                 coef(m.right_coeff_offset + r * m.right_cols + j),
                                 coo_part::O_right,
                                 line});
        }
    }

    // B: rows in D, columns in R(dir)
    if (mv.B.size() > 0) {
        const int dir = d.direction();
        if (dir < 0 || dir > 2)
            throw std::logic_error("extract_coo: derivative has B entries but no direction");
        append_csr(mv.B, 0, L.off_R(dir), coo_part::B, out.A);
    }

    // Bf_r: rows in R(r), columns in D.  Br_r: rows and columns in R(r).
    const matrix::csr* Bf[3] = {&mv.Bfx, &mv.Bfy, &mv.Bfz};
    const matrix::csr* Br[3] = {&mv.Brx, &mv.Bry, &mv.Brz};
    for (int r = 0; r < 3; ++r) {
        append_csr(*Bf[r], L.off_R(r), 0, coo_part::Bf(r), out.A);
        append_csr(*Br[r], L.off_R(r), L.off_R(r), coo_part::Br(r), out.A);
    }

    // N: rows in D, columns in the separate Neumann data buffer (D indices)
    append_csr(mv.N, 0, 0, coo_part::N, out.N);

    return out;
}

compact_tables build_compact_tables(const laplacian& lap,
                                    const packed_layout& L,
                                    std::span<const int> fluid_idx,
                                    std::span<const int> dirichlet_face_idx)
{
    if (L.n_V > (integer)std::numeric_limits<int32_t>::max())
        throw std::length_error("build_compact_tables: V does not fit int32 columns");

    compact_tables T;
    T.layout = L;
    T.role.assign(L.n_D, 0);

    struct entry {
        int32_t row;
        int32_t col;
        real val;
    };
    std::vector<entry> exc;

    for (int d = 0; d < 3; ++d) {
        const derivative& der = lap.component(d);
        const auto mv = der.matrices();
        const auto& meta = mv.O.metadata_view();
        const auto& coef = mv.O.coefficients_view();
        const int n_blocks = mv.O.num_lines();

        // every line of a direction shares the same interior coefficients
        if (n_blocks > 0) {
            const auto m0 = meta(0);
            T.width = std::max(T.width, m0.stencil_width);
            const real* c = coef.data() + m0.interior_coeff_offset;
            T.taps[d].assign(c, c + m0.stencil_width);
        }

        for (int b = 0; b < n_blocks; ++b) {
            const auto m = meta(b);
            assert(m.stencil_width == T.width);
            for (int r = 0; r < m.interior_rows; ++r) {
                const integer row = m.row_offset + (integer)(m.left_rows + r) * m.stride;
                T.role[row] |= role_bit::interior(d);
            }
        }

        auto coo = extract_coo(der, L, {});
        for (const auto& e : coo.A)
            if (e.part != coo_part::O_interior)
                exc.push_back({(int32_t)e.row, (int32_t)e.col, e.val});
    }

    for (int d = 0; d < 3; ++d)
        if (T.taps[d].empty()) T.taps[d].assign(T.width, 0.0);

    for (int i : fluid_idx) T.role[i] |= role_bit::fluid;
    for (int i : dirichlet_face_idx) T.role[i] |= role_bit::dirichlet_face;

    // CSR: rows ascending and unique, duplicates summed in storage order
    std::stable_sort(exc.begin(), exc.end(), [](const entry& a, const entry& b) {
        return std::tie(a.row, a.col) < std::tie(b.row, b.col);
    });

    for (std::size_t k = 0; k < exc.size();) {
        std::size_t j = k;
        real sum = 0.0;
        while (j < exc.size() && exc[j].row == exc[k].row && exc[j].col == exc[k].col)
            sum += exc[j++].val;

        if (T.exc_rows.empty() || T.exc_rows.back() != exc[k].row) {
            T.exc_rows.push_back(exc[k].row);
            T.exc_rowptr.push_back((integer)T.exc_col.size());
        }
        T.exc_col.push_back(exc[k].col);
        T.exc_val.push_back(sum);
        k = j;
    }
    T.exc_rowptr.push_back((integer)T.exc_col.size());

    return T;
}

void apply_structured(const compact_tables& T, std::span<const real> u, std::span<real> out)
{
    const auto& L = T.layout;
    if ((integer)u.size() != L.n_V || (integer)out.size() != L.n_V)
        throw std::invalid_argument("apply_structured: vector length is not n_V");

    const int w = T.width;
    const int half = w / 2;
    const integer strides[3] = {L.stride(0), L.stride(1), L.stride(2)};
    const real* taps[3] = {T.taps[0].data(), T.taps[1].data(), T.taps[2].data()};

    for (integer i = 0; i < L.n_D; ++i) {
        const uint8_t bits = T.role[i];
        real acc = 0.0;
        for (int d = 0; d < 3; ++d) {
            if (!(bits & role_bit::interior(d))) continue;
            const integer s = strides[d];
            for (int j = 0; j < w; ++j) acc += taps[d][j] * u[i + (j - half) * s];
        }
        out[i] = acc;
    }
    std::fill(out.begin() + L.n_D, out.end(), 0.0);

    for (std::size_t k = 0; k < T.exc_rows.size(); ++k) {
        real acc = 0.0;
        for (integer p = T.exc_rowptr[k]; p < T.exc_rowptr[k + 1]; ++p)
            acc += T.exc_val[p] * u[T.exc_col[p]];
        out[T.exc_rows[k]] += acc;
    }
}

} // namespace ccs
