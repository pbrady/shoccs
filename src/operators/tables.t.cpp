#include "tables.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sol/sol.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <set>
#include <vector>

// Custom main: Kokkos must be initialized before any parallel_for.
int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

using namespace ccs;

namespace
{

// The two floating-object setups of laplacian.t.cpp.
constexpr const char* floating_3d = R"(
    simulation = {
        mesh = {
            index_extents = {25, 26, 27},
            domain_bounds = { min = {0.1, 0.2, 0.3}, max = {1, 2, 2.2} }
        },
        domain_boundaries = {
            xmin = "dirichlet", ymin = "neumann", ymax = "neumann", zmax = "dirichlet"
        },
        shapes = {
            { type = "sphere", center = {0.45, 1.011, 1.31}, radius = 0.25,
              boundary_condition = "floating" }
        },
        scheme = { order = 2, type = "E2" }
    }
)";

constexpr const char* floating_2d = R"(
    simulation = {
        mesh = {
            index_extents = {25, 26},
            domain_bounds = { min = {0.1, 0.2}, max = {1, 2} }
        },
        domain_boundaries = { xmin = "dirichlet", ymin = "neumann", ymax = "neumann" },
        shapes = {
            { type = "sphere", center = {0.45, 1.011}, radius = 0.25,
              boundary_condition = "floating" }
        },
        scheme = { order = 2, type = "E2" }
    }
)";

struct problem {
    mesh m;
    bcs::Grid grid_bcs;
    bcs::Object object_bcs;
    stencil st;
};

problem make_problem(const char* script)
{
    sol::state lua;
    lua.script(script);
    auto m_opt = mesh::from_lua(lua["simulation"]);
    REQUIRE(!!m_opt);
    auto bc_opt = bcs::from_lua(lua["simulation"], m_opt->extents());
    REQUIRE(!!bc_opt);
    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    return {MOVE(*m_opt), MOVE(bc_opt->first), MOVE(bc_opt->second), MOVE(*st_opt)};
}

std::vector<real> random_vector(std::size_t n, std::uint64_t seed)
{
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<real> dist(-1.0, 1.0);
    std::vector<real> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

scalar_view view_of(std::span<const real> V, const packed_layout& L)
{
    return {V.subspan(0, L.n_D),
            V.subspan(L.off_R(0), L.n_R[0]),
            V.subspan(L.off_R(1), L.n_R[1]),
            V.subspan(L.off_R(2), L.n_R[2])};
}

scalar_span span_of(std::span<real> V, const packed_layout& L)
{
    return {V.subspan(0, L.n_D),
            V.subspan(L.off_R(0), L.n_R[0]),
            V.subspan(L.off_R(1), L.n_R[1]),
            V.subspan(L.off_R(2), L.n_R[2])};
}

std::vector<int> fluid_indices(const mesh& m)
{
    const auto& fd = m.fluid_desc();
    std::vector<int> out(fd.count());
    for (int i = 0; i < fd.count(); ++i) out[i] = fd.element(i);
    return out;
}

std::vector<int> dirichlet_faces(const mesh& m, const bcs::Grid& g)
{
    std::vector<int> out;
    for_each_grid_bc_desc<bcs::Dirichlet>(g, m.extents(), [&](auto desc) {
        for (int i = 0; i < desc.count(); ++i) out.push_back(desc.element(i));
    });
    return out;
}

real max_abs(std::span<const real> v)
{
    real r = 0;
    for (auto x : v) r = std::max(r, std::abs(x));
    return r;
}

// Rows are compared on the scale of their own rounding error,
// |row|_1 * max|u|; a row without entries must agree exactly.
struct row_check {
    real worst_ratio = 0.0;
    integer empty_mismatches = 0;
};

row_check compare_rows(std::span<const real> got,
                       std::span<const real> want,
                       std::span<const real> rownorm,
                       real umax)
{
    row_check c;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const real diff = std::abs(got[i] - want[i]);
        const real scale = rownorm[i] * umax;
        if (scale > 0)
            c.worst_ratio = std::max(c.worst_ratio, diff / scale);
        else if (diff != 0)
            ++c.empty_mismatches;
    }
    return c;
}

void check_problem(const char* script)
{
    problem pb = make_problem(script);
    const mesh& m = pb.m;
    const laplacian lap{m, pb.st, pb.grid_bcs, pb.object_bcs};
    const packed_layout L{m};

    REQUIRE(L.n_D == m.size());
    REQUIRE(L.n_R[0] == (integer)m.Rx().size());
    REQUIRE(L.n_R[1] == (integer)m.Ry().size());
    REQUIRE(L.n_R[2] == (integer)m.Rz().size());
    REQUIRE(L.n_V == L.n_D + L.n_R[0] + L.n_R[1] + L.n_R[2]);
    REQUIRE(L.off_R(2) + L.n_R[2] == L.n_V);
    for (int d = 0; d < 3; ++d) REQUIRE(L.stride(d) == m.stride(d));

    const auto u = random_vector(L.n_V, 1);
    const auto nu = random_vector(L.n_V, 2);
    const std::vector<real> zero_nu(L.n_V, 0.0);
    const real umax = std::max(max_abs(u), max_abs(nu));

    // eager Laplacian with and without Neumann data
    std::vector<real> du(L.n_V), du0(L.n_V);
    {
        scalar_span s = span_of(du, L);
        s = lap(view_of(u, L), view_of(nu, L));
    }
    {
        scalar_span s = span_of(du0, L);
        s = lap(view_of(u, L), view_of(zero_nu, L));
    }

    // (a) the COO entries, summed, reproduce the eager operator
    std::vector<real> y(L.n_V, 0.0), y0(L.n_V, 0.0);
    std::vector<real> rownorm(L.n_V, 0.0), rownorm0(L.n_V, 0.0);
    std::set<integer> rows_with_entries;
    integer bad_index = 0, bad_line = 0, n_entries = 0, n_neumann = 0;
    std::array<integer, 3> interior_entries{};

    for (int d = 0; d < 3; ++d) {
        const derivative& der = lap.component(d);
        const auto bl = block_line_map(m, d, pb.grid_bcs);
        REQUIRE((int)bl.size() == der.matrices().O.num_lines());
        if (m.extents()[d] > 1) REQUIRE(der.direction() == d);

        const auto coo = extract_coo(der, L, bl);
        const integer n_lines = (integer)m.lines(d).size();
        for (const auto& e : coo.A) {
            if (e.row < 0 || e.row >= L.n_V || e.col < 0 || e.col >= L.n_V) ++bad_index;
            const bool is_O = e.part <= coo_part::O_right;
            if (is_O ? (e.line < 0 || e.line >= n_lines) : e.line != -1) ++bad_line;
            if (e.part == coo_part::O_interior) ++interior_entries[d];
            y[e.row] += e.val * u[e.col];
            y0[e.row] += e.val * u[e.col];
            rownorm[e.row] += std::abs(e.val);
            rownorm0[e.row] += std::abs(e.val);
            rows_with_entries.insert(e.row);
            ++n_entries;
        }
        for (const auto& e : coo.N) {
            if (e.row < 0 || e.row >= L.n_D || e.col < 0 || e.col >= L.n_D) ++bad_index;
            if (e.part != coo_part::N) ++bad_line;
            y[e.row] += e.val * nu[e.col];
            rownorm[e.row] += std::abs(e.val);
            ++n_neumann;
        }
    }
    REQUIRE(bad_index == 0);
    REQUIRE(bad_line == 0);
    REQUIRE(n_entries > 0);
    REQUIRE(n_neumann > 0); // both setups have Neumann walls

    const auto ca = compare_rows(y, du, rownorm, umax);
    CAPTURE(ca.worst_ratio, ca.empty_mismatches);
    REQUIRE(ca.empty_mismatches == 0);
    REQUIRE(ca.worst_ratio <= 1e-13);

    // (b) the structured form reproduces the same operator (nu = 0)
    const auto fluid = fluid_indices(m);
    const auto dfaces = dirichlet_faces(m, pb.grid_bcs);
    const compact_tables T = build_compact_tables(lap, L, fluid, dfaces);

    std::vector<real> z(L.n_V);
    apply_structured(T, u, z);
    const auto cb = compare_rows(z, du0, rownorm0, umax);
    CAPTURE(cb.worst_ratio, cb.empty_mismatches);
    REQUIRE(cb.empty_mismatches == 0);
    REQUIRE(cb.worst_ratio <= 1e-14);

    // taps are the stencil's interior coefficients, zero where there is no line
    REQUIRE(T.width == 2 * pb.st.query_max().p + 1);
    for (int d = 0; d < 3; ++d) {
        REQUIRE((int)T.taps[d].size() == T.width);
        if (m.extents()[d] > 1) {
            std::vector<real> c(T.width);
            pb.st.interior(m.h(d), c);
            REQUIRE(T.taps[d] == c);
        } else {
            REQUIRE(std::ranges::all_of(T.taps[d], [](real v) { return v == 0.0; }));
        }
    }

    // (c) exception rows plus interior rows account for every row with entries
    std::set<integer> covered;
    std::array<integer, 3> interior_rows{};
    for (integer i = 0; i < L.n_D; ++i) {
        for (int d = 0; d < 3; ++d)
            if (T.role[i] & role_bit::interior(d)) ++interior_rows[d];
        if (T.role[i] & role_bit::any_interior) covered.insert(i);
    }
    for (auto r : T.exc_rows) covered.insert(r);
    REQUIRE(covered == rows_with_entries);
    for (int d = 0; d < 3; ++d) REQUIRE(interior_rows[d] * T.width == interior_entries[d]);

    // exception CSR structure
    REQUIRE(std::adjacent_find(T.exc_rows.begin(), T.exc_rows.end(), std::greater_equal<>()) ==
            T.exc_rows.end());
    REQUIRE(T.exc_rowptr.size() == T.exc_rows.size() + 1);
    REQUIRE(T.exc_rowptr.front() == 0);
    REQUIRE(T.exc_rowptr.back() == (integer)T.exc_col.size());
    REQUIRE(T.exc_val.size() == T.exc_col.size());
    for (std::size_t k = 0; k < T.exc_rows.size(); ++k)
        REQUIRE(T.exc_rowptr[k + 1] > T.exc_rowptr[k]);
    REQUIRE(std::ranges::all_of(T.exc_col, [&](int32_t c) { return c >= 0 && c < L.n_V; }));
    REQUIRE(std::ranges::all_of(T.exc_rows, [&](int32_t r) { return r >= 0 && r < L.n_V; }));
    // columns within a row are unique (duplicates were summed)
    for (std::size_t k = 0; k < T.exc_rows.size(); ++k) {
        auto first = T.exc_col.begin() + T.exc_rowptr[k];
        auto last = T.exc_col.begin() + T.exc_rowptr[k + 1];
        REQUIRE(std::adjacent_find(first, last, std::greater_equal<>()) == last);
    }

    // role bits 3 and 4
    const std::set<int> fluid_set(fluid.begin(), fluid.end());
    const std::set<int> face_set(dfaces.begin(), dfaces.end());
    for (integer i = 0; i < L.n_D; ++i) {
        const bool f = T.role[i] & role_bit::fluid;
        const bool g = T.role[i] & role_bit::dirichlet_face;
        REQUIRE(f == fluid_set.contains((int)i));
        REQUIRE(g == face_set.contains((int)i));
        if (T.role[i] & role_bit::any_interior) {
            REQUIRE(f);
            REQUIRE(!g);
        }
    }
}

} // namespace

TEST_CASE("2D E2 with Floating Objects") { check_problem(floating_2d); }

TEST_CASE("E2 with Floating Objects") { check_problem(floating_3d); }
