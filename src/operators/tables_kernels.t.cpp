#include "tables_kernels.hpp"

#include "fields/field_registry.hpp"
#include "fields/scalar.hpp"
#include "mms/manufactured_solutions.hpp"
#include "systems/heat.hpp"
#include "temporal/step_controller.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sol/sol.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <string>
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

// The two floating-object setups of laplacian.t.cpp and tables.t.cpp.
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

// heat.lua geometry: [0,2]^dim, one floating sphere, E2 order 2, Gaussian MMS.
std::string heat_config(int dim, int N)
{
    const std::string n = std::to_string(N);
    const bool three = dim == 3;
    const std::string z_ext = three ? ", " + n : std::string();
    const std::string z_dom = three ? std::string(", 2") : std::string();
    const std::string z_bc =
        three ? std::string(", zmin = \"dirichlet\", zmax = \"dirichlet\"") : std::string();
    const std::string z_one = three ? std::string(", 1") : std::string();

    return "simulation = { mesh = { index_extents = {" + n + ", " + n + z_ext +
           "}, domain_bounds = {2, 2" + z_dom +
           "} }, domain_boundaries = { xmin = \"dirichlet\", xmax = \"dirichlet\", ymin = "
           "\"dirichlet\", ymax = \"dirichlet\"" +
           z_bc +
           " }, shapes = { { type = \"sphere\", center = {16 / 17, 25 / 22" + z_one +
           "}, radius = math.sqrt(3) / 10, boundary_condition = \"floating\" } }, scheme = { "
           "order = 2, type = \"E2\" }, system = { type = \"heat\", diffusivity = 1 / 30 }, "
           "manufactured_solution = { type = \"gaussian\", { center = {1, 1" +
           z_one + "}, variance = {1, 1" + z_one + "}, amplitude = 2, frequency = 0.5 } } }";
}

struct problem {
    mesh m;
    bcs::Grid grid_bcs;
    bcs::Object object_bcs;
    stencil st;
};

problem make_problem(sol::state& lua)
{
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

// The serial tables plus the heat source rule, with the scale of each row's
// rounding error: k (|taps|_1 + |exc row|_1) max|u| + |src| where added.
struct reference {
    std::vector<real> rhs;
    std::vector<real> scale;
};

reference reference_rhs(const compact_tables& T,
                        const problem& pb,
                        real k,
                        std::span<const real> u,
                        std::span<const real> src)
{
    const packed_layout& L = T.layout;
    const mesh& m = pb.m;
    std::vector<real> z(L.n_V);
    apply_structured(T, u, z);

    const auto fl = fluid_indices(m);
    const auto df = dirichlet_faces(m, pb.grid_bcs);
    const std::set<int> fluid(fl.begin(), fl.end());
    const std::set<int> dface(df.begin(), df.end());
    const real umax = max_abs(u);

    std::vector<real> rownorm(L.n_V, 0.0);
    for (integer i = 0; i < L.n_D; ++i)
        for (int d = 0; d < 3; ++d)
            if (T.role[i] & role_bit::interior(d))
                for (real t : T.taps[d]) rownorm[i] += std::abs(t);
    for (std::size_t r = 0; r < T.exc_rows.size(); ++r)
        for (integer p = T.exc_rowptr[r]; p < T.exc_rowptr[r + 1]; ++p)
            rownorm[T.exc_rows[r]] += std::abs(T.exc_val[p]);

    reference out{std::vector<real>(L.n_V), std::vector<real>(L.n_V)};
    for (integer i = 0; i < L.n_V; ++i) {
        bool dirichlet, add_src;
        if (i < L.n_D) {
            dirichlet = dface.contains((int)i);
            add_src = fluid.contains((int)i) && !dirichlet;
        } else {
            int d = 0;
            while (d < 2 && i >= L.off_R(d + 1)) ++d;
            const auto& rec = m.R(d)[i - L.off_R(d)];
            dirichlet = pb.object_bcs[rec.shape_id] == bcs::Dirichlet;
            add_src = !dirichlet;
        }
        if (dirichlet) {
            out.rhs[i] = 0.0;
            out.scale[i] = 0.0;
        } else {
            out.rhs[i] = k * z[i] + (add_src ? src[i] : 0.0);
            out.scale[i] = k * rownorm[i] * umax + (add_src ? std::abs(src[i]) : 0.0);
        }
    }
    return out;
}

struct row_check {
    real worst_ratio = 0.0;
    integer empty_mismatches = 0;
};

row_check compare_rows(std::span<const real> got,
                       std::span<const real> want,
                       std::span<const real> scale)
{
    row_check c;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const real diff = std::abs(got[i] - want[i]);
        if (scale[i] > 0)
            c.worst_ratio = std::max(c.worst_ratio, diff / scale[i]);
        else if (diff != 0)
            ++c.empty_mismatches;
    }
    return c;
}

device_view<real*> upload(const char* label, std::span<const real> v)
{
    device_view<real*> out(label, v.size());
    auto h = Kokkos::create_mirror_view(out);
    for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
    Kokkos::deep_copy(out, h);
    return out;
}

std::vector<real> download(const device_view<real*>& v)
{
    auto h = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h, v);
    return {h.data(), h.data() + h.extent(0)};
}

const std::array<launch_config, 5> launches{
    launch_config{launch_config::one, {}},
    launch_config{launch_config::two, {}},
    launch_config{launch_config::two_md, {2, 4, 64}},
    launch_config{launch_config::pencil, {}},
    launch_config{launch_config::pencil_one, {}}};

void check_kernels(const char* script, real k)
{
    sol::state lua;
    lua.script(script);
    problem pb = make_problem(lua);
    const mesh& m = pb.m;
    const laplacian lap{m, pb.st, pb.grid_bcs, pb.object_bcs};
    const packed_layout L{m};
    const compact_tables T =
        build_compact_tables(lap, L, fluid_indices(m), dirichlet_faces(m, pb.grid_bcs));
    const device_tables D = make_device_tables(T, m, pb.object_bcs, k);

    REQUIRE(D.n_D == L.n_D);
    REQUIRE(D.n_V == L.n_V);
    REQUIRE(D.n_exc + D.n_exc_dropped == (int)T.exc_rows.size());
    REQUIRE(D.n_exc_dropped == 0); // no Dirichlet row carries entries in these setups

    const auto u = random_vector(L.n_V, 11);
    const auto src = random_vector(L.n_V, 12);
    const reference ref = reference_rhs(T, pb, k, u, src);

    auto u_d = upload("u", u);
    auto src_d = upload("src", src);
    device_view<real*> out_d("out", L.n_V);

    // (a) the structured heat RHS, every launch form
    for (const auto& cfg : launches) {
        Kokkos::deep_copy(out_d, -7.0); // every row must be written
        structured_rhs(D, u_d.data(), src_d.data(), out_d.data(), cfg);
        const auto out = download(out_d);
        const auto c = compare_rows(out, ref.rhs, ref.scale);
        CAPTURE((int)cfg.kind, c.worst_ratio, c.empty_mismatches);
        REQUIRE(c.empty_mismatches == 0);
        REQUIRE(c.worst_ratio <= 1e-14);
    }

    // (b) the fused RK stage against a hand-written stage, first, middle and last
    const auto u0 = random_vector(L.n_V, 13);
    const auto acc_in = random_vector(L.n_V, 14);
    auto u0_d = upload("u0", u0);
    const real dt = 0.37;
    const rk_stage stages[] = {{dt / 6, dt / 2, true, false},
                               {dt / 3, dt / 2, false, false},
                               {dt / 3, dt, false, false},
                               {dt / 6, 0.0, false, true}};
    for (const auto& cfg : launches) {
        for (const auto& s : stages) {
            auto acc_d = upload("acc", acc_in);
            device_view<real*> next_d("u_next", L.n_V);
            Kokkos::deep_copy(next_d, -7.0);
            fused_rk_stage(D, u_d.data(), src_d.data(), u0_d.data(), acc_d.data(),
                           next_d.data(), s, cfg);
            const auto acc = download(acc_d);
            const auto next = download(next_d);

            std::vector<real> acc_ref(L.n_V), next_ref(L.n_V), acc_scale(L.n_V),
                next_scale(L.n_V);
            for (integer i = 0; i < L.n_V; ++i) {
                const real r = ref.rhs[i];
                acc_ref[i] = (s.first ? 0.0 : acc_in[i]) + s.w * r;
                acc_scale[i] = (s.first ? 0.0 : std::abs(acc_in[i])) + s.w * ref.scale[i];
                next_ref[i] = u0[i] + (s.last ? acc_ref[i] : s.c * r);
                next_scale[i] = std::abs(u0[i]) + (s.last ? acc_scale[i] : s.c * ref.scale[i]);
            }
            const auto ca = compare_rows(acc, acc_ref, acc_scale);
            const auto cn = compare_rows(next, next_ref, next_scale);
            CAPTURE((int)cfg.kind, s.w, s.c, s.first, s.last, ca.worst_ratio, cn.worst_ratio);
            REQUIRE(ca.empty_mismatches == 0);
            REQUIRE(cn.empty_mismatches == 0);
            REQUIRE(ca.worst_ratio <= 1e-14);
            REQUIRE(cn.worst_ratio <= 1e-14);
        }
    }

    // the last stage without a next buffer leaves acc as the only output
    {
        auto acc_d = upload("acc", acc_in);
        fused_rk_stage(
            D, u_d.data(), src_d.data(), u0_d.data(), acc_d.data(), nullptr, stages[3]);
        const auto acc = download(acc_d);
        std::vector<real> acc_ref(L.n_V), acc_scale(L.n_V);
        for (integer i = 0; i < L.n_V; ++i) {
            acc_ref[i] = acc_in[i] + stages[3].w * ref.rhs[i];
            acc_scale[i] = std::abs(acc_in[i]) + stages[3].w * ref.scale[i];
        }
        const auto ca = compare_rows(acc, acc_ref, acc_scale);
        REQUIRE(ca.empty_mismatches == 0);
        REQUIRE(ca.worst_ratio <= 1e-14);
    }
}

// The heat source at every V location at time t, in the order of eval_at_locations.
std::vector<real>
heat_source(const mesh& m, const packed_layout& L, const manufactured_solution& ms, real k, real t)
{
    std::vector<real> out(L.n_V);
    const auto x = m.x();
    const auto y = m.y();
    const auto z = m.z();
    integer idx = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
        for (std::size_t j = 0; j < y.size(); ++j)
            for (std::size_t kk = 0; kk < z.size(); ++kk) {
                const real3 p{x[i], y[j], z[kk]};
                out[idx++] = ms.ddt(t, p) - k * ms.laplacian(t, p);
            }
    for (int d = 0; d < 3; ++d)
        for (const auto& rec : m.R(d))
            out[idx++] = ms.ddt(t, rec.position) - k * ms.laplacian(t, rec.position);
    return out;
}

void unpack(std::span<const real> V, scalar_span s)
{
    std::size_t off = 0;
    for (auto dst : {s.D, s.Rx, s.Ry, s.Rz}) {
        std::copy_n(V.begin() + off, dst.size(), dst.begin());
        off += dst.size();
    }
}

std::vector<real> pack(scalar_view v)
{
    std::vector<real> out;
    for (auto s : {v.D, v.Rx, v.Ry, v.Rz}) out.insert(out.end(), s.begin(), s.end());
    return out;
}

// The Kokkos structured RHS against the live heat RHS graph on random input.
void check_against_heat(int dim, int N)
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(heat_config(dim, N));
    const sol::table tbl = lua["simulation"];
    const real k = tbl["system"]["diffusivity"].get_or(1.0);

    problem pb = make_problem(lua);
    const mesh& m = pb.m;
    const laplacian lap{m, pb.st, pb.grid_bcs, pb.object_bcs};
    const packed_layout L{m};
    const compact_tables T =
        build_compact_tables(lap, L, fluid_indices(m), dirichlet_faces(m, pb.grid_bcs));
    const device_tables D = make_device_tables(T, m, pb.object_bcs, k);
    auto ms_opt = manufactured_solution::from_lua(tbl, m.dims());
    REQUIRE(!!ms_opt);

    auto heat_opt = systems::heat::from_lua(tbl);
    REQUIRE(!!heat_opt);
    systems::heat& sys = *heat_opt;
    const auto sz = sys.size();
    sim_registry reg;
    const field_ref u0 = reg.allocate_scalar(0, 0, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
    const field_ref u1 = reg.allocate_scalar(1, 0, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
    constexpr auto sh = scalar_handle{0};
    const step_controller step{};
    sys.initialize(reg, u0, step);
    sys.update_boundary(reg, u0, (real)step);

    const real t0 = (real)step;
    const auto u = random_vector(L.n_V, 21);
    const auto src = heat_source(m, L, *ms_opt, k, t0);
    unpack(u, extract_scalar_span(reg, u0, sh));
    sys.build_rhs_graph(extract_scalar_view(reg, u0, sh), extract_scalar_span(reg, u1, sh));
    sys.fill_source(t0);
    sys.submit_rhs_graph();
    const auto live = pack(extract_scalar_view(reg, u1, sh));

    const reference ref = reference_rhs(T, pb, k, u, src);
    auto u_d = upload("u", u);
    auto src_d = upload("src", src);
    device_view<real*> out_d("out", L.n_V);
    for (const auto& cfg : launches) {
        structured_rhs(D, u_d.data(), src_d.data(), out_d.data(), cfg);
        const auto out = download(out_d);
        const auto c = compare_rows(out, live, ref.scale);
        CAPTURE(dim, N, (int)cfg.kind, c.worst_ratio, c.empty_mismatches);
        REQUIRE(c.empty_mismatches == 0);
        REQUIRE(c.worst_ratio <= 1e-13);
    }
}

} // namespace

TEST_CASE("2D E2 with Floating Objects") { check_kernels(floating_2d, 1.0 / 30); }

TEST_CASE("E2 with Floating Objects") { check_kernels(floating_3d, 1.0 / 30); }

TEST_CASE("heat.lua 2D against the live RHS graph") { check_against_heat(2, 51); }

TEST_CASE("heat.lua 3D against the live RHS graph") { check_against_heat(3, 33); }
