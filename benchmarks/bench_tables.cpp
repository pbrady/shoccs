// Benchmark: the compact-table kernels of src/operators/tables_kernels.hpp against
// the live heat RHS graph on the heat.lua geometry (2D 51, 71, 201, 401; 3D 65),
// the C++ fused-stage arm of docs/PYTHON_DSL_SURVEY.md Section 4.3.
//
//   graph_rhs            heat::submit_rhs_graph (the fixture of bench_rhs_cutcell)
//   structured_rhs_*     the structured RHS over the tables: two launches
//                        (RangePolicy, MDRangePolicy or fast-axis pencils over D,
//                        then the rows) or one launch over both index sets
//   graph_step           one RK4 step from the live graph with slot_ops, the source
//                        and boundary vectors held fixed (no MMS evaluation)
//   fused_step_*         the same step from four fused RK stages plus four
//                        Dirichlet refreshes, on packed vectors
//   stream_triad         a = b + s c over three 134 MB buffers, median of 10
//
// Every table benchmark checks its output against the live graph once at setup.

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <sol/sol.hpp>

#include "fields/expr.hpp"
#include "fields/field_registry.hpp"
#include "fields/scalar.hpp"
#include "fields/selection_desc.hpp"
#include "operators/tables.hpp"
#include "operators/tables_kernels.hpp"
#include "systems/heat.hpp"
#include "temporal/slot_ops.hpp"
#include "temporal/step_controller.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ccs;

namespace
{

// heat.lua geometry: [0,2]^dim, one floating sphere, E2 order 2, Gaussian MMS.
std::string heat_config(int dim, int N)
{
    const std::string n = std::to_string(N);
    const bool three = dim == 3;
    const std::string z_ext = three ? std::string(", ") + n : std::string();
    const std::string z_dom = three ? std::string(", 2") : std::string();
    const std::string z_bc =
        three ? std::string(", zmin = \"dirichlet\", zmax = \"dirichlet\"") : std::string();
    const std::string z_one = three ? std::string(", 1") : std::string();

    return std::string(R"(
        simulation = {
            mesh = {
                index_extents = {)") + n + ", " + n + z_ext + R"(},
                domain_bounds = {2, 2)" + z_dom + R"(}
            },
            domain_boundaries = {
                xmin = "dirichlet", xmax = "dirichlet",
                ymin = "dirichlet", ymax = "dirichlet")" + z_bc + R"(
            },
            shapes = {
                {
                    type = "sphere",
                    center = {16 / 17, 25 / 22)" + z_one + R"(},
                    radius = math.sqrt(3) / 10,
                    boundary_condition = "floating"
                }
            },
            scheme = { order = 2, type = "E2" },
            system = { type = "heat", diffusivity = 1 / 30 },
            manufactured_solution = {
                type = "gaussian",
                {
                    center = {1, 1)" + z_one + R"(},
                    variance = {1, 1)" + z_one + R"(},
                    amplitude = 2,
                    frequency = 0.5
                }
            }
        }
    )";
}

constexpr std::array rki{0.0, 0.5, 0.5, 1.0};
constexpr std::array rkf{1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};

std::vector<real> pack(scalar_view v)
{
    std::vector<real> out;
    for (auto s : {v.D, v.Rx, v.Ry, v.Rz}) out.insert(out.end(), s.begin(), s.end());
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

template <typename T>
device_view<T*> upload(const char* label, const std::vector<T>& v)
{
    device_view<T*> out(label, v.size());
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

real max_rel_diff(std::span<const real> got, std::span<const real> want)
{
    real diff = 0, scale = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        diff = std::max(diff, std::abs(got[i] - want[i]));
        scale = std::max(scale, std::abs(want[i]));
    }
    return scale > 0 ? diff / scale : diff;
}

// The live heat system with a four-slot registry (as integrators::rk4 uses it)
// beside the compact tables of the same geometry on packed vectors.
struct fixture {
    systems::heat sys;
    system_size sz{};
    sim_registry reg;
    field_ref u0{0};
    field_ref u1{1};
    field_ref rk{2};
    field_ref srhs{3};
    step_controller step{};
    real k{};
    real dt{};
    real t0{};

    mesh m;
    bcs::Grid grid_bcs;
    bcs::Object object_bcs;
    packed_layout L;
    compact_tables T;
    device_tables dev;

    // packed vectors: random input, output, fixed source, stage ping-pong, the
    // accumulator and the fixed boundary values with their index list
    device_view<real*> u, out, src, ua, ub, acc, bnd;
    device_view<int*> bnd_idx;
    real rhs_check{}; // structured against live RHS, max relative difference

    // the same boundary refresh on the registry buffers
    gather_selection bnd_D;
    std::array<gather_selection, 3> bnd_R;
    std::vector<real> bnd_host;
    std::array<std::vector<real>, 3> bnd_R_host;

    fixture(int dim, int N)
    {
        sol::state lua;
        lua.open_libraries(sol::lib::base, sol::lib::math);
        lua.script(heat_config(dim, N));
        const sol::table tbl = lua["simulation"];
        k = tbl["system"]["diffusivity"].get_or(1.0);

        auto sys_opt = systems::heat::from_lua(tbl);
        if (!sys_opt) throw std::runtime_error("failed to build heat system");
        sys = std::move(*sys_opt);
        sz = sys.size();
        for (field_ref* r : {&u0, &u1, &rk, &srhs})
            *r = reg.allocate_scalar(r->slot, 0, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
        sys.initialize(reg, u0, step);
        reg.deep_copy_slot(u1.slot, u0.slot);
        t0 = (real)step;
        sys.update_boundary(reg, u0, t0);
        sys.fill_source(t0);

        auto m_opt = mesh::from_lua(tbl);
        auto bc_opt = bcs::from_lua(tbl, m_opt->extents());
        auto st_opt = stencil::from_lua(tbl);
        if (!m_opt || !bc_opt || !st_opt) throw std::runtime_error("failed to build the mesh");
        m = std::move(*m_opt);
        grid_bcs = std::move(bc_opt->first);
        object_bcs = std::move(bc_opt->second);
        L = packed_layout{m};
        const real h_min = std::ranges::min(m.h());
        dt = 0.5 * h_min * h_min / (4 * k);

        std::vector<int> fluid_idx(m.fluid_desc().count());
        for (int i = 0; i < m.fluid_desc().count(); ++i) fluid_idx[i] = m.fluid_desc().element(i);
        std::vector<int> face_idx;
        for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
            for (int i = 0; i < desc.count(); ++i) face_idx.push_back(desc.element(i));
        });
        {
            const laplacian lap{m, *st_opt, grid_bcs, object_bcs};
            T = build_compact_tables(lap, L, fluid_idx, face_idx);
        }
        dev = make_device_tables(T, m, object_bcs, k);

        constexpr auto sh = scalar_handle{0};

        // the boundary values: the state after initialize + update_boundary
        bnd_host = pack(extract_scalar_view(reg, u0, sh));
        bnd = upload("bnd", bnd_host);
        std::vector<int> idx = face_idx;
        bnd_D = gather_selection{upload("bnd_D", face_idx)};
        for (int d = 0; d < 3; ++d) {
            bnd_R[d] = m.dirichlet_object_desc(d, object_bcs);
            for (int i = 0; i < bnd_R[d].count(); ++i)
                idx.push_back((int)L.off_R(d) + bnd_R[d].element(i));
            bnd_R_host[d].assign(bnd_host.begin() + L.off_R(d),
                                 bnd_host.begin() + L.off_R(d) + L.n_R[d]);
        }
        bnd_idx = upload("bnd_idx", idx);

        // the fixed source where the live RHS adds it: the graph applied to u = 0
        build_graph(u0, u1);
        slot_zero(reg, u0);
        sys.submit_rhs_graph();
        src = upload("src", pack(extract_scalar_view(reg, u1, sh)));

        // random input through the live graph and the tables
        std::mt19937_64 gen(20261013);
        std::uniform_real_distribution<real> dist(-1.0, 1.0);
        std::vector<real> rnd(L.n_V);
        for (auto& v : rnd) v = dist(gen);
        unpack(rnd, extract_scalar_span(reg, u0, sh));
        sys.submit_rhs_graph();
        const auto live = pack(extract_scalar_view(reg, u1, sh));

        u = upload("u", rnd);
        out = device_view<real*>("out", L.n_V);
        ua = device_view<real*>("ua", L.n_V);
        ub = device_view<real*>("ub", L.n_V);
        acc = device_view<real*>("acc", L.n_V);
        structured_rhs(dev, u.data(), src.data(), out.data(), {launch_config::one, {}});
        rhs_check = max_rel_diff(download(out), live);
        if (rhs_check > 1e-12)
            throw std::runtime_error("structured RHS differs from the live graph: " +
                                     std::to_string(rhs_check));

        // restore the initial state in u0 and the step graph u1 -> srhs
        unpack(bnd_host, extract_scalar_span(reg, u0, sh));
        build_graph(u1, srhs);
    }

    void build_graph(field_ref in, field_ref o)
    {
        constexpr auto sh = scalar_handle{0};
        sys.build_rhs_graph(extract_scalar_view(reg, in, sh), extract_scalar_span(reg, o, sh));
        sys.submit_rhs_graph(); // warm up
    }

    // Dirichlet faces and Dirichlet-object R points of a registry slot from the
    // fixed boundary values (update_boundary without the MMS).
    void refresh(field_ref ref)
    {
        constexpr auto sh = scalar_handle{0};
        if (bnd_D.count() > 0)
            assign_selected(reg.data(ref, sh.D()), bnd_D, handle_expr{bnd_host.data()});
        const auto R = sh.R();
        for (int d = 0; d < 3; ++d)
            if (bnd_R[d].count() > 0)
                assign_selected(reg.data(ref, R[d]), bnd_R[d], handle_expr{bnd_R_host[d].data()});
        Kokkos::fence();
    }

    // integrators::rk4::operator() with the fixed source and boundary vectors
    void graph_step()
    {
        slot_zero(reg, rk);
        slot_zero(reg, srhs);
        reg.deep_copy_slot(u1.slot, u0.slot);
        for (int i = 0; i < 4; ++i) {
            if (i > 0) {
                slot_assign_lc(reg, u1, u0, dt * rki[i], srhs);
                refresh(u1);
            }
            sys.submit_rhs_graph();
            slot_accumulate(reg, rk, dt * rkf[i], srhs);
        }
        slot_assign_lc(reg, u1, u0, 1.0, rk);
        refresh(u1);
    }

    // the same step from four fused stages; the stages ping-pong between ua and
    // ub and the fourth writes the result into ub
    void fused_step(const launch_config& cfg)
    {
        const real* u0_p = bnd.data(); // the initial state is the boundary vector
        real* bufs[2] = {ub.data(), ua.data()};
        const real* in = u0_p;
        for (int i = 0; i < 4; ++i) {
            real* next = bufs[(i + 1) % 2];
            fused_rk_stage(dev,
                           in,
                           src.data(),
                           u0_p,
                           acc.data(),
                           next,
                           {dt * rkf[i], i < 3 ? dt * rki[i + 1] : 0.0, i == 0, i == 3},
                           cfg);
            assign_at(bnd_idx, bnd.data(), next);
            in = next;
        }
    }

    void counters(benchmark::State& state) const
    {
        state.counters["points"] = static_cast<double>(sz.d_size);
        state.counters["R"] = static_cast<double>(sz.rx_size + sz.ry_size + sz.rz_size);
        state.counters["exc"] = static_cast<double>(dev.n_exc);
        state.counters["rhs_check"] = rhs_check;
    }
};

// ---- RHS -------------------------------------------------------------------

void BM_graph_rhs(benchmark::State& state)
{
    fixture f{(int)state.range(0), (int)state.range(1)};
    f.build_graph(f.u0, f.u1);
    for (auto _ : state) f.sys.submit_rhs_graph();
    f.counters(state);
}

void run_structured_rhs(benchmark::State& state, launch_config cfg)
{
    fixture f{(int)state.range(0), (int)state.range(1)};
    for (auto _ : state) structured_rhs(f.dev, f.u.data(), f.src.data(), f.out.data(), cfg);
    f.counters(state);
}

void BM_structured_rhs_two(benchmark::State& state)
{
    run_structured_rhs(state, {launch_config::two, {}});
}

void BM_structured_rhs_two_md(benchmark::State& state)
{
    const int nz = state.range(0) == 3 ? 64 : 1;
    const int ny = state.range(0) == 3 ? 4 : 64;
    run_structured_rhs(state, {launch_config::two_md, {2, ny, nz}});
}

void BM_structured_rhs_one(benchmark::State& state)
{
    run_structured_rhs(state, {launch_config::one, {}});
}

void BM_structured_rhs_pencil(benchmark::State& state)
{
    run_structured_rhs(state, {launch_config::pencil, {}});
}

void BM_structured_rhs_pencil_one(benchmark::State& state)
{
    run_structured_rhs(state, {launch_config::pencil_one, {}});
}

// ---- step ------------------------------------------------------------------

void BM_graph_step(benchmark::State& state)
{
    fixture f{(int)state.range(0), (int)state.range(1)};
    for (auto _ : state) f.graph_step();
    f.counters(state);
}

void run_fused_step(benchmark::State& state, launch_config cfg)
{
    fixture f{(int)state.range(0), (int)state.range(1)};
    constexpr auto sh = scalar_handle{0};

    f.graph_step();
    const auto want = pack(extract_scalar_view(f.reg, f.u1, sh));
    f.fused_step(cfg);
    const real step_check = max_rel_diff(download(f.ub), want);
    if (step_check > 1e-12)
        throw std::runtime_error("fused step differs from the graph step: " +
                                 std::to_string(step_check));

    for (auto _ : state) f.fused_step(cfg);
    f.counters(state);
    state.counters["step_check"] = step_check;
}

void BM_fused_step_two(benchmark::State& state)
{
    run_fused_step(state, {launch_config::two, {}});
}

void BM_fused_step_one(benchmark::State& state)
{
    run_fused_step(state, {launch_config::one, {}});
}

void BM_fused_step_pencil(benchmark::State& state)
{
    run_fused_step(state, {launch_config::pencil, {}});
}

void BM_fused_step_pencil_one(benchmark::State& state)
{
    run_fused_step(state, {launch_config::pencil_one, {}});
}

// ---- bandwidth -------------------------------------------------------------

void BM_stream_triad(benchmark::State& state)
{
    const int64_t n = state.range(0);
    device_view<real*> a("a", n), b("b", n), c("c", n);
    Kokkos::parallel_for(
        "triad_init", Kokkos::RangePolicy<execution_space>(0, n), KOKKOS_LAMBDA(int64_t i) {
            a(i) = 0.0;
            b(i) = 1.0 + 1e-9 * (real)i;
            c(i) = 2.0 - 1e-9 * (real)i;
        });
    Kokkos::fence();
    const real s = 3.0;
    auto triad = [&] {
        Kokkos::parallel_for(
            "triad", Kokkos::RangePolicy<execution_space>(0, n), KOKKOS_LAMBDA(int64_t i) {
                a(i) = b(i) + s * c(i);
            });
        Kokkos::fence();
    };
    triad();

    std::vector<double> medians;
    for (auto _ : state) {
        std::array<double, 10> t{};
        for (auto& ti : t) {
            const auto start = std::chrono::steady_clock::now();
            triad();
            ti = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        std::sort(t.begin(), t.end());
        const double median = 0.5 * (t[4] + t[5]);
        medians.push_back(median);
        state.SetIterationTime(median);
    }
    std::sort(medians.begin(), medians.end());
    const double bytes = 3.0 * sizeof(real) * (double)n;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    state.counters["GiBps_median"] = bytes / medians[medians.size() / 2] / gib;
    state.counters["GiBps_best"] = bytes / medians.front() / gib;
    state.counters["doubles"] = (double)n;
    benchmark::DoNotOptimize(a.data());
}

#define HEAT_CASES                                                                        \
    ->Args({2, 51})->Args({2, 71})->Args({2, 201})->Args({2, 401})->Args({3, 65})         \
        ->Unit(benchmark::kMicrosecond)

BENCHMARK(BM_graph_rhs) HEAT_CASES;
BENCHMARK(BM_structured_rhs_two) HEAT_CASES;
BENCHMARK(BM_structured_rhs_two_md) HEAT_CASES;
BENCHMARK(BM_structured_rhs_one) HEAT_CASES;
BENCHMARK(BM_structured_rhs_pencil) HEAT_CASES;
BENCHMARK(BM_structured_rhs_pencil_one) HEAT_CASES;
BENCHMARK(BM_graph_step) HEAT_CASES;
BENCHMARK(BM_fused_step_two) HEAT_CASES;
BENCHMARK(BM_fused_step_one) HEAT_CASES;
BENCHMARK(BM_fused_step_pencil) HEAT_CASES;
BENCHMARK(BM_fused_step_pencil_one) HEAT_CASES;
BENCHMARK(BM_stream_triad)->Arg(16777216)->UseManualTime()->Unit(benchmark::kMillisecond);

} // namespace

// Custom main: Kokkos must be initialized before any Kokkos calls.
int main(int argc, char** argv)
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
