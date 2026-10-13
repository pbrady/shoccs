// Benchmark: RHS graph submit and per-step host work on the shipped cut-cell
// configurations (the heat.lua sphere and the scalar_wave.lua pair of spheres),
// in 2D and 3D.
//
// bench_rhs covers a shape-free cubic grid only. This benchmark embeds the
// shipped geometry so the cut-cell paths (R buffers, O/B CSR coupling,
// interpolation rows) are exercised, and it times the host-side work the time
// loop performs every stage or step (fill_source, update_boundary, stats) so
// the cost of one RK4 step can be decomposed:
//
//   step ~= 4 * (rhs + fill_source + update_boundary) + stats + slot copies

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <sol/sol.hpp>

#include "fields/field_registry.hpp"
#include "fields/scalar.hpp"
#include "systems/heat.hpp"
#include "systems/scalar_wave.hpp"
#include "temporal/step_controller.hpp"

#include <stdexcept>
#include <string>
#include <utility>

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

// scalar_wave.lua geometry: two spheres, E2-poly order 1 with the shipped alphas.
std::string wave_config(int N)
{
    const std::string n = std::to_string(N);
    return std::string(R"(
        simulation = {
            mesh = {
                index_extents = {)") + n + ", " + n + R"(},
                domain_bounds = {2, 2}
            },
            domain_boundaries = { xmin = "dirichlet", ymin = "dirichlet" },
            shapes = {
                { type = "sphere", center = {0, 0}, radius = math.pi / 10,
                  boundary_condition = "dirichlet" },
                { type = "sphere", center = {2, 2}, radius = math.pi / 10,
                  boundary_condition = "floating" }
            },
            scheme = {
                order = 1,
                type = "E2-poly",
                floating_alpha = {0.7039278390946743, 0.5390086175376538,
                                  -0.647109821986589, 0.2051508287133347,
                                  0.6062051039572746, 0.8148425279273044},
                dirichlet_alpha = {-0.10739761225713096, 0.8736492896991024,
                                   0.40413606410467495},
                interpolant_alpha = {0.01, -0.01, 0.01, 0.04}
            },
            system = { type = "scalar wave", center = {-1, -1}, radius = 0,
                       max_error = 2.0 }
        }
    )";
}

systems::heat make_heat(int dim, int N)
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(heat_config(dim, N));
    auto opt = systems::heat::from_lua(lua["simulation"]);
    if (!opt) throw std::runtime_error("failed to build heat system");
    return std::move(*opt);
}

systems::scalar_wave make_wave(int N)
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(wave_config(N));
    auto opt = systems::scalar_wave::from_lua(lua["simulation"]);
    if (!opt) throw std::runtime_error("failed to build scalar_wave system");
    return std::move(*opt);
}

// Two-slot registry (u0 input, u1 output) initialised the way simulation_cycle
// does it, with the RHS graph built from u0 into u1.
template <typename System>
struct fixture {
    System sys;
    system_size sz;
    sim_registry reg;
    field_ref u0{0};
    field_ref u1{1};
    step_controller step{};

    explicit fixture(System&& s) : sys{std::move(s)}, sz{sys.size()}
    {
        for (int i = 0; i < sz.nscalars; ++i) {
            u0 = reg.allocate_scalar(0, i, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
            u1 = reg.allocate_scalar(1, i, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
        }
        sys.initialize(reg, u0, step);
        reg.deep_copy_slot(u1.slot, u0.slot);
        sys.update_boundary(reg, u0, (real)step);
    }

    void build_graph()
    {
        constexpr auto sh = scalar_handle{0};
        auto u = extract_scalar_view(reg, u0, sh);
        auto du = extract_scalar_span(reg, u1, sh);
        sys.build_rhs_graph(u, du);
        sys.submit_rhs_graph(); // warm up
    }

    void counters(benchmark::State& state) const
    {
        state.counters["points"] = static_cast<double>(sz.d_size);
        state.counters["R"] = static_cast<double>(sz.rx_size + sz.ry_size + sz.rz_size);
    }
};

// ---- heat ------------------------------------------------------------------

void BM_heat_cut_rhs(benchmark::State& state)
{
    fixture<systems::heat> f{make_heat(state.range(0), state.range(1))};
    f.sys.fill_source((real)f.step);
    f.build_graph();
    for (auto _ : state) f.sys.submit_rhs_graph();
    f.counters(state);
}

void BM_heat_cut_fill_source(benchmark::State& state)
{
    fixture<systems::heat> f{make_heat(state.range(0), state.range(1))};
    for (auto _ : state) f.sys.fill_source(0.1);
    f.counters(state);
}

void BM_heat_cut_update_boundary(benchmark::State& state)
{
    fixture<systems::heat> f{make_heat(state.range(0), state.range(1))};
    for (auto _ : state) f.sys.update_boundary(f.reg, f.u0, 0.1);
    f.counters(state);
}

void BM_heat_cut_stats(benchmark::State& state)
{
    fixture<systems::heat> f{make_heat(state.range(0), state.range(1))};
    for (auto _ : state) {
        auto s = f.sys.stats(f.reg, f.u0, f.u1, f.step);
        benchmark::DoNotOptimize(s);
    }
    f.counters(state);
}

// ---- scalar wave -----------------------------------------------------------

void BM_wave_cut_rhs(benchmark::State& state)
{
    fixture<systems::scalar_wave> f{make_wave(state.range(0))};
    f.build_graph();
    for (auto _ : state) f.sys.submit_rhs_graph();
    f.counters(state);
}

void BM_wave_cut_update_boundary(benchmark::State& state)
{
    fixture<systems::scalar_wave> f{make_wave(state.range(0))};
    for (auto _ : state) f.sys.update_boundary(f.reg, f.u0, 0.1);
    f.counters(state);
}

void BM_wave_cut_stats(benchmark::State& state)
{
    fixture<systems::scalar_wave> f{make_wave(state.range(0))};
    for (auto _ : state) {
        auto s = f.sys.stats(f.reg, f.u0, f.u1, f.step);
        benchmark::DoNotOptimize(s);
    }
    f.counters(state);
}

BENCHMARK(BM_heat_cut_rhs)
    ->Args({2, 51})
    ->Args({2, 71})
    ->Args({2, 201})
    ->Args({2, 401})
    ->Args({3, 65})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_heat_cut_fill_source)
    ->Args({2, 51})
    ->Args({2, 201})
    ->Args({3, 65})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_heat_cut_update_boundary)
    ->Args({2, 51})
    ->Args({2, 201})
    ->Args({3, 65})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_heat_cut_stats)
    ->Args({2, 51})
    ->Args({2, 201})
    ->Args({3, 65})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_wave_cut_rhs)->Arg(71)->Arg(201)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_wave_cut_update_boundary)->Arg(71)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_wave_cut_stats)->Arg(71)->Unit(benchmark::kMicrosecond);

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
