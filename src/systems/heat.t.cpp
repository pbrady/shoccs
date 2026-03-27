#include "system.hpp"

#include "fields/field_registry.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <numeric>

using namespace ccs;

// Custom main: Kokkos must be initialized before any test allocates Views.
int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

// Helper: set up a sim_registry with 2 slots (u0 and rhs) from system size.
static std::pair<field_ref, field_ref> setup_registry(sim_registry& reg, ccs::system& sys)
{
    auto sz = sys.size();
    auto& ss = sz.scalar_size;
    int d_sz  = get<0>(get<0>(ss));
    int rx_sz = get<0>(get<1>(ss));
    int ry_sz = get<1>(get<1>(ss));
    int rz_sz = get<2>(get<1>(ss));

    field_ref u0_ref{0}, rhs_ref{1};
    for (int s = 0; s < sz.nscalars; ++s) {
        u0_ref  = reg.allocate_scalar(0, s, d_sz, rx_sz, ry_sz, rz_sz);
        rhs_ref = reg.allocate_scalar(1, s, d_sz, rx_sz, ry_sz, rz_sz);
    }
    return {u0_ref, rhs_ref};
}

TEST_CASE("heat - E2")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {21, 22, 23},
                domain_bounds = {
                    min = {1, 1.1, 0.3},
                    max = {3, 3.3, 2.2}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                ymin = "neumann",
                ymax = "neumann",
                zmax = "dirichlet"
            },
            shapes = {
                {
                    type = "sphere",
                    center = {2.0001, 2.5656565, 1.313131311},
                    radius = 0.25,
                    boundary_condition = "dirichlet"
                }
            },
            scheme = {
                order = 2,
                type = "E2"
            },
            system = {
                type = "heat",
                diffusivity = 0.1
            },
            manufactured_solution = {
                type = "lua",
                call = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return (math.sin(time) +
                        x * x * (y + z) + y * y * (x + z) + z * z * (x + y) +
                        3 * x * y * z + x + y + z)
                end,
                ddt = function(time, loc)
                    return math.cos(time)
                end,
                grad = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return 2. * x * (y + z) + y * y + z * z + 3. * y * z + 1,
                            x * x + 2. * y * (x + z) + z * z + 3. * x * z + 1,
                            x * x + y * y + 2. * z * (x + y) + 3. * x * y + 1
                end,
                lap = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return 2. * (y + z) + 2. * (x + z) + 2. * (x + y)
                end,
                div = function(time, loc)
                    return 0.0
                end
            }
        }
    )");

    auto sys_opt = system::from_lua(lua["simulation"]);
    REQUIRE(!!sys_opt);
    auto& sys = *sys_opt;
    step_controller step{};

    // Set up registry and initialize
    sim_registry reg;
    auto [u0_ref, rhs_ref] = setup_registry(reg, sys);
    sys.initialize(reg, u0_ref, step);

    constexpr auto sh = scalar_handle{0};
    auto u0_scalar = extract_scalar_span(reg, u0_ref, sh);

    // only solid points will be zero
    const integer solid_points = std::ranges::count(u0_scalar | sel::D, 0.0);
    // maximum error should be zero
    auto st = sys.stats(reg, u0_ref, u0_ref, step);
    REQUIRE(st.stats[0] == 0);

    // prepare for rhs calculation
    sys.update_boundary(reg, u0_ref, (real)step);
    sys.rhs(reg, u0_ref, reg, rhs_ref, (real)step);

    auto u_rhs = extract_scalar_span(reg, rhs_ref, sh);

    // at this point, all fluid points in rhs should have a value of cos(time) -> 1
    // and solid points should remain at zero
    const integer rhs_solid_points = std::ranges::count(u_rhs | sel::D, 0.0);
    // these zeros include the zeroed rhs contributions to the dirichlet planar bcs
    int3 n{21, 22, 23};
    integer x_sz = n[1] * n[2];
    integer z_sz = n[0] * n[1];
    REQUIRE(solid_points == rhs_solid_points - (x_sz + z_sz - n[1]));

    auto d_rng = u_rhs | sel::D;
    real sum = std::accumulate(std::ranges::begin(d_rng), std::ranges::end(d_rng), 0.0);

    REQUIRE(sum == Catch::Approx((real)n[0] * n[1] * n[2] - rhs_solid_points));
}

TEST_CASE("heat - E2 - floating")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {21, 22, 23},
                domain_bounds = {
                    min = {1, 1.1, 0.3},
                    max = {3, 3.3, 2.2}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                ymin = "neumann",
                ymax = "neumann",
                zmax = "dirichlet"
            },
            shapes = {
                {
                    type = "sphere",
                    center = {2.0001, 2.5656565, 1.313131311},
                    radius = 0.25,
                    boundary_condition = "floating"
                }
            },
            scheme = {
                order = 2,
                type = "E2"
            },
            system = {
                type = "heat",
                diffusivity = 0.1
            },
            manufactured_solution = {
                type = "lua",
                call = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return (math.sin(time) +
                        x * (y + z) + y * (x + z) + z * (x + y) +
                        x + y + z)
                    --[[
                        return (math.sin(time) +
                        x * x * (y + z) + y * y * (x + z) + z * z * (x + y) +
                        3 * x * y * z + x + y + z)
                        ]]
                end,
                ddt = function(time, loc)
                    return math.cos(time)
                end,
                grad = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return 2. * (y + z) + 1,
                           2. * (x + z) + 1,
                           2. * (x + y) + 1
                    --[[
                    return 2. * x * (y + z) + y * y + z * z + 3. * y * z + 1,
                            x * x + 2. * y * (x + z) + z * z + 3. * x * z + 1,
                            x * x + y * y + 2. * z * (x + y) + 3. * x * y + 1
                    ]]
                end,
                lap = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return 0.0
                    --return 2. * (y + z) + 2. * (x + z) + 2. * (x + y)
                end,
                div = function(time, loc)
                    return 0.0
                end
            }
        }
    )");

    auto sys_opt = system::from_lua(lua["simulation"]);
    REQUIRE(!!sys_opt);
    auto& sys = *sys_opt;
    step_controller step{};

    // Set up registry and initialize
    sim_registry reg;
    auto [u0_ref, rhs_ref] = setup_registry(reg, sys);
    sys.initialize(reg, u0_ref, step);

    constexpr auto sh = scalar_handle{0};
    auto u0_scalar = extract_scalar_span(reg, u0_ref, sh);

    // only solid points will be zero
    const integer solid_points = std::ranges::count(u0_scalar | sel::D, 0.0);
    // maximum error should be zero
    auto st = sys.stats(reg, u0_ref, u0_ref, step);
    REQUIRE(st.stats[0] == 0);

    // prepare for rhs calculation
    sys.update_boundary(reg, u0_ref, (real)step);
    sys.rhs(reg, u0_ref, reg, rhs_ref, (real)step);

    auto u_rhs = extract_scalar_span(reg, rhs_ref, sh);

    // at this point, all fluid points in rhs should have a value of cos(time) -> 1
    // and solid points should remain at zero
    const integer rhs_solid_points = std::ranges::count(u_rhs | sel::D, 0.0);
    // these zeros include the zeroed rhs contributions to the dirichlet planar bcs
    int3 n{21, 22, 23};
    integer x_sz = n[1] * n[2];
    integer z_sz = n[0] * n[1];
    REQUIRE(solid_points == rhs_solid_points - (x_sz + z_sz - n[1]));

    auto sum = transform([](auto&& ui) { return std::accumulate(std::ranges::begin(ui), std::ranges::end(ui), 0.0); }, u_rhs);

    REQUIRE(get<si::D>(sum) ==
            Catch::Approx((real)n[0] * n[1] * n[2] - rhs_solid_points));

    auto ss = sys.size().scalar_size;

    REQUIRE(get<si::Rx>(sum) == Catch::Approx((real)get<si::Rx>(ss)));
    REQUIRE(get<si::Ry>(sum) == Catch::Approx((real)get<si::Ry>(ss)));
    REQUIRE(get<si::Rz>(sum) == Catch::Approx((real)get<si::Rz>(ss)));
}

TEST_CASE("2D heat - E2 - floating")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {21, 22},
                domain_bounds = {
                    min = {1, 1.1},
                    max = {3, 3.3}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                ymin = "neumann",
                ymax = "neumann",
            },
            shapes = {
                {
                    type = "sphere",
                    center = {2.0001, 2.5656565},
                    radius = 0.25,
                    boundary_condition = "floating"
                }
            },
            scheme = {
                order = 2,
                type = "E2"
            },
            system = {
                type = "heat",
                diffusivity = 0.1
            },
            manufactured_solution = {
                type = "lua",
                call = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    --[[
                    return (math.sin(time) +
                        x * x * y + y * y * x + 3 * x * y + x + y)
                    ]]
                    return (math.sin(time) + x * y + x + y)
                end,
                ddt = function(time, loc)
                    return math.cos(time)
                end,
                grad = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return y + 1, x + 1, 0.0
                    --[[
                        return 2. * x * y + y * y + 3. * y + 1,
                            x * x + 2. * y * x + 3. * x + 1,
                            0
                        ]]
                end,
                lap = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    --return 2. * y + 2. * x
                    return 0.0
                end,
                div = function(time, loc)
                    return 0.0
                end
            }
        }
    )");

    auto sys_opt = system::from_lua(lua["simulation"]);
    REQUIRE(!!sys_opt);
    auto& sys = *sys_opt;
    step_controller step{};

    // Set up registry and initialize
    sim_registry reg;
    auto [u0_ref, rhs_ref] = setup_registry(reg, sys);
    sys.initialize(reg, u0_ref, step);

    constexpr auto sh = scalar_handle{0};
    auto u0_scalar = extract_scalar_span(reg, u0_ref, sh);

    // only solid points will be zero
    const integer solid_points = std::ranges::count(u0_scalar | sel::D, 0.0);
    // maximum error should be zero
    auto st = sys.stats(reg, u0_ref, u0_ref, step);
    REQUIRE(st.stats[0] == 0);

    // prepare for rhs calculation
    sys.update_boundary(reg, u0_ref, (real)step);
    sys.rhs(reg, u0_ref, reg, rhs_ref, (real)step);

    auto u_rhs = extract_scalar_span(reg, rhs_ref, sh);

    // at this point, all fluid points in rhs should have a value of cos(time) -> 1
    // and solid points should remain at zero
    const integer rhs_solid_points = std::ranges::count(u_rhs | sel::D, 0.0);
    // these zeros include the zeroed rhs contributions to the dirichlet planar bcs
    int3 n{21, 22, 1};
    integer x_sz = n[1];
    REQUIRE(solid_points == rhs_solid_points - x_sz);

    auto sum = transform([](auto&& ui) { return std::accumulate(std::ranges::begin(ui), std::ranges::end(ui), 0.0); }, u_rhs);

    REQUIRE(get<si::D>(sum) ==
            Catch::Approx((real)n[0] * n[1] * n[2] - rhs_solid_points));

    auto ss = sys.size().scalar_size;

    REQUIRE(get<si::Rx>(sum) == Catch::Approx((real)get<si::Rx>(ss)));
    REQUIRE(get<si::Ry>(sum) == Catch::Approx((real)get<si::Ry>(ss)));
}
