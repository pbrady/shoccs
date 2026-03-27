#include "heat.hpp"
#include "fields/algorithms.hpp"
#include "fields/expr.hpp"
#include "fields/selector.hpp"
#include "real3_operators.hpp"
#include <cmath>
#include <numbers>

#include <fmt/ranges.h>
#include <sol/sol.hpp>

#include "operators/discrete_operator.hpp"

#include <iterator>

namespace ccs::systems
{

constexpr auto abs = lift([](auto&& x) { return std::abs(x); });
enum class scalars : int { u };

heat::heat(mesh&& m,
           bcs::Grid&& grid_bcs,
           bcs::Object&& object_bcs,
           manufactured_solution&& m_sol,
           stencil st,
           real diffusivity,
           const logs& build_logger)
    : m{MOVE(m)},
      grid_bcs{MOVE(grid_bcs)},
      object_bcs{MOVE(object_bcs)},
      m_sol{MOVE(m_sol)},
      lap{this->m, st, this->grid_bcs, this->object_bcs, build_logger},
      diffusivity{diffusivity},
      neumann_u{this->m.ss()},
      error{this->m.ss()},
      logger{build_logger, "system", "system.csv"}
{
    assert(!!(this->m_sol));

    logger.set_pattern("%v");
    logger(spdlog::level::info,
           "Timestamp,Time,Step,Linf,Min,Max,Domain_Linf,Domain_ic,Rx_Linf,Rx_ic,Ry_"
           "Linf,Ry_ic,Rz_Linf,Rz_ic");

    logger.set_pattern("%Y-%m-%d %H:%M:%S.%f,%v");
}


bool heat::valid(const system_stats& stats) const
{
    const auto& v = stats.stats[0];
    return std::isfinite(v) && std::abs(v) <= 1e6;
}

void heat::log(const system_stats& stats, const step_controller& step)
{
    logger(spdlog::level::info,
           "{},{},{}",
           (real)step,
           (int)step,
           fmt::join(stats.stats, ","));
}

//
// Convert the system statistics into a real3 summary
//
real3 heat::summary(const system_stats& stats) const
{
    return {stats.stats[0], stats.stats[1], stats.stats[2]};
}

std::optional<heat> heat::from_lua(const sol::table& tbl, const logs& logger)
{
    // assume we can only get here if simulation.system.type == "heat" so check
    // for the rest
    real diff = tbl["system"]["diffusivity"].get_or(1.0);

    auto mesh_opt = mesh::from_lua(tbl, logger);
    if (!mesh_opt) return std::nullopt;

    auto bc_opt = bcs::from_lua(tbl, mesh_opt->extents(), logger);
    auto st_opt = stencil::from_lua(tbl, logger);

    if (bc_opt && st_opt) {
        auto ms_opt = manufactured_solution::from_lua(tbl, mesh_opt->dims(), logger);
        auto t = ms_opt ? MOVE(*ms_opt) : manufactured_solution{};

        return heat{MOVE(*mesh_opt),
                    MOVE(bc_opt->first),
                    MOVE(bc_opt->second),
                    MOVE(t),
                    *st_opt,
                    diff,
                    logger};
    }

    return std::nullopt;
}

system_size heat::size() const { return {1, 0, m.ss()}; }

void heat::rhs(const sim_registry& reg, field_ref input,
               sim_registry& out_reg, field_ref output, real time) const
{
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, input, sh);
    auto u_rhs = extract_scalar_span(out_reg, output, sh);

    // rhs = diffusivity * lap(u) + (dS/dt - diffusivity * lap(S))
    u_rhs = lap(u, neumann_u);
    times_assign_scalar(out_reg, output, sh, diffusivity);

    if (m_sol) {
        const auto src =
            (m.xyz | m_sol.ddt(time)) - (diffusivity * (m.xyz | m_sol.laplacian(time)));

        u_rhs | m.fluid_all(object_bcs) += src;
        u_rhs | m.dirichlet(grid_bcs, object_bcs) = 0;
    }
}

void heat::update_boundary(sim_registry& reg, field_ref ref, real time)
{
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_span(reg, ref, sh);
    auto l = m.xyz;

    u | m.dirichlet(grid_bcs, object_bcs) = l | m_sol(time);

    // set possible neumann bcs
    neumann_u | m.neumann<0>(grid_bcs) = l | m_sol.gradient(0, time);
    neumann_u | m.neumann<1>(grid_bcs) = l | m_sol.gradient(1, time);
    neumann_u | m.neumann<2>(grid_bcs) = l | m_sol.gradient(2, time);
}

real heat::timestep_size(const sim_registry&, field_ref,
                         const step_controller& step) const
{
    const auto h_min = std::ranges::min(m.h());
    return step.parabolic_cfl() * h_min * h_min / (4 * diffusivity);
}

system_stats heat::stats(const sim_registry& reg, field_ref /*u0*/,
                          field_ref u1, const step_controller& step) const
{
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, u1, sh);

    auto sol = m.xyz | m_sol(step.simulation_time());
    auto [u_min, u_max] = minmax(u | m.fluid_all(object_bcs));

    real err = max(abs(u - sol) | m.fluid_all(object_bcs));
    auto linf = abs(u - sol);
    auto fluid_error = linf | m.fluid_all(object_bcs);
    auto max_el = transform(std::ranges::max_element, fluid_error);
    auto err_pairs = transform(
        [](auto&& rng, auto&& max_el) {
            if (std::ranges::end(rng) != max_el)
                return std::pair{
                    *max_el,
                    (real)std::ranges::distance(std::ranges::begin(rng.base()),
                                                max_el.base())};
            else
                return std::pair{0.0, (real)0};
        },
        fluid_error,
        max_el);

    auto&& [d, rx, ry, rz] = err_pairs;
    return system_stats{.stats = {err,
                                  u_min,
                                  u_max,
                                  d.first,
                                  d.second,
                                  rx.first,
                                  rx.second,
                                  ry.first,
                                  ry.second,
                                  rz.first,
                                  rz.second}};
}

void heat::initialize(sim_registry& reg, field_ref ref, const step_controller& c)
{
    if (!m_sol) return;

    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_span(reg, ref, sh);
    auto sol = m.xyz | m_sol(c.simulation_time());

    u | sel::D = 0;
    u | m.fluid = sol;
    u | sel::R = sol;
}

bool heat::write(field_io& io, const sim_registry& reg, field_ref ref,
                 const step_controller& c, real dt)
{
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, ref, sh);
    auto sol = m.xyz | m_sol(c.simulation_time());

    error = 0;
    error | m.fluid_all(object_bcs) = abs(u - sol);
    error | m.dirichlet(grid_bcs, object_bcs) = 0;

    field_view io_view{std::vector<scalar_view>{u, error}, std::vector<vector_view>{}};

    return io.write(io_names, io_view, c, dt, m.R());
}

} // namespace ccs::systems
