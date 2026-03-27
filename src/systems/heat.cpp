#include "heat.hpp"
#include "fields/expr.hpp"
#include "fields/selection_desc.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include <fmt/ranges.h>
#include <sol/sol.hpp>

#include "operators/discrete_operator.hpp"

#include <iterator>

namespace ccs::systems
{

enum class scalars : int { u };

namespace
{
// Evaluate func(loc) at every mesh location, storing results in out.
void eval_at_locations(const mesh& m, auto&& func, scalar_span out)
{
    int idx = 0;
    for (auto&& loc : ccs::cartesian_product(m.x(), m.y(), m.z()))
        out.D[idx++] = func(real3{std::get<0>(loc), std::get<1>(loc), std::get<2>(loc)});
    for (size_t i = 0; i < m.Rx().size(); ++i)
        out.Rx[i] = func(m.Rx()[i].position);
    for (size_t i = 0; i < m.Ry().size(); ++i)
        out.Ry[i] = func(m.Ry()[i].position);
    for (size_t i = 0; i < m.Rz().size(); ++i)
        out.Rz[i] = func(m.Rz()[i].position);
}
} // namespace

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
      neumann_d(this->m.size()), neumann_rx(this->m.Rx().size()),
      neumann_ry(this->m.Ry().size()), neumann_rz(this->m.Rz().size()),
      error_d(this->m.size()), error_rx(this->m.Rx().size()),
      error_ry(this->m.Ry().size()), error_rz(this->m.Rz().size()),
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

system_size heat::size() const
{
    return {1, 0, m.size(), (integer)m.Rx().size(), (integer)m.Ry().size(), (integer)m.Rz().size()};
}

void heat::rhs(const sim_registry& reg, field_ref input,
               sim_registry& out_reg, field_ref output, real time) const
{
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, input, sh);
    auto u_rhs = extract_scalar_span(out_reg, output, sh);

    // rhs = diffusivity * lap(u) + (dS/dt - diffusivity * lap(S))
    u_rhs = lap(u, scalar_view{neumann_d, neumann_rx, neumann_ry, neumann_rz});
    times_assign_scalar(out_reg, output, sh, diffusivity);

    if (m_sol) {
        // Evaluate source expression at all mesh locations
        std::vector<real> src_d(m.size());
        std::vector<real> src_rx(m.Rx().size());
        std::vector<real> src_ry(m.Ry().size());
        std::vector<real> src_rz(m.Rz().size());
        scalar_span src{src_d, src_rx, src_ry, src_rz};
        eval_at_locations(m, [&](const real3& loc) {
            return m_sol.ddt(time, loc) - diffusivity * m_sol.laplacian(time, loc);
        }, src);

        // Shared destination pointers
        real* rhs_D = out_reg.data(output, sh.D());
        auto R = sh.R();

        // Fluid on D buffer: plus_assign from gather_selection of fluid indices
        plus_assign_selected(rhs_D, m.fluid_desc(), handle_expr{src.D.data()});

        // Non-dirichlet objects on Rx/Ry/Rz buffers
        real* src_R[] = {src.Rx.data(), src.Ry.data(), src.Rz.data()};
        for (int dir = 0; dir < 3; ++dir) {
            auto gd = m.non_dirichlet_object_desc(dir, object_bcs);
            plus_assign_selected(out_reg.data(output, R[dir]), gd,
                                 handle_expr{src_R[dir]});
        }

        // Grid Dirichlet: fill plane subsets of D buffer with zero
        for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
            fill_selected(rhs_D, desc, 0.0);
        });

        // Object Dirichlet: fill predicate subsets of Rx/Ry/Rz buffers
        for (int dir = 0; dir < 3; ++dir) {
            auto gd = m.dirichlet_object_desc(dir, object_bcs);
            fill_selected(out_reg.data(output, R[dir]), gd, 0.0);
        }
    }
}

void heat::update_boundary(sim_registry& reg, field_ref ref, real time)
{
    constexpr auto sh = scalar_handle{0};
    // Evaluate manufactured solution at all mesh locations
    std::vector<real> sol_d(m.size());
    std::vector<real> sol_rx(m.Rx().size());
    std::vector<real> sol_ry(m.Ry().size());
    std::vector<real> sol_rz(m.Rz().size());
    scalar_span sol{sol_d, sol_rx, sol_ry, sol_rz};
    eval_at_locations(m, [&](const real3& loc) {
        return m_sol(time, loc);
    }, sol);

    // Grid Dirichlet: assign plane subsets of D buffer
    real* u_D = reg.data(ref, sh.D());
    for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
        assign_selected(u_D, desc, handle_expr{sol.D.data()});
    });

    // Object Dirichlet: assign predicate subsets of Rx/Ry/Rz buffers
    auto R = sh.R();
    real* sol_R[] = {sol.Rx.data(), sol.Ry.data(), sol.Rz.data()};
    for (int dir = 0; dir < 3; ++dir) {
        auto gd = m.dirichlet_object_desc(dir, object_bcs);
        assign_selected(reg.data(ref, R[dir]), gd, handle_expr{sol_R[dir]});
    }

    // Set Neumann BCs: evaluate gradient component at domain locations, assign at faces
    scalar_span neu{neumann_d, neumann_rx, neumann_ry, neumann_rz};
    auto ext = m.extents();
    for (int dir = 0; dir < 3; ++dir) {
        bool need_left = grid_bcs[dir].left == bcs::Neumann;
        bool need_right = grid_bcs[dir].right == bcs::Neumann;
        if (!need_left && !need_right) continue;

        std::vector<real> grad_d(m.size());
        int idx = 0;
        for (auto&& loc : ccs::cartesian_product(m.x(), m.y(), m.z()))
            grad_d[idx++] = m_sol.gradient(time,
                real3{std::get<0>(loc), std::get<1>(loc), std::get<2>(loc)})[dir];
        auto src = handle_expr{grad_d.data()};

        auto assign_face = [&](int face_idx) {
            if (dir == 0)
                assign_selected(neu.D.data(), make_x_plane_desc(ext, face_idx), src);
            else if (dir == 1)
                assign_selected(neu.D.data(), make_y_plane_desc(ext, face_idx), src);
            else
                assign_selected(neu.D.data(), make_z_plane_desc(ext, face_idx), src);
        };

        if (need_left) assign_face(0);
        if (need_right) assign_face(ext[dir] - 1);
    }
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

    // Evaluate manufactured solution at all mesh locations
    std::vector<real> sol_d(m.size());
    std::vector<real> sol_rx(m.Rx().size());
    std::vector<real> sol_ry(m.Ry().size());
    std::vector<real> sol_rz(m.Rz().size());
    scalar_span sol{sol_d, sol_rx, sol_ry, sol_rz};
    eval_at_locations(m, [&](const real3& loc) {
        return m_sol(step.simulation_time(), loc);
    }, sol);

    // Compute min/max and per-component error over fluid D indices
    real u_min = std::numeric_limits<real>::max();
    real u_max = std::numeric_limits<real>::lowest();
    real err_d = 0.0;
    real err_d_idx = 0.0;

    const auto& fd = m.fluid_desc();
    for (int k = 0; k < fd.count(); ++k) {
        int i = fd.element(k);
        u_min = std::min(u_min, u.D[i]);
        u_max = std::max(u_max, u.D[i]);
        real e = std::abs(u.D[i] - sol.D[i]);
        if (e > err_d) {
            err_d = e;
            err_d_idx = (real)i;
        }
    }

    // Per-component stats for Rx/Ry/Rz over non-dirichlet object indices
    auto component_stats = [&](std::span<const real> u_R,
                               std::span<const real> sol_R,
                               int dir) -> std::pair<real, real> {
        auto nd = m.non_dirichlet_object_desc(dir, object_bcs);
        real comp_err = 0.0;
        real comp_idx = 0.0;
        for (int k = 0; k < nd.count(); ++k) {
            int i = nd.element(k);
            u_min = std::min(u_min, u_R[i]);
            u_max = std::max(u_max, u_R[i]);
            real e = std::abs(u_R[i] - sol_R[i]);
            if (e > comp_err) {
                comp_err = e;
                comp_idx = (real)i;
            }
        }
        return {comp_err, comp_idx};
    };

    auto [err_rx, idx_rx] = component_stats(u.Rx, sol.Rx, 0);
    auto [err_ry, idx_ry] = component_stats(u.Ry, sol.Ry, 1);
    auto [err_rz, idx_rz] = component_stats(u.Rz, sol.Rz, 2);

    real err = std::max({err_d, err_rx, err_ry, err_rz});
    return system_stats{.stats = {err,
                                  u_min,
                                  u_max,
                                  err_d,
                                  err_d_idx,
                                  err_rx,
                                  idx_rx,
                                  err_ry,
                                  idx_ry,
                                  err_rz,
                                  idx_rz}};
}

void heat::initialize(sim_registry& reg, field_ref ref, const step_controller& c)
{
    if (!m_sol) return;

    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_span(reg, ref, sh);

    // Evaluate manufactured solution at all mesh locations
    std::vector<real> sol_d(m.size());
    std::vector<real> sol_rx(m.Rx().size());
    std::vector<real> sol_ry(m.Ry().size());
    std::vector<real> sol_rz(m.Rz().size());
    scalar_span sol{sol_d, sol_rx, sol_ry, sol_rz};
    eval_at_locations(m, [&](const real3& loc) {
        return m_sol(c.simulation_time(), loc);
    }, sol);

    // Fill D with zeros, then copy sol at fluid indices
    std::ranges::fill(u.D, 0.0);
    const auto& fd = m.fluid_desc();
    for (int k = 0; k < fd.count(); ++k) {
        int i = fd.element(k);
        u.D[i] = sol.D[i];
    }

    // Copy sol's R components to u's R components
    std::ranges::copy(sol.Rx, u.Rx.begin());
    std::ranges::copy(sol.Ry, u.Ry.begin());
    std::ranges::copy(sol.Rz, u.Rz.begin());
}

bool heat::write(field_io& io, const sim_registry& reg, field_ref ref,
                 const step_controller& c, real dt)
{
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, ref, sh);

    // Evaluate manufactured solution at all mesh locations
    std::vector<real> sol_d(m.size());
    std::vector<real> sol_rx(m.Rx().size());
    std::vector<real> sol_ry(m.Ry().size());
    std::vector<real> sol_rz(m.Rz().size());
    scalar_span sol{sol_d, sol_rx, sol_ry, sol_rz};
    eval_at_locations(m, [&](const real3& loc) {
        return m_sol(c.simulation_time(), loc);
    }, sol);

    // Zero all error buffers
    std::ranges::fill(error_d, 0.0);
    std::ranges::fill(error_rx, 0.0);
    std::ranges::fill(error_ry, 0.0);
    std::ranges::fill(error_rz, 0.0);

    // Compute |u - sol| at fluid D indices
    const auto& fd = m.fluid_desc();
    for (int k = 0; k < fd.count(); ++k) {
        int i = fd.element(k);
        error_d[i] = std::abs(u.D[i] - sol.D[i]);
    }

    // Compute |u - sol| at non-dirichlet R indices
    std::span<const real> u_R[] = {u.Rx, u.Ry, u.Rz};
    std::span<const real> sol_R[] = {sol.Rx, sol.Ry, sol.Rz};
    std::span<real> err_R[] = {std::span{error_rx}, std::span{error_ry}, std::span{error_rz}};
    for (int dir = 0; dir < 3; ++dir) {
        auto nd = m.non_dirichlet_object_desc(dir, object_bcs);
        for (int k = 0; k < nd.count(); ++k) {
            int i = nd.element(k);
            err_R[dir][i] = std::abs(u_R[dir][i] - sol_R[dir][i]);
        }
    }

    // Zero Dirichlet grid faces on D buffer
    for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
        fill_selected(error_d.data(), desc, 0.0);
    });

    // Zero Dirichlet object entries on Rx/Ry/Rz buffers
    for (int dir = 0; dir < 3; ++dir) {
        auto gd = m.dirichlet_object_desc(dir, object_bcs);
        fill_selected(err_R[dir].data(), gd, 0.0);
    }

    scalar_view err_view{error_d, error_rx, error_ry, error_rz};
    std::vector<scalar_view> io_scalars{u, err_view};

    return io.write(io_names, io_scalars, c, dt, m.R());
}

} // namespace ccs::systems
