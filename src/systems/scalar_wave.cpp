#include "scalar_wave.hpp"
#include "fields/expr.hpp"
#include "fields/selection_desc.hpp"
#include "real3_operators.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include <sol/sol.hpp>

#include "operators/discrete_operator.hpp"

#include <fmt/ranges.h>
#include <iterator>
#include <ranges>

namespace ccs::systems
{
namespace
{

constexpr real twoPI = 2 * std::numbers::pi_v<real>;

// negative gradient coefficient for spatial component `comp`
constexpr auto neg_G_at(int comp, const real3& center)
{
    return [=](const real3& location) -> real {
        return -(location[comp] - center[comp]) / length(location - center);
    };
}

constexpr auto solution_at(const real3& center, real radius, real time)
{
    return [=](const real3& location) {
        return std::sin(twoPI * (length(location - center) - radius - time));
    };
}

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

scalar_wave::scalar_wave(mesh&& m_,
                         bcs::Grid&& grid_bcs,
                         bcs::Object&& object_bcs,
                         stencil st,
                         real3 center,
                         real radius,
                         real max_error,
                         const logs& build_logger)
    : m{MOVE(m_)},
      grid_bcs{MOVE(grid_bcs)},
      object_bcs{MOVE(object_bcs)},
      grad{gradient(this->m, st, this->grid_bcs, this->object_bcs, build_logger)},
      center{center},
      radius{radius},
      gG_xd(m.size()), gG_xrx(m.Rx().size()), gG_xry(m.Ry().size()), gG_xrz(m.Rz().size()),
      gG_yd(m.size()), gG_yrx(m.Rx().size()), gG_yry(m.Ry().size()), gG_yrz(m.Rz().size()),
      gG_zd(m.size()), gG_zrx(m.Rx().size()), gG_zry(m.Ry().size()), gG_zrz(m.Rz().size()),
      du_xd(m.size()), du_xrx(m.Rx().size()), du_xry(m.Ry().size()), du_xrz(m.Rz().size()),
      du_yd(m.size()), du_yrx(m.Rx().size()), du_yry(m.Ry().size()), du_yrz(m.Rz().size()),
      du_zd(m.size()), du_zrx(m.Rx().size()), du_zry(m.Ry().size()), du_zrz(m.Rz().size()),
      error_d(m.size()), error_rx(m.Rx().size()),
      error_ry(m.Ry().size()), error_rz(m.Rz().size()),
      max_error{max_error},
      logger{build_logger, "system", "system.csv"}
{

    // Initialize wave speed coefficients at all mesh locations
    scalar_span gG_x{gG_xd, gG_xrx, gG_xry, gG_xrz};
    scalar_span gG_y{gG_yd, gG_yrx, gG_yry, gG_yrz};
    scalar_span gG_z{gG_zd, gG_zrx, gG_zry, gG_zrz};

    eval_at_locations(m, neg_G_at(0, center), gG_x);
    eval_at_locations(m, neg_G_at(1, center), gG_y);
    eval_at_locations(m, neg_G_at(2, center), gG_z);

    // Zero Dirichlet grid boundaries on D buffers
    for_each_grid_bc_desc<bcs::Dirichlet>(this->grid_bcs, m.extents(), [&](auto desc) {
        fill_selected(gG_xd.data(), desc, 0.0);
        fill_selected(gG_yd.data(), desc, 0.0);
        fill_selected(gG_zd.data(), desc, 0.0);
    });

    // Zero Dirichlet object boundaries on Rx/Ry/Rz buffers
    for (int dir = 0; dir < 3; ++dir) {
        auto gd = m.dirichlet_object_desc(dir, this->object_bcs);
        real* x_r = dir == 0 ? gG_xrx.data() : dir == 1 ? gG_xry.data() : gG_xrz.data();
        real* y_r = dir == 0 ? gG_yrx.data() : dir == 1 ? gG_yry.data() : gG_yrz.data();
        real* z_r = dir == 0 ? gG_zrx.data() : dir == 1 ? gG_zry.data() : gG_zrz.data();
        fill_selected(x_r, gd, 0.0);
        fill_selected(y_r, gd, 0.0);
        fill_selected(z_r, gd, 0.0);
    }

    spdlog::debug("-grad_G {}\n", gG_xrx[0]);

    logger.set_pattern("%v");
    logger(spdlog::level::info,
           "Timestamp,Time,Step,Linf,Min,Max,Domain_Linf,Domain_ic,Rx_Linf,Rx_ic,Ry_"
           "Linf,Ry_ic,Rz_Linf,Rz_ic");
    logger.set_pattern("%Y-%m-%d %H:%M:%S.%f,%v");
}


//
// Determine if the computed field is valid by checking the linf error
//
bool scalar_wave::valid(const system_stats& stats) const
{
    const auto& v = stats.stats[0];
    return std::isfinite(v) && std::abs(v) <= max_error;
}

real3 scalar_wave::summary(const system_stats& stats) const
{
    return {stats.stats[0], stats.stats[1], stats.stats[2]};
}

void scalar_wave::log(const system_stats& stats, const step_controller& step)
{
    logger(spdlog::level::info,
           "{},{},{}",
           (real)step,
           (int)step,
           fmt::join(stats.stats, ","));
}

system_size scalar_wave::size() const
{
    return {1, 0, m.size(), (integer)m.Rx().size(), (integer)m.Ry().size(), (integer)m.Rz().size()};
}

std::optional<scalar_wave> scalar_wave::from_lua(const sol::table& tbl,
                                                 const logs& logger)
{
    real max_error = tbl["system"]["max_error"].get_or(100.0);
    // assume we can only get here if simulation.system.type == "scalar_wave" so check
    // for the rest
    real3 center;
    real radius;
    // if the center/radius was specified in the system table, use it.
    if (tbl["system"]["center"].valid() && tbl["system"]["radius"].valid()) {

        auto c = tbl["system"]["center"];
        center = {c[1].get_or(0.0), c[2].get_or(0.0), c[3].get_or(0.0)};
        radius = tbl["system"]["radius"];

    } else if (tbl["shapes"].valid()) {
        // attempt to extract the center/radius from the first specified shape in the
        // shapes table
        bool found{false};
        auto t = tbl["shapes"];
        for (int i = 1; t[i].valid() && !found; i++) {
            found = (t[i]["type"].get_or(std::string{}) == "sphere");
            if (found) {
                center = {t[i]["center"][1].get_or(0.0),
                          t[i]["center"][2].get_or(0.0),
                          t[i]["center"][3].get_or(0.0)};
                radius = t[i]["radius"].get_or(0.0);
            }
        }
        if (!found) {
            logger(spdlog::level::err,
                   "No valid spheres found in simulation.shapes for scalar_wave");
            return std::nullopt;
        }
    } else {
        logger(spdlog::level::err,
               "a system.center / system.radius must be specified for scalar_wave");
        return std::nullopt;
    }

    auto mesh_opt = mesh::from_lua(tbl, logger);
    if (!mesh_opt) return std::nullopt;

    auto bc_opt = bcs::from_lua(tbl, mesh_opt->extents(), logger);
    auto st_opt = stencil::from_lua(tbl, logger);

    if (bc_opt && st_opt) {

        return scalar_wave{MOVE(*mesh_opt),
                           MOVE(bc_opt->first),
                           MOVE(bc_opt->second),
                           *st_opt,
                           center,
                           radius,
                           max_error,
                           logger};
    }

    return std::nullopt;
}

void scalar_wave::rhs(const sim_registry& reg, field_ref input,
                      sim_registry& out_reg, field_ref output, real /*time*/)
{
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, input, sh);

    scalar_span dux{du_xd, du_xrx, du_xry, du_xrz};
    scalar_span duy{du_yd, du_yrx, du_yry, du_yrz};
    scalar_span duz{du_zd, du_zrx, du_zry, du_zrz};

    grad(u)(dux, duy, duz);

    // u_rhs = dot(grad_G, du) = gG_x * du_x + gG_y * du_y + gG_z * du_z
    auto sp = [&](buf_handle bh) -> std::span<real> {
        return {out_reg.data(output, bh),
                static_cast<std::size_t>(out_reg.size(output, bh))};
    };
    scalar_span u_rhs{sp(sh.D()), sp(sh.Rx()), sp(sh.Ry()), sp(sh.Rz())};

    scalar_view gGx{gG_xd, gG_xrx, gG_xry, gG_xrz};
    scalar_view gGy{gG_yd, gG_yrx, gG_yry, gG_yrz};
    scalar_view gGz{gG_zd, gG_zrx, gG_zry, gG_zrz};

    auto dot_spans = [](std::span<real> out,
                        std::span<const real> gx, std::span<const real> dx,
                        std::span<const real> gy, std::span<const real> dy,
                        std::span<const real> gz, std::span<const real> dz) {
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = gx[i] * dx[i] + gy[i] * dy[i] + gz[i] * dz[i];
    };

    dot_spans(u_rhs.D,  gGx.D,  dux.D,  gGy.D,  duy.D,  gGz.D,  duz.D);
    dot_spans(u_rhs.Rx, gGx.Rx, dux.Rx, gGy.Rx, duy.Rx, gGz.Rx, duz.Rx);
    dot_spans(u_rhs.Ry, gGx.Ry, dux.Ry, gGy.Ry, duy.Ry, gGz.Ry, duz.Ry);
    dot_spans(u_rhs.Rz, gGx.Rz, dux.Rz, gGy.Rz, duy.Rz, gGz.Rz, duz.Rz);
}

void scalar_wave::update_boundary(sim_registry& reg, field_ref ref, real time)
{
    constexpr auto sh = scalar_handle{0};

    // Evaluate solution at all mesh locations
    std::vector<real> sol_d(m.size());
    std::vector<real> sol_rx(m.Rx().size());
    std::vector<real> sol_ry(m.Ry().size());
    std::vector<real> sol_rz(m.Rz().size());
    scalar_span sol{sol_d, sol_rx, sol_ry, sol_rz};
    eval_at_locations(m, solution_at(center, radius, time), sol);

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
}

real scalar_wave::timestep_size(const sim_registry&, field_ref,
                                const step_controller& step) const
{
    const auto h_min = std::ranges::min(m.h());
    return step.hyperbolic_cfl() * h_min;
}

system_stats scalar_wave::stats(const sim_registry& reg, field_ref /*u0*/,
                                field_ref u1, const step_controller& c) const
{
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, u1, sh);

    // Evaluate solution at all mesh locations
    std::vector<real> sol_d(m.size());
    std::vector<real> sol_rx(m.Rx().size());
    std::vector<real> sol_ry(m.Ry().size());
    std::vector<real> sol_rz(m.Rz().size());
    scalar_span sol{sol_d, sol_rx, sol_ry, sol_rz};
    eval_at_locations(m, solution_at(center, radius, c), sol);

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

void scalar_wave::initialize(sim_registry& reg, field_ref ref, const step_controller& c)
{
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_span(reg, ref, sh);

    // Evaluate solution at all mesh locations
    std::vector<real> sol_d(m.size());
    std::vector<real> sol_rx(m.Rx().size());
    std::vector<real> sol_ry(m.Ry().size());
    std::vector<real> sol_rz(m.Rz().size());
    scalar_span sol{sol_d, sol_rx, sol_ry, sol_rz};
    eval_at_locations(m, solution_at(center, radius, c), sol);

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

bool scalar_wave::write(field_io& io, const sim_registry& reg, field_ref ref,
                        const step_controller& c, real dt)
{
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, ref, sh);

    // Evaluate solution at all mesh locations
    std::vector<real> sol_d(m.size());
    std::vector<real> sol_rx(m.Rx().size());
    std::vector<real> sol_ry(m.Ry().size());
    std::vector<real> sol_rz(m.Rz().size());
    scalar_span sol{sol_d, sol_rx, sol_ry, sol_rz};
    eval_at_locations(m, solution_at(center, radius, (real)c), sol);

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
