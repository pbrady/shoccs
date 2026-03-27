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
// Uses Kokkos::parallel_for for D and R buffers.
void eval_at_locations(const mesh& m, auto&& func, scalar_span out)
{
    const auto* xv = m.x().data();
    const auto* yv = m.y().data();
    const auto* zv = m.z().data();
    int nx = (int)m.x().size(), ny = (int)m.y().size(), nz = (int)m.z().size();

    // D-buffer: flat parallel_for over cartesian product of x, y, z
    auto* d = out.D.data();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, nx * ny * nz),
        [=, &func](int idx) {
            int i = idx / (ny * nz);
            int j = (idx / nz) % ny;
            int k = idx % nz;
            d[idx] = func(real3{xv[i], yv[j], zv[k]});
        });

    // Rx buffer
    const auto* rx_data = m.Rx().data();
    auto* rx_out = out.Rx.data();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, (int)m.Rx().size()),
        [=, &func](int i) { rx_out[i] = func(rx_data[i].position); });

    // Ry buffer
    const auto* ry_data = m.Ry().data();
    auto* ry_out = out.Ry.data();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, (int)m.Ry().size()),
        [=, &func](int i) { ry_out[i] = func(ry_data[i].position); });

    // Rz buffer
    const auto* rz_data = m.Rz().data();
    auto* rz_out = out.Rz.data();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, (int)m.Rz().size()),
        [=, &func](int i) { rz_out[i] = func(rz_data[i].position); });

    Kokkos::fence();
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
        auto* o = out.data();
        const auto* gxp = gx.data(); const auto* dxp = dx.data();
        const auto* gyp = gy.data(); const auto* dyp = dy.data();
        const auto* gzp = gz.data(); const auto* dzp = dz.data();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, (int)out.size()),
            [=](int i) {
                o[i] = gxp[i] * dxp[i] + gyp[i] * dyp[i] + gzp[i] * dzp[i];
            });
        Kokkos::fence();
    };

    dot_spans(u_rhs.D,  gGx.D,  dux.D,  gGy.D,  duy.D,  gGz.D,  duz.D);
    dot_spans(u_rhs.Rx, gGx.Rx, dux.Rx, gGy.Rx, duy.Rx, gGz.Rx, duz.Rx);
    dot_spans(u_rhs.Ry, gGx.Ry, dux.Ry, gGy.Ry, duy.Ry, gGz.Ry, duz.Ry);
    dot_spans(u_rhs.Rz, gGx.Rz, dux.Rz, gGy.Rz, duy.Rz, gGz.Rz, duz.Rz);
}

void scalar_wave::build_rhs_graph(scalar_view u, scalar_span du)
{
    // Gradient scratch buffers (member data — stable pointers)
    scalar_span dux{du_xd, du_xrx, du_xry, du_xrz};
    scalar_span duy{du_yd, du_yrx, du_yry, du_yrz};
    scalar_span duz{du_zd, du_zrx, du_zry, du_zrz};

    // Wave speed coefficients (member data — stable pointers)
    const real* gx_d = gG_xd.data();
    const real* gy_d = gG_yd.data();
    const real* gz_d = gG_zd.data();
    const real* gx_rx = gG_xrx.data();
    const real* gy_rx = gG_yrx.data();
    const real* gz_rx = gG_zrx.data();
    const real* gx_ry = gG_xry.data();
    const real* gy_ry = gG_yry.data();
    const real* gz_ry = gG_zry.data();
    const real* gx_rz = gG_xrz.data();
    const real* gy_rz = gG_yrz.data();
    const real* gz_rz = gG_zrz.data();

    // Gradient scratch pointers
    const real* dux_d = du_xd.data();
    const real* duy_d = du_yd.data();
    const real* duz_d = du_zd.data();
    const real* dux_rx = du_xrx.data();
    const real* duy_rx = du_yrx.data();
    const real* duz_rx = du_zrx.data();
    const real* dux_ry = du_xry.data();
    const real* duy_ry = du_yry.data();
    const real* duz_ry = du_zry.data();
    const real* dux_rz = du_xrz.data();
    const real* duy_rz = du_yrz.data();
    const real* duz_rz = du_zrz.data();

    // Output pointers and sizes
    real* d_ptr = du.D.data();
    real* rx_ptr = du.Rx.data();
    real* ry_ptr = du.Ry.data();
    real* rz_ptr = du.Rz.data();
    const int n_d = static_cast<int>(du.D.size());
    const int n_rx = static_cast<int>(du.Rx.size());
    const int n_ry = static_cast<int>(du.Ry.size());
    const int n_rz = static_cast<int>(du.Rz.size());

    rhs_graph_ = Kokkos::Experimental::create_graph(
        execution_space{}, [&](auto root) {
            using rp_t = Kokkos::RangePolicy<execution_space>;

            // 1. Gradient: zeros scratch, then dx/dy/dz in parallel
            auto grad_done = grad.add_graph_nodes(root, u, dux, duy, duz);

            // 2. Dot product: du[i] = gGx[i]*dux[i] + gGy[i]*duy[i] + gGz[i]*duz[i]
            grad_done.then_parallel_for(
                "sw_dot_D", rp_t(0, n_d),
                KOKKOS_LAMBDA(int i) {
                    d_ptr[i] = gx_d[i] * dux_d[i] + gy_d[i] * duy_d[i] +
                               gz_d[i] * duz_d[i];
                });
            grad_done.then_parallel_for(
                "sw_dot_Rx", rp_t(0, n_rx),
                KOKKOS_LAMBDA(int i) {
                    rx_ptr[i] = gx_rx[i] * dux_rx[i] + gy_rx[i] * duy_rx[i] +
                                gz_rx[i] * duz_rx[i];
                });
            grad_done.then_parallel_for(
                "sw_dot_Ry", rp_t(0, n_ry),
                KOKKOS_LAMBDA(int i) {
                    ry_ptr[i] = gx_ry[i] * dux_ry[i] + gy_ry[i] * duy_ry[i] +
                                gz_ry[i] * duz_ry[i];
                });
            grad_done.then_parallel_for(
                "sw_dot_Rz", rp_t(0, n_rz),
                KOKKOS_LAMBDA(int i) {
                    rz_ptr[i] = gx_rz[i] * dux_rz[i] + gy_rz[i] * duy_rz[i] +
                                gz_rz[i] * duz_rz[i];
                });
        });

    rhs_graph_->instantiate();
}

void scalar_wave::submit_rhs_graph()
{
    rhs_graph_->submit();
    Kokkos::fence("scalar_wave::submit_rhs_graph() complete");
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
    const auto fd = m.fluid_desc(); // copy for lambda capture
    const real* u_D = u.D.data();
    const real* sol_D = sol.D.data();

    // MinMax reduction for u_min/u_max
    Kokkos::MinMaxScalar<real> minmax_result;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, fd.count()),
        KOKKOS_LAMBDA(int k, Kokkos::MinMaxScalar<real>& update) {
            int i = fd.element(k);
            if (u_D[i] < update.min_val) update.min_val = u_D[i];
            if (u_D[i] > update.max_val) update.max_val = u_D[i];
        },
        Kokkos::MinMax<real>(minmax_result));

    real u_min = minmax_result.min_val;
    real u_max = minmax_result.max_val;

    // MaxLoc reduction for err_d/err_d_idx
    Kokkos::ValLocScalar<real, int> maxloc_result;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, fd.count()),
        KOKKOS_LAMBDA(int k, Kokkos::ValLocScalar<real, int>& update) {
            int i = fd.element(k);
            real e = Kokkos::abs(u_D[i] - sol_D[i]);
            if (e > update.val) {
                update.val = e;
                update.loc = i;
            }
        },
        Kokkos::MaxLoc<real, int>(maxloc_result));

    real err_d = fd.count() > 0 ? maxloc_result.val : 0.0;
    real err_d_idx = fd.count() > 0 ? (real)maxloc_result.loc : 0.0;
    Kokkos::fence();

    // Per-component stats for Rx/Ry/Rz over non-dirichlet object indices
    std::span<const real> u_Rs[] = {u.Rx, u.Ry, u.Rz};
    std::span<const real> sol_Rs[] = {sol.Rx, sol.Ry, sol.Rz};
    real comp_errs[3] = {0.0, 0.0, 0.0};
    real comp_idxs[3] = {0.0, 0.0, 0.0};

    for (int dir = 0; dir < 3; ++dir) {
        auto nd = m.non_dirichlet_object_desc(dir, object_bcs);
        if (nd.count() == 0) continue;

        const real* u_R_ptr = u_Rs[dir].data();
        const real* sol_R_ptr = sol_Rs[dir].data();

        // MinMax reduction for this R component
        Kokkos::MinMaxScalar<real> r_minmax;
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<execution_space>(0, nd.count()),
            KOKKOS_LAMBDA(int k, Kokkos::MinMaxScalar<real>& update) {
                int i = nd.element(k);
                if (u_R_ptr[i] < update.min_val) update.min_val = u_R_ptr[i];
                if (u_R_ptr[i] > update.max_val) update.max_val = u_R_ptr[i];
            },
            Kokkos::MinMax<real>(r_minmax));

        u_min = std::min(u_min, r_minmax.min_val);
        u_max = std::max(u_max, r_minmax.max_val);

        // MaxLoc reduction for error
        Kokkos::ValLocScalar<real, int> r_maxloc;
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<execution_space>(0, nd.count()),
            KOKKOS_LAMBDA(int k, Kokkos::ValLocScalar<real, int>& update) {
                int i = nd.element(k);
                real e = Kokkos::abs(u_R_ptr[i] - sol_R_ptr[i]);
                if (e > update.val) {
                    update.val = e;
                    update.loc = i;
                }
            },
            Kokkos::MaxLoc<real, int>(r_maxloc));

        comp_errs[dir] = r_maxloc.val;
        comp_idxs[dir] = (real)r_maxloc.loc;
    }
    Kokkos::fence();

    real err_rx = comp_errs[0], idx_rx = comp_idxs[0];
    real err_ry = comp_errs[1], idx_ry = comp_idxs[1];
    real err_rz = comp_errs[2], idx_rz = comp_idxs[2];

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

    // Fill D with zeros via parallel_for
    real* u_D = u.D.data();
    int u_D_size = (int)u.D.size();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, u_D_size),
        KOKKOS_LAMBDA(int i) { u_D[i] = 0.0; });

    // Copy sol at fluid indices
    const auto fd = m.fluid_desc(); // copy for lambda capture
    assign_selected(u_D, fd, handle_expr{sol.D.data()});

    // Copy sol's R components to u's R components via parallel_for
    real* u_Rx = u.Rx.data();
    const real* sol_Rx_ptr = sol.Rx.data();
    int rx_size = (int)u.Rx.size();
    real* u_Ry = u.Ry.data();
    const real* sol_Ry_ptr = sol.Ry.data();
    int ry_size = (int)u.Ry.size();
    real* u_Rz = u.Rz.data();
    const real* sol_Rz_ptr = sol.Rz.data();
    int rz_size = (int)u.Rz.size();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, rx_size),
        KOKKOS_LAMBDA(int i) { u_Rx[i] = sol_Rx_ptr[i]; });
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, ry_size),
        KOKKOS_LAMBDA(int i) { u_Ry[i] = sol_Ry_ptr[i]; });
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, rz_size),
        KOKKOS_LAMBDA(int i) { u_Rz[i] = sol_Rz_ptr[i]; });
    Kokkos::fence();
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
    const auto fd = m.fluid_desc(); // copy for lambda capture
    const real* u_D = u.D.data();
    const real* sol_D = sol.D.data();
    real* err_d_ptr = error_d.data();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, fd.count()),
        KOKKOS_LAMBDA(int k) {
            int i = fd.element(k);
            err_d_ptr[i] = Kokkos::abs(u_D[i] - sol_D[i]);
        });

    // Compute |u - sol| at non-dirichlet R indices
    std::span<const real> u_R[] = {u.Rx, u.Ry, u.Rz};
    std::span<const real> sol_R[] = {sol.Rx, sol.Ry, sol.Rz};
    real* err_R_ptrs[] = {error_rx.data(), error_ry.data(), error_rz.data()};
    for (int dir = 0; dir < 3; ++dir) {
        auto nd = m.non_dirichlet_object_desc(dir, object_bcs);
        if (nd.count() == 0) continue;
        const real* u_R_ptr = u_R[dir].data();
        const real* sol_R_ptr = sol_R[dir].data();
        real* err_R_ptr = err_R_ptrs[dir];
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, nd.count()),
            KOKKOS_LAMBDA(int k) {
                int i = nd.element(k);
                err_R_ptr[i] = Kokkos::abs(u_R_ptr[i] - sol_R_ptr[i]);
            });
    }
    Kokkos::fence();

    // Zero Dirichlet grid faces on D buffer
    for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
        fill_selected(error_d.data(), desc, 0.0);
    });

    // Zero Dirichlet object entries on Rx/Ry/Rz buffers
    for (int dir = 0; dir < 3; ++dir) {
        auto gd = m.dirichlet_object_desc(dir, object_bcs);
        fill_selected(err_R_ptrs[dir], gd, 0.0);
    }

    scalar_view err_view{error_d, error_rx, error_ry, error_rz};
    std::vector<scalar_view> io_scalars{u, err_view};

    return io.write(io_names, io_scalars, c, dt, m.R());
}

} // namespace ccs::systems
