// shoccs-export: writes the operator tables and oracle dumps of
// docs/reference/pyshoccs-export.md for a heat configuration.

#include "fields/field_registry.hpp"
#include "fields/scalar.hpp"
#include "fields/selection_desc.hpp"
#include "io/npy.hpp"
#include "mesh/mesh.hpp"
#include "mms/manufactured_solutions.hpp"
#include "operators/boundaries.hpp"
#include "operators/laplacian.hpp"
#include "operators/tables.hpp"
#include "stencils/stencil.hpp"
#include "systems/heat.hpp"
#include "temporal/slot_ops.hpp"
#include "temporal/step_controller.hpp"

#include <Kokkos_Core.hpp>
#include <cxxopts.hpp>
#include <fmt/core.h>
#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace ccs
{
namespace
{

struct options {
    std::string input;
    std::string script;
    fs::path out;
    std::uint64_t seed;
};

// ---- JSON helpers -----------------------------------------------------------

std::string json_real(real x)
{
    if (!std::isfinite(x)) return "null";
    return fmt::format("{:.17g}", x);
}

std::string json_string(std::string_view s)
{
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20)
                out += fmt::format("\\u{:04x}", static_cast<unsigned>(c));
            else
                out += c;
        }
    }
    out += '"';
    return out;
}

template <typename R, typename F>
std::string json_array(const R& r, F&& f)
{
    std::string out = "[";
    bool first = true;
    for (const auto& x : r) {
        if (!first) out += ", ";
        first = false;
        out += f(x);
    }
    out += "]";
    return out;
}

std::string json_reals(std::span<const real> v)
{
    return json_array(v, [](real x) { return json_real(x); });
}

std::string json_ints(std::span<const integer> v)
{
    return json_array(v, [](integer i) { return std::to_string(i); });
}

const char* bc_name(bcs::type t)
{
    switch (t) {
    case bcs::Dirichlet:
        return "dirichlet";
    case bcs::Neumann:
        return "neumann";
    default:
        return "floating";
    }
}

// ---- packed vector helpers --------------------------------------------------

std::vector<real> pack(scalar_view v)
{
    std::vector<real> out;
    out.reserve(v.D.size() + v.Rx.size() + v.Ry.size() + v.Rz.size());
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

// f at every V location: D grid points (z fastest), then the R positions, in
// the order of eval_at_locations (src/systems/detail/scalar_system_utils.hpp).
template <typename F>
std::vector<real> eval_V(const mesh& m, const packed_layout& L, F&& f)
{
    std::vector<real> out(L.n_V);
    const auto x = m.x();
    const auto y = m.y();
    const auto z = m.z();
    integer idx = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
        for (std::size_t j = 0; j < y.size(); ++j)
            for (std::size_t k = 0; k < z.size(); ++k)
                out[idx++] = f(real3{x[i], y[j], z[k]});
    for (int d = 0; d < 3; ++d)
        for (const auto& rec : m.R(d)) out[idx++] = f(rec.position);
    return out;
}

std::string git_commit()
{
    std::string out;
    if (FILE* p = ::popen("git rev-parse HEAD 2>/dev/null", "r")) {
        char buf[128];
        while (std::fgets(buf, sizeof buf, p)) out += buf;
        ::pclose(p);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    if (out.size() != 40 || out.find_first_not_of("0123456789abcdef") != std::string::npos)
        return {};
    return out;
}

// ---- the export ---------------------------------------------------------------

struct exporter {
    const options& opt;
    const fs::path& out;
    const mesh& m;
    const bcs::Grid& grid_bcs;
    const bcs::Object& object_bcs;
    const stencil& st;
    const packed_layout L;

    std::vector<int32_t> fluid_idx;
    std::vector<int32_t> dirichlet_face_idx;

    exporter(const options& opt,
             const fs::path& out,
             const mesh& m,
             const bcs::Grid& g,
             const bcs::Object& o,
             const stencil& st)
        : opt{opt}, out{out}, m{m}, grid_bcs{g}, object_bcs{o}, st{st}, L{m}
    {
        const auto& fd = m.fluid_desc();
        fluid_idx.resize(fd.count());
        for (int i = 0; i < fd.count(); ++i) fluid_idx[i] = fd.element(i);

        // the flattening of heat::build_rhs_graph
        for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
            for (int i = 0; i < desc.count(); ++i)
                dirichlet_face_idx.push_back(desc.element(i));
        });
    }

    fs::path file(std::string_view name) const { return out / name; }

    // Section 2.1
    void write_geometry() const
    {
        npy::write(file("x.npy"), std::vector<real>(m.x().begin(), m.x().end()));
        npy::write(file("y.npy"), std::vector<real>(m.y().begin(), m.y().end()));
        npy::write(file("z.npy"), std::vector<real>(m.z().begin(), m.z().end()));

        for (int d = 0; d < 3; ++d) {
            const auto R = m.R(d);
            const std::size_t n = R.size();
            std::vector<real> psi(n), position(3 * n), normal(3 * n);
            std::vector<uint8_t> ray_outside(n);
            std::vector<int32_t> solid_coord(3 * n), shape_id(n), dir_idx, non_dir_idx;
            std::vector<integer> solid_ic(n);
            for (std::size_t i = 0; i < n; ++i) {
                const auto& r = R[i];
                psi[i] = r.psi;
                for (int c = 0; c < 3; ++c) {
                    position[3 * i + c] = r.position[c];
                    normal[3 * i + c] = r.normal[c];
                    solid_coord[3 * i + c] = r.solid_coord[c];
                }
                ray_outside[i] = r.ray_outside ? 1 : 0;
                solid_ic[i] = m.ic(r.solid_coord);
                shape_id[i] = r.shape_id;
                (object_bcs[r.shape_id] == bcs::Dirichlet ? dir_idx : non_dir_idx)
                    .push_back((int32_t)i);
            }
            npy::write(file(fmt::format("R{}_psi.npy", d)), psi);
            npy::write(file(fmt::format("R{}_position.npy", d)), position, n, 3);
            npy::write(file(fmt::format("R{}_normal.npy", d)), normal, n, 3);
            npy::write(file(fmt::format("R{}_ray_outside.npy", d)), ray_outside);
            npy::write(file(fmt::format("R{}_solid_coord.npy", d)), solid_coord, n, 3);
            npy::write(file(fmt::format("R{}_solid_ic.npy", d)), solid_ic);
            npy::write(file(fmt::format("R{}_shape_id.npy", d)), shape_id);
            npy::write(file(fmt::format("R{}_dirichlet_idx.npy", d)), dir_idx);
            npy::write(file(fmt::format("R{}_non_dirichlet_idx.npy", d)), non_dir_idx);

            const auto& lines = m.lines(d);
            const std::size_t nl = lines.size();
            std::vector<integer> ln(7 * nl);
            std::vector<real> lpsi(2 * nl);
            std::vector<uint8_t> skipped(nl);
            const real nan = std::numeric_limits<real>::quiet_NaN();
            for (std::size_t i = 0; i < nl; ++i) {
                const auto& [stride, start, end] = lines[i];
                ln[7 * i + 0] = stride;
                ln[7 * i + 1] = m.ic(start.mesh_coordinate);
                ln[7 * i + 2] = m.ic(end.mesh_coordinate);
                ln[7 * i + 3] = start.object ? start.object->object_coordinate : -1;
                ln[7 * i + 4] = start.object ? start.object->objectID : -1;
                ln[7 * i + 5] = end.object ? end.object->object_coordinate : -1;
                ln[7 * i + 6] = end.object ? end.object->objectID : -1;
                lpsi[2 * i + 0] = start.object ? start.object->psi : nan;
                lpsi[2 * i + 1] = end.object ? end.object->psi : nan;
                skipped[i] = m.dirichlet_line(start.mesh_coordinate, d, grid_bcs) ? 1 : 0;
            }
            npy::write(file(fmt::format("lines{}.npy", d)), ln, nl, 7);
            npy::write(file(fmt::format("lines{}_psi.npy", d)), lpsi, nl, 2);
            npy::write(file(fmt::format("lines{}_skipped.npy", d)), skipped);
        }

        npy::write(file("fluid_idx.npy"), fluid_idx);
        npy::write(file("dirichlet_face_idx.npy"), dirichlet_face_idx);
    }

    // Section 2.2
    std::vector<real> write_stencil_data() const
    {
        const auto mx = st.query_max();
        const int width = 2 * mx.p + 1;
        std::vector<real> taps(3 * width, 0.0);
        for (int d = 0; d < 3; ++d)
            if (m.extents()[d] > 1)
                st.interior(m.h(d), std::span{taps}.subspan(d * width, width));
        npy::write(file("taps.npy"), taps, 3, width);

        const std::array<std::pair<bcs::type, char>, 3> kinds{
            {{bcs::Dirichlet, 'D'}, {bcs::Floating, 'F'}, {bcs::Neumann, 'N'}}};
        for (int d = 0; d < 3; ++d) {
            if (m.extents()[d] < 2) continue;
            for (bool right : {false, true}) {
                for (auto [kind, tag] : kinds) {
                    const auto q = st.query(kind);
                    if (q.r == 0) continue;
                    std::vector<real> c(mx.r * mx.t), ex(mx.nextra);
                    st.nbs(m.h(d), kind, 1.0, right, c, ex);
                    std::vector<real> block(c.begin(), c.begin() + q.r * q.t);
                    const char side = right ? 'R' : 'L';
                    npy::write(file(fmt::format("wall_{}_{}_{}.npy", d, side, tag)),
                               block,
                               q.r,
                               q.t);
                    if (kind == bcs::Neumann && q.nextra > 0) {
                        std::vector<real> extra(ex.begin(), ex.begin() + q.nextra);
                        npy::write(file(fmt::format("wall_{}_{}_N_extra.npy", d, side)), extra);
                    }
                }
            }
        }
        return taps;
    }

    // Section 2.3
    void write_operators(const laplacian& lap) const
    {
        for (int d = 0; d < 3; ++d) {
            const auto block_lines = block_line_map(m, d, grid_bcs);
            const auto coo = extract_coo(lap.component(d), L, block_lines);

            const std::size_t n = coo.A.size();
            std::vector<integer> row(n), col(n);
            std::vector<real> val(n);
            std::vector<uint8_t> part(n);
            std::vector<int32_t> line(n);
            for (std::size_t i = 0; i < n; ++i) {
                row[i] = coo.A[i].row;
                col[i] = coo.A[i].col;
                val[i] = coo.A[i].val;
                part[i] = coo.A[i].part;
                line[i] = coo.A[i].line;
            }
            npy::write(file(fmt::format("A{}_row.npy", d)), row);
            npy::write(file(fmt::format("A{}_col.npy", d)), col);
            npy::write(file(fmt::format("A{}_val.npy", d)), val);
            npy::write(file(fmt::format("A{}_part.npy", d)), part);
            npy::write(file(fmt::format("A{}_line.npy", d)), line);

            const std::size_t nn = coo.N.size();
            std::vector<integer> nrow(nn), ncol(nn);
            std::vector<real> nval(nn);
            for (std::size_t i = 0; i < nn; ++i) {
                nrow[i] = coo.N[i].row;
                ncol[i] = coo.N[i].col;
                nval[i] = coo.N[i].val;
            }
            npy::write(file(fmt::format("N{}_row.npy", d)), nrow);
            npy::write(file(fmt::format("N{}_col.npy", d)), ncol);
            npy::write(file(fmt::format("N{}_val.npy", d)), nval);

            spdlog::info("A{}: {} entries, N{}: {} entries", d, n, d, nn);
        }
    }

    // Section 4
    void write_compact(const laplacian& lap, std::span<const real> taps) const
    {
        const compact_tables T = build_compact_tables(lap, L, fluid_idx, dirichlet_face_idx);

        for (int d = 0; d < 3; ++d) {
            const auto expected = taps.subspan(d * T.width, T.width);
            if (!std::ranges::equal(T.taps[d], expected))
                spdlog::warn("compact taps of direction {} differ from stencil::interior", d);
        }

        npy::write(file("role.npy"), T.role);
        npy::write(file("exc_rows.npy"), T.exc_rows);
        npy::write(file("exc_rowptr.npy"), T.exc_rowptr);
        npy::write(file("exc_col.npy"), T.exc_col);
        npy::write(file("exc_val.npy"), T.exc_val);

        std::array<integer, 3> interior_rows{};
        for (auto bits : T.role)
            for (int d = 0; d < 3; ++d)
                if (bits & role_bit::interior(d)) ++interior_rows[d];
        spdlog::info("interior rows per direction: {} {} {}; exception rows: {}, nnz: {}",
                     interior_rows[0],
                     interior_rows[1],
                     interior_rows[2],
                     T.exc_rows.size(),
                     T.exc_val.size());
    }

    // Section 2.4
    void write_oracle(const sol::table& tbl,
                      const laplacian& lap,
                      const manufactured_solution& ms,
                      real k,
                      step_controller sc,
                      real dt) const
    {
        auto heat_opt = systems::heat::from_lua(tbl);
        if (!heat_opt) throw std::runtime_error("systems::heat::from_lua failed");
        systems::heat& heat = *heat_opt;

        const real t0 = sc.simulation_time();
        constexpr auto sh = scalar_handle{0};

        sim_registry reg;
        const auto sz = heat.size();
        auto alloc = [&](int slot) {
            return reg.allocate_scalar(
                slot, 0, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
        };
        const field_ref u0 = alloc(0);
        const field_ref u1 = alloc(1);
        const field_ref rk = alloc(2);
        const field_ref srhs = alloc(3);

        // the state simulation_cycle starts from
        heat.initialize(reg, u0, sc);
        reg.deep_copy_slot(u1.slot, u0.slot);
        heat.update_boundary(reg, u0, t0);
        const auto u0_V = pack(extract_scalar_view(reg, u0, sh));
        npy::write(file("u0_V.npy"), u0_V);

        // live RHS graph u0 -> u1 at t0
        heat.build_rhs_graph(extract_scalar_view(reg, u0, sh), extract_scalar_span(reg, u1, sh));
        heat.fill_source(t0);
        heat.submit_rhs_graph();
        npy::write(file("rhs_u0_V.npy"), pack(extract_scalar_view(reg, u1, sh)));

        // random input through the same graph
        std::mt19937_64 gen(opt.seed);
        std::uniform_real_distribution<real> dist(-1.0, 1.0);
        std::vector<real> rand_V(L.n_V);
        for (auto& v : rand_V) v = dist(gen);
        npy::write(file("rand_V.npy"), rand_V);

        unpack(rand_V, extract_scalar_span(reg, u0, sh));
        heat.submit_rhs_graph();
        npy::write(file("rhs_rand_V.npy"), pack(extract_scalar_view(reg, u1, sh)));
        unpack(u0_V, extract_scalar_span(reg, u0, sh));

        // eager Laplacian of the random vector with nu = 0
        {
            const std::vector<real> nu(L.n_V, 0.0);
            std::vector<real> du(L.n_V);
            scalar_span du_sp = span_of(du, L);
            du_sp = lap(view_of(rand_V, L), view_of(nu, L));
            npy::write(file("lap_rand_V.npy"), du);
        }

        // manufactured solution and source at t0 and t0 + dt/2
        if (ms) {
            for (auto [tag, t] : {std::pair{"t0", t0}, std::pair{"t1", t0 + dt / 2}}) {
                npy::write(file(fmt::format("sol_{}_V.npy", tag)),
                           eval_V(m, L, [&](const real3& x) { return ms(t, x); }));
                npy::write(file(fmt::format("src_{}_V.npy", tag)),
                           eval_V(m, L, [&](const real3& x) {
                               return ms.ddt(t, x) - k * ms.laplacian(t, x);
                           }));
            }
        } else {
            spdlog::warn("no manufactured solution: sol/src dumps skipped");
        }

        // one RK4 step from u0, as integrators::rk4 performs it: the graph reads
        // the output slot and writes the system rhs slot, fill_source precedes
        // every submit (system::submit_rhs_graph)
        heat.build_rhs_graph(extract_scalar_view(reg, u1, sh),
                             extract_scalar_span(reg, srhs, sh));
        constexpr std::array rki{0.0, 0.5, 0.5, 1.0};
        constexpr std::array rkf{1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};

        slot_zero(reg, rk);
        slot_zero(reg, srhs);
        reg.deep_copy_slot(u1.slot, u0.slot);
        for (int i = 0; i < 4; ++i) {
            if (i > 0) {
                slot_assign_lc(reg, u1, u0, dt * rki[i], srhs);
                heat.update_boundary(reg, u1, t0 + dt * rki[i]);
            }
            heat.fill_source(t0 + dt * rki[i]);
            heat.submit_rhs_graph();
            slot_accumulate(reg, rk, dt * rkf[i], srhs);
        }
        slot_assign_lc(reg, u1, u0, 1.0, rk);
        heat.update_boundary(reg, u1, t0 + dt);
        npy::write(file("u1_V.npy"), pack(extract_scalar_view(reg, u1, sh)));

        sc.advance(dt);
        const auto stats = heat.stats(reg, u0, u1, sc);
        std::ofstream f(file("stats_u1.json"));
        f << fmt::format("{{\"time\": {}, \"stats\": {}}}\n",
                         json_real(sc.simulation_time()),
                         json_reals(stats.stats));
    }

    void write_meta(const sol::table& tbl, real k, real t0, real dt) const
    {
        std::string meta = "{\n";
        auto field = [&](std::string_view key, const std::string& value) {
            meta += fmt::format("  \"{}\": {},\n", key, value);
        };

        const int3& ext = m.extents();
        const std::array<real, 3> dmin{m.x().front(), m.y().front(), m.z().front()};
        const std::array<real, 3> dmax{m.x().back(), m.y().back(), m.z().back()};
        std::array<real, 3> h{};
        for (int d = 0; d < 3; ++d) h[d] = ext[d] > 1 ? m.h(d) : 0.0;

        field("export_version", "1");
        field("config", json_string(opt.input));
        field("script", json_string(opt.script));
        field("dims", std::to_string(m.dims()));
        field("extents", fmt::format("[{}, {}, {}]", ext[0], ext[1], ext[2]));
        field("domain_min", json_reals(dmin));
        field("domain_max", json_reals(dmax));
        field("h", json_reals(h));
        field("n_D", std::to_string(L.n_D));
        field("n_R", json_ints(L.n_R));
        field("offsets",
              fmt::format("{{\"D\": {}, \"Rx\": {}, \"Ry\": {}, \"Rz\": {}}}",
                          L.offsets[0],
                          L.offsets[1],
                          L.offsets[2],
                          L.offsets[3]));
        field("n_V", std::to_string(L.n_V));
        field("grid_bcs", json_array(grid_bcs, [](const bcs::Line& l) {
                  return fmt::format("[\"{}\", \"{}\"]", bc_name(l.left), bc_name(l.right));
              }));
        field("object_bcs", json_array(object_bcs, [](bcs::type t) {
                  return fmt::format("\"{}\"", bc_name(t));
              }));

        const auto mx = st.query_max();
        const auto ip = st.query_interp();
        auto query = [&](bcs::type t) {
            const auto q = st.query(t);
            return fmt::format(
                "{{\"r\": {}, \"t\": {}, \"nextra\": {}}}", q.r, q.t, q.nextra);
        };
        field("scheme",
              fmt::format("{{\"type\": {}, \"order\": {}, \"p\": {}, \"interp\": {{\"p\": {}, "
                          "\"t\": {}}}, \"query\": {{\"dirichlet\": {}, \"floating\": {}, "
                          "\"neumann\": {}}}}}",
                          json_string(tbl["scheme"]["type"].get_or(std::string{})),
                          tbl["scheme"]["order"].get_or(1),
                          mx.p,
                          ip.p,
                          ip.t,
                          query(bcs::Dirichlet),
                          query(bcs::Floating),
                          query(bcs::Neumann)));
        field("system",
              fmt::format("{{\"type\": {}, \"diffusivity\": {}}}",
                          json_string(tbl["system"]["type"].get_or(std::string{})),
                          json_real(k)));

        // the manufactured solution as manufactured_solution::from_lua reads it
        std::string mms = "null";
        if (auto ms = tbl["manufactured_solution"]; ms.valid()) {
            std::string type = ms["type"].get_or(std::string{});
            std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) {
                return std::tolower(c);
            });
            if (type == "gaussian") {
                std::vector<real3> center, variance;
                std::vector<real> amplitude, frequency;
                for (int i = 1; ms[i].valid(); ++i) {
                    auto t = ms[i];
                    center.push_back({t["center"][1].get_or(0.0),
                                      t["center"][2].get_or(0.0),
                                      t["center"][3].get_or(0.0)});
                    variance.push_back({t["variance"][1].get_or(0.0),
                                        t["variance"][2].get_or(0.0),
                                        t["variance"][3].get_or(0.0)});
                    amplitude.push_back(t["amplitude"].get_or(0.0));
                    frequency.push_back(t["frequency"].get_or(0.0));
                }
                auto r3 = [](const real3& c) { return json_reals(c); };
                mms = fmt::format("{{\"type\": \"gaussian\", \"center\": {}, \"variance\": {}, "
                                  "\"amplitude\": {}, \"frequency\": {}}}",
                                  json_array(center, r3),
                                  json_array(variance, r3),
                                  json_reals(amplitude),
                                  json_reals(frequency));
            } else {
                mms = fmt::format("{{\"type\": {}}}", json_string(type));
            }
        }
        field("mms", mms);
        field("t0", json_real(t0));
        field("dt", json_real(dt));
        field("seed", std::to_string(opt.seed));
        if (auto commit = git_commit(); !commit.empty())
            field("git_commit", json_string(commit));

        meta.erase(meta.size() - 2); // the last ",\n"
        meta += "\n}\n";
        std::ofstream f(file("meta.json"));
        f << meta;
    }
};

int run(const options& opt)
{
    const auto t_start = std::chrono::steady_clock::now();

    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script_file(opt.input);
    if (!opt.script.empty()) lua.script(opt.script);

    sol::object sim = lua["simulation"];
    if (!sim.is<sol::table>()) {
        spdlog::error("the input does not define a `simulation` table");
        return 1;
    }
    const sol::table tbl = sim.as<sol::table>();

    // the factories heat::from_lua uses
    auto mesh_opt = mesh::from_lua(tbl);
    if (!mesh_opt) {
        spdlog::error("mesh::from_lua failed");
        return 1;
    }
    const mesh& m = *mesh_opt;
    auto bc_opt = bcs::from_lua(tbl, m.extents());
    auto st_opt = stencil::from_lua(tbl);
    if (!bc_opt || !st_opt) {
        spdlog::error("bcs::from_lua or stencil::from_lua failed");
        return 1;
    }
    const bcs::Grid& grid_bcs = bc_opt->first;
    const bcs::Object& object_bcs = bc_opt->second;
    const stencil& st = *st_opt;
    auto ms_opt = manufactured_solution::from_lua(tbl, m.dims());
    const manufactured_solution ms = ms_opt ? MOVE(*ms_opt) : manufactured_solution{};

    const real k = tbl["system"]["diffusivity"].get_or(1.0);
    auto sc_opt = step_controller::from_lua(tbl);
    const step_controller sc = sc_opt ? *sc_opt : step_controller{};
    const real t0 = sc.simulation_time();
    const real h_min = std::ranges::min(m.h());
    const real dt = sc.parabolic_cfl() * h_min * h_min / (4 * k);

    const laplacian lap{m, st, grid_bcs, object_bcs};

    fs::create_directories(opt.out);
    const exporter ex{opt, opt.out, m, grid_bcs, object_bcs, st};
    spdlog::info("n_D = {}, n_R = [{}, {}, {}], n_V = {}, t0 = {}, dt = {}",
                 ex.L.n_D,
                 ex.L.n_R[0],
                 ex.L.n_R[1],
                 ex.L.n_R[2],
                 ex.L.n_V,
                 t0,
                 dt);

    ex.write_geometry();
    const auto taps = ex.write_stencil_data();
    ex.write_operators(lap);
    ex.write_compact(lap, taps);
    ex.write_oracle(tbl, lap, ms, k, sc, dt);
    ex.write_meta(tbl, k, t0, dt);

    const auto t_end = std::chrono::steady_clock::now();
    spdlog::info("wrote {} in {:.3f} s",
                 opt.out.string(),
                 std::chrono::duration<double>(t_end - t_start).count());
    return 0;
}

} // namespace
} // namespace ccs

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);

    cxxopts::Options options(
        "shoccs-export",
        "Export the operator tables and oracle dumps of a heat configuration");

    // clang-format off
    options.add_options()
        ("input-file", "Main lua input file", cxxopts::value<std::string>())
        ("script", "Supplementary lua options take precedence over file",
             cxxopts::value<std::string>()->default_value(""))
        ("out", "Output directory", cxxopts::value<std::string>())
        ("seed", "Seed of the random vector",
             cxxopts::value<std::uint64_t>()->default_value("20261013"))
        ("help", "Print usage");
    // clang-format on
    options.parse_positional("input-file");

    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("input-file") || !result.count("out")) {
        std::cout << options.help() << '\n';
        return result.count("help") ? 0 : 1;
    }

    const ccs::options opt{result["input-file"].as<std::string>(),
                           result["script"].as<std::string>(),
                           result["out"].as<std::string>(),
                           result["seed"].as<std::uint64_t>()};
    try {
        return ccs::run(opt);
    } catch (const std::exception& e) {
        spdlog::error("export failed: {}", e.what());
        return 1;
    }
}
