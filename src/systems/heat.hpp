#pragma once

#include "fields/field.hpp"
#include "fields/field_registry.hpp"
#include "io/field_io.hpp"
#include "mesh/mesh.hpp"
#include "mms/manufactured_solutions.hpp"
#include "operators/laplacian.hpp"
#include "temporal/step_controller.hpp"
#include <sol/forward.hpp>

namespace ccs::systems
{
//
// solve dT/dt = k lap T
//
class heat
{

    mesh m;
    bcs::Grid grid_bcs;
    bcs::Object object_bcs;
    manufactured_solution m_sol;

    laplacian lap;
    real diffusivity;

    scalar_real neumann_u;
    scalar_real error;

    logs logger;

    std::vector<std::string> io_names = {"U", "Error"};

public:
    heat() = default;

    heat(mesh&& m,
         bcs::Grid&& grid_bcs,
         bcs::Object&& object_bcs,
         manufactured_solution&& m_sol,
         stencil st,
         real diffusivity,
         const logs& = {});

    static std::optional<heat> from_lua(const sol::table&, const logs& = {});

    bool valid(const system_stats&) const;

    void log(const system_stats&, const step_controller&);

    real3 summary(const system_stats&) const;

    system_size size() const;

    void rhs(const sim_registry& reg, field_ref input,
             sim_registry& out_reg, field_ref output, real time) const;
    void update_boundary(sim_registry& reg, field_ref ref, real time);
    real timestep_size(const sim_registry& reg, field_ref ref,
                       const step_controller&) const;
    system_stats stats(const sim_registry& reg, field_ref u0,
                       field_ref u1, const step_controller&) const;
    void initialize(sim_registry& reg, field_ref ref, const step_controller&);
    bool write(field_io& io, const sim_registry& reg, field_ref ref,
               const step_controller& c, real dt);
};
} // namespace ccs::systems
