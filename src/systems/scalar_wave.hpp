#pragma once

#include "fields/field.hpp"
#include "fields/field_registry.hpp"
#include "io/field_io.hpp"
#include "operators/gradient.hpp"
#include "temporal/step_controller.hpp"
#include "types.hpp"

#include <sol/forward.hpp>

namespace ccs::systems
{

// the system of pdes to solve is in this class
class scalar_wave
{
    mesh m;
    bcs::Grid grid_bcs;
    bcs::Object object_bcs;

    gradient grad;

    // required data
    // std::vector<double> grad_c;
    // std::vector<double> grad_u;

    real3 center; // center of the circular wave
    real radius;

    vector_real grad_G;
    vector_real du;

    scalar_real error;

    real max_error;

    logs logger;
    std::vector<std::string> io_names = {"U", "Error"};

public:
    scalar_wave() = default;

    scalar_wave(mesh&&,
                bcs::Grid&&,
                bcs::Object&&,
                stencil,
                real3 center,
                real radius,
                real max_error = 100.0,
                const logs& = {});

    bool valid(const system_stats&) const;

    real3 summary(const system_stats&) const;

    void log(const system_stats&, const step_controller&);

    system_size size() const;

    static std::optional<scalar_wave> from_lua(const sol::table&, const logs& = {});

    void rhs(const sim_registry& reg, field_ref input,
             sim_registry& out_reg, field_ref output, real time);
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
