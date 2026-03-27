#include "euler.hpp"
#include "slot_ops.hpp"
#include "step_controller.hpp"
#include "systems/system.hpp"

namespace ccs::integrators
{

void euler::operator()(system& sys, sim_registry& reg,
                       field_ref u0, field_ref output,
                       field_ref system_rhs_ref,
                       const step_controller& ctrl, real dt)
{
    const real time = ctrl;

    slot_zero(reg, system_rhs_ref);
    sys.rhs(reg, u0, reg, system_rhs_ref, time);
    slot_assign_lc(reg, output, u0, dt, system_rhs_ref);
    sys.update_boundary(reg, output, time + dt);
}

} // namespace ccs::integrators
