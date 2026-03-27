#include "rk4.hpp"
#include "slot_ops.hpp"
#include "step_controller.hpp"
#include "systems/system.hpp"

namespace ccs::integrators
{

constexpr std::array rki{0.0, 0.5, 0.5, 1.0};
constexpr std::array rkf{1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};

void rk4::operator()(system& sys, sim_registry& reg,
                     field_ref u0, field_ref output,
                     field_ref rk_rhs_ref, field_ref system_rhs_ref,
                     const step_controller& ctrl, real dt)
{
    slot_zero(reg, rk_rhs_ref);
    slot_zero(reg, system_rhs_ref);
    const real time = ctrl;

    reg.deep_copy_slot(output.slot, u0.slot);

    for (int i = 0; i < 4; ++i) {
        if (i > 0) {
            slot_assign_lc(reg, output, u0, dt * rki[i], system_rhs_ref);
            sys.update_boundary(reg, output, time + dt * rki[i]);
        }
        sys.rhs(reg, output, reg, system_rhs_ref, time + dt * rki[i]);
        slot_accumulate(reg, rk_rhs_ref, dt * rkf[i], system_rhs_ref);
    }

    // final update
    slot_assign_lc(reg, output, u0, 1.0, rk_rhs_ref);
    sys.update_boundary(reg, output, time + dt);
}
} // namespace ccs::integrators
