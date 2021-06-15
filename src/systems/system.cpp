#include "system.hpp"

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

namespace ccs
{

std::function<void(field&)> system::operator()(const step_controller& step)
{
    return std::visit(
        [&step](auto&& sys) {
            return std::function<void(field&)>{[&step, &sys](field& f) {
                // ensure the proper size for f
                if (ssize(f) != sys.size()) { f = field{sys.size()}; }
                sys(f, step);
            }};
        },
        v);
}

std::function<void(field_span)> system::rhs(field_view field, real time)
{
    return std::visit(
        [field, time](auto&& current_system) {
            return std::function<void(field_span)>{
                [&current_system, field, time](field_span view) {
                    current_system.rhs(field, time, view);
                }};
        },
        v);
}

void system::update_boundary(field_span view, real time)
{
    std::visit([view, time](
                   auto&& current_system) { current_system.update_boundary(view, time); },
               v);
}

std::optional<real> system::timestep_size(const field& field,
                                          const step_controller& controller) const
{
    const auto predicted_dt = std::visit(
        [&field, &controller](auto&& current_system) {
            return current_system.timestep_size(field, controller);
        },
        v);
    // the controller may adjust or invalidate this timestep size
    return controller.check_timestep_size(predicted_dt);
}

bool system::valid(const system_stats& stats) const
{
    return std::visit([&stats](auto&& sys) { return sys.valid(stats); }, v);
}

system_stats
system::stats(const field& u0, const field& u1, const step_controller& controller) const
{
    return std::visit(
        [&u0, &u1, &controller](auto&& sys) { return sys.stats(u0, u1, controller); }, v);
}

void system::log(const system_stats& stats, const step_controller& controller)
{
    return std::visit(
        [&stats, &controller](auto&& current_system) {
            current_system.log(stats, controller);
        },
        v);
}

system_size system::size() const
{
    return std::visit([](auto&& current_system) { return current_system.size(); }, v);
}

std::optional<system> system::from_lua(const sol::table& tbl)
{
    auto m = tbl["system"];
    if (!m.valid()) {
        spdlog::error("simulation.system must be specified");
        return std::nullopt;
    }

    auto type = m["type"].get_or(std::string{});

    if (type == "heat") {
        if (auto opt = systems::heat::from_lua(tbl); opt) return system(MOVE(*opt));
    } else if (type == "scalar wave") {
        return system(systems::scalar_wave{});
    } else if (type == "inviscid vortex") {
        return system(systems::inviscid_vortex{});
    } else {
        spdlog::error("unrecognized system.type");
    }
    return std::nullopt;
}
} // namespace ccs
