#pragma once

#include "types.hpp"

#include <algorithm>

namespace ccs
{

// ---------------------------------------------------------------------------
// scalar_span / scalar_view — lightweight structs for 4-component field access.
// ---------------------------------------------------------------------------

struct scalar_span {
    std::span<real> D{}, Rx{}, Ry{}, Rz{};

    scalar_span() = default;
    scalar_span(std::span<real> d, std::span<real> rx,
                std::span<real> ry, std::span<real> rz)
        : D(d), Rx(rx), Ry(ry), Rz(rz) {}

    // Converting constructor from tuple-based scalar types (e.g., scalar_real).
    // ADL (P0846R0) finds ccs::get at instantiation time, avoiding
    // #include "tuple.hpp" in this header.
    template <typename T>
        requires(!std::same_as<std::remove_cvref_t<T>, scalar_span> &&
                 !std::is_arithmetic_v<std::remove_cvref_t<T>> &&
                 !std::invocable<T, scalar_span&>)
    scalar_span(T& s)  // NOLINT(google-explicit-constructor)
        : D(get<0>(get<0>(s))), Rx(get<0>(get<1>(s))),
          Ry(get<1>(get<1>(s))), Rz(get<2>(get<1>(s))) {}

    // Broadcast fill: du = 0.
    template <typename T>
        requires std::is_arithmetic_v<T>
    scalar_span& operator=(T val)
    {
        std::ranges::fill(D, static_cast<real>(val));
        std::ranges::fill(Rx, static_cast<real>(val));
        std::ranges::fill(Ry, static_cast<real>(val));
        std::ranges::fill(Rz, static_cast<real>(val));
        return *this;
    }

    // Functional assignment: u_rhs = lap(u, nu).
    template <std::invocable<scalar_span&> Fn>
    scalar_span& operator=(Fn&& fn)
    {
        FWD(fn)(*this);
        return *this;
    }
};

struct scalar_view {
    std::span<const real> D{}, Rx{}, Ry{}, Rz{};

    scalar_view() = default;
    scalar_view(std::span<const real> d, std::span<const real> rx,
                std::span<const real> ry, std::span<const real> rz)
        : D(d), Rx(rx), Ry(ry), Rz(rz) {}

    // Converting constructor from tuple-based scalar types (e.g., scalar_real).
    template <typename T>
        requires(!std::same_as<std::remove_cvref_t<T>, scalar_view> &&
                 !std::same_as<std::remove_cvref_t<T>, scalar_span>)
    scalar_view(const T& s)  // NOLINT(google-explicit-constructor)
        : D(get<0>(get<0>(s))), Rx(get<0>(get<1>(s))),
          Ry(get<1>(get<1>(s))), Rz(get<2>(get<1>(s))) {}

    // Converting constructor from scalar_span.
    scalar_view(const scalar_span& s)  // NOLINT(google-explicit-constructor)
        : D(s.D), Rx(s.Rx), Ry(s.Ry), Rz(s.Rz) {}
};

} // namespace ccs
