#pragma once

#include "cartesian.hpp"
#include "fields/selector.hpp"
#include "mesh_types.hpp"
#include "object_geometry.hpp"

#include "fields/ccs_range_utils.hpp"
#include "fields/lazy_views.hpp"

namespace ccs::views
{

// Concepts and traits for the utility functions below (xmin, xmax, ..., location, F).
// These were part of the original selection system but lost during refactoring.
// selections.hpp is dead code (not #include'd by any production TU); these stubs
// make the header self-contained for compile-verification purposes.
template <typename T>
concept Selection = requires { typename std::remove_cvref_t<T>::index; };

template <typename T>
concept DomainSelection =
    Selection<T> && std::same_as<typename std::remove_cvref_t<T>::index, si::D>;

template <typename T>
inline constexpr bool is_domain_selection_v =
    Selection<T> && std::same_as<typename std::remove_cvref_t<T>::index, si::D>;

template <typename T>
inline constexpr bool is_Rx_selection_v =
    Selection<T> && std::same_as<typename std::remove_cvref_t<T>::index, si::Rx>;

template <typename T>
inline constexpr bool is_Ry_selection_v =
    Selection<T> && std::same_as<typename std::remove_cvref_t<T>::index, si::Ry>;

template <typename T>
inline constexpr bool is_Rz_selection_v =
    Selection<T> && std::same_as<typename std::remove_cvref_t<T>::index, si::Rz>;

namespace detail
{
// y-plane view: selects elements where y == j from a flat 3D grid.
// Formulated as a custom view because naive range-v3 building blocks
// (chunk | stride | join) did not produce output iterators.
// This will need to be revisited if the layout changes.
template <typename Rng>
class YPlaneView : public std::ranges::view_interface<YPlaneView<Rng>>
{
    using diff_t = std::ranges::range_difference_t<Rng>;

    Rng base_{};
    diff_t nx{}, ny{}, nz{}, j{};

public:
    class iterator
    {
        using base_iter = std::ranges::iterator_t<Rng>;

        base_iter base_it_{};
        diff_t nx_{}, ny_{}, nz_{};
        diff_t i_{}, k_{};

    public:
        using difference_type = diff_t;
        using value_type = std::ranges::range_value_t<Rng>;
        using reference = std::ranges::range_reference_t<Rng>;
        using iterator_concept = std::random_access_iterator_tag;
        using iterator_category = std::random_access_iterator_tag;

        iterator() = default;

        constexpr iterator(base_iter it,
                           diff_t nx,
                           diff_t ny,
                           diff_t nz,
                           diff_t i,
                           diff_t k)
            : base_it_{std::move(it)}, nx_{nx}, ny_{ny}, nz_{nz}, i_{i}, k_{k}
        {
        }

        constexpr reference operator*() const { return *base_it_; }

        constexpr iterator& operator++()
        {
            ++k_;
            ++base_it_;
            if (k_ == nz_ && i_ != nx_ - 1) {
                k_ = 0;
                ++i_;
                base_it_ += (ny_ - 1) * nz_;
            }
            return *this;
        }

        constexpr iterator operator++(int)
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr iterator& operator--()
        {
            --k_;
            --base_it_;
            if (k_ < 0) {
                k_ = nz_ - 1;
                --i_;
                base_it_ -= (ny_ - 1) * nz_;
            }
            return *this;
        }

        constexpr iterator operator--(int)
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        constexpr iterator& operator+=(difference_type n)
        {
            if (n == 0) return *this;

            const auto line_offset = ny_ * nz_;

            if (n > 0) {
                n += k_;

                auto qr = std::div(n, nz_);
                diff_t i1 = i_ + qr.quot;
                diff_t k1 = qr.rem;

                if (i1 == nx_) {
                    i1 = nx_ - 1;
                    k1 = nz_;
                }

                base_it_ += line_offset * (i1 - i_) + (k1 - k_);
                i_ = i1;
                k_ = k1;
            } else {
                n -= (nz_ - 1 - k_);

                auto qr = std::div(n, nz_);
                diff_t i1 = i_ + qr.quot;
                diff_t k1 = nz_ - 1 + qr.rem;

                base_it_ += line_offset * (i1 - i_) + (k1 - k_);
                i_ = i1;
                k_ = k1;
            }
            return *this;
        }

        constexpr iterator& operator-=(difference_type n) { return *this += -n; }

        constexpr reference operator[](difference_type n) const
        {
            auto tmp = *this;
            tmp += n;
            return *tmp;
        }

        friend constexpr iterator operator+(iterator it, difference_type n)
        {
            it += n;
            return it;
        }

        friend constexpr iterator operator+(difference_type n, iterator it)
        {
            it += n;
            return it;
        }

        friend constexpr iterator operator-(iterator it, difference_type n)
        {
            it -= n;
            return it;
        }

        friend constexpr difference_type operator-(const iterator& a, const iterator& b)
        {
            return (a.i_ - b.i_) * a.nz_ + (a.k_ - b.k_);
        }

        friend constexpr bool operator==(const iterator& a, const iterator& b)
        {
            return a.base_it_ == b.base_it_;
        }

        friend constexpr auto operator<=>(const iterator& a, const iterator& b)
        {
            auto diff = a - b;
            return diff <=> decltype(diff){0};
        }
    };

    YPlaneView() = default;
    explicit constexpr YPlaneView(Rng rng, const int3& extents, int j)
        : base_{std::move(rng)},
          nx{extents[0]},
          ny{extents[1]},
          nz{extents[2]},
          j{j}
    {
    }

    constexpr iterator begin()
    {
        auto it = std::ranges::begin(base_);
        std::ranges::advance(it, j * nz);
        return iterator(std::move(it), nx, ny, nz, 0, 0);
    }

    constexpr iterator end()
    {
        auto it = std::ranges::begin(base_);
        std::ranges::advance(it, (nx - 1) * ny * nz + j * nz + nz);
        return iterator(std::move(it), nx, ny, nz, nx - 1, nz);
    }

    constexpr auto size() const { return nx * nz; }
};

template <typename Rng>
YPlaneView(Rng&&, const int3&, int) -> YPlaneView<std::views::all_t<Rng>>;

struct y_plane_base_fn {
    template <typename Rng>
    constexpr auto operator()(Rng&& rng, const int3& extents, int j) const
    {
        return YPlaneView(std::views::all(FWD(rng)), extents, j);
    }
};

struct y_plane_fn : y_plane_base_fn {
    using y_plane_base_fn::operator();

    constexpr auto operator()(const int3& extents, int j) const
    {
        return ccs::make_view_closure(ccs::bind_back(y_plane_base_fn{}, extents, j));
    }
};

constexpr auto y_plane_view = y_plane_fn{};

} // namespace detail

template <auto I>
struct plane_fn;

template <>
struct plane_fn<0> {
    constexpr auto operator()(const int3& extents) const
    {
        return std::views::take(extents[1] * extents[2]);
    }

    constexpr auto operator()(const int3& extents, int i) const
    {
        return std::views::drop(i * extents[1] * extents[2]) | (*this)(extents);
    }
};

template <>
struct plane_fn<1> {
    constexpr auto operator()(const int3& extents, int j = 0) const
    {
        return detail::y_plane_view(extents, j);
    }
};

template <>
struct plane_fn<2> {
    constexpr auto operator()(const int3& extents) const
    {
        return ccs::make_view_closure([n = extents[2]](auto&& rng) {
            return ccs::stride(FWD(rng), n);
        });
    }

    constexpr auto operator()(const int3& extents, int k) const
    {
        return ccs::make_view_closure([n = extents[2], k](auto&& rng) {
            return ccs::stride(std::views::drop(FWD(rng), k), n);
        });
    }
};

template <auto I>
constexpr auto plane_view = plane_fn<I>{};

constexpr auto xmin(int3 extents)
{
    return ccs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | plane_view<0>(extents)};
    });
}

constexpr auto xmax(int3 extents)
{
    return ccs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | plane_view<0>(extents, extents[0] - 1)};
    });
}

constexpr auto ymin(int3 extents)
{
    return ccs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | plane_view<1>(extents)};
    });
}

constexpr auto ymax(int3 extents)
{
    return ccs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | plane_view<1>(extents, extents[1] - 1)};
    });
}

constexpr auto zmin(int3 extents)
{
    return ccs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | plane_view<2>(extents)};
    });
}

constexpr auto zmax(int3 extents)
{
    return ccs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | plane_view<2>(extents, extents[2] - 1)};
    });
}

constexpr auto location(const cartesian& cart, const object_geometry& geometry)
{
    return ccs::make_view_closure([&]<Selection S>(S&&) {
        if constexpr (is_domain_selection_v<S>)
            return ccs::cartesian_product(cart.x(), cart.y(), cart.z());
        else if constexpr (is_Rx_selection_v<S>)
            return geometry.Rx() |
                   std::views::transform([](auto&& o) { return o.position; });
        else if constexpr (is_Ry_selection_v<S>)
            return geometry.Ry() |
                   std::views::transform([](auto&& o) { return o.position; });
        else if constexpr (is_Rz_selection_v<S>)
            return geometry.Rz() |
                   std::views::transform([](auto&& o) { return o.position; });

        else
            static_assert("unaccounted selection type");
    });
}

namespace detail
{
// fluid domain view, need to write our own adaptor so m.F() can be an output view
template <typename Rng>
class FView : public std::ranges::view_interface<FView<Rng>>
{
    using diff_t = std::ranges::range_difference_t<Rng>;

    Rng base_{};
    index_extents extents{};
    std::span<const line> lines{};

public:
    class iterator
    {
        using base_iter = std::ranges::iterator_t<Rng>;

        base_iter base_it_{};
        index_extents extents{};
        std::span<const line> lines{};
        unsigned long l{};
        integer i{}, i0{}, i1{}, local_off{};

        constexpr void set_line()
        {
            auto&& [_, start, end] = lines[l];

            i0 = start.object ? extents(start.mesh_coordinate) + 1
                              : extents(start.mesh_coordinate);
            i1 = end.object ? extents(end.mesh_coordinate)
                            : extents(end.mesh_coordinate) + 1;
        }

    public:
        using difference_type = diff_t;
        using value_type = std::ranges::range_value_t<Rng>;
        using reference = std::ranges::range_reference_t<Rng>;
        using iterator_concept = std::random_access_iterator_tag;
        using iterator_category = std::random_access_iterator_tag;

        iterator() = default;

        constexpr iterator(base_iter it,
                           index_extents extents,
                           std::span<const line> lines,
                           unsigned long l,
                           integer i,
                           integer i0,
                           integer i1,
                           integer local_off)
            : base_it_{std::move(it)}
            , extents{extents}
            , lines{lines}
            , l{l}
            , i{i}
            , i0{i0}
            , i1{i1}
            , local_off{local_off}
        {
        }

        constexpr reference operator*() const { return *base_it_; }

        constexpr iterator& operator++()
        {
            ++i;
            ++base_it_;
            ++local_off;
            if (i == i1 && l != lines.size() - 1) {
                ++l;
                set_line();

                base_it_ += (i0 - i);
                i = i0;
            }
            return *this;
        }

        constexpr iterator operator++(int)
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr iterator& operator--()
        {
            --i;
            --base_it_;
            --local_off;
            if (i < i0) {
                --l;
                set_line();

                base_it_ -= (i - (i1 - 1));
                i = i1 - 1;
            }
            return *this;
        }

        constexpr iterator operator--(int)
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        constexpr iterator& operator+=(difference_type n)
        {
            if (n == 0) return *this;

            local_off += n;
            difference_type it_off = 0;
            const auto last_line = lines.size() - 1;

            if (n > 0) {
                // move iterator to i0 to make life easier
                it_off = (i0 - i);
                i = i0;
                n -= it_off;
                while (l != last_line && n > ((i1 - 1) - i0)) {
                    // advance the line and reset i0/i1
                    n -= (i1 - i0);
                    ++l;
                    set_line();
                }
                it_off += i0 + n - i;
                i = i0 + n;
            } else {
                // move iterator to i1 to make life easier
                it_off = (i1 - i);
                i = i1;
                n -= it_off;

                while (n < (i0 - i1)) {
                    n -= (i0 - i1);
                    --l;
                    set_line();
                }
                it_off += i1 + n - i;
                i = i1 + n;
            }

            std::ranges::advance(base_it_, it_off);
            return *this;
        }

        constexpr iterator& operator-=(difference_type n) { return *this += -n; }

        constexpr reference operator[](difference_type n) const
        {
            auto tmp = *this;
            tmp += n;
            return *tmp;
        }

        friend constexpr iterator operator+(iterator it, difference_type n)
        {
            it += n;
            return it;
        }

        friend constexpr iterator operator+(difference_type n, iterator it)
        {
            it += n;
            return it;
        }

        friend constexpr iterator operator-(iterator it, difference_type n)
        {
            it -= n;
            return it;
        }

        friend constexpr difference_type operator-(const iterator& a, const iterator& b)
        {
            return a.local_off - b.local_off;
        }

        friend constexpr bool operator==(const iterator& a, const iterator& b)
        {
            return a.local_off == b.local_off;
        }

        friend constexpr auto operator<=>(const iterator& a, const iterator& b)
        {
            return a.local_off <=> b.local_off;
        }
    };

    FView() = default;
    explicit constexpr FView(Rng rng,
                             index_extents extents,
                             std::span<const line> lines)
        : base_{std::move(rng)}, extents{MOVE(extents)}, lines{MOVE(lines)}
    {
    }

    constexpr iterator begin()
    {
        auto it = std::ranges::begin(base_);

        integer local_off = 0;
        unsigned long l = 0;

        auto&& [_, start, end] = lines[l];
        integer i0 = start.object ? extents(start.mesh_coordinate) + 1
                                  : extents(start.mesh_coordinate);
        integer i1 = end.object ? extents(end.mesh_coordinate)
                                : extents(end.mesh_coordinate) + 1;

        integer i = i0;
        std::ranges::advance(it, i);
        return iterator(std::move(it), extents, lines, l, i, i0, i1, local_off);
    }

    constexpr iterator end()
    {
        auto it = std::ranges::begin(base_);

        integer local_off = 0;
        unsigned long l = 0;
        integer i0 = 0, i1 = 0;
        for (l = 0; l < lines.size(); l++) {
            auto&& [_, start, end] = lines[l];
            i0 = start.object ? extents(start.mesh_coordinate) + 1
                              : extents(start.mesh_coordinate);
            i1 = end.object ? extents(end.mesh_coordinate)
                            : extents(end.mesh_coordinate) + 1;
            local_off += (i1 - i0);
        }
        // Use last valid line index (not one-past-end) so that backward
        // traversal (operator--, operator+=(-n)) retreats to the previous
        // line instead of re-reading the last line.  This matches the
        // state that operator++ produces when it reaches the end.
        l = lines.size() - 1;
        integer i = i1;

        std::ranges::advance(it, i);
        return iterator(std::move(it), extents, lines, l, i, i0, i1, local_off);
    }

    constexpr auto size() const
    {
        integer total = 0;
        for (unsigned long l = 0; l < lines.size(); l++) {
            auto&& [_, start, end] = lines[l];
            integer i0 = start.object ? extents(start.mesh_coordinate) + 1
                                      : extents(start.mesh_coordinate);
            integer i1 = end.object ? extents(end.mesh_coordinate)
                                    : extents(end.mesh_coordinate) + 1;
            total += (i1 - i0);
        }
        return total;
    }
};

template <typename Rng>
FView(Rng&&, index_extents, std::span<const line>) -> FView<std::views::all_t<Rng>>;

struct fview_base_fn {
    template <typename Rng>
    constexpr auto
    operator()(Rng&& rng, index_extents extents, std::span<const line> lines) const
    {
        return FView(std::views::all(FWD(rng)), MOVE(extents), MOVE(lines));
    }
};

struct fview_fn : fview_base_fn {
    using fview_base_fn::operator();

    constexpr auto operator()(index_extents extents, std::span<const line> lines) const
    {
        return ccs::make_view_closure(
            ccs::bind_back(fview_base_fn{}, MOVE(extents), MOVE(lines)));
    }
};

constexpr auto fview = fview_fn{};
} // namespace detail

constexpr auto F(index_extents extents, std::span<const line> lines)
{
    return ccs::make_view_closure([=]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | detail::fview(extents, lines)};
    });
}

} // namespace ccs::views