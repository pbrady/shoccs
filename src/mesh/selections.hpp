#pragma once

#include "cartesian.hpp"
#include "fields/selector.hpp"
#include "mesh_types.hpp"
#include "object_geometry.hpp"

#include "fields/ccs_range_utils.hpp"
#include "fields/lazy_views.hpp"

#include <range/v3/view/drop_exactly.hpp>
#include <range/v3/view/stride.hpp>
#include <range/v3/view/take_exactly.hpp>

namespace ccs::views
{

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
        return vs::take_exactly(extents[1] * extents[2]);
    }

    constexpr auto operator()(const int3& extents, int i) const
    {
        return vs::drop_exactly(i * extents[1] * extents[2]) | (*this)(extents);
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
        return vs::stride(extents[2]);
    }

    constexpr auto operator()(const int3& extents, int k) const
    {
        return vs::drop_exactly(k) | (*this)(extents);
    }
};

template <auto I>
constexpr auto plane_view = plane_fn<I>{};

constexpr auto xmin(int3 extents)
{
    return rs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | plane_view<0>(extents)};
    });
}

constexpr auto xmax(int3 extents)
{
    return rs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | plane_view<0>(extents, extents[0] - 1)};
    });
}

constexpr auto ymin(int3 extents)
{
    return rs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | plane_view<1>(extents)};
    });
}

constexpr auto ymax(int3 extents)
{
    return rs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | plane_view<1>(extents, extents[1] - 1)};
    });
}

constexpr auto zmin(int3 extents)
{
    return rs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | plane_view<2>(extents)};
    });
}

constexpr auto zmax(int3 extents)
{
    return rs::make_view_closure([extents = MOVE(extents)]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | plane_view<2>(extents, extents[2] - 1)};
    });
}

constexpr auto location(const cartesian& cart, const object_geometry& geometry)
{
    return rs::make_view_closure([&]<Selection S>(S&&) {
        if constexpr (is_domain_selection_v<S>)
            return vs::cartesian_product(cart.x(), cart.y(), cart.z());
        else if constexpr (is_Rx_selection_v<S>)
            return geometry.Rx() | vs::transform([](auto&& o) { return o.position; });
        else if constexpr (is_Ry_selection_v<S>)
            return geometry.Ry() | vs::transform([](auto&& o) { return o.position; });
        else if constexpr (is_Rz_selection_v<S>)
            return geometry.Rz() | vs::transform([](auto&& o) { return o.position; });

        else
            static_assert("unaccounted selection type");
    });
}

namespace detail
{
// fluid domain view, need to write our own adaptor so m.F() can be an output view
template <typename Rng>
class FView : public rs::view_adaptor<FView<Rng>, Rng>
{
    using diff_t = rs::range_difference_t<Rng>;

    friend rs::range_access;

    index_extents extents;
    std::span<const line> lines;

    class adaptor : public rs::adaptor_base
    {
        index_extents extents;
        std::span<const line> lines;
        unsigned long l;
        integer i, i0, i1, local_off;

        constexpr void set_line()
        {
            auto&& [_, start, end] = lines[l];

            i0 = start.object ? extents(start.mesh_coordinate) + 1
                              : extents(start.mesh_coordinate);
            i1 = end.object ? extents(end.mesh_coordinate)
                            : extents(end.mesh_coordinate) + 1;
        }

    public:
        adaptor() = default;
        adaptor(index_extents extents, std::span<const line> lines)
            : extents{MOVE(extents)}, lines{MOVE(lines)}
        {
            assert(this->lines.size() > 0);
        }

        template <typename R>
        constexpr auto begin(R& rng)
        {
            auto it = rs::begin(rng.base());

            local_off = 0;
            l = 0;
            set_line();
            // this doesn't really handle the case of stride > 1
            i = i0;
            rs::advance(it, i);
            return it;
        }

        template <typename R>
        constexpr auto end(R& rng)
        {
            auto it = rs::begin(rng.base());

            local_off = 0;
            for (l = 0; l < lines.size(); l++) {
                set_line();
                local_off += (i1 - i0);
            }
            i = i1;

            rs::advance(it, i);
            return it;
        }

        template <typename I>
        void next(I& it)
        {
            // advance to next point on this line, or to the next line
            ++i;
            ++it;
            ++local_off;
            if (i == i1 && l != lines.size() - 1) {
                ++l;
                set_line();

                it += (i0 - i);
                i = i0;
            }
        }

        template <typename I>
        void prev(I& it)
        {
            --i;
            --it;
            --local_off;
            if (i < i0) {
                --l;
                set_line();

                it -= (i - (i1 - 1));
                i = i1 - 1;
            }
        }

        template <typename I>
        void advance(I& it, rs::difference_type_t<I> n)
        {
            if (n == 0) return;

            local_off += n;
            rs::difference_type_t<I> it_off = 0;
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

            rs::advance(it, it_off);
        }

        template <typename I>
        diff_t distance_to(const I&, const I&, const adaptor& that) const
        {
            return that.local_off - local_off;
        }
    };

    adaptor begin_adaptor() { return {extents, lines}; }

    adaptor end_adaptor() { return {extents, lines}; }

public:
    FView() = default;
    // This really hints that we should extract the extents into their own object
    // and give them a call operator that converts an ijk tuple to a single index
    // It should also convert so a simple int3
    explicit constexpr FView(Rng&& rng,
                             index_extents extents,
                             std::span<const line> lines)
        : FView::view_adaptor{FWD(rng)}, extents{MOVE(extents)}, lines{MOVE(lines)}
    {
    }
};

template <typename Rng>
FView(Rng&&, index_extents, std::span<const line>) -> FView<Rng>;

struct fview_base_fn {
    template <typename Rng>
    constexpr auto
    operator()(Rng&& rng, index_extents extents, std::span<const line> lines) const
    {
        return FView(FWD(rng), MOVE(extents), MOVE(lines));
    }
};

struct fview_fn : fview_base_fn {
    using fview_base_fn::operator();

    constexpr auto operator()(index_extents extents, std::span<const line> lines) const
    {
        return rs::make_view_closure(
            rs::bind_back(fview_base_fn{}, MOVE(extents), MOVE(lines)));
    }
};

constexpr auto fview = fview_fn{};
} // namespace detail

constexpr auto F(index_extents extents, std::span<const line> lines)
{
    return rs::make_view_closure([=]<DomainSelection S>(S&& s) {
        return tuple{FWD(s) | detail::fview(extents, lines)};
    });
}

} // namespace ccs::views