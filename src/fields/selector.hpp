#pragma once

#include <cassert>
#include <optional>

#include "tuple.hpp"

#include "scalar.hpp"
#include "vector.hpp"

#include "selector_fwd.hpp"

#include "index_extents.hpp"

#include <ranges>

#include "ccs_range_utils.hpp"
#include "lazy_views.hpp"

namespace ccs
{

// traits for mapping scalar and vector types to appropriate selection mp_list
namespace detail
{
template <typename>
struct selection_fn_index_impl;

template <Scalar T>
struct selection_fn_index_impl<T> {
    using type = mp_size_t<0>;
};

template <Vector T>
struct selection_fn_index_impl<T> {
    using type = mp_size_t<1>;
};

template <ListIndex L, typename R>
constexpr auto make_selection(R);

} // namespace detail

template <typename T>
using selection_fn_index = detail::selection_fn_index_impl<T>::type;

template <ListIndex I, Tuple R, typename Fn>
struct selection : R {
    using index = I;
    ccs::semiregular_box<Fn> f;

    selection() = default; // default construction needed for semi-regular concept

    constexpr selection(R r, Fn f) : R{MOVE(r)}, f{MOVE(f)} {}

    selection(const selection&) = default;
    selection(selection&&) = default;

    selection& operator=(const selection&) = default;
    selection& operator=(selection&&) = default;

    template <TupleLike T>
    constexpr auto apply(T&& t) const
    {
        return f(FWD(t));
    }
};

namespace detail
{
template <typename... Lists>
struct selection_view_fn;

template <ListIndex L, typename R>
constexpr auto make_selection(R r)
{
    return selection<L, R, selection_view_fn<L>>{MOVE(r), selection_view_fn<L>{}};
}

//
// selection views are designed for extracting the major components of scalars and vectors
// such as the domain - D, or the sets for boundary data - R
//
template <typename... Lists>
struct selection_view_fn {
    using Indices = mp_list<Lists...>;

    template <TupleLike U>
    constexpr auto operator()(U&& u) const requires(sizeof...(Lists) > 1)
    {

        // for now just grab the first element of the list
        using List = mp_at<Indices, selection_fn_index<U>>;
        static_assert(!mp_empty<List>::value, "selection operation not permitted");

        // Build a selection using the tuples indexed in the list
        return []<auto... Is>(std::index_sequence<Is...>, auto&& u)
        {
            if constexpr (sizeof...(Is) == 1)
                return tuple{detail::make_selection<mp_at_c<List, 0>>(
                    tuple{get<mp_at_c<List, 0>>(FWD(u))})};
            else
                return tuple{detail::make_selection<mp_at_c<List, Is>>(
                    tuple{get<mp_at_c<List, Is>>(FWD(u))})...};
        }
        (sequence<List>, FWD(u));
    }

    template <TupleLike U>
    constexpr auto operator()(U&& u) const requires(sizeof...(Lists) == 1)
    {

        // for now just grab the first element of the list
        using List = mp_at<Indices, mp_size_t<0>>;
        static_assert(!mp_empty<List>::value, "selection operation not permitted");

        return tuple{detail::make_selection<List>(tuple{get<List>(FWD(u))})};
    }
};

} // namespace detail

template <typename... Lists>
inline constexpr auto
    selection_view = ccs::make_view_closure(detail::selection_view_fn<Lists...>{});

//
// selectors for main components of scalars and vectors
//

namespace sel
{
inline constexpr auto Dx = selection_view<mp_list<>, mp_list<vi::Dx>>;
inline constexpr auto Dy = selection_view<mp_list<>, mp_list<vi::Dy>>;
inline constexpr auto Dz = selection_view<mp_list<>, mp_list<vi::Dz>>;
inline constexpr auto Rx =
    selection_view<mp_list<si::Rx>, mp_list<vi::xRx, vi::yRx, vi::zRx>>;
inline constexpr auto Ry =
    selection_view<mp_list<si::Ry>, mp_list<vi::xRy, vi::yRy, vi::zRy>>;
inline constexpr auto Rz =
    selection_view<mp_list<si::Rz>, mp_list<vi::xRz, vi::yRz, vi::zRz>>;
inline constexpr auto xR = selection_view<mp_list<>, mp_list<vi::xRx, vi::xRy, vi::xRz>>;
inline constexpr auto yR = selection_view<mp_list<>, mp_list<vi::yRx, vi::yRy, vi::yRz>>;
inline constexpr auto zR = selection_view<mp_list<>, mp_list<vi::zRx, vi::zRy, vi::zRz>>;
inline constexpr auto D = selection_view<mp_list<si::D>, mp_list<vi::Dx, vi::Dy, vi::Dz>>;
inline constexpr auto R = selection_view<mp_list<si::Rx, si::Ry, si::Rz>,
                                         mp_list<vi::xRx,
                                                 vi::yRx,
                                                 vi::zRx,
                                                 vi::xRy,
                                                 vi::yRy,
                                                 vi::zRy,
                                                 vi::xRz,
                                                 vi::yRz,
                                                 vi::zRz>>;
inline constexpr auto xRx = selection_view<mp_list<>, mp_list<vi::xRx>>;
inline constexpr auto xRy = selection_view<mp_list<>, mp_list<vi::xRy>>;
inline constexpr auto xRz = selection_view<mp_list<>, mp_list<vi::xRz>>;
inline constexpr auto yRx = selection_view<mp_list<>, mp_list<vi::yRx>>;
inline constexpr auto yRy = selection_view<mp_list<>, mp_list<vi::yRy>>;
inline constexpr auto yRz = selection_view<mp_list<>, mp_list<vi::yRz>>;
inline constexpr auto zRx = selection_view<mp_list<>, mp_list<vi::zRx>>;
inline constexpr auto zRy = selection_view<mp_list<>, mp_list<vi::zRy>>;
inline constexpr auto zRz = selection_view<mp_list<>, mp_list<vi::zRz>>;

} // namespace sel

namespace detail
{

template <int I, typename Rng, typename Fn>
class plane_view;

template <typename Rng>
using x_plane_t =
    decltype(std::declval<Rng>() | std::views::drop(int{}) | std::views::take(integer{}));

template <typename Rng, typename Fn>
class plane_view<0, Rng, Fn> : public x_plane_t<Rng>
{
    using base = x_plane_t<Rng>;

    ccs::semiregular_box<Fn> f;

    template <Range U>
    static constexpr auto apply_(U&& u, integer n, int i)
    {
        return FWD(u) | std::views::drop(i * n) | std::views::take(n);
    }

public:
    plane_view() = default;
    explicit constexpr plane_view(Rng&& rng, index_extents extents, int i, Fn f)
        : base{apply_(FWD(rng), extents[1] * extents[2], i)}, f{MOVE(f)}
    {
    }

    template <typename U>
        requires std::invocable<Fn, U>
    constexpr auto apply(U&& u) const
    {
        return f(FWD(u)); // plane_view<0, U>(FWD(u), extents, i);
    }
};

// y-plane view do not result in output iterator when formulated in an intuitive fashion
// using range building blocks: grid | chunk(nz) | stride(ny) | join; To workaround this
// limitation, we define a custom y-plane view with a hand-rolled iterator that
// implements the non-contiguous stride pattern.  This will need to be revisited if the
// layout changes
template <typename Rng, typename Fn>
class plane_view<1, Rng, Fn>
    : public std::ranges::view_interface<plane_view<1, Rng, Fn>>
{
    using diff_t = std::ranges::range_difference_t<Rng>;

    Rng base_;
    index_extents n;
    diff_t j;
    ccs::semiregular_box<Fn> f;

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
                std::ranges::advance(base_it_, (ny_ - 1) * nz_);
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
                std::ranges::advance(base_it_, -((ny_ - 1) * nz_));
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

                std::ranges::advance(base_it_, line_offset * (i1 - i_) + (k1 - k_));
                i_ = i1;
                k_ = k1;
            } else {
                n -= (nz_ - 1 - k_);

                auto qr = std::div(n, nz_);
                diff_t i1 = i_ + qr.quot;
                diff_t k1 = nz_ - 1 + qr.rem;

                std::ranges::advance(base_it_, line_offset * (i1 - i_) + (k1 - k_));
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

    plane_view() = default;
    explicit constexpr plane_view(Rng&& rng, index_extents extents, int j, Fn f)
        : base_{FWD(rng)}, n{extents}, j{j}, f{MOVE(f)}
    {
    }

    constexpr auto begin()
    {
        auto it = std::ranges::begin(base_);
        std::ranges::advance(it, j * n[2]);
        return iterator(std::move(it), n[0], n[1], n[2], 0, 0);
    }

    constexpr auto end()
    {
        auto it = std::ranges::begin(base_);
        std::ranges::advance(
            it, (diff_t{n[0]} - 1) * n[1] * n[2] + j * n[2] + n[2]);
        return iterator(std::move(it), n[0], n[1], n[2], n[0] - 1, n[2]);
    }

    constexpr auto size() const
    {
        return static_cast<diff_t>(n[0]) * static_cast<diff_t>(n[2]);
    }

    template <typename U>
        requires std::invocable<Fn, U>
    constexpr auto apply(U&& u) const
    {
        return f(FWD(u));
    }
};

template <typename Rng>
using z_plane_t =
    decltype(ccs::stride(std::declval<Rng>() | std::views::drop(int{}), integer{}));

template <typename Rng, typename Fn>
class plane_view<2, Rng, Fn> : public z_plane_t<Rng>
{
    using base = z_plane_t<Rng>;

    ccs::semiregular_box<Fn> f;

public:
    plane_view() = default;
    explicit constexpr plane_view(Rng&& rng, index_extents extents, int k, Fn f)
        : base{ccs::stride(FWD(rng) | std::views::drop(k), extents[2])}, f{MOVE(f)}
    {
    }

    template <typename U>
        requires std::invocable<Fn, U>
    constexpr auto apply(U&& u) const
    {
        return f(FWD(u)); // plane_view<2, U>(FWD(u), extents, k);
    }
};

template <int I, typename R, typename F>
constexpr auto make_plane_view(R&& r, index_extents extents, int plane_coord, F f)
{
    return plane_view<I, R, F>{FWD(r), extents, plane_coord, MOVE(f)};
}

// template <int I, typename Rng>
// plane_view(mp_int<I>, Rng&&, index_extents, int) -> plane_view<I, Rng>;
template <auto>
struct plane_selection_fn;

template <int I>
struct plane_selection_base_fn {
    template <Range R>
    constexpr auto operator()(R&& r, index_extents extents, int plane_coord) const
    {
        return make_plane_view<I>(
            FWD(r),
            extents,
            plane_coord,
            ccs::bind_back(plane_selection_fn<I>{}, extents, plane_coord));
    }

    template <Range R, typename F>
    constexpr auto operator()(R&& r, index_extents extents, int plane_coord, F&& f) const
    {
        return make_plane_view<I>(
            FWD(r),
            extents,
            plane_coord,
            ccs::compose(ccs::bind_back(plane_selection_fn<I>{}, extents, plane_coord),
                         FWD(f)));
    }
};

// First parameter indicate the direction of the plane {0, 1, 2}
template <auto I>
struct plane_selection_fn : plane_selection_base_fn<I> {
    using base = plane_selection_base_fn<I>;

    template <TupleLike U>
    constexpr auto operator()(U&& u, index_extents extents, int plane_coord) const
    {
        if (plane_coord < 0) { plane_coord += extents[I]; }
        if constexpr (Scalar<U>) {
            return tuple{base::operator()(FWD(u) | sel::D, MOVE(extents), plane_coord)};
        } else if constexpr (Vector<U>) {
            return tuple{
                base::operator()(FWD(u) | sel::Dx, extents, plane_coord, sel::Dx),
                base::operator()(FWD(u) | sel::Dy, extents, plane_coord, sel::Dy),
                base::operator()(FWD(u) | sel::Dz, extents, plane_coord, sel::Dz)};
        } else
            return tuple{base::operator()(FWD(u), MOVE(extents), plane_coord)};
    }

    constexpr auto operator()(index_extents extents, int plane_coord) const
    {
        return ccs::make_view_closure(ccs::bind_back(*this, MOVE(extents), plane_coord));
    }

    constexpr auto operator()(int plane_coord) const
    {
        return ccs::bind_back(*this, plane_coord);
    }
};
} // namespace detail

template <auto I>
constexpr auto plane_selection_fn = detail::plane_selection_fn<I>{};

//
// Selectors for planes of data for Tuples, Scalars, and Vectors
//

namespace sel
{

inline constexpr auto x_plane = plane_selection_fn<0>;
inline constexpr auto y_plane = plane_selection_fn<1>;
inline constexpr auto z_plane = plane_selection_fn<2>;

inline constexpr auto xmin = x_plane(0);
inline constexpr auto xmax = x_plane(-1);
inline constexpr auto ymin = y_plane(0);
inline constexpr auto ymax = y_plane(-1);
inline constexpr auto zmin = z_plane(0);
inline constexpr auto zmax = z_plane(-1);

using xmin_t = decltype(sel::xmin(index_extents{}));
using xmax_t = decltype(sel::xmax(index_extents{}));
using ymin_t = decltype(sel::ymin(index_extents{}));
using ymax_t = decltype(sel::ymax(index_extents{}));
using zmin_t = decltype(sel::zmin(index_extents{}));
using zmax_t = decltype(sel::zmax(index_extents{}));

} // namespace sel

namespace detail
{
// multi_slice view is used to select multiple slices of data and treat them as a single
// range. Used to construct the `fluid` selector
template <typename Rng, typename Fn>
class multi_slice_view : public std::ranges::view_interface<multi_slice_view<Rng, Fn>>
{
    using diff_t = std::ranges::range_difference_t<Rng>;

    Rng base_;
    std::span<const index_slice> slices_;

    ccs::semiregular_box<Fn> f;

public:
    class iterator
    {
        using base_iter = std::ranges::iterator_t<Rng>;
        using slice_it = typename std::span<const index_slice>::iterator;

        base_iter base_it_{};
        slice_it slice_{};
        slice_it last_slice_{};

        integer i_{};       // current index in the base range [slice->first, slice->last)
        integer multi_i_{}; // index in the multi_slice, allows for quick size computation

    public:
        using difference_type = diff_t;
        using value_type = std::ranges::range_value_t<Rng>;
        using reference = std::ranges::range_reference_t<Rng>;
        using iterator_concept = std::random_access_iterator_tag;
        using iterator_category = std::random_access_iterator_tag;

        iterator() = default;

        constexpr iterator(base_iter it,
                           slice_it slice,
                           slice_it last_slice,
                           integer i,
                           integer multi_i)
            : base_it_{std::move(it)}
            , slice_{slice}
            , last_slice_{last_slice}
            , i_{i}
            , multi_i_{multi_i}
        {
        }

        constexpr reference operator*() const { return *base_it_; }

        constexpr iterator& operator++()
        {
            ++i_;
            ++base_it_;
            ++multi_i_;
            if (i_ == slice_->last && ++slice_ != last_slice_) {
                std::ranges::advance(base_it_, slice_->first - i_);
                i_ = slice_->first;
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
            --i_;
            --base_it_;
            --multi_i_;
            if (slice_ == last_slice_) {
                // Stepping back from the end position
                --slice_;
            } else if (i_ < slice_->first) {
                --slice_;
                std::ranges::advance(base_it_, slice_->last - 1 - i_);
                i_ = slice_->last - 1;
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

            multi_i_ += n;
            difference_type it_off = 0;

            if (n > 0) {
                // move iterator to slice start to make life easier
                it_off = (slice_->first - i_);
                i_ = slice_->first;
                n -= it_off;
                while (slice_ != last_slice_ &&
                       n > ((slice_->last - 1) - slice_->first)) {
                    n -= (slice_->last - slice_->first);
                    ++slice_;
                }
                if (slice_ != last_slice_) {
                    it_off += slice_->first + n - i_;
                    i_ = slice_->first + n;
                } else {
                    // Reached end position
                    auto prev = slice_ - 1;
                    it_off += prev->last + n - i_;
                    i_ = prev->last + n;
                }
            } else {
                // Handle end position
                if (slice_ == last_slice_) --slice_;
                // move iterator to slice end to make life easier
                it_off = (slice_->last - i_);
                i_ = slice_->last;
                n -= it_off;

                while (n < (slice_->first - slice_->last)) {
                    n -= (slice_->first - slice_->last);
                    --slice_;
                }
                it_off += slice_->last + n - i_;
                i_ = slice_->last + n;
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
            return a.multi_i_ - b.multi_i_;
        }

        constexpr auto base() const { return base_it_; }

        friend constexpr bool operator==(const iterator& a, const iterator& b)
        {
            return a.multi_i_ == b.multi_i_;
        }

        friend constexpr auto operator<=>(const iterator& a, const iterator& b)
        {
            return a.multi_i_ <=> b.multi_i_;
        }
    };

    multi_slice_view() = default;

    explicit constexpr multi_slice_view(Rng rng,
                                        std::span<const index_slice> slices,
                                        Fn f)
        : base_{MOVE(rng)}, slices_{MOVE(slices)}, f{MOVE(f)}
    {
    }

    constexpr auto begin()
    {
        auto it = std::ranges::begin(base_);
        auto first = slices_.begin();
        auto last = slices_.end();
        integer i = first != last ? first->first : 0;
        std::ranges::advance(it, i);
        return iterator(std::move(it), first, last, i, 0);
    }

    constexpr auto end()
    {
        auto it = std::ranges::begin(base_);
        integer total = 0;
        integer i = 0;
        auto first = slices_.begin();
        auto last = slices_.end();
        for (auto s = first; s != last; ++s) {
            total += s->last - s->first;
            i = s->last;
        }
        std::ranges::advance(it, i);
        return iterator(std::move(it), last, last, i, total);
    }

    constexpr auto& base() & { return base_; }
    constexpr const auto& base() const& { return base_; }

    template <typename U>
        requires std::invocable<Fn, U>
    constexpr auto apply(U&& u) const { return f(FWD(u)); }
};

template <typename Rng, typename Fn>
multi_slice_view(Rng&&, std::span<const index_slice>, Fn) -> multi_slice_view<std::views::all_t<Rng>, Fn>;

struct multi_slice_base_fn {
    template <typename Rng>
    constexpr auto operator()(Rng&& rng, std::span<const index_slice> slices) const;

    template <typename Rng, typename F>
    constexpr auto
    operator()(Rng&& rng, std::span<const index_slice> slices, F&& f) const;
};

struct multi_slice_fn : multi_slice_base_fn {
    using base = multi_slice_base_fn;

    template <TupleLike U>
    constexpr auto operator()(U&& u, std::span<const index_slice> slices) const
    {
        if constexpr (Scalar<U>) {
            return tuple{base::operator()(FWD(u) | sel::D, MOVE(slices))};
        } else if constexpr (Vector<U>) {
            return tuple{base::operator()(FWD(u) | sel::Dx, slices, sel::Dx),
                         base::operator()(FWD(u) | sel::Dy, slices, sel::Dy),
                         base::operator()(FWD(u) | sel::Dz, slices, sel::Dz)};
        } else {
            return tuple{base::operator()(FWD(u), MOVE(slices))};
        }
    }

    constexpr auto operator()(std::span<const index_slice> slices) const
    {
        return ccs::make_view_closure(ccs::bind_back(*this, MOVE(slices)));
    }
};

template <typename Rng>
constexpr auto multi_slice_base_fn::operator()(Rng&& rng,
                                               std::span<const index_slice> slices) const
{
    return multi_slice_view(FWD(rng), slices, ccs::bind_back(multi_slice_fn{}, slices));
}

template <typename Rng, typename F>
constexpr auto multi_slice_base_fn::operator()(Rng&& rng,
                                               std::span<const index_slice> slices,
                                               F&& f) const
{
    return multi_slice_view(
        FWD(rng), slices, ccs::compose(ccs::bind_back(multi_slice_fn{}, slices), FWD(f)));
}

} // namespace detail

namespace sel
{
constexpr inline auto multi_slice = ::ccs::detail::multi_slice_fn{};
using multi_slice_t = decltype(multi_slice(std::span<const index_slice>{}));
} // namespace sel

namespace detail
{
// optional_view is used to make a range appear as zero sized
template <typename Rng, typename Fn>
class optional_view : public std::ranges::view_interface<optional_view<Rng, Fn>>
{
    Rng base_;
    bool keep_bounds_;

    ccs::semiregular_box<Fn> f;

public:
    optional_view() = default;

    explicit constexpr optional_view(Rng rng, bool keep_bounds, Fn f)
        : base_{MOVE(rng)}, keep_bounds_{keep_bounds}, f{MOVE(f)}
    {
    }

    constexpr auto begin()
    {
        auto it = std::ranges::begin(base_);
        if (!keep_bounds_) std::ranges::advance(it, std::ranges::end(base_));
        return it;
    }

    constexpr auto end() { return std::ranges::end(base_); }

    constexpr auto& base() & { return base_; }
    constexpr const auto& base() const& { return base_; }

    template <typename U>
    constexpr auto apply(U&& u) const
    {
        constexpr bool nested = requires(optional_view o, U u) { o.base().apply(u); };
        if constexpr (nested)
        {
            return f(this->base().apply(FWD(u)));
        }
        else { return f(FWD(u)); }
    }
};

template <typename Rng, typename Fn>
optional_view(Rng&&, bool, Fn) -> optional_view<std::views::all_t<Rng>, Fn>;

struct optional_view_fn {

    template <Range U>
    constexpr auto operator()(U&& u, bool keep_bounds) const
    {
        return tuple{
            optional_view(FWD(u), keep_bounds, ccs::bind_back(*this, keep_bounds))};
    }

    template <TupleLike U>
        requires(!Range<U>)
    constexpr auto operator()(U&& u, bool keep_bounds) const
    {
        return transform(
            [keep_bounds](auto&& rng) {
                return optional_view(FWD(rng),
                                     keep_bounds,
                                     ccs::bind_back(optional_view_fn{}, keep_bounds));
            },
            FWD(u));
    }

    constexpr auto operator()(bool keep_bounds) const
    {
        return ccs::make_view_closure(ccs::bind_back(*this, keep_bounds));
    }
};
} // namespace detail

namespace sel
{
constexpr inline auto optional_view = ::ccs::detail::optional_view_fn{};
// using multi_slice_t = decltype(multi_slice(std::span<const index_slice>{}));
} // namespace sel

//
// indirect selection based on predicate ranges
//
namespace detail
{
// predicate view is used to select elements from a different range if the predicate range
// is true
template <typename Rng, typename Pred, typename Fn>
class predicate_view
    : public std::ranges::view_interface<predicate_view<Rng, Pred, Fn>>
{
    using base_iter_t = std::ranges::iterator_t<Rng>;
    using pred_iter_t = std::ranges::iterator_t<Pred>;

    Rng base_;
    Pred pred_;
    ccs::semiregular_box<Fn> f;

    std::optional<base_iter_t> cached_begin_;
    pred_iter_t cached_pred_begin_{};

public:
    class iterator
    {
        base_iter_t base_it_{};
        base_iter_t base_end_{};
        pred_iter_t pred_it_{};
        pred_iter_t pred_end_{};

        constexpr void satisfy_forward()
        {
            while (base_it_ != base_end_ && pred_it_ != pred_end_ && !(*pred_it_)) {
                ++base_it_;
                ++pred_it_;
            }
        }

    public:
        using difference_type = std::ranges::range_difference_t<Rng>;
        using value_type = std::ranges::range_value_t<Rng>;
        using reference = std::ranges::range_reference_t<Rng>;
        using iterator_concept = std::bidirectional_iterator_tag;
        using iterator_category = std::bidirectional_iterator_tag;

        iterator() = default;

        constexpr iterator(base_iter_t base_it,
                           base_iter_t base_end,
                           pred_iter_t pred_it,
                           pred_iter_t pred_end,
                           bool do_satisfy = false)
            : base_it_{std::move(base_it)}
            , base_end_{std::move(base_end)}
            , pred_it_{std::move(pred_it)}
            , pred_end_{std::move(pred_end)}
        {
            if (do_satisfy) satisfy_forward();
        }

        constexpr reference operator*() const { return *base_it_; }

        constexpr iterator& operator++()
        {
            ++base_it_;
            ++pred_it_;
            satisfy_forward();
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
            do {
                --base_it_;
                --pred_it_;
            } while (!(*pred_it_));
            return *this;
        }

        constexpr iterator operator--(int)
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        constexpr auto base() const { return base_it_; }

        friend constexpr bool operator==(const iterator& a, const iterator& b)
        {
            return a.base_it_ == b.base_it_;
        }
    };

    predicate_view() = default;

    explicit constexpr predicate_view(Rng rng, Pred p, Fn f)
        : base_{MOVE(rng)}, pred_{MOVE(p)}, f{MOVE(f)}
    {
        assert(std::ranges::size(base_) == std::ranges::size(pred_));
    }

    constexpr auto begin()
    {
        if (!cached_begin_) {
            auto it = std::ranges::begin(base_);
            auto pit = std::ranges::begin(pred_);
            auto base_end = std::ranges::end(base_);
            auto pred_end = std::ranges::end(pred_);
            // skip initial false predicates
            while (it != base_end && pit != pred_end && !(*pit)) {
                ++it;
                ++pit;
            }
            cached_begin_.emplace(it);
            cached_pred_begin_ = pit;
        }
        return iterator(*cached_begin_,
                         std::ranges::end(base_),
                         cached_pred_begin_,
                         std::ranges::end(pred_));
    }

    constexpr auto end()
    {
        return iterator(std::ranges::end(base_),
                         std::ranges::end(base_),
                         std::ranges::end(pred_),
                         std::ranges::end(pred_));
    }

    constexpr auto& base() & { return base_; }
    constexpr const auto& base() const& { return base_; }

    template <typename U>
    constexpr auto apply(U&& u) const
    {
        constexpr bool nested = requires(predicate_view o, U u) { o.base().apply(u); };
        if constexpr (nested)
        {
            return f(this->base().apply(FWD(u)));
        }
        else { return f(FWD(u)); }
    }
};

template <typename Rng, typename Pred, typename Fn>
predicate_view(Rng&&, Pred, Fn) -> predicate_view<std::views::all_t<Rng>, Pred, Fn>;

struct predicate_view_base_fn {
    template <typename Rng, typename Pred>
    constexpr auto operator()(Rng&& rng, Pred&& pred) const;

    template <typename Rng, typename Pred, typename F>
    constexpr auto operator()(Rng&& rng, Pred&&, F&& f) const;
};

struct predicate_view_fn : predicate_view_base_fn {
    using base = predicate_view_base_fn;

    template <TupleLike U, typename P>
    constexpr auto operator()(U&& u, P&& p) const
    {
        if constexpr (SimilarTuples<U, P>)
            return transform(
                [this](auto&& ui, auto&& pi) {
                    return this->base::operator()(tuple{FWD(ui)}, FWD(pi));
                },
                FWD(u),
                FWD(p));
        else
            return transform(
                [this, pi = FWD(p)](auto&& ui) {
                    return this->base::operator()(tuple{FWD(ui)}, pi);
                },
                FWD(u));
        // if constexpr (Scalar<U>) {
        //     return tuple{base::operator()(FWD(u) | sel::D, MOVE(slices))};
        // } else if constexpr (Vector<U>) {
        //     return tuple{base::operator()(FWD(u) | sel::Dx, slices, sel::Dx),
        //                  base::operator()(FWD(u) | sel::Dy, slices, sel::Dy),
        //                  base::operator()(FWD(u) | sel::Dz, slices, sel::Dz)};
        // } else {
        //     return tuple{base::operator()(FWD(u), MOVE(slices))};
        // }
    }

    template <typename P>
    constexpr auto operator()(P&& p) const
    {
        return ccs::make_view_closure(ccs::bind_back(*this, FWD(p)));
    }
};

template <typename Rng, typename Pred>
constexpr auto predicate_view_base_fn::operator()(Rng&& rng, Pred&& pred) const
{
    return predicate_view(FWD(rng), pred, ccs::bind_back(predicate_view_fn{}, pred));
}

template <typename Rng, typename Pred, typename F>
constexpr auto predicate_view_base_fn::operator()(Rng&& rng, Pred&& pred, F&& f) const
{
    return predicate_view(
        FWD(rng), pred, ccs::compose(ccs::bind_back(predicate_view_fn{}, pred), FWD(f)));
}

} // namespace detail

namespace sel
{
constexpr inline auto predicate = ::ccs::detail::predicate_view_fn{};

} // namespace sel

} // namespace ccs
