#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <utility>

#include "ccs_range_utils.hpp"

namespace ccs
{

// ===========================================================================
// zip_transform_view<F, Rngs...>
//
// Lazy view that yields f(*it1, *it2, ...).
// Models std::ranges::view_interface. Propagates the weakest iterator
// category from base ranges (like C++23 zip_transform_view).
// ===========================================================================
namespace detail
{

template <typename F, typename... Rngs>
class zip_transform_iterator
{
    using tuple_of_iters = std::tuple<std::ranges::iterator_t<Rngs>...>;
    tuple_of_iters iters_;
    const F* f_;

    static constexpr bool all_bidir =
        (std::ranges::bidirectional_range<Rngs> && ...);
    static constexpr bool all_random =
        (std::ranges::random_access_range<Rngs> && ...);

public:
    using difference_type = std::ptrdiff_t;
    using value_type =
        std::remove_cvref_t<std::invoke_result_t<const F&,
                                                  std::ranges::range_reference_t<Rngs>...>>;
    using reference =
        std::invoke_result_t<const F&, std::ranges::range_reference_t<Rngs>...>;
    using iterator_concept = std::conditional_t<
        all_random,
        std::random_access_iterator_tag,
        std::conditional_t<
            all_bidir,
            std::bidirectional_iterator_tag,
            std::conditional_t<(std::ranges::forward_range<Rngs> && ...),
                               std::forward_iterator_tag,
                               std::input_iterator_tag>>>;
    using iterator_category = iterator_concept;

    zip_transform_iterator() = default;

    constexpr zip_transform_iterator(const F* f,
                                     std::ranges::iterator_t<Rngs>... its)
        : iters_{std::move(its)...}, f_{f}
    {
    }

    constexpr reference operator*() const
    {
        return std::apply(
            [this](const auto&... its) -> reference { return (*f_)(*its...); }, iters_);
    }

    // -- forward --

    constexpr zip_transform_iterator& operator++()
    {
        std::apply([](auto&... its) { (++its, ...); }, iters_);
        return *this;
    }

    constexpr zip_transform_iterator operator++(int)
    {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    // -- bidirectional --

    constexpr zip_transform_iterator& operator--()
        requires all_bidir
    {
        std::apply([](auto&... its) { (--its, ...); }, iters_);
        return *this;
    }

    constexpr zip_transform_iterator operator--(int)
        requires all_bidir
    {
        auto tmp = *this;
        --*this;
        return tmp;
    }

    // -- random access --

    constexpr zip_transform_iterator& operator+=(difference_type n)
        requires all_random
    {
        std::apply([n](auto&... its) { ((its += n), ...); }, iters_);
        return *this;
    }

    constexpr zip_transform_iterator& operator-=(difference_type n)
        requires all_random
    {
        return *this += -n;
    }

    constexpr reference operator[](difference_type n) const
        requires all_random
    {
        return *(*this + n);
    }

    friend constexpr zip_transform_iterator operator+(zip_transform_iterator it,
                                                      difference_type n)
        requires all_random
    {
        it += n;
        return it;
    }

    friend constexpr zip_transform_iterator operator+(difference_type n,
                                                      zip_transform_iterator it)
        requires all_random
    {
        it += n;
        return it;
    }

    friend constexpr zip_transform_iterator operator-(zip_transform_iterator it,
                                                      difference_type n)
        requires all_random
    {
        it -= n;
        return it;
    }

    friend constexpr difference_type operator-(const zip_transform_iterator& a,
                                               const zip_transform_iterator& b)
        requires all_random
    {
        return std::get<0>(a.iters_) - std::get<0>(b.iters_);
    }

    // -- comparison --

    friend constexpr bool operator==(const zip_transform_iterator& a,
                                     const zip_transform_iterator& b)
    {
        // Equal if ANY pair of iterators matches (like zip semantics)
        return eq_impl(a.iters_, b.iters_, std::index_sequence_for<Rngs...>{});
    }

    friend constexpr auto operator<=>(const zip_transform_iterator& a,
                                      const zip_transform_iterator& b)
        requires all_random
    {
        auto diff = std::get<0>(a.iters_) - std::get<0>(b.iters_);
        return diff <=> decltype(diff){0};
    }

private:
    template <std::size_t... Is>
    static constexpr bool eq_impl(const tuple_of_iters& a,
                                  const tuple_of_iters& b,
                                  std::index_sequence<Is...>)
    {
        return ((std::get<Is>(a) == std::get<Is>(b)) || ...);
    }
};

} // namespace detail

template <typename F, typename... Rngs>
class zip_transform_view
    : public std::ranges::view_interface<zip_transform_view<F, Rngs...>>
{
    semiregular_box<F> f_;
    std::tuple<Rngs...> rngs_;

    using iterator = detail::zip_transform_iterator<F, Rngs...>;

public:
    zip_transform_view() = default;

    constexpr explicit zip_transform_view(F f, Rngs... rngs)
        : f_{std::move(f)}, rngs_{std::move(rngs)...}
    {
    }

    constexpr auto begin()
    {
        return std::apply(
            [this](auto&... rngs) {
                return iterator{&f_.get(), std::ranges::begin(rngs)...};
            },
            rngs_);
    }

    constexpr auto begin() const
    {
        return std::apply(
            [this](const auto&... rngs) {
                return iterator{&f_.get(), std::ranges::begin(rngs)...};
            },
            rngs_);
    }

    constexpr auto end()
    {
        return std::apply(
            [this](auto&... rngs) {
                return iterator{&f_.get(), std::ranges::end(rngs)...};
            },
            rngs_);
    }

    constexpr auto end() const
    {
        return std::apply(
            [this](const auto&... rngs) {
                return iterator{&f_.get(), std::ranges::end(rngs)...};
            },
            rngs_);
    }

    constexpr auto size()
        requires(std::ranges::sized_range<Rngs> && ...)
    {
        return std::apply(
            [](auto&... rngs) {
                return std::ranges::min(
                    {static_cast<std::size_t>(std::ranges::size(rngs))...});
            },
            rngs_);
    }

    constexpr auto size() const
        requires(std::ranges::sized_range<const Rngs> && ...)
    {
        return std::apply(
            [](const auto&... rngs) {
                return std::ranges::min(
                    {static_cast<std::size_t>(std::ranges::size(rngs))...});
            },
            rngs_);
    }
};

template <typename F, typename... Rngs>
zip_transform_view(F, Rngs...) -> zip_transform_view<F, std::views::all_t<Rngs>...>;

// Factory function
struct zip_transform_fn {
    template <typename F, std::ranges::viewable_range... Rngs>
    constexpr auto operator()(F f, Rngs&&... rngs) const
    {
        return zip_transform_view(std::move(f), std::views::all(std::forward<Rngs>(rngs))...);
    }
};
inline constexpr zip_transform_fn zip_transform{};

// ===========================================================================
// repeat_n_view<T>
//
// Lazy view of n copies of value v.
// Models random_access_range and sized_range.
// ===========================================================================
template <typename T>
class repeat_n_view : public std::ranges::view_interface<repeat_n_view<T>>
{
    T value_;
    std::ptrdiff_t count_;

public:
    class iterator
    {
        const T* value_;
        std::ptrdiff_t pos_;

    public:
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using reference = const T&;
        using pointer = const T*;
        using iterator_concept = std::random_access_iterator_tag;
        using iterator_category = std::random_access_iterator_tag;

        iterator() = default;
        constexpr iterator(const T* v, std::ptrdiff_t p) : value_{v}, pos_{p} {}

        constexpr reference operator*() const { return *value_; }
        constexpr reference operator[](difference_type) const { return *value_; }

        constexpr iterator& operator++()
        {
            ++pos_;
            return *this;
        }
        constexpr iterator operator++(int)
        {
            auto tmp = *this;
            ++pos_;
            return tmp;
        }
        constexpr iterator& operator--()
        {
            --pos_;
            return *this;
        }
        constexpr iterator operator--(int)
        {
            auto tmp = *this;
            --pos_;
            return tmp;
        }
        constexpr iterator& operator+=(difference_type n)
        {
            pos_ += n;
            return *this;
        }
        constexpr iterator& operator-=(difference_type n)
        {
            pos_ -= n;
            return *this;
        }

        friend constexpr iterator operator+(iterator it, difference_type n)
        {
            return {it.value_, it.pos_ + n};
        }
        friend constexpr iterator operator+(difference_type n, iterator it)
        {
            return {it.value_, it.pos_ + n};
        }
        friend constexpr iterator operator-(iterator it, difference_type n)
        {
            return {it.value_, it.pos_ - n};
        }
        friend constexpr difference_type operator-(const iterator& a, const iterator& b)
        {
            return a.pos_ - b.pos_;
        }

        friend constexpr bool operator==(const iterator& a, const iterator& b)
        {
            return a.pos_ == b.pos_;
        }
        friend constexpr auto operator<=>(const iterator& a, const iterator& b)
        {
            return a.pos_ <=> b.pos_;
        }
    };

    repeat_n_view() = default;
    constexpr repeat_n_view(T value, std::ptrdiff_t count)
        : value_{std::move(value)}, count_{count}
    {
    }

    constexpr iterator begin() const { return {&value_, 0}; }
    constexpr iterator end() const { return {&value_, count_}; }
    constexpr std::ptrdiff_t size() const { return count_; }
};

// Factory function
struct repeat_n_fn {
    template <typename T>
    constexpr auto operator()(T value, std::ptrdiff_t n) const
    {
        return repeat_n_view<std::decay_t<T>>(std::move(value), n);
    }
};
inline constexpr repeat_n_fn repeat_n{};

// ===========================================================================
// stride_view<Rng>
//
// Lazy view that yields every n-th element of Rng.
// Models view_interface. Supports random-access if base range does.
// ===========================================================================
template <std::ranges::input_range Rng>
    requires std::ranges::view<Rng>
class stride_view : public std::ranges::view_interface<stride_view<Rng>>
{
    Rng base_;
    std::ranges::range_difference_t<Rng> stride_;

public:
    class iterator
    {
        using base_iter = std::ranges::iterator_t<Rng>;
        using base_sent = std::ranges::sentinel_t<Rng>;
        using diff_t = std::ranges::range_difference_t<Rng>;

        base_iter current_{};
        base_sent end_{};
        diff_t stride_{};
        diff_t missing_{}; // how many steps short of stride we were at end

    public:
        using difference_type = diff_t;
        using value_type = std::ranges::range_value_t<Rng>;
        using reference = std::ranges::range_reference_t<Rng>;
        using iterator_concept = std::conditional_t<
            std::ranges::random_access_range<Rng>,
            std::random_access_iterator_tag,
            std::conditional_t<std::ranges::bidirectional_range<Rng>,
                               std::bidirectional_iterator_tag,
                               std::conditional_t<std::ranges::forward_range<Rng>,
                                                  std::forward_iterator_tag,
                                                  std::input_iterator_tag>>>;
        using iterator_category = iterator_concept;

        iterator() = default;

        constexpr iterator(base_iter current, base_sent end, diff_t stride,
                           diff_t missing = 0)
            : current_{std::move(current)}
            , end_{std::move(end)}
            , stride_{stride}
            , missing_{missing}
        {
        }

        constexpr reference operator*() const { return *current_; }

        constexpr iterator& operator++()
        {
            missing_ = std::ranges::advance(current_, stride_, end_);
            return *this;
        }

        constexpr iterator operator++(int)
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr iterator& operator--()
            requires std::ranges::bidirectional_range<Rng>
        {
            std::ranges::advance(current_, missing_ - stride_);
            missing_ = 0;
            return *this;
        }

        constexpr iterator operator--(int)
            requires std::ranges::bidirectional_range<Rng>
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        constexpr iterator& operator+=(difference_type n)
            requires std::ranges::random_access_range<Rng>
        {
            if (n > 0) {
                missing_ = std::ranges::advance(current_, stride_ * n, end_);
            } else if (n < 0) {
                std::ranges::advance(current_, stride_ * n + missing_);
                missing_ = 0;
            }
            return *this;
        }

        constexpr iterator& operator-=(difference_type n)
            requires std::ranges::random_access_range<Rng>
        {
            return *this += -n;
        }

        constexpr reference operator[](difference_type n) const
            requires std::ranges::random_access_range<Rng>
        {
            return *(*this + n);
        }

        friend constexpr iterator operator+(iterator it, difference_type n)
            requires std::ranges::random_access_range<Rng>
        {
            it += n;
            return it;
        }

        friend constexpr iterator operator+(difference_type n, iterator it)
            requires std::ranges::random_access_range<Rng>
        {
            it += n;
            return it;
        }

        friend constexpr iterator operator-(iterator it, difference_type n)
            requires std::ranges::random_access_range<Rng>
        {
            it -= n;
            return it;
        }

        friend constexpr difference_type operator-(const iterator& a, const iterator& b)
            requires std::ranges::random_access_range<Rng>
        {
            auto dist = a.current_ - b.current_;
            if (dist > 0)
                return (dist + a.missing_) / a.stride_;
            else if (dist < 0)
                return -((-dist + b.missing_) / b.stride_);
            else
                return 0;
        }

        friend constexpr bool operator==(const iterator& a, const iterator& b)
        {
            return a.current_ == b.current_;
        }

        friend constexpr auto operator<=>(const iterator& a, const iterator& b)
            requires std::ranges::random_access_range<Rng>
        {
            return a.current_ <=> b.current_;
        }
    };

    stride_view() = default;

    constexpr stride_view(Rng base, std::ranges::range_difference_t<Rng> stride)
        : base_{std::move(base)}, stride_{stride}
    {
    }

    constexpr auto begin()
    {
        return iterator{std::ranges::begin(base_), std::ranges::end(base_), stride_};
    }

    constexpr auto begin() const
        requires std::ranges::range<const Rng>
    {
        return iterator{std::ranges::begin(base_), std::ranges::end(base_), stride_};
    }

    constexpr auto end()
    {
        if constexpr (std::ranges::sized_range<Rng>) {
            auto sz = std::ranges::size(base_);
            auto d = static_cast<std::ranges::range_difference_t<Rng>>(sz);
            auto missing = (stride_ - d % stride_) % stride_;
            auto end_it = std::ranges::end(base_);
            return iterator{std::move(end_it),
                            std::ranges::end(base_),
                            stride_,
                            missing};
        } else {
            auto end_it = std::ranges::end(base_);
            return iterator{std::move(end_it), std::ranges::end(base_), stride_};
        }
    }

    constexpr auto end() const
        requires std::ranges::range<const Rng>
    {
        if constexpr (std::ranges::sized_range<const Rng>) {
            auto sz = std::ranges::size(base_);
            auto d = static_cast<std::ranges::range_difference_t<const Rng>>(sz);
            auto missing = (stride_ - d % stride_) % stride_;
            auto end_it = std::ranges::end(base_);
            return iterator{std::move(end_it),
                            std::ranges::end(base_),
                            stride_,
                            missing};
        } else {
            auto end_it = std::ranges::end(base_);
            return iterator{std::move(end_it), std::ranges::end(base_), stride_};
        }
    }

    constexpr auto size()
        requires std::ranges::sized_range<Rng>
    {
        auto sz = std::ranges::size(base_);
        auto d = static_cast<std::ranges::range_difference_t<Rng>>(sz);
        return (d + stride_ - 1) / stride_;
    }

    constexpr auto size() const
        requires std::ranges::sized_range<const Rng>
    {
        auto sz = std::ranges::size(base_);
        auto d = static_cast<std::ranges::range_difference_t<const Rng>>(sz);
        return (d + stride_ - 1) / stride_;
    }
};

template <typename Rng>
stride_view(Rng&&, std::ranges::range_difference_t<Rng>) -> stride_view<std::views::all_t<Rng>>;

// Factory function (takes range + stride)
struct stride_fn {
    template <std::ranges::viewable_range Rng>
    constexpr auto operator()(Rng&& rng,
                              std::ranges::range_difference_t<std::remove_cvref_t<Rng>> n) const
    {
        return stride_view(std::views::all(std::forward<Rng>(rng)), n);
    }
};
inline constexpr stride_fn stride{};

} // namespace ccs
