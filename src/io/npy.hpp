#pragma once

// Minimal writer for NumPy .npy files (format version 1.0, little-endian,
// C order) for rank-1 and rank-2 arrays of f64, i64, i32 and u8.

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ccs::npy
{

static_assert(std::endian::native == std::endian::little,
              "npy writer assumes a little-endian host");

template <typename T>
constexpr std::string_view descr()
{
    if constexpr (std::is_same_v<T, double>)
        return "<f8";
    else if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && sizeof(T) == 8)
        return "<i8";
    else if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && sizeof(T) == 4)
        return "<i4";
    else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> && sizeof(T) == 1)
        return "|u1";
    else
        static_assert(sizeof(T) == 0, "unsupported npy dtype");
}

inline std::string header(std::string_view dtype, std::span<const std::size_t> shape)
{
    std::string dict = "{'descr': '";
    dict += dtype;
    dict += "', 'fortran_order': False, 'shape': (";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        dict += std::to_string(shape[i]);
        if (i + 1 < shape.size()) dict += ", ";
    }
    if (shape.size() == 1) dict += ",";
    dict += "), }";

    // magic (6) + version (2) + header length (2) + dict + padding + newline,
    // padded to a multiple of 64 bytes
    const std::size_t prefix = 10;
    const std::size_t total = prefix + dict.size() + 1;
    const std::size_t pad = (64 - total % 64) % 64;
    dict.append(pad, ' ');
    dict += '\n';
    if (dict.size() > 65535) throw std::length_error("npy header too long");

    std::string out;
    out.reserve(prefix + dict.size());
    out += "\x93NUMPY";
    out += char(1);
    out += char(0);
    const auto len = static_cast<std::uint16_t>(dict.size());
    out += char(len & 0xff);
    out += char(len >> 8);
    out += dict;
    return out;
}

template <typename T>
void write(const std::filesystem::path& path,
           std::span<const T> data,
           std::initializer_list<std::size_t> shape)
{
    std::size_t n = 1;
    for (auto s : shape) n *= s;
    if (n != data.size())
        throw std::invalid_argument("npy::write: shape does not match data size for " +
                                    path.string());

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy::write: cannot open " + path.string());
    std::vector<std::size_t> sh(shape);
    const auto h = header(descr<T>(), sh);
    f.write(h.data(), h.size());
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
    if (!f) throw std::runtime_error("npy::write: write failed for " + path.string());
}

// rank 1
template <typename T>
void write(const std::filesystem::path& path, std::span<const T> data)
{
    write<T>(path, data, {data.size()});
}

template <typename T>
void write(const std::filesystem::path& path, const std::vector<T>& data)
{
    write<T>(path, std::span<const T>(data), {data.size()});
}

// rank 2, C order
template <typename T>
void write(const std::filesystem::path& path,
           const std::vector<T>& data,
           std::size_t rows,
           std::size_t cols)
{
    write<T>(path, std::span<const T>(data), {rows, cols});
}

} // namespace ccs::npy
