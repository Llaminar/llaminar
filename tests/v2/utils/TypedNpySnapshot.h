/**
 * @file TypedNpySnapshot.h
 * @brief Authenticate NumPy scalar descriptors before canonical parity payload loading.
 *
 * cnpy reports word width without the descriptor's integer/float distinction.
 * This adapter authenticates dtype, row order and payload extent first, then
 * delegates payload loading to cnpy. An i8 tensor can never enter its f8 path.
 */
#pragma once
#include "snapshots/SnapshotValues.h"
#include <cnpy.h>
#include <algorithm>
#include <bit>
#include <fstream>
#include <limits>
#include <regex>
#include <string>

namespace llaminar2::test
{
/** @brief Dtype identity authenticated directly from the NumPy header. */
enum class NpySnapshotScalar { FP32, FP64, INT64 };
/** @brief Exact NumPy descriptor and payload geometry before any scalar interpretation. */
struct NpySnapshotDescriptor
{
    NpySnapshotScalar scalar;
    std::vector<std::size_t> shape;
    std::size_t count = 1;
};
/** @brief Authenticate supported little-endian, C-order NumPy snapshots and complete file extent. */
inline NpySnapshotDescriptor inspectNpySnapshot(const std::string &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("snapshot file unavailable: " + path);
    unsigned char prefix[12]{};
    input.read(reinterpret_cast<char *>(prefix), 8);
    if (!input || std::string(reinterpret_cast<char *>(prefix), 6) != std::string("\x93NUMPY", 6) ||
        prefix[7] != 0 || prefix[6] != 1)
        throw std::invalid_argument("invalid NumPy snapshot magic/version");
    const std::size_t width = 2;
    input.read(reinterpret_cast<char *>(prefix + 8), width);
    std::size_t header_bytes = 0;
    for (std::size_t i = 0; i < width; ++i) header_bytes |= std::size_t(prefix[8 + i]) << (8 * i);
    if (!input || header_bytes == 0 || header_bytes > 256)
        throw std::invalid_argument("invalid NumPy snapshot header extent");
    std::string header(header_bytes, '\0');
    input.read(header.data(), header.size());
    if (!input || header.back() != '\n') throw std::invalid_argument("truncated NumPy snapshot header");
    std::smatch match;
    if (!std::regex_search(header, match, std::regex(R"(['"]descr['"]\s*:\s*['"]([<>=|])([fi])([48])['"])")))
        throw std::invalid_argument("unsupported NumPy snapshot scalar descriptor");
    const char endian = match[1].str()[0], kind = match[2].str()[0], bytes = match[3].str()[0];
    if (endian != '<' || std::endian::native != std::endian::little ||
        (kind == 'i' && bytes != '8'))
        throw std::invalid_argument("snapshot requires supported little-endian FP32/FP64/INT64 values");
    // The installed cnpy parser accepts the canonical v1 single-quoted
    // descriptor only. Authenticate its fixed descriptor offset before calling
    // that dependency, whose unchecked endian assertion must never fire.
    const auto descriptor_key = header.find("descr");
    if (descriptor_key == std::string::npos || descriptor_key + 9 >= header.size() ||
        header[descriptor_key + 9] != '<')
        throw std::invalid_argument("snapshot header is outside canonical cnpy v1 descriptor syntax");
    NpySnapshotDescriptor descriptor{kind == 'i' ? NpySnapshotScalar::INT64 :
        bytes == '4' ? NpySnapshotScalar::FP32 : NpySnapshotScalar::FP64};
    if (!std::regex_search(header, match, std::regex(R"(['"]fortran_order['"]\s*:\s*(False|True))")) ||
        match[1].str() != "False")
        throw std::invalid_argument("snapshot requires C-order row storage");
    if (!std::regex_search(header, match, std::regex(R"(['"]shape['"]\s*:\s*\(([0-9,\s]*)\))")))
        throw std::invalid_argument("snapshot requires an explicit nonnegative shape");
    const auto dimensions = match[1].str();
    const std::regex integer("[0-9]+");
    for (auto i = std::sregex_iterator(dimensions.begin(), dimensions.end(), integer);
         i != std::sregex_iterator(); ++i)
    {
        const auto value = std::stoull(i->str());
        if (value > std::numeric_limits<std::size_t>::max() ||
            (value != 0 && descriptor.count > std::numeric_limits<std::size_t>::max() / value))
            throw std::overflow_error("snapshot shape exceeds addressable capacity");
        descriptor.shape.push_back(static_cast<std::size_t>(value));
        descriptor.count *= value;
    }
    const auto element_bytes = std::size_t(bytes - '0');
    if (descriptor.count > std::numeric_limits<std::size_t>::max() / element_bytes)
        throw std::overflow_error("snapshot byte extent exceeds addressable capacity");
    const auto payload_start = input.tellg();
    input.seekg(0, std::ios::end);
    if (input.tellg() - payload_start != static_cast<std::streamoff>(descriptor.count * element_bytes))
        throw std::invalid_argument("snapshot payload extent disagrees with its authenticated shape");
    return descriptor;
}
/** @brief Load authenticated values through the canonical cnpy payload reader. */
inline SnapshotValues loadTypedNpySnapshot(const std::string &path)
{
    const auto descriptor = inspectNpySnapshot(path);
    auto array = cnpy::npy_load(path);
    if (array.shape != descriptor.shape || array.num_vals != descriptor.count || array.fortran_order)
        throw std::invalid_argument("canonical NumPy payload and authenticated header disagree");
    if (descriptor.scalar == NpySnapshotScalar::INT64)
    {
        const auto *data = array.data<std::int64_t>();
        return std::vector<std::int64_t>(data, data + array.num_vals);
    }
    if (descriptor.scalar == NpySnapshotScalar::FP32)
    {
        const auto *data = array.data<float>();
        return std::vector<float>(data, data + array.num_vals);
    }
    const auto *data = array.data<double>();
    std::vector<float> converted(array.num_vals);
    for (std::size_t i = 0; i < array.num_vals; ++i) converted[i] = static_cast<float>(data[i]);
    return converted;
}
/** @return Exact mismatched identity count, including missing/extra scalar positions. */
inline std::size_t integerSnapshotMismatches(std::span<const std::int64_t> actual,
                                            std::span<const std::int64_t> reference)
{
    const auto common = std::min(actual.size(), reference.size());
    std::size_t mismatches = std::max(actual.size(), reference.size()) - common;
    for (std::size_t i = 0; i < common; ++i) mismatches += actual[i] != reference[i];
    return mismatches;
}
} // namespace llaminar2::test
