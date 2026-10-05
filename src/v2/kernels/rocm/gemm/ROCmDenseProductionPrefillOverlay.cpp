/**
 * @file ROCmDenseProductionPrefillOverlay.cpp
 * @brief Validate and search the immutable common ROCm dense-prefill policy.
 *
 * The generated include is the only exact geometry/launch authority. Compile-
 * time admission proves its sorted unique keys and producer ABIs before a
 * binary can publish them. Lookup returns the measured value unchanged; the
 * caller's existing generic policy continues to own every unmeasured geometry.
 */
#include "ROCmDenseProductionPrefillOverlay.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include "tensors/NativeVnniFormatInfo.h"
namespace llaminar2::rocm
{
namespace
{
thread_local bool exact_enabled = true;
/** @return Stable lexicographic identity used by generation and lookup. */
constexpr auto key(const generated::ROCmDensePrefillOverlayEntry &entry) noexcept
{ return std::tuple(entry.codebook, entry.m, entry.n, entry.k); }
/** @return Whether the canonical source catalog owns this execution decoder. */
constexpr bool sourceOwned(std::uint8_t codebook) noexcept
{
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
        if (canonicalDeviceVnniCodebookId(source.metadata->codebook_id) == codebook) return true;
    return false;
}
/** @return Whether one immutable launch is supported by its physical producer. */
constexpr bool admitted(const generated::ROCmDensePrefillOverlayEntry &entry) noexcept
{
    const auto &launch = entry.config;
    if (!sourceOwned(entry.codebook) ||
        entry.m <= 0 || entry.n <= 0 || entry.k <= 0 || entry.k % 32 != 0 ||
        launch.m_tile <= 0 || launch.n_tile <= 0 ||
        (launch.unroll != 0 && launch.unroll != 1 && launch.unroll != 2 && launch.unroll != 4)) return false;
    if (launch.full_tiles && (entry.m % launch.m_tile || entry.n % launch.n_tile || entry.k % 64)) return false;
    switch (launch.producer)
    {
    case VNNIPrefillProducer::NativeCooperative:
        return entry.codebook != 19 && (launch.n_tile == 64 || launch.n_tile == 128) &&
            (launch.m_tile == 16 || launch.m_tile == 32 || (launch.n_tile == 64 && launch.m_tile == 64 && !launch.full_tiles)) &&
            (launch.min_blocks == 1 || launch.min_blocks == 2) &&
            (!launch.full_tiles || entry.codebook == 8) &&
            !(entry.codebook == 8 && !launch.full_tiles && launch.min_blocks == 2 &&
              ((launch.n_tile == 64 && launch.m_tile == 64) || (launch.n_tile == 128 && launch.m_tile == 32)));
    case VNNIPrefillProducer::NativeStreaming:
        return launch.n_tile == 256 &&
            ((launch.m_tile == 8 && (entry.codebook == 8 || entry.codebook == 9 ||
              entry.codebook == 13 || entry.codebook == 14 || entry.codebook == 17)) ||
             (launch.m_tile == 16 && (entry.codebook == 0 || entry.codebook == 4 ||
              entry.codebook == 5 || entry.codebook == 6 || entry.codebook == 11 ||
              entry.codebook == 12 || entry.codebook == 15 || entry.codebook == 16))) &&
            launch.min_blocks == 3 && launch.unroll == 0 && !launch.full_tiles;
    case VNNIPrefillProducer::Int8BlockwiseV3:
        return entry.codebook == 19 && launch.n_tile == 64 && (launch.m_tile == 16 || launch.m_tile == 32) &&
            launch.min_blocks == 1 && !launch.full_tiles;
    case VNNIPrefillProducer::Int8BlockwiseV7:
        return entry.codebook == 19 && launch.n_tile == 128 && (launch.m_tile == 16 || launch.m_tile == 32 || launch.m_tile == 64) &&
            launch.min_blocks == 1 && !launch.full_tiles;
    }
    return false;
}
/** @return Whether every common-generated entry has one valid distinct key. */
consteval bool completeInventory()
{
    constexpr auto &entries = generated::kROCmDensePrefillOverlayEntries;
    for (std::size_t index = 0; index < std::size(entries); ++index)
        if (!admitted(entries[index]) || (index && !(key(entries[index - 1]) < key(entries[index])))) return false;
    return std::size(entries) != 0;
}
static_assert(completeInventory(), "ROCm ordinary-prefill overlay violates key or producer admission");
}
bool densePrefillExactOverlayEnabled() noexcept { return exact_enabled; }
void setDensePrefillExactOverlayEnabled(bool enabled) noexcept { exact_enabled = enabled; }
std::optional<DensePrefillExactConfig> selectDensePrefillExactConfig(
    std::uint8_t codebook, int m, int n, int k, DensePrefillWeightFamily family)
{
    if (family != DensePrefillWeightFamily::NativeLowBit && family != DensePrefillWeightFamily::Int8)
        throw std::invalid_argument("Unknown ROCm dense prefill weight family");
    if (m <= 0 || n <= 0 || k <= 0) throw std::invalid_argument("ROCm dense prefill requires positive geometry");
    if (!exact_enabled) return std::nullopt;
    const auto requested = std::tuple(codebook, m, n, k);
    const auto &entries = generated::kROCmDensePrefillOverlayEntries;
    const auto found = std::lower_bound(std::begin(entries), std::end(entries), requested,
        [](const auto &entry, const auto &lookup) { return key(entry) < lookup; });
    if (found == std::end(entries) || key(*found) != requested) return std::nullopt;
    const bool int8 = found->config.producer == VNNIPrefillProducer::Int8BlockwiseV3 ||
        found->config.producer == VNNIPrefillProducer::Int8BlockwiseV7;
    if (int8 != (family == DensePrefillWeightFamily::Int8))
        throw std::logic_error("ROCm measured prefill producer disagrees with prepared weight family");
    return found->config;
}
}
