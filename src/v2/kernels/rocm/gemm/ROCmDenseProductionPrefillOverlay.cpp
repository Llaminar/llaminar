/**
 * @file ROCmDenseProductionPrefillOverlay.cpp
 * @brief Validate and search the immutable common ROCm dense-prefill policy.
 *
 * The generated include is the only exact geometry/launch authority. Compile-
 * time admission proves its sorted unique keys and producer ABIs before a
 * binary can publish them. Lookup returns the measured value unchanged; the
 * caller's existing generic policy continues to own every unmeasured geometry.
 * Cold evidence readers borrow this same compiled table; a trainer-local copy
 * cannot authenticate the linked selector's installed policy generation.
 */
#include "ROCmDenseProductionPrefillOverlay.h"
#include "ROCmDensePrefillPolicyEvidence.h"
// The generated inline object must remain private to this library. Otherwise
// ELF executable interposition can substitute a stale trainer-local table for
// both production lookup and its evidence view despite different core bytes.
#pragma GCC visibility push(hidden)
#include "ROCmDenseProductionPrefillOverlayGenerated.inc"
#pragma GCC visibility pop
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include "tensors/NativeQuantisedFormatInfo.h"
namespace llaminar2::rocm
{
namespace
{
thread_local bool exact_enabled = true;
/** @return Stable lexicographic identity used by generation and lookup. */
constexpr auto key(const generated::ROCmDensePrefillOverlayEntry &entry) noexcept
{ return std::tuple(entry.codebook, entry.m, entry.n, entry.k, entry.family); }
/** @return The new architecture-qualified producer domain; historical rows stay unobserved. */
constexpr bool nativeQ8(const generated::ROCmDensePrefillOverlayEntry &entry) noexcept
{ return entry.codebook == 19 && entry.family == DensePrefillWeightFamily::NativeQuantised; }
/** @return Complete immutable identity without reinterpreting legacy observations. */
constexpr auto identity(const generated::ROCmDensePrefillOverlayEntry &entry) noexcept
{ return entry.orderingKey(); }
/** @return Whether the canonical source catalog owns this execution decoder. */
constexpr bool sourceOwned(std::uint8_t codebook) noexcept
{
    for (const auto &source : native_quantised_formats::kAllSourceFormats)
        if (canonicalDeviceVnniCodebookId(*source.metadata) == codebook) return true;
    return false;
}
/** @return Whether one immutable launch is supported by its physical producer. */
constexpr bool admitted(const generated::ROCmDensePrefillOverlayEntry &entry) noexcept
{
    const auto &launch = entry.config;
    // New native measurements retain actual architecture for every codebook.
    // Historical rows keep their original unobserved identity; WMMA may never
    // enter through such a row or inherit another architecture's instruction set.
    const bool qualified = entry.architecture.observed();
    if (qualified && (entry.architecture.backend !=
            native_quantised::NativeQuantisedArchitectureBackend::ROCm ||
            entry.family != DensePrefillWeightFamily::NativeQuantised)) return false;
    if (!qualified && (nativeQ8(entry) || launch.producer == VNNIPrefillProducer::NativeWmma ||
        entry.architecture != native_quantised::NativeQuantisedArchitecture{})) return false;
    if (!sourceOwned(entry.codebook) ||
        !densePrefillWeightFamilyValid(entry.family) ||
        entry.m <= 0 || entry.n <= 0 || entry.k <= 0 || entry.k % 32 != 0 ||
        launch.m_tile <= 0 || launch.n_tile <= 0 ||
        (launch.unroll != 0 && launch.unroll != 1 && launch.unroll != 2 && launch.unroll != 4)) return false;
    if (launch.full_tiles && (entry.m % launch.m_tile || entry.n % launch.n_tile || entry.k % 64)) return false;
    if (!densePrefillProducerMatchesFamily(entry.family, launch.producer)) return false;
    switch (launch.producer)
    {
    case VNNIPrefillProducer::NativeWmma:
        return entry.architecture == native_quantised::NativeQuantisedArchitecture::rocm(11, 0, 0) &&
            ((launch.n_tile == 32 && launch.m_tile == 32) ||
             (launch.n_tile == 64 && (launch.m_tile == 32 || launch.m_tile == 64)) ||
             (launch.n_tile == 128 && launch.m_tile == 64)) &&
            launch.min_blocks == 1 && launch.unroll == 0 && !launch.full_tiles;
    case VNNIPrefillProducer::NativeCooperative:
        return (launch.n_tile == 64 || launch.n_tile == 128) &&
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
              entry.codebook == 12 || entry.codebook == 15 || entry.codebook == 16 || entry.codebook == 24))) &&
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
        if (!admitted(entries[index]) || (index && !(identity(entries[index - 1]) < identity(entries[index])))) return false;
    return std::size(entries) != 0;
}
static_assert(completeInventory(), "ROCm ordinary-prefill overlay violates key or producer admission");
}
bool densePrefillExactOverlayEnabled() noexcept { return exact_enabled; }
std::span<const generated::ROCmDensePrefillOverlayEntry> densePrefillInstalledEntries() noexcept
{ return generated::kROCmDensePrefillOverlayEntries; }
void setDensePrefillExactOverlayEnabled(bool enabled) noexcept { exact_enabled = enabled; }
std::optional<DensePrefillExactConfig> selectDensePrefillExactConfigFromEntries(
    std::span<const generated::ROCmDensePrefillOverlayEntry> entries,
    std::uint8_t codebook, int m, int n, int k, DensePrefillWeightFamily family,
    const native_quantised::NativeQuantisedGPUHardware *actual_hardware)
{
    if (!densePrefillWeightFamilyValid(family))
        throw std::invalid_argument("Unknown ROCm dense prefill weight family");
    if (m <= 0 || n <= 0 || k <= 0) throw std::invalid_argument("ROCm dense prefill requires positive geometry");
    if (!exact_enabled) return std::nullopt;
    const auto requested = std::tuple(codebook, m, n, k, family);
    const auto first = std::lower_bound(entries.begin(), entries.end(), requested,
        [](const auto &entry, const auto &lookup) { return key(entry) < lookup; });
    const auto last = std::upper_bound(first, entries.end(), requested,
        [](const auto &lookup, const auto &entry) { return lookup < key(entry); });
    if (first == last) return std::nullopt;
    for (auto entry = first; entry != last; ++entry)
        if (!admitted(*entry))
            throw std::logic_error("ROCm measured prefill entry violates architecture or producer admission");
    const auto legacy = !first->architecture.observed()
        ? std::optional<DensePrefillExactConfig>{first->config} : std::nullopt;
    if (!actual_hardware) return legacy;
    actual_hardware->requireObserved();
    if (actual_hardware->architecture.backend != native_quantised::NativeQuantisedArchitectureBackend::ROCm)
        throw std::invalid_argument("ROCm dense prefill needs original ROCm hardware metadata");
    const auto *selected = native_quantised::selectNativeQuantisedArchitectureEntry(
        actual_hardware->architecture,
        std::span<const generated::ROCmDensePrefillOverlayEntry>(&*first, static_cast<std::size_t>(last - first)),
        [&](const auto &entry) {
            if (entry.config.producer == VNNIPrefillProducer::NativeWmma)
            {
                const auto &launch = entry.config;
                // Conservative bound including all correction metadata planes.
                const std::uint64_t lds = 4ull * (52ull * launch.m_tile + 42ull * launch.n_tile);
                return actual_hardware->architecture ==
                    native_quantised::NativeQuantisedArchitecture::rocm(11, 0, 0) &&
                    actual_hardware->admits(256, lds);
            }
            if (entry.config.producer == VNNIPrefillProducer::NativeStreaming)
            {
                // Streaming stages only row correction sums. Unlike the
                // cooperative tiles, this live extent grows with K. IQ1_M
                // retains four eight-value sums; asymmetric sources need one.
                const std::uint64_t groups = entry.codebook == 17 ? 4 :
                    (entry.codebook == 5 || entry.codebook == 7 || entry.codebook == 16 ? 1 : 0);
                const std::uint64_t lds = groups * entry.config.m_tile * (entry.k / 32) * sizeof(std::int32_t);
                return actual_hardware->admits(256, lds);
            }
            // Source-declared Q8 cooperative LDS, BPT=2, double buffered:
            // A/B int32 tiles + half scale plane + conservative small dummy
            // arrays (four uint16[2][1], one int32[2][1]). Compilers may remove
            // dummy storage; this bound cannot admit an oversized launch.
            const auto &launch = entry.config;
            const std::uint64_t lds = entry.codebook == 19
                ? 128ull * (launch.m_tile + launch.n_tile) + 8ull * launch.n_tile + 24ull
                : 160ull * launch.m_tile + 168ull * launch.n_tile + 24ull;
            // Native preferred wave32 on gfx1100 does not contradict the
            // separately compiled wave64 kernel. Signed dot4 availability is
            // already enforced by the actual per-target compilation contract.
            return actual_hardware->admits(256, lds);
        });
    return selected ? std::optional<DensePrefillExactConfig>{selected->config} : legacy;
}
std::optional<DensePrefillExactConfig> selectDensePrefillExactConfig(
    std::uint8_t codebook, int m, int n, int k, DensePrefillWeightFamily family,
    const native_quantised::NativeQuantisedGPUHardware *actual_hardware)
{
    return selectDensePrefillExactConfigFromEntries(generated::kROCmDensePrefillOverlayEntries,
        codebook, m, n, k, family, actual_hardware);
}
}
