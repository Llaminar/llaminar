/**
 * @file NativeQuantisedArchitecture.h
 * @brief Typed native ISA identity and deterministic measured-architecture proximity.
 *
 * Actual hardware comes from the admitted immutable backend inventory. A selected
 * tuning architecture is a separate observation: nearest selection never changes
 * hardware capabilities, compiled kernel availability or resource admission.
 * Generated exact lookup uses this common comparator only after a requested-key
 * candidate passes the actual device's native compatibility predicate. Existing
 * general policies continue to own an empty/incompatible measured domain.
 */
#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <span>
#include <tuple>

namespace llaminar2::native_quantised
{
    /** @brief Native ISA namespace; backend ordinals are not architecture identities. */
    enum class NativeQuantisedArchitectureBackend : std::uint32_t
    { CUDA = 0, ROCm = 1, Unobserved = 2 };

    /** @brief Lossless ISA coordinates, independent of marketing names and device order. */
    struct NativeQuantisedArchitecture
    {
        NativeQuantisedArchitectureBackend backend = NativeQuantisedArchitectureBackend::Unobserved;
        std::uint32_t major = 0, minor = 0, stepping = 0;

        /** @brief Bind the actual CUDA compute-capability coordinates.
         * @param major Positive native SM major version.
         * @param minor Native single-digit SM minor version.
         * @return Checked CUDA architecture, with no invented stepping.
         * @throws std::invalid_argument For malformed native coordinates.
         */
        [[nodiscard]] static constexpr NativeQuantisedArchitecture cuda(std::uint32_t major, std::uint32_t minor)
        {
            if (major == 0 || minor > 9)
                throw std::invalid_argument("CUDA architecture requires native major/minor coordinates");
            return {NativeQuantisedArchitectureBackend::CUDA, major, minor, 0};
        }

        /** @brief Bind the actual AMD GFX ISA version, including hexadecimal stepping.
         * @param major Positive decimal ISA generation (9,10,11,...).
         * @param minor One hexadecimal ISA minor digit.
         * @param stepping One hexadecimal ISA stepping digit (gfx90a retains10).
         * @return Checked ROCm ISA identity, with no device/marketing inference.
         * @throws std::invalid_argument For malformed native coordinates.
         */
        [[nodiscard]] static constexpr NativeQuantisedArchitecture rocm(
            std::uint32_t major, std::uint32_t minor, std::uint32_t stepping)
        {
            if (major == 0 || minor > 15 || stepping > 15)
                throw std::invalid_argument("ROCm architecture requires native GFX coordinates");
            return {NativeQuantisedArchitectureBackend::ROCm, major, minor, stepping};
        }

        /** @brief Parse the immutable SDK ISA target without allocation or a runtime query.
         * @param target Native gfx target, optionally followed by colon feature qualifiers.
         * @return Original ISA generation/minor/stepping. Feature qualifiers do not
         * rename the architecture; native instruction/resource admission remains separate.
         * @throws std::invalid_argument For malformed, truncated or overflowing targets.
         */
        [[nodiscard]] static constexpr NativeQuantisedArchitecture fromROCmISATarget(std::string_view target)
        {
            target = target.substr(0, target.find(':'));
            if (!target.starts_with("gfx") || target.size() < 6)
                throw std::invalid_argument("ROCm architecture needs a complete native gfx target");
            const auto digits = target.substr(3);
            std::uint32_t generation = 0;
            for (std::size_t i = 0; i + 2 < digits.size(); ++i)
            {
                const char digit = digits[i];
                if (digit < '0' || digit > '9' || (i == 0 && digit == '0') ||
                    generation > (std::numeric_limits<std::uint32_t>::max() - std::uint32_t(digit - '0')) / 10)
                    throw std::invalid_argument("ROCm gfx generation is malformed or overflows");
                generation = generation * 10 + std::uint32_t(digit - '0');
            }
            const auto hex = [](char digit) constexpr -> std::uint32_t {
                if (digit >= '0' && digit <= '9') return std::uint32_t(digit - '0');
                if (digit >= 'a' && digit <= 'f') return std::uint32_t(digit - 'a' + 10);
                throw std::invalid_argument("ROCm gfx revision is not canonical hexadecimal");
            };
            return rocm(generation, hex(digits[digits.size() - 2]), hex(digits.back()));
        }

        /** @return Whether this is a complete physical ISA observation. */
        [[nodiscard]] constexpr bool observed() const noexcept
        {
            return major > 0 &&
                ((backend == NativeQuantisedArchitectureBackend::CUDA && minor <= 9 && stepping == 0) ||
                 (backend == NativeQuantisedArchitectureBackend::ROCm && minor <= 15 && stepping <= 15));
        }
        /** @return Full-width structured generated-key order; no lossy packed architecture hash. */
        [[nodiscard]] constexpr std::array<std::uint32_t, 4> scalars() const noexcept
        { return {static_cast<std::uint32_t>(backend), major, minor, stepping}; }
        /** @brief Compare complete ISA coordinates, independently of participant ordinal. */
        bool operator==(const NativeQuantisedArchitecture &) const = default;
    };

    /** @brief Rank same-backend measured candidates using actual ISA-version coordinates.
     * @param actual Admitted physical architecture, never the last selected tuning architecture.
     * @param candidate Architecture of one geometry-matching measured entry that has
     * already passed actual native compatibility/resource admission.
     * @param current Current winner, or explicitly unobserved when none exists.
     * @return Whether candidate is nearer: exact actual first, then lexicographic
     * absolute major distance, full monotonic native-version distance within
     * that class, then lexical ISA tuple on ties. CUDA versions use major*10+minor;
     * ROCm versions use major*256+minor*16+stepping, all computed in uint64.
     * Cross-backend or unqualified candidate records are never eligible. Input
     * order and device ordinal cannot affect equal-distance results.
     */
    [[nodiscard]] constexpr bool isNearerNativeQuantisedArchitecture(
        const NativeQuantisedArchitecture &actual, const NativeQuantisedArchitecture &candidate,
        const NativeQuantisedArchitecture &current)
    {
        if (!actual.observed())
            throw std::invalid_argument("Measured GPU dispatch requires the actual native architecture");
        if (!candidate.observed() || candidate.backend != actual.backend) return false;
        if (!current.observed() || current.backend != actual.backend) return true;
        const auto difference = [](std::uint64_t a, std::uint64_t b) constexpr { return a > b ? a - b : b - a; };
        const auto version = [](const NativeQuantisedArchitecture &architecture) constexpr -> std::uint64_t {
            return architecture.backend == NativeQuantisedArchitectureBackend::CUDA
                ? std::uint64_t(architecture.major) * 10u + architecture.minor
                : std::uint64_t(architecture.major) * 256u + std::uint64_t(architecture.minor) * 16u + architecture.stepping;
        };
        const auto rank = [&](const NativeQuantisedArchitecture &architecture) constexpr {
            return std::tuple(difference(actual.major, architecture.major),
                difference(version(actual), version(architecture)), architecture.scalars());
        };
        return rank(candidate) < rank(current);
    }

    /** @brief Select a geometry-matching measured row using one shared policy.
     * @tparam Entry Generated row exposing its typed architecture field.
     * @tparam Compatible Actual native ISA/resource validator, not an architecture-name guess.
     * @param actual Original admitted physical architecture.
     * @param candidates Only records matching the requested operation/geometry key.
     * @param compatible Returns false for an unsupported actual native launch;
     * malformed generated records must fail through their normal admission.
     * @return Nearest compatible row, or null so the unchanged complete general
     * policy owns an empty/incompatible measured domain. No native work/allocation.
     */
    template<class Entry, class Compatible>
    [[nodiscard]] constexpr const Entry *selectNativeQuantisedArchitectureEntry(
        const NativeQuantisedArchitecture &actual, std::span<const Entry> candidates, Compatible &&compatible)
    {
        if (!actual.observed()) throw std::invalid_argument("GPU dispatch needs actual native architecture before selection");
        const Entry *best = nullptr;
        for (const auto &entry : candidates)
            if (entry.architecture.backend == actual.backend && entry.architecture.observed() && compatible(entry) &&
                isNearerNativeQuantisedArchitecture(actual, entry.architecture,
                    best ? best->architecture : NativeQuantisedArchitecture{}))
                best = &entry;
        return best;
    }

    /** @brief Render native target spelling at cold receipt/log boundaries only.
     * @param architecture Checked physical or selected tuning identity.
     * @return sm86, gfx906, gfx90a, gfx1100, etc.; never a marketing/device name.
     * @throws std::invalid_argument For an unobserved or malformed architecture.
     */
    [[nodiscard]] inline std::string nativeQuantisedArchitectureName(const NativeQuantisedArchitecture &architecture)
    {
        if (!architecture.observed()) throw std::invalid_argument("Cannot name an unobserved GPU architecture");
        if (architecture.backend == NativeQuantisedArchitectureBackend::CUDA)
            return "sm" + std::to_string(architecture.major) + std::to_string(architecture.minor);
        constexpr char digits[] = "0123456789abcdef";
        return "gfx" + std::to_string(architecture.major) + digits[architecture.minor] + digits[architecture.stepping];
    }

    /** @brief Immutable dispatch facts projected from the existing hardware authority.
     * CPU/absent records remain explicitly unobserved. Native GPU execution and
     * workspace selection require the same checked participant facts; this is
     * neither device discovery nor a storage/capacity ledger.
     */
    struct NativeQuantisedGPUHardware
    {
        NativeQuantisedArchitecture architecture;
        int ordinal = -1;
        int compute_units = 0;
        int native_lane_width = 0;
        int maximum_threads_per_block = 0;
        std::uint64_t shared_memory_per_block = 0;

        /** @brief Validate actual or rank-published facts before policy lookup.
         * @throws std::invalid_argument For an absent/contradictory participant.
         */
        constexpr void requireObserved() const
        {
            if (!architecture.observed() || ordinal < 0 || compute_units <= 0 ||
                maximum_threads_per_block <= 0 || shared_memory_per_block == 0 ||
                (architecture.backend == NativeQuantisedArchitectureBackend::CUDA
                    ? native_lane_width != 32 : (native_lane_width != 32 && native_lane_width != 64)))
                throw std::invalid_argument("NativeQuantised GPU policy needs complete actual hardware metadata");
        }
        /** @brief Check a compiled launch footprint against actual physical limits.
         * @param threads Exact native CTA/workgroup size.
         * @param shared_bytes Exact/bounded compiled shared-memory extent.
         * @return Whether both resources fit; ISA-specific admission remains explicit at the caller.
         */
        [[nodiscard]] constexpr bool admits(int threads, std::uint64_t shared_bytes) const
        {
            requireObserved();
            return threads > 0 && threads <= maximum_threads_per_block && shared_bytes <= shared_memory_per_block;
        }
        /** @brief Compare complete actual participant facts, independently of tuning provenance. */
        bool operator==(const NativeQuantisedGPUHardware &) const = default;
    };

}
