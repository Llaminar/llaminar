/**
 * @file GenerationRequestSeeds.h
 * @brief Immutable request randomness and its shared arena/BOM geometry.
 *
 * Seeds are admission data, not changing PRNG state. Each device draw combines
 * its seed with the device-owned logical position. Reset changes seed bytes
 * behind one stable address; graphs therefore never embed request seed values.
 * The geometry contributes bytes to PhysicalMemoryAuthority, not a second ledger.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2
{
/** @brief Validated, immutable per-request seeds retained through admission. */
class GenerationRequestSeeds final
{
public:
    /**
     * @brief Own an already resolved seed for each independent request.
     * @param seeds Nonempty batch; zero (unresolved random seed) is forbidden.
     * @throws std::invalid_argument when any seed is unresolved or no row exists.
     */
    explicit GenerationRequestSeeds(std::vector<uint64_t> seeds) : seeds_(std::move(seeds))
    {
        if (seeds_.empty() || std::ranges::find(seeds_, uint64_t{0}) != seeds_.end())
            throw std::invalid_argument("Generation admission requires nonzero resolved request seeds");
    }

    /** @return Immutable metadata, never a host shadow of device progress. */
    [[nodiscard]] std::span<const uint64_t> values() const noexcept { return seeds_; }

private:
    std::vector<uint64_t> seeds_; ///< Owned request metadata, independent of caller lifetime.
};

/**
 * @brief One shared storage description for arena construction and admission BOM.
 *
 * The arena's integer storage type is INT32. Two contiguous words carry each
 * UINT64 seed byte-for-byte; kernels consume the aligned allocation as UINT64.
 * There is no numeric conversion, padding between seeds or extra device copy.
 */
class GenerationRequestSeedGeometry final
{
public:
    static constexpr std::size_t words_per_request = sizeof(uint64_t) / sizeof(int32_t);

    /** @brief Retain every ordinary and speculative request row, even with MTP off.
     *  @throws std::invalid_argument for nonpositive capacity inputs. */
    GenerationRequestSeedGeometry(int ordinary_requests, int retained_requests)
    {
        if (ordinary_requests <= 0 || retained_requests <= 0)
            throw std::invalid_argument("Generation seed capacity requires positive request counts");
        requests_ = static_cast<std::size_t>(std::max(ordinary_requests, retained_requests));
        if (requests_ > std::numeric_limits<std::size_t>::max() / sizeof(uint64_t))
            throw std::overflow_error("Generation seed capacity exceeds addressable bytes");
    }

    /** @return Physical seed rows; inactive rows are cleared at each admission. */
    [[nodiscard]] std::size_t requests() const noexcept { return requests_; }
    /** @return Exact payload bytes contributed to the canonical memory BOM. */
    [[nodiscard]] std::size_t bytes() const noexcept { return requests_ * sizeof(uint64_t); }

private:
    std::size_t requests_ = 0; ///< Geometry only; never a live allocation balance.
};
} // namespace llaminar2
