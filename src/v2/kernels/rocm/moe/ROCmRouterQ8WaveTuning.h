/**
 * @file ROCmRouterQ8WaveTuning.h
 * @brief Explicit K-wave experiments for the captured native Q8 router.
 *
 * Cooperating waves own adjacent subtrees of the same ordered 128-leaf
 * reduction. This scope changes recording only; a retained graph embeds its
 * concrete kernel and launch geometry. It neither reads live rows on the host
 * nor changes the installed Automatic engine policy.
 */
#pragma once
#include <cstdint>

namespace llaminar2::rocm
{
/** @brief Maximum cooperating wave64 waves; tiny K uses only nonempty subtrees. */
enum class RouterQ8WaveTile : uint8_t { Four, Eight, Sixteen };

/** @param tile Explicit experiment. @return Wave count, or zero for an invalid enum. */
[[nodiscard]] constexpr int routerQ8WaveCount(RouterQ8WaveTile tile) noexcept
{
    switch (tile)
    {
    case RouterQ8WaveTile::Four: return 4;
    case RouterQ8WaveTile::Eight: return 8;
    case RouterQ8WaveTile::Sixteen: return 16;
    }
    return 0;
}

/** @brief Nestable worker-local native recording choice with checked resources. */
class ScopedRouterQ8WaveTile final
{
public:
    /** @brief Select an explicit native K-wave experiment until scope exit.
     * @param tile Four waves preserve the original native candidate.
     * @throws std::invalid_argument For an unknown tile, or a larger tile without
     * an enclosing explicit NativeWmma recording scope. */
    explicit ScopedRouterQ8WaveTile(RouterQ8WaveTile tile);
    /** @brief Restore the enclosing choice; retained graphs remain unchanged. */
    ~ScopedRouterQ8WaveTile();
    ScopedRouterQ8WaveTile(const ScopedRouterQ8WaveTile &) = delete;
    ScopedRouterQ8WaveTile &operator=(const ScopedRouterQ8WaveTile &) = delete;
private:
    RouterQ8WaveTile previous_;
};
}
