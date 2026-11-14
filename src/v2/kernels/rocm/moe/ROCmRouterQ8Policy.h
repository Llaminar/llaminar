/**
 * @file ROCmRouterQ8Policy.h
 * @brief Device-free admission for exact Q8 router engines and K-wave tiles.
 *
 * Automatic admission binds whole-matrix capacity and ownership, never a host
 * observation of live rows. The installed whole-row M512/K2560/E512 tile is
 * economical through empty and one-row retained replays in both paired bases.
 * Compact-owned matrices require their own evidence even when their local
 * capacity happens to match. Diagnostic choices cannot alter an Automatic tile.
 */
#pragma once
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "config/MoERouterArithmeticPolicy.h"
#include "ROCmRouterQ8WaveTuning.h"
#include <optional>
#include <stdexcept>

namespace llaminar2::rocm
{
/** @brief Recording policy; retained execution never consults this host value. */
enum class RouterQ8PrefillMode { Automatic, OrderedDot4, NativeWmma };
/** @brief Physical output ownership, independent of the local row count. */
enum class RouterQ8RowOwnership { WholeRows, CompactOwned };
/** @brief Immutable admission dimensions, using original whole-matrix capacity. */
struct RouterQ8PrefillShape
{
    int capacity, width, experts;
    RouterQ8Basis basis;
    RouterQ8RowOwnership ownership;
};

/** @brief Check the exact native instruction target and requested launch resources.
 * @param hardware Admitted immutable device metadata. @param tile Maximum K waves.
 * @return Whether this device can execute the requested native tile. */
[[nodiscard]] inline bool supportsRouterQ8Tile(const ROCmDeviceHardware &hardware, RouterQ8WaveTile tile)
{
    const int waves = routerQ8WaveCount(tile);
    return waves > 0 && (hardware.isa_target == "gfx1100" || hardware.isa_target.starts_with("gfx1100:")) &&
        hardware.maximum_threads_per_block >= 64 * waves && hardware.lds_per_block >= 1024u * waves;
}

/** @brief Resolved engine and its bound shape; an ordered engine has no native tile. */
class RouterQ8PrefillSelection final
{
public:
    /** @return Whether the resolved engine is native WMMA. */
    [[nodiscard]] bool usesWmma() const noexcept { return tile_.has_value(); }
    /** @return Resolved native tile. @throws std::logic_error For an ordered engine. */
    [[nodiscard]] RouterQ8WaveTile nativeTile() const
    {
        if (!tile_) throw std::logic_error("ordered Q8 router has no native launch tile");
        return *tile_;
    }
    /** @return Original admitted geometry and ownership. */
    [[nodiscard]] const RouterQ8PrefillShape &shape() const noexcept { return shape_; }
    /** @return Admitted visible ordinal, frozen with the launch geometry. */
    [[nodiscard]] int device() const noexcept { return device_; }

    /** @brief Resolve one first-class engine without querying or touching a device.
     * @param mode Explicit recording policy. @param requested Diagnostic native tile.
     * @param hardware Immutable device inventory entry. @param shape Original geometry.
     * @return Validated selection, independent of any device-owned live count.
     * @throws std::invalid_argument For invalid geometry/enums or unsupported forced WMMA. */
    [[nodiscard]] static RouterQ8PrefillSelection resolve(RouterQ8PrefillMode mode,
        RouterQ8WaveTile requested, const ROCmDeviceHardware &hardware, RouterQ8PrefillShape shape)
    {
        if (hardware.ordinal < 0 || shape.capacity <= 0 || shape.width <= 0 || shape.width % 32 || shape.experts <= 0 ||
            (shape.basis != RouterQ8Basis::Original && shape.basis != RouterQ8Basis::SignedHadamard32) ||
            (shape.ownership != RouterQ8RowOwnership::WholeRows && shape.ownership != RouterQ8RowOwnership::CompactOwned) ||
            !routerQ8WaveCount(requested))
            throw std::invalid_argument("Q8 router admission requires valid device, ownership, basis and block32 geometry");
        switch (mode)
        {
        case RouterQ8PrefillMode::OrderedDot4: return {hardware.ordinal,shape,std::nullopt};
        case RouterQ8PrefillMode::NativeWmma:
            if (!supportsRouterQ8Tile(hardware,requested))
                throw std::invalid_argument("native Q8 router requires gfx1100 and resources for the requested K-wave tile");
            return {hardware.ordinal,shape,requested};
        case RouterQ8PrefillMode::Automatic:
            if (shape.ownership == RouterQ8RowOwnership::WholeRows && shape.capacity == 512 &&
                shape.width == 2560 && shape.experts == 512 && supportsRouterQ8Tile(hardware,RouterQ8WaveTile::Sixteen))
                return {hardware.ordinal,shape,RouterQ8WaveTile::Sixteen};
            return {hardware.ordinal,shape,std::nullopt};
        }
        throw std::invalid_argument("unknown Q8 router recording policy");
    }
private:
    /** @brief Only resolve() may construct a shape-bound, resource-checked selection. */
    RouterQ8PrefillSelection(int device, RouterQ8PrefillShape shape, std::optional<RouterQ8WaveTile> tile)
        : device_(device), shape_(shape), tile_(tile) {}
    int device_;
    RouterQ8PrefillShape shape_;
    std::optional<RouterQ8WaveTile> tile_;
};
}
