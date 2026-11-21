/**
 * @file ROCmRouterQ8Policy.h
 * @brief Device-free admission for exact Q8 router engines and K-wave tiles.
 *
 * Automatic admission binds whole-matrix capacity and ownership, never a host
 * observation of live rows. The installed whole-row M512/K2560/E512 tile is
 * economical through empty and one-row retained replays in both paired bases.
 * Compact-owned matrices require their own evidence even when their local
 * capacity happens to match. Short verifier keys additionally bind the existing
 * exact-hosted extent to the original count pointer, and submission revalidates
 * that binding. Diagnostic choices cannot alter an Automatic tile.
 */
#pragma once
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "config/MoERouterArithmeticPolicy.h"
#include "kernels/common/CapturedVerifierExtent.h"
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

    /** @brief Authenticate original row authority at native submission.
     * @param rows Original submitted matrix rows, before compact partitioning.
     * @return Whether capacity and any retained exact-hosted count identity match.
     * Guarded M512 and explicit candidates retain their established live-prefix contract. */
    [[nodiscard]] bool matchesRows(const DeviceRowRange &rows) const noexcept
    {
        const auto identity=rows.captureIdentity();
        return rows.physicalRows()==shape_.capacity && identity.first_physical_row==0 &&
            identity.repeats_per_row==1 &&
            (!extent_.isExact() || extent_.matches(DeviceId::rocm(device_),rows));
    }

    /** @brief Resolve one first-class engine without querying or touching a device.
     * @param mode Explicit recording policy. @param requested Diagnostic native tile.
     * @param hardware Immutable device inventory entry. @param shape Original geometry.
     * @param rows Original whole-matrix row authority, before any compact partition.
     * @param extent Existing captured transaction fact; ordinary capacity supplies an empty fact.
     * @return Validated selection, independent of any device-owned live count.
     * @throws std::invalid_argument For invalid geometry/enums or unsupported forced WMMA. */
    [[nodiscard]] static RouterQ8PrefillSelection resolve(RouterQ8PrefillMode mode,
        RouterQ8WaveTile requested, const ROCmDeviceHardware &hardware, RouterQ8PrefillShape shape,
        const DeviceRowRange &rows, CapturedVerifierExtent extent)
    {
        const auto identity=rows.captureIdentity();
        if (hardware.ordinal < 0 || shape.capacity <= 0 || shape.width <= 0 || shape.width % 32 || shape.experts <= 0 ||
            rows.physicalRows()!=shape.capacity || identity.first_physical_row!=0 || identity.repeats_per_row!=1 ||
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
            if(shape.ownership==RouterQ8RowOwnership::WholeRows &&
                extent.matches(DeviceId::rocm(hardware.ordinal),rows) && measuredVerifierShape(shape) &&
                supportsRouterQ8Tile(hardware,RouterQ8WaveTile::Sixteen))
                return {hardware.ordinal,shape,RouterQ8WaveTile::Sixteen,extent};
            return {hardware.ordinal,shape,std::nullopt};
        }
        throw std::invalid_argument("unknown Q8 router recording policy");
    }
private:
    /** @brief Admit only complete-operation wins measured at every M2..16 in both bases.
     * @param shape Original immutable geometry, including the paired Q8 basis.
     * @return Whether sixteen waves won its median and at least22 of31 pairs.
     * Near-tie M4/K2560, Original M5/K2048 and both M5/K3072 remain ordered. */
    [[nodiscard]] static bool measuredVerifierShape(const RouterQ8PrefillShape &shape) noexcept
    {
        if(shape.capacity<5 || shape.capacity>16)return false;
        if(shape.width==2048 && shape.experts==256)
            return shape.basis==RouterQ8Basis::SignedHadamard32 || shape.capacity>=6;
        if(shape.width==2560 && shape.experts==512)return true;
        if(shape.width==3072 && shape.experts==256)
            return shape.capacity>=6;
        return shape.width==8192 && shape.experts==512 && shape.capacity>=9;
    }

    /** @brief Only resolve() may construct a shape-bound, resource-checked selection.
     * @param device Original participant. @param shape Validated whole-matrix geometry.
     * @param tile Native tile or the ordered engine. @param extent Qualifying exact-row fact, if required. */
    RouterQ8PrefillSelection(int device, RouterQ8PrefillShape shape, std::optional<RouterQ8WaveTile> tile,
        CapturedVerifierExtent extent={})
        : device_(device), shape_(shape), tile_(tile), extent_(extent) {}
    int device_;
    RouterQ8PrefillShape shape_;
    std::optional<RouterQ8WaveTile> tile_;
    CapturedVerifierExtent extent_; ///< Exact original count identity, retained only by short-width Automatic.
};
}
