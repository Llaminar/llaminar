/**
 * @file GPUQ8ResidualNative.h
 * @brief Exact prepared-source validation and cold borrowed IQ views for TwoTerm.
 *
 * Physical descriptors remain borrowed from the original prepared engine. No
 * allocation decision or weight conversion lives here. Optional allocation
 * planes are included in consumed spans even when the current decoder ignores
 * them, so writable scratch cannot alias a migration slot's resident metadata.
 */
#pragma once
#include "kernels/common/NativeEmbeddingIQTables.h"
#include <limits>
#include "kernels/common/Q8ResidualCorrection.h"
namespace llaminar2
{
    /** @return Whether this actual physical decoder consumes immutable IQ grids. */
    inline constexpr bool q8ResidualNeedsIQTables(std::uint8_t codebook) noexcept
    {
        return codebook >= 11 && codebook <= 17;
    }
    /**
     * @brief Visit one compiled physical decoder, before any activation publication.
     * @tparam Visitor Host setup/launch visitor with a codebook template parameter.
     * @param codebook Actual prepared execution format, separate from source identity.
     * @param visitor Receives the exact compile-time decoder.
     * @throws std::invalid_argument For every uncompiled physical format.
     *
     * Validation and kernel instantiation use this same switch. Extending a source
     * catalog cannot silently permit publication without a compiled projection.
     */
    template <class Visitor> void visitQ8ResidualNativeCodebook(std::uint8_t codebook, Visitor &&visitor)
    {
        switch (codebook)
        {
        case 0:
            visitor.template operator()<0>();
            return;
        case 4:
            visitor.template operator()<4>();
            return;
        case 5:
            visitor.template operator()<5>();
            return;
        case 6:
            visitor.template operator()<6>();
            return;
        case 7:
            visitor.template operator()<7>();
            return;
        case 8:
            visitor.template operator()<8>();
            return;
        case 9:
            visitor.template operator()<9>();
            return;
        case 10:
            visitor.template operator()<10>();
            return;
        case 11:
            visitor.template operator()<11>();
            return;
        case 12:
            visitor.template operator()<12>();
            return;
        case 13:
            visitor.template operator()<13>();
            return;
        case 14:
            visitor.template operator()<14>();
            return;
        case 15:
            visitor.template operator()<15>();
            return;
        case 16:
            visitor.template operator()<16>();
            return;
        case 17:
            visitor.template operator()<17>();
            return;
        case 19:
            visitor.template operator()<19>();
            return;
        case 23:
            visitor.template operator()<23>();
            return;
        case 24:
            visitor.template operator()<24>();
            return;
        default:
            throw std::invalid_argument("TwoTerm GPU execution codebook has no compiled native projection");
        }
    }
    /** @brief Authenticate original source, physical decoder and every metadata plane.
     * @param d Original prepared matrix descriptor, without pointer dereferences.
     * @param m Physical retained rows.
     * @param n Required local output width.
     * @param k Required local reduction width in complete native32 blocks.
     * @return Exact consumed physical plane geometry.
     * @throws std::invalid_argument Before publication for an unsupported or malformed descriptor.
     */
    inline NativeVnniMigrationStableDeviceFormat requireQ8ResidualNativeMatrix(
        const DeviceNativeVNNIMatrixDesc &d, int m, int n, int k)
    {
        visitQ8ResidualNativeCodebook(d.codebook_id, [&]<std::uint8_t>() {});
        const auto &source = requireQ8ResidualSourceFormat(
            NativeVnniSourceIdentity{d.source_codebook_id, d.source_is_superblock != 0,
                                     d.source_identity_present != 0},
            m, n, k);
        if (!d.valid() || d.n != n || d.k != k || d.blocks_per_row != static_cast<unsigned>(k / 32) ||
            !deviceVnniExecutionCompatibleWithSource(source, d.codebook_id) || d.reserved ||
            d.reserved_tail || d.source_is_superblock > 1 || d.source_identity_present != 1 ||
            d.allocation_has_mins > 1 || d.allocation_has_emins > 1)
            throw std::invalid_argument("TwoTerm GPU prepared source/execution identity or shape mismatch");
        auto physical = canonicalDeviceVnniFormat(source);
        if (d.codebook_id != physical.codebook_id)
            physical = {d.codebook_id, 32, d.codebook_id == kNativeVnniExpandedInt8MinCodebook, false};
        // The original constructor permits a declared reusable superset. Its
        // real capacity/plane flags remain authoritative; source geometry only
        // proves that the current live decoder fits, never invents a slot size.
        const bool allocated = d.allocation_payload_bytes_per_block != 0;
        if ((physical.is_asymmetric && !d.mins) || (physical.has_emins && !d.emins) ||
            (!allocated && (bool(d.mins) != physical.is_asymmetric || bool(d.emins) != physical.has_emins ||
                            d.allocation_has_mins || d.allocation_has_emins)) ||
            (allocated && (d.allocation_payload_bytes_per_block < physical.payload_bytes_per_block ||
                           (physical.is_asymmetric && !d.allocation_has_mins) ||
                           (physical.has_emins && !d.allocation_has_emins) ||
                           bool(d.mins) != bool(d.allocation_has_mins) ||
                           bool(d.emins) != bool(d.allocation_has_emins))))
            throw std::invalid_argument(
                "TwoTerm GPU prepared source has malformed native metadata/allocation planes");
        // Bounds on M*N and M*K do not bound N*(K/32)*slot capacity.
        // Prove every original weight-plane span before the factory forms any
        // byte interval; a wrapped alias span could otherwise admit a writer
        // inside the same original allocation. No pointer is dereferenced here.
        constexpr auto maximum = std::numeric_limits<std::size_t>::max();
        const auto columns = static_cast<std::size_t>(n);
        const auto blocksPerRow = static_cast<std::size_t>(d.blocks_per_row);
        if (columns > maximum / blocksPerRow)
            throw std::invalid_argument("TwoTerm GPU native weight-block extent exceeds addressable storage");
        const auto weightBlocks = columns * blocksPerRow;
        const auto payloadCapacity = allocated ? d.allocation_payload_bytes_per_block
                                               : physical.payload_bytes_per_block;
        if (weightBlocks > maximum / payloadCapacity ||
            weightBlocks > maximum / sizeof(std::uint16_t) ||
            (d.emins && weightBlocks > maximum / sizeof(std::uint32_t)))
            throw std::invalid_argument("TwoTerm GPU native weight-plane byte extent exceeds addressable storage");
        return physical;
    }
    /** @brief Internal launcher retaining a cold generation-bound immutable table view.
     * @param args Unchanged public projection ABI.
     * @param stream Exact non-default producer stream.
     * @param tables Backend-owned codebook addresses authenticated by the prepared engine.
     * No launch initializes or rebinds these addresses; the original backend generation
     * must outlive every prepared engine and retained graph borrowing them.
     */
    void cudaQ8ResidualCorrectedProjection(const GPUQ8ResidualCorrectionArgs &args, void *stream,
                                           NativeEmbeddingIQTables tables);
    /** @copydoc cudaQ8ResidualCorrectedProjection */
    void rocmQ8ResidualCorrectedProjection(const GPUQ8ResidualCorrectionArgs &args, void *stream,
                                           NativeEmbeddingIQTables tables);
} // namespace llaminar2
