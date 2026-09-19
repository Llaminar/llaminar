/**
 * @file MoEOverlayNodeLocalRouteTransport.h
 * @brief Typed transport policy for rank-local ExpertOverlay route publication.
 *
 * A distributed ExpertOverlay may resume its continuation on several GPUs in
 * one process.  Homogeneous domains with any driver-reported P2P opportunity
 * retain their native NCCL/RCCL collective because that library owns topology
 * selection.  A homogeneous domain with no P2P, or a mixed CUDA/ROCm domain
 * that no native collective can span, uses the captured mapped sparse fabric.
 * The decision is made once before graph construction and becomes part of the
 * declarative graph contract; replay never probes topology or changes paths.
 */

#pragma once

#include "backends/DeviceId.h"
#include "backends/PeerAccessCoverage.h"

#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>

namespace llaminar2
{
    /** Rank-local physical publication selected before graph construction. */
    enum class MoEOverlayNodeLocalRouteTransport
    {
        Unresolved,       ///< Invalid for a multi-GPU continuation graph.
        NativeCollective, ///< Rooted NCCL/RCCL reduction and compact broadcast.
        MappedSparse      ///< Device-epoch sparse publication through mapped pages.
    };

    /** Logical graph phase that owns one continuation-route publication. */
    enum class MoEOverlayNodeLocalRoutePhase
    {
        OrdinaryPrefill, ///< Multi-token prompt ingestion, excluding MTP verification.
        Decode           ///< Serial decode, grouped verification, and MTP sidecars.
    };

    /**
     * @brief Immutable phase-aware transport policy for one continuation cell.
     *
     * Sparse route publication has a different economy curve at prefill and
     * decode shapes. Keeping both choices in one typed value prevents a rank,
     * child runner, or graph builder from silently applying a prefill result to
     * decode. A policy is either wholly unresolved or wholly resolved.
     */
    struct MoEOverlayNodeLocalRouteTransportPolicy
    {
        /** Transport embedded in ordinary prefill graphs. */
        MoEOverlayNodeLocalRouteTransport ordinary_prefill =
            MoEOverlayNodeLocalRouteTransport::Unresolved;

        /** Transport embedded in decode, verifier, and sidecar graphs. */
        MoEOverlayNodeLocalRouteTransport decode =
            MoEOverlayNodeLocalRouteTransport::Unresolved;

        /**
         * @brief Return whether both graph phases have an explicit transport.
         * @return True only for a complete, graph-buildable policy.
         */
        [[nodiscard]] constexpr bool resolved() const noexcept
        {
            return ordinary_prefill !=
                       MoEOverlayNodeLocalRouteTransport::Unresolved &&
                   decode !=
                       MoEOverlayNodeLocalRouteTransport::Unresolved;
        }

        /**
         * @brief Return whether the policy is the default unconfigured value.
         * @return True only when neither phase has been selected.
         */
        [[nodiscard]] constexpr bool unresolved() const noexcept
        {
            return ordinary_prefill ==
                       MoEOverlayNodeLocalRouteTransport::Unresolved &&
                   decode ==
                       MoEOverlayNodeLocalRouteTransport::Unresolved;
        }

        /**
         * @brief Resolve the exact transport embedded in one graph family.
         * @param phase Typed graph phase; grouped MTP work is decode.
         * @return Transport selected before graph construction.
         */
        [[nodiscard]] constexpr MoEOverlayNodeLocalRouteTransport forPhase(
            MoEOverlayNodeLocalRoutePhase phase) const noexcept
        {
            return phase == MoEOverlayNodeLocalRoutePhase::OrdinaryPrefill
                       ? ordinary_prefill
                       : decode;
        }

        /**
         * @brief Return whether any retained graph needs the mapped fabric.
         * @return True when prefill or decode embeds mapped sparse transport.
         */
        [[nodiscard]] constexpr bool requiresMappedExchange() const noexcept
        {
            return ordinary_prefill ==
                       MoEOverlayNodeLocalRouteTransport::MappedSparse ||
                   decode ==
                       MoEOverlayNodeLocalRouteTransport::MappedSparse;
        }

        friend constexpr bool operator==(
            const MoEOverlayNodeLocalRouteTransportPolicy &,
            const MoEOverlayNodeLocalRouteTransportPolicy &) noexcept =
            default;
    };

    /**
     * @brief Default policy for no-P2P dense root publication.
     *
     * Production retains the native collective until topology calibration has
     * proved a positive mapped-publication crossover. On the reference
     * no-P2P CUDA topology, mapped publication remained slower even for a
     * 7.4 MiB prefill payload. Zero therefore means "not selected by default";
     * callers with measured topology evidence may supply a positive threshold
     * to @ref shouldUseMoEOverlayMappedDensePublication.
     */
    inline constexpr std::size_t
        kMoEOverlayMappedDensePublicationMinimumBytes = 0u;

    /**
     * @brief Return a stable transport name for diagnostics and PerfStats.
     * @param transport Transport value to format.
     * @return Lower-case stable identifier.
     */
    [[nodiscard]] constexpr const char *
    moeOverlayNodeLocalRouteTransportName(
        MoEOverlayNodeLocalRouteTransport transport) noexcept
    {
        switch (transport)
        {
        case MoEOverlayNodeLocalRouteTransport::Unresolved:
            return "unresolved";
        case MoEOverlayNodeLocalRouteTransport::NativeCollective:
            return "native_collective";
        case MoEOverlayNodeLocalRouteTransport::MappedSparse:
            return "mapped_sparse";
        }
        return "invalid";
    }

    /**
     * @brief Select the production route transport for one exact GPU cell.
     *
     * Native collectives win for both partial and complete P2P.  This is
     * intentionally conservative: NCCL/RCCL can exploit an asymmetric or
     * partially connected topology, whereas forcing every byte through host
     * memory would discard that opportunity.  Mapped sparse transport bypasses
     * the native library only when the homogeneous matrix proves `None`.
     * Mixed CUDA/ROCm cells cannot have a shared native collective and use the
     * node-local mapped protocol regardless of per-backend matrices.
     *
     * @param devices Distinct process-local GPU endpoints in the continuation.
     * @param homogeneous_coverage Driver result for a homogeneous cell; empty
     *        is required for a heterogeneous cell.
     * @return Immutable graph transport policy.
     * @throws std::invalid_argument for an incomplete or contradictory policy.
     */
    [[nodiscard]] inline MoEOverlayNodeLocalRouteTransportPolicy
    selectMoEOverlayNodeLocalRouteTransportPolicy(
        std::span<const DeviceId> devices,
        std::optional<PeerAccessCoverage> homogeneous_coverage)
    {
        if (devices.size() < 2u)
        {
            throw std::invalid_argument(
                "node-local MoE route transport requires at least two devices");
        }

        const DeviceType first_type = devices.front().type;
        bool homogeneous = true;
        for (std::size_t index = 0; index < devices.size(); ++index)
        {
            if (!devices[index].is_gpu())
            {
                throw std::invalid_argument(
                    "node-local MoE route transport accepts GPU endpoints only");
            }
            for (std::size_t prior = 0; prior < index; ++prior)
            {
                if (devices[prior] == devices[index])
                {
                    throw std::invalid_argument(
                        "node-local MoE route transport devices must be distinct");
                }
            }
            homogeneous = homogeneous && devices[index].type == first_type;
        }

        if (!homogeneous)
        {
            if (homogeneous_coverage.has_value())
            {
                throw std::invalid_argument(
                    "heterogeneous MoE route transport cannot carry one homogeneous P2P matrix");
            }
            return {
                .ordinary_prefill =
                    MoEOverlayNodeLocalRouteTransport::MappedSparse,
                .decode = MoEOverlayNodeLocalRouteTransport::MappedSparse,
            };
        }
        if (!homogeneous_coverage)
        {
            throw std::invalid_argument(
                "homogeneous MoE route transport requires driver-backed P2P coverage");
        }
        if (*homogeneous_coverage == PeerAccessCoverage::None)
        {
            /* Without a peer edge the native collective necessarily stages
             * routed rows through its host transport.  The capture-stable
             * mapped exchange avoids that bounce for both prompt and decode
             * shapes.  P2P-capable cells remain native below. */
            return {
                .ordinary_prefill =
                    MoEOverlayNodeLocalRouteTransport::MappedSparse,
                .decode =
                    MoEOverlayNodeLocalRouteTransport::MappedSparse,
            };
        }
        return {
            .ordinary_prefill =
                MoEOverlayNodeLocalRouteTransport::NativeCollective,
            .decode = MoEOverlayNodeLocalRouteTransport::NativeCollective,
        };
    }

    /**
     * @brief Select mapped publication for one dense rooted payload.
     *
     * Sparse route transport and dense continuation publication have
     * different economy curves. A no-P2P topology still uses mapped sparse
     * packets, while decode-sized dense rows stay on the lower-fixed-cost
     * native collective. A calibrated positive crossover may select the mapped
     * device-epoch channel for large dense blocks. A P2P-capable native route
     * can never select the host-mapped channel, regardless of payload size.
     *
     * @param route_transport Topology-owned route transport selected before
     *        graph construction.
     * @param payload_bytes Exact fixed payload embedded in this graph.
     * @param minimum_payload_bytes Tunable crossover supplied by topology
     *        calibration; defaults to the conservative production value.
     * @return True only when mapped dense publication is both topology-safe
     *         and economical for this graph identity.
     */
    [[nodiscard]] constexpr bool
    shouldUseMoEOverlayMappedDensePublication(
        MoEOverlayNodeLocalRouteTransport route_transport,
        std::size_t payload_bytes,
        std::size_t minimum_payload_bytes =
            kMoEOverlayMappedDensePublicationMinimumBytes) noexcept
    {
        return route_transport ==
                   MoEOverlayNodeLocalRouteTransport::MappedSparse &&
               minimum_payload_bytes > 0u &&
               payload_bytes >= minimum_payload_bytes;
    }
} // namespace llaminar2
