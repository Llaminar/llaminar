/**
 * @file MoEOverlayResidencyProposalPublisher.h
 * @brief Device-free publication boundary for authoritative residency plans.
 *
 * Distributed ExpertOverlay placement has exactly one histogram authority: the
 * rank owning the continuation/router root. This interface publishes that
 * authority's complete immutable execution proposal rather than asking peers
 * to reconstruct policy from a histogram. Production uses a private MPI lane;
 * CPU unit tests use an adversarial in-process publisher.
 */

#pragma once

#include "MoEOverlayDistributedResidencyProtocol.h"

#include <cstdint>
#include <memory>
#include <string>

namespace llaminar2
{
    /** @brief Asynchronous coordinator-to-peer canonical-plan publication. */
    class IMoEOverlayResidencyProposalPublisher
    {
    public:
        virtual ~IMoEOverlayResidencyProposalPublisher() = default;

        /**
         * @brief Begin sending one coordinator-owned immutable proposal.
         * @param proposal Valid exact-model execution proposal.
         * @param error Receives a precise validation/transport diagnostic.
         * @return True after the asynchronous publication owns its buffers.
         */
        virtual bool beginPublish(
            const MoEOverlayDistributedResidencyProposal &proposal,
            std::string *error = nullptr) = 0;

        /**
         * @brief Progress one active send or receive without waiting.
         * @param received_proposal Peer output on Ready; empty on coordinator.
         * @param error Receives a precise transport/authentication diagnostic.
         * @return Pending, Ready, or Failed for the current generation.
         */
        virtual MoEOverlayResidencyWaveProgress poll(
            std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
                *received_proposal,
            std::string *error = nullptr) = 0;

        /**
         * @brief Acknowledge one peer proposal after semantic adoption succeeds.
         * @param histogram_generation Exact decoded evidence generation.
         * @param error Receives a precise state or transport diagnostic.
         * @return True only after the acknowledgement owns its async resources.
         *
         * Packet authentication is not enough to cross the publication edge.
         * The peer must first reconstruct the exact transaction against its
         * local topology and verify the coordinator execution fingerprint.
         */
        virtual bool acceptReceivedProposal(
            std::uint64_t histogram_generation,
            std::string *error = nullptr) = 0;

        /**
         * @brief Fatally reject a decoded proposal that cannot be adopted.
         * @param histogram_generation Exact rejected evidence generation.
         * @param diagnostic Stable semantic-adoption failure reason.
         *
         * A production distributed publisher terminates its private rank lane:
         * after bytes have arrived there is no safe process-local recovery that
         * would leave every participant in the same collective order. Device-
         * free publishers retain the rejection for adversarial unit tests.
         */
        virtual void rejectReceivedProposal(
            std::uint64_t histogram_generation,
            std::string diagnostic) = 0;

        /**
         * @brief Arm the peer's next fixed-buffer receive without waiting.
         * @param error Receives a precise state or transport diagnostic.
         * @return True only after the next receive owns its buffer.
         */
        virtual bool armReceive(std::string *error = nullptr) = 0;

        /** @brief Drain/cancel setup traffic during explicit runner teardown. */
        virtual void stopAndDrain() = 0;

        /** @return Whether this process is the declared routing coordinator. */
        [[nodiscard]] virtual bool isCoordinator() const noexcept = 0;
    };
} // namespace llaminar2
