/**
 * @file MoEOverlayHistogramPublisher.h
 * @brief Device-free publication boundary for authoritative routing evidence.
 *
 * Distributed ExpertOverlay placement has exactly one histogram authority: the
 * rank owning the continuation/router root. This interface lets the maintenance
 * state machine publish or receive that frozen window without depending on a
 * particular network transport. Production uses a private MPI lane; CPU unit
 * tests use an adversarial in-process publisher.
 */

#pragma once

#include "MoEOverlayDistributedResidencyProtocol.h"

#include <memory>
#include <string>

namespace llaminar2
{
    /** @brief Asynchronous coordinator-to-peer frozen-histogram publication. */
    class IMoEOverlayHistogramPublisher
    {
    public:
        virtual ~IMoEOverlayHistogramPublisher() = default;

        /**
         * @brief Begin sending one coordinator-owned immutable window.
         * @param window Valid exact-model routing evidence.
         * @param error Receives a precise validation/transport diagnostic.
         * @return True after the asynchronous publication owns its buffers.
         */
        virtual bool beginPublish(
            const DecodeExpertHistogramWindow &window,
            std::string *error = nullptr) = 0;

        /**
         * @brief Progress one active send or receive without waiting.
         * @param received_window Peer output on Ready; empty on coordinator.
         * @param error Receives a precise transport/authentication diagnostic.
         * @return Pending, Ready, or Failed for the current generation.
         */
        virtual MoEOverlayResidencyWaveProgress poll(
            std::shared_ptr<const DecodeExpertHistogramWindow>
                *received_window,
            std::string *error = nullptr) = 0;

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
